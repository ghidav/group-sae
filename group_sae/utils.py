from __future__ import annotations

import json
import os
import re
from importlib.metadata import version
from typing import Any, Dict, Iterable, Type, TypeVar, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate.utils import send_to_device
from matplotlib.colors import LinearSegmentedColormap
from sae_lens import SAE
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformerConfig
from transformers import PreTrainedModel

from group_sae.config import TrainConfig
from group_sae.export import to_sae_lens
from group_sae.hooks import forward_hook_wrapper

palette = ["#FFC533", "#f48c06", "#DD5703", "#d00000", "#6A040F"]
cmap = LinearSegmentedColormap.from_list("paper", palette)

T = TypeVar("T")


MODEL_MAP = {
    "pythia-160m": {
        "short_name": "pythia_160m",
        "n_layers": 12,
        "d_model": 768,
        "A": 207.62,
        "T": 94.37,
    },
    "pythia-410m": {
        "short_name": "pythia_410m",
        "n_layers": 24,
        "d_model": 1024,
        "A": 704.64,
        "T": 167.77,
    },
    "pythia-1b": {
        "short_name": "pythia_1b",
        "n_layers": 16,
        "d_model": 2048,
        "A": 1744.83,
        "T": 671.09,
    },
    "pythia-1.4b": {
        "short_name": "pythia_1.4b",
        "n_layers": 32,
        "d_model": 2048,
        "A": None,
        "T": None,
    },
}


def load_amds(size, include_baseline=False):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(package_dir, "groups", f"pythia-{size}.json")
    clusters = json.load(open(file_path))
    nl = MODEL_MAP[f"pythia-{size}"]["n_layers"]
    A = MODEL_MAP[f"pythia-{size}"]["A"]
    T = MODEL_MAP[f"pythia-{size}"]["T"]
    amd = pd.Series([clusters[str(i)]["amd"] for i in range(1, nl - 1)]).reset_index()
    amd.columns = ["G", "AMD"]
    amd["G"] += 1
    if include_baseline:
        amd = pd.concat([amd, pd.DataFrame([{"G": nl - 1, "AMD": 0}])])
    amd["C"] = A + T * amd["G"]
    return amd


def load_cluster_map(size):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(package_dir, "groups", f"pythia-{size}.json")
    clusters = json.load(open(file_path))
    training_clusters = clusters.pop("training_clusters")

    clusters_grouped = {}
    for cluster in clusters:
        labels = clusters[cluster]["labels"]
        layers_grouped = [[] for _ in range(len(np.unique(labels)))]
        layer_idx = 0
        group_index = 0
        last_label = labels[0]
        layers_grouped[0].append(str(layer_idx))
        for label in labels[1:]:
            layer_idx += 1
            if label != last_label:
                group_index += 1
                layers_grouped[group_index].append(str(layer_idx))
                last_label = label
            else:
                layers_grouped[group_index].append(str(layer_idx))
        clusters_grouped[cluster] = layers_grouped

    clusters_to_saes = {}
    for cluster in clusters_grouped:
        saes = []
        for group in clusters_grouped[cluster]:
            if len(group) == 1:
                saes.extend([f"layers.{group[0]}"])
            for cluster_name, layers in training_clusters.items():
                if group == layers:
                    saes.extend([cluster_name for _ in range(len(layers))])
                    break
        clusters_to_saes[cluster] = saes

    return clusters_to_saes


def load_training_clusters(size):
    package_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(package_dir, "groups", f"pythia-{size}.json")
    clusters = json.load(open(file_path))
    return clusters["training_clusters"]


def get_lr_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Get the learning rate scheduler from the Transformers library.

    Args:
        scheduler_name (str): Name of the scheduler to use, one of
            'constant', 'cosine', or 'linear'.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """
    from transformers import (
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    elif scheduler_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            "The supported schedulers are 'constant', 'linear', and 'cosine'."
        )
    return scheduler


class L1Scheduler:
    def __init__(
        self,
        l1_warmup_steps: float,
        total_steps: int,
        final_l1_coefficient: float,
    ):
        self.l1_warmup_steps = l1_warmup_steps
        # assume using warm-up
        if self.l1_warmup_steps != 0:
            self.current_l1_coefficient = 0.0
        else:
            self.current_l1_coefficient = final_l1_coefficient

        self.final_l1_coefficient = final_l1_coefficient

        self.current_step = 0
        self.total_steps = total_steps
        assert isinstance(self.final_l1_coefficient, float | int)

    def __repr__(self) -> str:
        return (
            f"L1Scheduler(final_l1_value={self.final_l1_coefficient}, "
            f"l1_warmup_steps={self.l1_warmup_steps}, "
            f"total_steps={self.total_steps})"
        )

    def step(self):
        """
        Updates the l1 coefficient of the sparse autoencoder.
        """
        step = self.current_step
        if step < self.l1_warmup_steps:
            self.current_l1_coefficient = self.final_l1_coefficient * (
                (1 + step) / self.l1_warmup_steps
            )  # type: ignore
        else:
            self.current_l1_coefficient = self.final_l1_coefficient  # type: ignore

        self.current_step += 1

    def state_dict(self):
        """State dict for serializing as part of an SAETrainContext."""
        return {
            "l1_warmup_steps": self.l1_warmup_steps,
            "total_steps": self.total_steps,
            "current_l1_coefficient": self.current_l1_coefficient,
            "final_l1_coefficient": self.final_l1_coefficient,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads all state apart from attached SAE."""
        for k in state_dict:
            setattr(self, k, state_dict[k])


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> "CycleIterator":
        return self


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Get the machine epsilon for the data type
    eps = torch.finfo(points.dtype).eps

    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.clamp(torch.norm(points - guess, dim=1), min=eps)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


@torch.inference_mode()
def resolve_widths(
    cfg: TrainConfig,
    model: PreTrainedModel,
    module_names: list[str],
    dataloader: DataLoader | None = None,
) -> dict[str, torch.Size]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {model.get_submodule(name): name for name in module_names}
    hidden_dict: Dict[str, Tensor] = {}

    hook = forward_hook_wrapper(
        cfg.hook,
        module_to_name=module_to_name,
        hidden_dict=hidden_dict,
    )

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    if dataloader is None:
        dummy = model.dummy_inputs
    else:
        dummy = next(iter(dataloader))
    dummy = send_to_device(dummy, model.device)
    try:
        model(**dummy)
    finally:
        for handle in handles:
            handle.remove()

    shapes = {name: hidden.shape for name, hidden in hidden_dict.items()}
    return shapes


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode


def get_device_for_block(layer: int, cfg: HookedTransformerConfig, device: str | None = None):
    """Equally and sequentially distribute the blocks across the devices"""
    if device is None:
        if cfg.device is None:
            raise ValueError("No device specified in either the config or the function")
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    devices = list(range(cfg.n_devices))
    layers_split = np.array_split(range(cfg.n_layers), cfg.n_devices)
    for i, layers in enumerate(layers_split):
        if layer in layers:
            return torch.device(device.type, devices[i])


def chunk_almost_equal_sum(numbers: list[int], num_chunks: int) -> list[list[int]]:
    """
    Divide a list of numbers into `num_chunks` so that
    the sum of the chunks is as equal as possible.

    Args:
        numbers (list[int]): List of numbers to divide.
        num_chunks (int): Number of chunks to divide the numbers into.

    Returns:
        list[list[int]]: List of chunks, each containing a subset of the input numbers.
    """
    # Sort numbers in descending order for better balancing
    numbers = sorted(numbers, reverse=True)

    # Initialize empty chunks and their corresponding sums
    chunks = [[] for _ in range(num_chunks)]
    chunk_sums = [0 for _ in range(num_chunks)]

    # Distribute each number into the chunk with the smallest current sum
    for num in numbers:
        smallest_chunk_index = chunk_sums.index(min(chunk_sums))
        chunks[smallest_chunk_index].append(num)
        chunk_sums[smallest_chunk_index] += num

    return chunks


def load_saes(
    sae_folder_path: str,
    device: str = "cuda",
    debug: bool = False,
    layer: int | None = None,
    cluster: str | None = None,
    load_from_sae_lens: bool = False,
    model_name: str | None = None,
    dtype: str = "float32",
    dataset_path: str = "NeelNanda/pile-small-tokenized-2b",
):
    from group_sae.sae import Sae, SaeConfig

    dictionaries = {}

    if not os.path.exists(sae_folder_path):
        raise ValueError(f"SAE path {sae_folder_path} does not exist. ")
    else:
        # Load all available SAEs in `sae_folder_path`
        def get_paths(folder_path):
            paths = []
            for path in os.listdir(folder_path):
                sae_path = os.path.join(folder_path, path)
                if load_from_sae_lens:
                    sae_path = os.path.join(sae_path, "sae_lens")
                if os.path.isdir(os.path.join(folder_path, path)):
                    if not os.path.exists(sae_path):
                        raise FileNotFoundError(f"SAE path {sae_path} does not existorch. ")
                    paths.append(sae_path)
            return paths

        baseline_paths = get_paths(sae_folder_path + "/baseline")
        cluster_paths = get_paths(sae_folder_path + "/cluster")

        # Map modules to paths, converting paths to corresponding sae_lens hookpoints
        if cluster is not None:
            if model_name is None:
                raise ValueError("model_name must be specified when cluster is not None")
            CLUSTER_MAP = load_cluster_map(model_name.split("-")[1])
            cluster_layers = CLUSTER_MAP[cluster]
            modules_to_paths = {}
            for layer_num, cluster_layer in enumerate(cluster_layers):
                if "layers." in cluster_layer:
                    for path in baseline_paths:
                        if cluster_layer == path.split(os.sep)[-1]:
                            modules_to_paths[f"blocks.{layer_num}.hook_resid_post"] = path
                            break
                else:
                    for path in cluster_paths:
                        if cluster_layer == path.split(os.sep)[-1]:
                            modules_to_paths[f"blocks.{layer_num}.hook_resid_post"] = path
                            break
        else:
            modules_to_paths = {}
            for path in baseline_paths:
                layer_num = re.findall(r"\d+", path.split(os.sep)[-1])[0]
                if f"layers.{layer_num}" in path:
                    modules_to_paths[f"blocks.{layer_num}.hook_resid_post"] = path

        # Grab only the specified layer, if specified
        if layer is not None:
            modules_to_paths = {
                hook_name: path
                for hook_name, path in modules_to_paths.items()
                if str(layer) == re.findall(r"\d+", hook_name)[0]
            }

        # Load SAEs
        if load_from_sae_lens:
            for hook_name, path in modules_to_paths:
                if debug:
                    print(f"Loading SAE for {hook_name} from {path}")
                dictionaries[hook_name] = SAE.load_from_pretrained(
                    path, device=device, dtype=dtype
                )
        else:
            if model_name is None:
                raise ValueError("model_name must be specified when load_from_sae_lens is False")
            for hook_name, path in modules_to_paths.items():
                if debug:
                    print(f"Loading SAE for {hook_name} from {path}")
                sae = Sae.load_from_disk(path)
                sae_cfg = SaeConfig.load_json(os.path.join(path, "cfg.json"))
                cfg = TrainConfig.load_json(os.path.join(path, "..", "config.json"))
                norm_scaling_factors = torch.load(
                    os.path.join(sae_folder_path, "scaling_factors.pt")
                )
                hookpoint = "layers." + re.findall(r"\d+", hook_name)[0]
                norm_scaling_factor = norm_scaling_factors[hookpoint]
                sae_lens = to_sae_lens(
                    sae=sae,
                    sae_cfg=sae_cfg,
                    model_name=model_name,
                    dataset_path=dataset_path,
                    norm_scaling_factor=norm_scaling_factor,
                    max_seq_len=cfg.max_seq_len,
                    hook_name=hook_name,
                    hook_layer=int(re.findall(r"\d+", hook_name)[0]),
                    dtype=dtype,
                    device=device,
                    sae_lens_version=version("sae_lens"),
                )
                dictionaries[hook_name] = sae_lens
    return dictionaries


def load_saes_by_training_clusters(
    sae_folder_path: str,
    device: str = "cuda",
    debug: bool = False,
    cluster: bool = False,
    load_from_sae_lens: bool = False,
    model_name: str | None = None,
    dtype: str = "float32",
    dataset_path: str = "NeelNanda/pile-small-tokenized-2b",
):
    from group_sae.sae import Sae, SaeConfig

    dictionaries = {}

    if not os.path.exists(sae_folder_path):
        raise ValueError(f"SAE path {sae_folder_path} does not exist. ")
    else:
        # Load all available SAEs in `sae_folder_path`
        def get_paths(folder_path):
            paths = []
            for path in os.listdir(folder_path):
                sae_path = os.path.join(folder_path, path)
                if load_from_sae_lens:
                    sae_path = os.path.join(sae_path, "sae_lens")
                if os.path.isdir(os.path.join(folder_path, path)):
                    if not os.path.exists(sae_path):
                        raise FileNotFoundError(f"SAE path {sae_path} does not existorch. ")
                    paths.append(sae_path)
            return paths

        baseline_paths = get_paths(sae_folder_path + "/baseline")
        cluster_paths = get_paths(sae_folder_path + "/cluster")

        # Map modules to paths, converting paths to corresponding sae_lens hookpoints
        if cluster:
            if model_name is None:
                raise ValueError("model_name must be specified when cluster is not None")
            CLUSTER_MAP = load_training_clusters(model_name.split("-")[1])
            modules_to_paths = {}
            for cluster_id, cluster_layers in CLUSTER_MAP.items():
                for path in cluster_paths:
                    if cluster_id == path.split(os.sep)[-1]:
                        modules_to_paths[cluster_id] = {
                            "sae": path,
                            "layers": ["layers." + layer for layer in cluster_layers],
                        }
                        break
        else:
            modules_to_paths = {}
            for path in baseline_paths:
                layer_num = re.findall(r"\d+", path.split(os.sep)[-1])[0]
                if f"layers.{layer_num}" in path:
                    modules_to_paths[f"layers.{layer_num}"] = {
                        "sae": path,
                        "layers": [f"layers.{layer_num}"],
                    }

        # Load SAEs
        if load_from_sae_lens:
            for cluster_id, sae_dict in modules_to_paths:
                path = sae_dict["sae"]
                if debug:
                    print(f"Loading SAE for {cluster_id} from {path}")
                dictionaries[cluster_id] = {
                    "sae": SAE.load_from_pretrained(path, device=device, dtype=dtype),
                    "layers": sae_dict["layers"],
                }
        else:
            if model_name is None:
                raise ValueError("model_name must be specified when load_from_sae_lens is False")
            for cluster_id, sae_dict in modules_to_paths.items():
                path = sae_dict["sae"]
                hook_name = f"blocks.{sae_dict['layers'][0].split('.')[1]}.hook_resid_post"
                if debug:
                    print(f"Loading SAE for {cluster_id} from {path}")
                sae = Sae.load_from_disk(path)
                sae_cfg = SaeConfig.load_json(os.path.join(path, "cfg.json"))
                cfg = TrainConfig.load_json(os.path.join(path, "..", "config.json"))
                norm_scaling_factors = torch.load(
                    os.path.join(sae_folder_path, "scaling_factors.pt")
                )
                hookpoint = "layers." + re.findall(r"\d+", hook_name)[0]
                norm_scaling_factor = norm_scaling_factors[hookpoint]
                sae_lens = to_sae_lens(
                    sae=sae,
                    sae_cfg=sae_cfg,
                    model_name=model_name,
                    dataset_path=dataset_path,
                    norm_scaling_factor=norm_scaling_factor,
                    max_seq_len=cfg.max_seq_len,
                    hook_name=hook_name,
                    hook_layer=int(re.findall(r"\d+", hook_name)[0]),
                    dtype=dtype,
                    device=device,
                    sae_lens_version=version("sae_lens"),
                )
                dictionaries[cluster_id] = {"sae": sae_lens, "layers": sae_dict["layers"]}
    return dictionaries