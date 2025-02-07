import json
import os
import random
import re
from importlib.metadata import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE

from group_sae.config import TrainConfig
from group_sae.export import to_sae_lens
from group_sae.sae import Sae, SaeConfig
from group_sae.utils import CLUSTER_MAP, MODEL_MAP


class IdentitySAE(nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, **kwargs):
        if output_features:
            return x, x
        else:
            return x


def load_examples(dataset, num_examples, model, seed=12, pad_to_length=None, length=None):
    examples = []
    dataset_items = open(dataset).readlines()
    random.seed(seed)
    random.shuffle(dataset_items)
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = model.tokenizer(
            data["clean_prefix"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).input_ids
        patch_prefix = model.tokenizer(
            data["patch_prefix"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).input_ids
        clean_answer = model.tokenizer(
            data["clean_answer"],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids
        patch_answer = model.tokenizer(
            data["patch_answer"],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids

        # only keep examples where answers are single tokens
        if clean_prefix.shape[1] != patch_prefix.shape[1]:
            continue
        # only keep examples where clean and patch inputs are the same length
        if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
            continue
        # if we specify a `length`, filter examples if they don't match
        if length and clean_prefix.shape[1] != length:
            continue
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            model.tokenizer.padding_side = "right"
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = torch.flip(
                F.pad(
                    torch.flip(clean_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )
            patch_prefix = torch.flip(
                F.pad(
                    torch.flip(patch_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )

        example_dict = {
            "clean_prefix": clean_prefix,
            "patch_prefix": patch_prefix,
            "clean_answer": clean_answer.item(),
            "patch_answer": patch_answer.item(),
            "prefix_length_wo_pad": prefix_length_wo_pad,
        }
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples


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
    dictionaries = {}

    if not os.path.exists(sae_folder_path):
        raise ValueError(f"SAE path {sae_folder_path} does not existorch. ")
    else:
        # Load all available SAEs in `sae_folder_path`
        paths = []
        for path in os.listdir(sae_folder_path):
            sae_path = os.path.join(sae_folder_path, path)
            if load_from_sae_lens:
                sae_path = os.path.join(sae_path, "sae_lens")
            if os.path.isdir(os.path.join(sae_folder_path, path)):
                if not os.path.exists(sae_path):
                    raise FileNotFoundError(f"SAE path {sae_path} does not existorch. ")
                paths.append(sae_path)

        # Map modules to paths, converting paths to corresponding sae_lens hookpoints
        if cluster is not None:
            if model_name is None:
                raise ValueError("model_name must be specified when cluster is not None")
            cluster_layers = CLUSTER_MAP[MODEL_MAP[model_name]][cluster]
            modules_to_paths = {}
            for layer_num, cluster_layer in enumerate(cluster_layers):
                for path in paths:
                    if cluster_layer in path:
                        modules_to_paths[f"blocks.{layer_num}.hook_resid_post"] = path
                        break
        else:
            modules_to_paths = {}
            for path in paths:
                layer_num = re.findall(r"\d+", path.split(os.sep)[-1])[0]
                if f"layers.{layer_num}" in path:
                    modules_to_paths[f"blocks.{layer_num}.hook_resid_post"] = path

        # Grab only the specified layer, if specified
        if layer is not None:
            modules_to_paths = {
                hook_name: path
                for hook_name, path in modules_to_paths.items()
                if str(layer) in hook_name
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
                cfg = TrainConfig.load_json(os.path.join(sae_folder_path, "config.json"))
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
