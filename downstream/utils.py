import json
import os
import random
import re

import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE
from transformer_lens import HookedTransformerConfig


def get_device_for_block(layer, cfg: HookedTransformerConfig, device: str | None = None):
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
            clean_prefix = t.flip(
                F.pad(
                    t.flip(clean_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )
            patch_prefix = t.flip(
                F.pad(
                    t.flip(patch_prefix, (1,)),
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
    cfg: HookedTransformerConfig,
    modules: list[str],
    device: str = "cuda",
    debug: bool = False,
    layer: int | None = None,
):

    dictionaries = {}
    component = modules[0].split(".")[-1]
    print(f"Loading {component} dictionaries...")

    if "attn" in component:
        dim = cfg.d_head * cfg.n_heads
    else:
        dim = cfg.d_model

    if not os.path.exists(sae_folder_path):
        dictionaries = {module: IdentitySAE(dim) for module in modules}
    else:
        paths = []
        for path in os.listdir(sae_folder_path):
            sae_path = os.path.join(sae_folder_path, path, "sae_lens")
            if not os.path.exists(sae_path):
                raise FileNotFoundError(
                    f"SAE path {sae_path} does not exist. "
                    f"Please make sure that the every folder contains the folder `sae_lens`, where "
                    "the converted SAE into the `sae_lens` format has to be found."
                )
            if os.path.isdir(os.path.join(sae_folder_path, path)):
                paths.append(sae_path)
        if layer is not None and layer > 0:
            paths = [path for path in paths if re.findall(r"\d+", path)[0] == str(layer)]
        saes = [SAE.load_from_pretrained(path, device=device, dtype="float32") for path in paths]
        hooks = set()
        dictionaries = {}
        for i, sae in enumerate(saes):
            hook = sae.cfg.hook_name
            if debug:
                print(hook, "->", paths[i])
            dictionaries[hook] = sae
            hooks.add(hook)
        remaining_modules = set(modules) - hooks
        print(
            f"For the remaining modules {remaining_modules}, no SAE will be attached,"
            " therefore no effects will be computed."
        )
    return dictionaries
