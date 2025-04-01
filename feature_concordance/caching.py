import argparse
import logging
import os
import re
from functools import partial

import numpy as np
import torch
import transformer_lens
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from group_sae.hooks import from_tokens
from group_sae.utils import get_device_for_block, load_saes


def sae_hook(act, hook, sae, cache):
    original_shape = act.shape
    if len(original_shape) == 4:
        x = act.reshape(act.shape[0], act.shape[1], -1).clone()
    else:
        x = act.clone()
    x = x.to(sae.device)

    f = sae.encode(x)
    x_hat = sae.decode(f)

    if torch.is_grad_enabled():
        f.retain_grad()

    residual = x - x_hat
    cache[hook.name] = f

    x_recon = x_hat + residual.detach()

    if len(original_shape) == 4:
        return x_recon.reshape(original_shape)

    return x_recon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-160m",
        help="Name of the model to load (default: %(default)s)",
    )
    parser.add_argument(
        "--n_devices",
        type=int,
        default=1,
        help="Number of devices to use (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the DataLoader (default: %(default)s)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            "Specific layer to use; if not provided, "
            "all layers will be processed (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--component",
        type=str,
        default="resid_post",
        help="Component name from which to extract activations (default: %(default)s)",
    )
    parser.add_argument(
        "--sae_root_folder",
        type=str,
        default="/home/fbelotti/group-sae/saes/pythia_160m-topk",
        help="Path to the root folder for SAE dictionaries (default: %(default)s)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="Value of K for clustering (set to -1 for no clustering; default: %(default)s)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1_000_000,
        help="Maximum number of tokens to process (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="NeelNanda/pile-small-tokenized-2b",
        help="Name of the dataset to load (default: %(default)s)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for DataLoader (default: %(default)s)",
    )

    args = parser.parse_args()

    # If using more than one device, adjust the device-getter function.
    if args.n_devices > 1:
        transformer_lens.utilities.devices.get_device_for_block_index = get_device_for_block

    # Load the model
    model = HookedTransformer.from_pretrained(
        args.model_name, device="cuda", n_devices=args.n_devices
    )
    device = model.cfg.device
    if device is None:
        device = "cuda"
        model.cfg.device = device

    # Determine which layers to process
    if args.layer is None:
        layers = list(range(model.cfg.n_layers - 1))
    else:
        layers = [args.layer]

    modules = [get_act_name(args.component, layer) for layer in layers]

    # Load SAE dictionaries
    dictionaries = load_saes(
        args.sae_root_folder,
        device=device,
        debug=True,
        layer=args.layer,
        cluster=None if args.K == -1 else str(args.K),
        load_from_sae_lens=False,
        dtype="float32",
        model_name=args.model_name,
    )
    dictionaries = {
        k: v.to(get_device_for_block(int(re.findall(r"\d+", k)[0]), model.cfg, device=device))
        for k, v in dictionaries.items()
    }
    if len(dictionaries) == 0:
        raise ValueError("No dictionaries were loaded. Check the path to the dictionaries.")
    elif len(dictionaries) != len(modules):
        logging.warning(
            f"Loaded {len(dictionaries)} dictionaries, but expected {len(modules)}. "
            "Some modules may not have been loaded."
        )
        modules = [k for k in modules if k in dictionaries.keys()]
        dictionaries = {k: v for k, v in dictionaries.items() if k in modules}

    # Load dataset and prepare DataLoader
    dataset = load_dataset(args.dataset_name, streaming=False, split="train").shuffle(seed=42)
    dataset = dataset.select(range(len(dataset) // 2, len(dataset)))

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=from_tokens,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Set up hooks for the model
    hooks = []
    feature_cache = {}
    for hook_name in dictionaries.keys():
        hooks.append(
            (hook_name, partial(sae_hook, sae=dictionaries[hook_name], cache=feature_cache))
        )

    processed_tokens = 0
    feature_save = {hook_name: [] for hook_name in dictionaries.keys()}
    for tokens in tqdm(dl, total=args.max_tokens // (1024 * args.batch_size)):
        if processed_tokens >= args.max_tokens:
            break
        with torch.no_grad():
            model.run_with_hooks(tokens["input_ids"].to(device), fwd_hooks=hooks)
        for hook_name, features in feature_cache.items():
            _, features_idxes = torch.topk(features, k=128, dim=-1)
            feature_save[hook_name].append(features_idxes.view(-1).cpu().numpy().astype(np.int16))
        feature_cache.clear()
        processed_tokens += tokens["input_ids"].numel()

    # Save the features to disk
    for hook_name, features in feature_save.items():
        features = np.concatenate(features, axis=0)
        os.makedirs(
            os.path.join(
                "features",
                args.model_name,
                f"{args.K if args.K != -1 else 'baseline'}",
            ),
            exist_ok=True,
        )
        np.save(
            os.path.join(
                "features",
                args.model_name,
                f"{args.K if args.K != -1 else 'baseline'}",
                f"{hook_name}.npy",
            ),
            features,
        )
