import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import transformer_lens
import transformer_lens.utilities
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.utils import get_act_name

from group_sae.distance import SVCCA, AngularDistance, ApproxCKA
from group_sae.utils import MODEL_MAP, get_device_for_block

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_tokens", type=float, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--ctx_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs("dist", exist_ok=True)
    print(BASE_DIR)

    if args.n_devices > 1:
        transformer_lens.utilities.devices.get_device_for_block_index = get_device_for_block

    # Load model and dataset
    model_name = "google/" + args.model if "gemma" in args.model else "EleutherAI/" + args.model
    model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=args.dtype,
        default_prepend_bos=False,
        default_padding_side="right",
        n_devices=args.n_devices,
    )
    d = model.cfg.d_model

    # We use directly the Pile dataset as it's the one used to pre-train models from the pythia family
    dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Caching activations
    def cache_hook(x, hook, cache):
        cache[hook.name].append(x.reshape(-1, d))
        return x

    alpha = 0.05
    avg_batch = None
    processed_tokens = 0
    n_layers = model.cfg.n_layers
    max_tokens = int(args.num_tokens)
    ctx_len = args.ctx_len or model.cfg.n_ctx
    n_pairs = int(n_layers * (n_layers - 1) / 2)

    # Initialize distances
    if args.method == "angular":
        distances = [AngularDistance() for _ in range(n_pairs)]
    elif args.method == "cka":
        distances = [ApproxCKA(kernel="linear") for _ in range(n_pairs)]
    elif args.method == "svcca":
        distances = [SVCCA(top_k=int(np.sqrt(d))) for _ in range(n_pairs)]
    else:
        raise NotImplementedError()

    with torch.no_grad():
        with tqdm(total=max_tokens) as pbar:
            for ex in dataloader:
                tokens = model.to_tokens(ex["text"], prepend_bos=False)[
                    :, :ctx_len
                ]  # 8 * 256 = 2048 tokens
                batch_size = tokens.size(0)
                cache = {get_act_name("resid_post", i): [] for i in range(n_layers)}
                hooks = [
                    (get_act_name("resid_post", i), partial(cache_hook, cache=cache))
                    for i in range(n_layers)
                ]
                _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

                cache = {k: torch.cat(v, dim=0) for k, v in cache.items()}
                activations_batch = torch.stack(
                    [cache[k].cpu() for k in cache], dim=0
                )  # [L, N, D]

                # Center activations
                if avg_batch is None:  # First iter
                    avg_batch = activations_batch.mean(dim=1, keepdim=True)  # [L, 1, D]
                else:  # Update
                    avg_update = activations_batch.mean(dim=1, keepdim=True)
                    avg_batch = avg_batch * (1 - alpha) + avg_update * alpha

                centered_batch = activations_batch - avg_batch  # [L, N, D]
                for i in range(n_layers):
                    for j in range(i):
                        A = centered_batch[i]  # [N, D]
                        B = centered_batch[j]  # [N, D]
                        tri_index = (i * (i - 1)) // 2 + j
                        distances[tri_index].update(A, B)
                processed_tokens += tokens.numel()
                pbar.update(tokens.numel())

                if processed_tokens >= max_tokens:
                    break

    final_distances = torch.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(i):
            tri_index = (i * (i - 1)) // 2 + j
            final_distances[i, j] = distances[tri_index].value().cpu()

    num_tokens_label = f"{float(args.num_tokens) / 1e6:.1f}M"
    np.save(
        f"dist/{MODEL_MAP[args.model]["short_name"]}_{num_tokens_label}_{args.method}.npy",
        final_distances.cpu().numpy(),
    )