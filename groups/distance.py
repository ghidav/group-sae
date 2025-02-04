import json
import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import transformer_lens
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.utils import get_act_name

from group_sae.utils import MODEL_MAP

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--num_tokens", type=float, required=True)
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--hf_token", type=str, default="")
parser.add_argument("--dtype", type=str, default="bfloat16")
args = parser.parse_args()

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if args.hf_token != "":
    os.environ["HF_TOKEN"] = args.hf_token
else:
    try:
        with open("keys.json", "r") as f:
            keys = json.load(f)
        os.environ["HF_TOKEN"] = keys["huggingface"]
    except FileNotFoundError:
        raise ValueError("No Hugging Face token provided")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs("dist", exist_ok=True)
print(BASE_DIR)


# Distance functions
def angular_distance(W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    cosine_similarity = F.cosine_similarity(W1, W2, dim=-1)  # [N]
    angular_distance = torch.arccos(cosine_similarity).mean()
    return angular_distance / torch.pi


def cka(W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    n = W1.size(0)
    H = (
        torch.eye(n, device=W1.device, dtype=W1.dtype)
        - torch.ones((n, n), device=W1.device, dtype=W1.dtype) / n
    )
    K = W1 @ W1.T
    L = W2 @ W2.T

    def hisc(K, L):
        return torch.trace(K @ H @ L @ H) / (n - 1) ** 2

    cka_ = hisc(K, L) / torch.sqrt(hisc(K, K) * hisc(L, L))
    return 1 - cka_


if args.method == "angular":
    distance_metric = angular_distance
elif args.method == "cka":
    distance_metric = cka
else:
    raise NotImplementedError()

# Load model and dataset
model_name = "google/" + args.model if "gemma" in args.model else "EleutherAI/" + args.model
model = transformer_lens.HookedTransformer.from_pretrained(
    model_name, device=device, dtype=args.dtype
)
d = model.cfg.d_model

dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
dataloader = DataLoader(dataset, batch_size=4)


# Caching activations
def cache_hook(x, hook, cache):
    cache[hook.name].append(x[:, 1:].reshape(-1, d))  # Skip the BOS
    return x


all_distances = []
max_tokens = int(args.num_tokens)
processed_tokens = 0
n_layers = model.cfg.n_layers
alpha = 0.05
avg_batch = None

with torch.no_grad():
    with tqdm(total=max_tokens) as pbar:
        for ex in dataloader:
            tokens = model.to_tokens(ex["text"])[:, :256]
            batch_size = tokens.size(0)
            cache = {get_act_name("resid_post", i): [] for i in range(n_layers)}
            hooks = [
                (get_act_name("resid_post", i), partial(cache_hook, cache=cache))
                for i in range(n_layers)
            ]

            _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

            cache = {k: torch.cat(v, dim=0) for k, v in cache.items()}
            activations_batch = torch.stack([cache[k] for k in cache], dim=0)  # [L, N, D]

            # Center activations
            if avg_batch is None:  # First iter
                avg_batch = activations_batch.mean(dim=1, keepdim=True)  # [L, 1, D]
            else:  # Update
                avg_update = activations_batch.mean(dim=1, keepdim=True)
                avg_batch = avg_batch * (1 - alpha) + avg_update * alpha

            activations_batch = activations_batch - avg_batch

            distances_tensor = torch.zeros(batch_size, n_layers, n_layers)

            for i in range(n_layers):
                for j in range(i):
                    W1 = activations_batch[i]  # [N, D]
                    W2 = activations_batch[j]  # [N, D]
                    angular_distances = distance_metric(W1, W2)
                    distances_tensor[:, i, j] = angular_distances

            mean_distances = distances_tensor.mean(dim=0)

            all_distances.append(mean_distances)
            processed_tokens += tokens.numel()
            pbar.update(tokens.numel())

            if processed_tokens >= max_tokens:
                break


num_tokens_label = f"{float(args.num_tokens) / 1e6:.1f}M"
np.save(
    f"dist/{MODEL_MAP[args.model]}_{num_tokens_label}_{args.method}.npy",
    torch.stack(all_distances).cpu().numpy(),
)
