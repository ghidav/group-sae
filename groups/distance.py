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
parser.add_argument("--hf_token", type=str, required=True)
args = parser.parse_args()

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

os.environ["HF_TOKEN"] = args.hf_token
os.makedirs("dist", exist_ok=True)

# Load model and dataset
model_name = "google/" + args.model if "gemma" in args.model else "EleutherAI/" + args.model
model = transformer_lens.HookedTransformer.from_pretrained(
    model_name, device=device, dtype=torch.bfloat16
)

dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
dataloader = DataLoader(dataset, batch_size=4)


# Caching activations
def cache_hook(x, hook, cache):
    cache[hook.name].append(x[:, -1])
    return x


def compute_angular_distance(W1, W2):
    cosine_similarity = F.cosine_similarity(W1, W2, dim=-1)
    angular_distance = torch.arccos(cosine_similarity)
    return angular_distance / torch.pi


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
            activations_batch = torch.stack([cache[k] for k in cache], dim=0)

            # Center activations
            if avg_batch is None:
                avg_batch = activations_batch.mean(dim=-1, keepdim=True)
            else:
                avg_batch = (
                    avg_batch * (1 - alpha) + activations_batch.mean(dim=-1, keepdim=True) * alpha
                )

            activations_batch = activations_batch - avg_batch

            distances_tensor = torch.zeros(batch_size, n_layers, n_layers)

            for i in range(n_layers):
                for j in range(i):
                    W1 = activations_batch[i]
                    W2 = activations_batch[j]
                    angular_distances = compute_angular_distance(W1, W2)
                    distances_tensor[:, i, j] = angular_distances

            mean_distances = distances_tensor.mean(dim=0)

            all_distances.append(mean_distances)
            processed_tokens += tokens.numel()
            pbar.update(tokens.numel())

            if processed_tokens >= max_tokens:
                break

np.save(
    f"dist/{MODEL_MAP[args.model]}_{args.num_tokens}.npy", torch.stack(all_distances).cpu().numpy()
)
