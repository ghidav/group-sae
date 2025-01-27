from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import random
import json
import os
import gc

from group_sae.hooks import from_tokens
from group_sae import Sae

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--n_layers", type=str, required=True)
parser.add_argument("--n_batches", type=str, required=True)
parser.add_argument("--n_features", type=str, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_MAP = {
    "pythia-160m-deduped": "pythia_160m",
    "pythia-410m-deduped": "pythia_410m",
    "gemma-2-2b": "gemma2_2b",
}

os.makedirs(f"data/{MODEL_MAP[args.model]}", exist_ok=True)

# Load model
model_name = "EleutherAI/" + args.model if "pythia" in args.model else "google/" + args.model
lm = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
lm.to(device)

W_U = lm.embed_out.weight.detach().clone()
d_model = W_U.size(1)
nl = int(args.n_layers)

if "pythia" in args.model:
    name_to_module = {
        name: lm.gpt_neox.get_submodule(name) for name in [f"layers.{i}" for i in range(nl)]
    }

module_to_name = {v: k for k, v in name_to_module.items()}

# Load dataset
batch_size = 32
seq_len = 16

dataset = load_dataset(
    "NeelNanda/pile-small-tokenized-2b",
    streaming=True,
    split="train",
    trust_remote_code=True,
)

dataloader = DataLoader(
    dataset,
    collate_fn=from_tokens,
    batch_size=batch_size,
)


# Cache activations with reduced precision
def get_act_dict(module_names):
    tokens = []
    activations = []

    n_batches = int(args.n_batches)

    for i, batch in tqdm(enumerate(dataloader), total=n_batches):
        tokens.append(batch["input_ids"].to(torch.int32))  # Reduced precision
        act_dict = {name: [] for name in module_names}

        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            name = module_to_name[module]
            # Store activations in FP16
            act_dict[name] = outputs.cpu().to(torch.float16).reshape(-1, seq_len, d_model)

        handles = [name_to_module[name].register_forward_hook(hook) for name in module_names]
        try:
            with torch.no_grad():
                lm(batch["input_ids"].to(device))
        finally:
            for handle in handles:
                handle.remove()

        new_activations = torch.stack(list(act_dict.values())) # [M, B, P, D]
        batch_sentences = batch_size * (1024 // seq_len)
        indices = torch.randint(0, len(module_names), (batch_sentences,))
        activations.append(new_activations[indices, torch.arange(batch_sentences)])

        if i == n_batches:
            break

    activations = torch.cat(activations)
    tokens = torch.cat(tokens, dim=0).reshape(-1, seq_len).to(torch.int32)  # Reduced precision

    return activations, tokens


num_selected_features = int(args.n_features)
num_sentences = int(args.n_batches) * batch_size * (1024 // seq_len)
sae_batch_size = 1024
sae_n_batches = num_sentences // sae_batch_size

print(f"Generating datasets from {num_sentences} sentences. In total, {num_sentences * seq_len / 1e6:.1f}M tokens will be processed...")


def compute_feature_dataset(latents_generator, max_acts, f_ids, W_dec, tokens):
    feature_dataset = {}
    logit_scores = (W_dec.to(torch.float32) @ W_U.T.to(torch.float32)).cpu()  # FP32 for stability

    # Precompute quantile ranges for each feature
    quantile_ranges = torch.tensor([0.8, 0.9, 0.95, 0.99], device=max_acts.device)
    quantile_data = {}

    for i, f_id in enumerate(f_ids):
        fid = f_id.item()
        feature_acts = max_acts[:, i]
        quantiles = torch.quantile(feature_acts, quantile_ranges)

        quantile_data[fid] = {
            "ranges": [
                (quantiles[0], quantiles[1], "80-90"),
                (quantiles[1], quantiles[2], "90-95"),
                (quantiles[2], quantiles[3], "95-99"),
                (quantiles[3], float("inf"), "99-100"),
            ],
            "top_tokens": [],
            "bottom_tokens": [],
        }

        # Compute token scores
        with torch.no_grad():
            top_scores, top_ids = torch.topk(logit_scores[i], 10)
            bottom_scores, bottom_ids = torch.topk(-logit_scores[i], 10)

        quantile_data[fid]["top_tokens"] = list(zip(top_ids.tolist(), top_scores.tolist()))
        quantile_data[fid]["bottom_tokens"] = list(
            zip(bottom_ids.tolist(), (-bottom_scores).tolist())
        )

    # Initialize feature dataset structure
    for fid in quantile_data:
        feature_dataset[fid] = {
            "top_tokens": quantile_data[fid]["top_tokens"],
            "bottom_tokens": quantile_data[fid]["bottom_tokens"],
            **{r[2]: [] for r in quantile_data[fid]["ranges"]},
            "99-100": [],
        }

    # Process batches incrementally
    for batch_idx, latents_batch in enumerate(latents_generator):
        batch_size = latents_batch.size(0)
        global_start = batch_idx * batch_size
        global_end = global_start + batch_size

        for i, f_id in enumerate(f_ids):
            fid = f_id.item()
            batch_features = latents_batch[..., i]  # [batch_size, seq_len]
            batch_max = max_acts[global_start:global_end, i]

            # Process each quantile range
            for lower, upper, label in quantile_data[fid]["ranges"]:
                mask = (batch_max > lower) & (batch_max <= upper)
                if not mask.any():
                    continue

                # Get matching samples
                for sample_idx in torch.where(mask)[0]:
                    global_idx = global_start + sample_idx.item()
                    feature_dataset[fid][label].append(
                        (tokens[global_idx].tolist(), batch_features[sample_idx].cpu().tolist())
                    )

            # Handle 99-100% separately
            mask = batch_max > quantile_data[fid]["ranges"][-1][0]
            for sample_idx in torch.where(mask)[0]:
                global_idx = global_start + sample_idx.item()
                feature_dataset[fid]["99-100"].append(
                    (tokens[global_idx].tolist(), batch_features[sample_idx].cpu().tolist())
                )

    # Final sampling
    for fid in feature_dataset:
        for key in ["80-90", "90-95", "95-99", "99-100"]:
            entries = feature_dataset[fid][key]
            feature_dataset[fid][key] = (
                random.sample(entries, min(5, len(entries))) if entries else []
            )

    return feature_dataset


def process_sae_group(path, prefix, is_cluster=False):
    print(f"Processing {prefix}...")
    saes = {}
    layer_ranges = []

    # Load SAEs
    if is_cluster:
        for folder in os.listdir(path):
            if "-" in folder:
                sae = Sae.load_from_disk(os.path.join(path, folder))
                sae.to(device)
                saes[folder] = sae
                layer_ranges.append(folder)
    else:
        saes = {str(i): Sae.load_from_disk(os.path.join(path, str(i))) for i in range(nl - 1)}
        [sae.to(device) for sae in saes.values()]
        layer_ranges = list(saes.keys())

    for l in layer_ranges:
        sae = saes[l]
        num_latents = sae.W_dec.size(0)
        f_ids = torch.randint(0, num_latents, (int(args.n_features),))
        W_dec = sae.W_dec.detach().clone()[f_ids]

        # First pass: Compute max_acts
        max_acts_list = []
        if is_cluster:
            l_start, l_end = map(int, l.split("-"))
            layers = list(range(l_start, l_end + 1))
            selected_acts, tokens = get_act_dict([f"layers.{i}" for i in layers])
        else:
            selected_acts, tokens = get_act_dict([f"layers.{l}"])

        # Process in batches to compute max_acts
        for i in range(sae_n_batches + 1):
            batch_acts = (
                selected_acts[i * sae_batch_size : (i + 1) * sae_batch_size].to(device).float()
            )
            with torch.no_grad():
                if "topk" in prefix:
                    sae_out = sae(batch_acts.reshape(-1, d_model))
                    buf = sae_out.topk_acts.new_zeros((*sae_out.topk_acts.shape[:-1], num_latents))
                    latents = buf.scatter_(-1, sae_out.topk_indices, sae_out.topk_acts)
                else:
                    latents = sae(batch_acts.reshape(-1, d_model)).feature_acts
            max_acts_list.append(latents.reshape(-1, seq_len, num_latents).max(1)[0].cpu())
            del batch_acts, latents
            torch.cuda.empty_cache()

        max_acts = torch.cat(max_acts_list)

        # Second pass: Generate latents incrementally
        def latents_generator():
            for i in range(sae_n_batches + 1):
                batch_acts = (
                    selected_acts[i * sae_batch_size : (i + 1) * sae_batch_size].to(device).float()
                )
                with torch.no_grad():
                    if "topk" in prefix:
                        sae_out = sae(batch_acts.reshape(-1, d_model))
                        buf = sae_out.topk_acts.new_zeros(
                            (*sae_out.topk_acts.shape[:-1], num_latents)
                        )
                        latents = buf.scatter_(-1, sae_out.topk_indices, sae_out.topk_acts)
                    else:
                        latents = sae(batch_acts.reshape(-1, d_model)).feature_acts
                yield latents.reshape(-1, seq_len, num_latents)[..., f_ids].cpu()
                del batch_acts, latents
                torch.cuda.empty_cache()

        # Compute and save feature dataset
        feature_dataset = compute_feature_dataset(
            latents_generator(), max_acts[:, f_ids], f_ids, W_dec, tokens
        )

        output_name = f"{prefix}-{l}" if is_cluster else f"{prefix}-{l}"
        with open(f"data/{MODEL_MAP[args.model]}/{output_name}.json", "w") as f:
            json.dump(feature_dataset, f)

        del max_acts, feature_dataset
        gc.collect()
        torch.cuda.empty_cache()


# Process each SAE group
# process_sae_group(f"../saes/{MODEL_MAP[args.model]}-jr/baseline/", "jr-baseline")
process_sae_group(f"../saes/{MODEL_MAP[args.model]}-jr/cluster/", "jr-cluster", is_cluster=True)
# process_sae_group(f"../saes/{MODEL_MAP[args.model]}-topk/baseline/", "topk-baseline")
# process_sae_group(f"../saes/{MODEL_MAP[args.model]}-topk/cluster/", "topk-cluster", is_cluster=True)

# Cleanup remaining resources
# del act_dict, tokens
gc.collect()
torch.cuda.empty_cache()
