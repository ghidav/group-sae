from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import random
import json
import os

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
batch_size = 8
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


# Cache topkens and activations
act_dict = {k: [] for k in name_to_module.keys()}
tokens = []

n_batches = int(args.n_batches)

for i, batch in tqdm(enumerate(dataloader), total=n_batches):
    tokens.append(batch["input_ids"])

    def hook(module, inputs, outputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        name = module_to_name[module]
        act_dict[name].append(outputs.cpu())

    handles = [mod.register_forward_hook(hook) for mod in name_to_module.values()]
    try:
        with torch.no_grad():
            lm(batch["input_ids"].to(device))
    finally:
        for handle in handles:
            handle.remove()

    if i == n_batches:
        break

act_dict = {k: torch.cat(v, dim=0).reshape(-1, seq_len, d_model) for k, v in act_dict.items()}
tokens = torch.cat(tokens, dim=0).reshape(-1, seq_len)


def get_feature_dataset(latents, max_acts, f_ids, W_dec):
    feature_dataset = {}
    logit_scores = W_dec @ W_U.T  # [F, V]

    for i, f_id in enumerate(f_ids):
        feature_dataset[f_id.item()] = {}

        # Cut activations in quantiles: (0.8, 0.9, 0.95, 0.99)
        quantiles = max_acts[:, i].quantile(
            torch.tensor([0.8, 0.9, 0.95, 0.99], device=max_acts.device)
        )

        lower_bound = quantiles[0]
        for j, (quantile, label) in enumerate(zip(quantiles[1:], ["80-90", "90-95", "95-99"])):
            mask = max_acts[:, i] > lower_bound
            mask &= max_acts[:, i] <= quantile

            sent_tokens = tokens[mask.cpu()].tolist()
            latent_acts = latents[mask, :, i].cpu().tolist()

            # Zip tokens and activations, then sample at most 5
            zipped_data = list(zip(sent_tokens, latent_acts))
            sampled_data = random.sample(zipped_data, min(5, len(zipped_data)))

            feature_dataset[f_id.item()][label] = sampled_data
            lower_bound = quantile

        mask = max_acts[:, i] > lower_bound
        sent_tokens = tokens[mask.cpu()].tolist()
        latent_acts = latents[mask, :, i].cpu().tolist()

        # Zip tokens and activations, then sample at most 5 for "99-100"
        zipped_data = list(zip(sent_tokens, latent_acts))
        sampled_data = random.sample(zipped_data, min(5, len(zipped_data)))

        # Find top and bottom token by attribution
        top_val, top_ids = torch.topk(logit_scores[i], 10)
        bottom_val, bottom_ids = torch.topk(-logit_scores[i], 10)

        feature_dataset[f_id.item()]["top_tokens"] = list(zip(top_ids.tolist(), top_val.tolist()))
        feature_dataset[f_id.item()]["bottom_tokens"] = list(
            zip(bottom_ids.tolist(), bottom_val.tolist())
        )

        feature_dataset[f_id.item()]["99-100"] = sampled_data

    return feature_dataset


num_selected_features = int(args.n_features)
sae_batch_size = 2048
num_sentences = act_dict[f"layers.0"].size(0)
sae_n_batches = num_sentences // sae_batch_size


# JumpReLU - Baseline

print("Running JR baselines...")
path = f"../saes/{MODEL_MAP[args.model]}-jr/baseline/"
saes = [Sae.load_from_disk(path + str(i)) for i in range(nl - 1)]
[sae.to(device) for sae in saes]

num_latents = saes[0].W_dec.size(0)

for l in tqdm(range(nl - 1)):
    f_ids = torch.randint(0, num_latents, (num_selected_features,))
    W_dec = saes[l].W_dec.clone().detach()

    latents = []

    for i in range(sae_n_batches + 1):
        with torch.no_grad():
            new_latents = saes[l](
                act_dict[f"layers.{l}"][i * sae_batch_size : (i + 1) * sae_batch_size].to(device).reshape(-1, d_model)
            ).feature_acts[..., f_ids]

        latents.append(new_latents)

    latents = torch.cat(latents).reshape(-1, seq_len, num_selected_features)

    max_acts, _ = latents.max(dim=1)

    feature_dataset = get_feature_dataset(latents, max_acts, f_ids, W_dec)

    with open(f"data/{MODEL_MAP[args.model]}/jr-baseline-{l}.json", "w") as f:
        json.dump(feature_dataset, f)


# JumpReLU - Cluster

print("Running JR clusters...")
path = f"../saes/{MODEL_MAP[args.model]}-jr/cluster/"
saes = {}
for folder in os.listdir(path):
    if "-" in folder:
        saes[folder] = Sae.load_from_disk(path + folder)
        saes[folder].to(device)


for l, sae in tqdm(saes.items()):
    f_ids = torch.randint(0, num_latents, (num_selected_features,))
    W_dec = sae.W_dec.clone().detach()

    latents = []
    l_start, l_end = l.split('-')
    layers = list(range(int(l_start), int(l_end) + 1))

    cluster_activations = [act_dict[f"layers.{i}"] for i in layers]
    cluster_activations = torch.stack(cluster_activations)
    indices = torch.randint(0, cluster_activations.size(0), (num_sentences,))
    cluster_activations = cluster_activations[indices, torch.arange(num_sentences), ...]

    for i in range(sae_n_batches + 1):
        with torch.no_grad():
            new_latents = sae(
                cluster_activations[i * sae_batch_size : (i + 1) * sae_batch_size].to(device).reshape(-1, d_model)
            ).feature_acts[..., f_ids]

        latents.append(new_latents)

    latents = torch.cat(latents).reshape(-1, seq_len, num_selected_features)

    max_acts, _ = latents.max(dim=1)

    feature_dataset = get_feature_dataset(latents, max_acts, f_ids, W_dec)

    with open(f"data/{MODEL_MAP[args.model]}/jr-cluster-{l}.json", "w") as f:
        json.dump(feature_dataset, f)


# TopK - Baseline

print("Running TopK baselines...")
path = f"../saes/{MODEL_MAP[args.model]}-topk/baseline/"
saes = [Sae.load_from_disk(path + str(i)) for i in range(nl - 1)]
[sae.to(device) for sae in saes]

num_latents = saes[0].W_dec.size(0)

for l in tqdm(range(nl - 1)):
    f_ids = torch.randint(0, num_latents, (num_selected_features,))
    W_dec = saes[l].W_dec.clone().detach()

    latents = []

    for i in range(sae_n_batches + 1):
        with torch.no_grad():
            sae_out = saes[l](
                act_dict[f"layers.{l}"][i * sae_batch_size : (i + 1) * sae_batch_size].to(device).reshape(-1, d_model)
            )
            ids = sae_out.topk_indices
            acts = sae_out.topk_acts
            buf = acts.new_zeros(acts.shape[:-1] + (num_latents,))
            new_latents = buf.scatter_(dim=-1, index=ids, src=acts)[..., f_ids]

        latents.append(new_latents)

    latents = torch.cat(latents).reshape(-1, seq_len, num_selected_features)

    max_acts, _ = latents.max(dim=1)

    feature_dataset = get_feature_dataset(latents, max_acts, f_ids, W_dec)

    with open(f"data/{MODEL_MAP[args.model]}/topk-baseline-{l}.json", "w") as f:
        json.dump(feature_dataset, f)


# TopK - Cluster

print("Running TopK clusters...")
path = f"../saes/{MODEL_MAP[args.model]}-topk/cluster/"
saes = {}
for folder in os.listdir(path):
    if "-" in folder:
        saes[folder] = Sae.load_from_disk(path + folder)
        saes[folder].to(device)


for l, sae in tqdm(saes.items()):
    f_ids = torch.randint(0, num_latents, (num_selected_features,))
    W_dec = sae.W_dec.clone().detach()

    latents = []
    l_start, l_end = l.split('-')
    layers = list(range(int(l_start), int(l_end) + 1))

    cluster_activations = [act_dict[f"layers.{i}"] for i in layers]
    cluster_activations = torch.stack(cluster_activations)
    indices = torch.randint(0, cluster_activations.size(0), (num_sentences,))
    cluster_activations = cluster_activations[indices, torch.arange(num_sentences), ...]

    for i in range(sae_n_batches + 1):
        with torch.no_grad():
            sae_out = saes[l](
                cluster_activations[i * sae_batch_size : (i + 1) * sae_batch_size].to(device).reshape(-1, d_model)
            )
            ids = sae_out.topk_indices
            acts = sae_out.topk_acts
            buf = acts.new_zeros(acts.shape[:-1] + (num_latents,))
            new_latents = buf.scatter_(dim=-1, index=ids, src=acts)[..., f_ids]

        latents.append(new_latents)

    latents = torch.cat(latents).reshape(-1, seq_len, num_selected_features)

    max_acts, _ = latents.max(dim=1)

    feature_dataset = get_feature_dataset(latents, max_acts, f_ids, W_dec)

    with open(f"data/{MODEL_MAP[args.model]}/topk-cluster-{l}.json", "w") as f:
        json.dump(feature_dataset, f)