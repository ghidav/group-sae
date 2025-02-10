import os
import json
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from group_sae.hooks import from_tokens
from group_sae.utils import MODEL_MAP, load_cluster_map, load_saes


def parse_args():
    parser = ArgumentParser(
        description="Extract SAE activations from a model and save them as safetensors, optionally in token splits."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-160m",
        help="Name of the model (e.g. 'pythia-160m').",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Use clustering when loading SAEs.",
    )
    parser.add_argument(
        "--G",
        type=int,
        default=None,
        help="G parameter for clustering (required if --cluster is set).",
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=1_000_000,
        help="Number of tokens to process.",
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=None,
        help="Context length to use. Default is None (full context).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Choose device.
    device = "cuda" if torch.cuda.is_available() else "mps"
    if device != "cuda":
        print("Warning: Running on MPS, performance may be suboptimal.")

    # Load the model and tokenizer.
    full_model_name = f"EleutherAI/{args.model_name}"
    model = AutoModel.from_pretrained(full_model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)

    MODEL_MAP[args.model_name]["n_layers"]
    d_model = MODEL_MAP[args.model_name]["d_model"]

    # Processing parameters.
    batch_size = args.batch_size
    ctx_len = args.ctx_len or tokenizer.model_max_length
    n_tokens = args.n_tokens
    # The activation hook produces a fixed number of outputs per token.
    k = 128

    print(f"Running for {n_tokens} tokens with batch size {batch_size}, context length {ctx_len}.")

    # Load the dataset.
    dataset = load_dataset(
        "NeelNanda/pile-small-tokenized-2b",
        streaming=False,
        split="train",
        trust_remote_code=True,
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=from_tokens,
        batch_size=batch_size,
    )

    # If clustering is enabled, load the corresponding cluster map.
    cluster_map = None
    if args.cluster:
        if args.G is None:
            raise ValueError("If clustering, --G must be specified")
        cluster_map = load_cluster_map(args.model_name.split("-")[1])[str(args.G)]
        print("Cluster map:", cluster_map)

    print(f"Model {args.model_name} loaded.")

    # Load SAEs.
    sae_folder_path = os.path.join(
        script_dir, "../saes", MODEL_MAP[args.model_name]["short_name"] + "-topk"
    )
    G = str(args.G) if args.cluster else None
    saes = load_saes(sae_folder_path, cluster=G, device=device, model_name=args.model_name)
    saes = {f"layers.{k.split('.')[1]}": v for k, v in saes.items()}

    # Get a mapping from submodule names to model modules.
    name_to_module = {name: model.get_submodule(name) for name in saes.keys()}

    # Prepare cache for storing hook outputs.
    cache = {name: {"ids": [], "acts": []} for name in saes.keys()}

    # Precompute a "locations" tensor (used in the hook) based on batch and context length.
    X, Y = torch.meshgrid(torch.arange(batch_size), torch.arange(ctx_len), indexing="ij")
    locations = (
        torch.stack([X.flatten(), Y.flatten()], dim=1)
        .repeat_interleave(k, dim=0)
        .type(torch.int64)
    )

    # Define a factory for the forward hook.
    def create_hook(name):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                outputs_ = outputs[0]
            else:
                outputs_ = outputs
            outputs_flat = outputs_.reshape(-1, d_model)
            with torch.no_grad():
                latents = saes[name].encode(outputs_flat)
            top_acts, top_indices = torch.topk(latents, k, dim=1)
            # Concatenate the fixed locations with the top indices.
            ids = torch.cat([locations.to(device), top_indices.flatten()[:, None]], dim=1)
            cache[name]["ids"].append(ids.cpu())
            cache[name]["acts"].append(top_acts.cpu().flatten())

        return hook

    # Register the hook functions for each relevant module (only once).
    for name, module in name_to_module.items():
        module.register_forward_hook(create_hook(name))

    token_counter = 0
    processed_tokens_batches = []
    pbar = tqdm(total=n_tokens, unit="tokens", desc="Processing tokens")

    # Process batches from the dataloader until at least n_tokens have been processed.
    for batch in dataloader:
        tokens = batch["input_ids"][:, :ctx_len]
        with torch.no_grad():
            model(tokens.to(device))
        processed_tokens_batches.append(tokens.cpu())
        token_counter += tokens.numel()
        pbar.update(tokens.numel())
        if token_counter >= n_tokens:
            break
    pbar.close()

    # Concatenate processed tokens and the cached hook outputs.
    processed_tokens = torch.cat(processed_tokens_batches)
    for name in cache:
        cache[name]["ids"] = torch.cat(cache[name]["ids"])
        cache[name]["acts"] = torch.cat(cache[name]["acts"])

    # Build the saving directory.
    save_dir = os.path.join(script_dir, "latents")
    save_dir = os.path.join(save_dir, MODEL_MAP[args.model_name]["short_name"])
    if args.cluster:
        save_dir = os.path.join(save_dir, str(args.G))
    else:
        save_dir = os.path.join(save_dir, "baseline")

    # Save the results for each submodule.
    cfg_dict = {
        "dataset_repo": "gngdb/subset_the_pile_deduplicated",
        "dataset_split": "train[:1%]",
        "dataset_name": "",
        "dataset_row": "text",
        "batch_size": args.batch_size,
        "ctx_len": args.ctx_len,
        "n_tokens": args.n_tokens,
        "n_splits": args.n_splits,
        "model_name": "EleutherAI/pythia-160m",
    }

    for submodule, data in cache.items():
        submodule_folder = os.path.join(save_dir, f".gpt_neox.{submodule}")
        os.makedirs(submodule_folder, exist_ok=True)
        file_name = f"0_{d_model * 16}.safetensors"
        file_path = os.path.join(submodule_folder, file_name)
        save_file(
            {
                "tokens": processed_tokens,
                "locations": data["ids"],
                "activations": data["acts"],
            },
            file_path,
        )
        with open(os.path.join(submodule_folder, "config.json"), "w") as f:
            json.dump(cfg_dict, f)

        print(f"Saved file: {file_path}")


if __name__ == "__main__":
    main()
