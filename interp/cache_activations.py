import json
import os
from argparse import ArgumentParser

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from group_sae.hooks import from_tokens
from group_sae.utils import MODEL_MAP, load_saes_by_training_clusters


def parse_args():
    parser = ArgumentParser(
        description="Extract SAE activations from a model and save them as safetensors, "
        "optionally in token splits."
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
        "--n_tokens",
        type=int,
        default=1_000_000,
        help="Number of tokens to process.",
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=1024,
        help="Context length to use. Default is None (full context).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--latents_dir",
        type=str,
        default="interp/latents",
    )
    parser.add_argument(
        "--max_feature_id",
        type=int,
        default=64,
    )
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    os.path.dirname(os.path.abspath(__file__))

    # Choose device.
    device = "cuda" if torch.cuda.is_available() else "mps"
    if device != "cuda":
        print("Warning: Running on MPS, performance may be suboptimal.")

    # Load the model and tokenizer.
    full_model_name = f"EleutherAI/{args.model_name}"
    model = AutoModel.from_pretrained(full_model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    print(f"Model {args.model_name} loaded.")

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
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=from_tokens,
        batch_size=batch_size,
    )

    # Load SAEs.
    #sae_folder_path = os.path.join(script_dir, "../saes", MODEL_MAP[args.model_name]["short_name"])
    sae_folder_path = os.path.join("saes", MODEL_MAP[args.model_name]["short_name"] + "-topk")
    saes = load_saes_by_training_clusters(
        sae_folder_path, cluster=args.cluster, device=device, model_name=args.model_name
    )
    if len(saes) == 0:
        raise ValueError("No SAEs found.")
    else:
        print(f"Loaded SAEs for: {list(saes.keys())}")
    saes_mapping = {}
    for cluster_id, cluster_data in saes.items():
        saes_mapping[cluster_id] = {}
        for layer in cluster_data["layers"]:
            # `layer` is in `layers.layer_num` format
            saes_mapping[cluster_id][layer] = cluster_data["sae"]

    # Prepare cache for storing hook outputs.
    cache = {
        cluster_id: {layer: {"ids": [], "acts": []} for layer in cluster_data["layers"]}
        for cluster_id, cluster_data in saes.items()
    }

    # Define a factory for the forward hook.
    def create_hook(cluster_id, name):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                outputs_ = outputs[0]
            else:
                outputs_ = outputs
            outputs_flat = outputs_.reshape(-1, d_model)
            with torch.no_grad():
                latents = saes_mapping[cluster_id][name].encode(outputs_flat)
            top_acts, top_indices = torch.topk(latents, k, dim=1)  # B*Pxk
            mask = top_indices < args.max_feature_id
            b, p, _ = torch.where(mask.view(-1, args.ctx_len, k))
            l = top_indices[mask]
            # Concatenate the fixed locations with the top indices.
            b += idx * args.batch_size
            ids = torch.stack([b, p, l], dim=-1)
            cache[cluster_id][name]["ids"].append(ids.long().cpu())
            cache[cluster_id][name]["acts"].append(top_acts[mask].cpu().flatten())

        return hook

    # Register the hook functions for each relevant module (only once).
    for cluster_id, cluster_data in saes_mapping.items():
        for name in cluster_data:
            module = model.get_submodule(name)
            module.register_forward_hook(create_hook(cluster_id, name))

    token_counter = 0
    processed_tokens_batches = []
    pbar = tqdm(total=n_tokens, unit="tokens", desc="Processing tokens")

    # Process batches from the dataloader until at least n_tokens have been processed.
    for idx, batch in enumerate(dataloader):
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
    for cluster_id, cluster_data in cache.items():
        for name in cluster_data:
            cache[cluster_id][name]["ids"] = torch.cat(cache[cluster_id][name]["ids"])
            cache[cluster_id][name]["acts"] = torch.cat(cache[cluster_id][name]["acts"])

    # Build the saving directory.
    save_dir = args.latents_dir
    save_dir = os.path.join(save_dir, MODEL_MAP[args.model_name]["short_name"])
    if args.cluster:
        save_dir = os.path.join(save_dir, "cluster")
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
        "n_splits": 1,
        "model_name": f"EleutherAI/{args.model_name}",
    }

    for cluster_id, cluster_data in cache.items():
        cluster_save_dir = os.path.join(save_dir, f"{cluster_id}")
        for submodule, data in cluster_data.items():
            submodule_folder = os.path.join(cluster_save_dir, f".gpt_neox.{submodule}")
            os.makedirs(submodule_folder, exist_ok=True)
            file_name = f"0_{d_model * 16 - 1}.safetensors"
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
