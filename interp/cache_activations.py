import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from safetensors.torch import save_file
from argparse import ArgumentParser
from tqdm import tqdm

from group_sae.sae import Sae
from group_sae.hooks import from_tokens
from group_sae.utils import MODEL_MAP, load_cluster_map


def parse_args():
    parser = ArgumentParser(
        description="Extract SAE activations from a model and save them as safetensors, optionally in token splits."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-160m-deduped",
        help="Name of the model (e.g. 'pythia-160m-deduped').",
    )
    parser.add_argument(
        "--sae_folder_path",
        type=str,
        default="../saes/pythia_160m-topk",
        help="Path to the folder containing the SAEs.",
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
        "--n_features",
        type=int,
        default=1024,
        help="Number of features to use in the SAE.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=1,
        help="Number of output file splits. Default is 1 (single file).",
    )
    return parser.parse_args()


def load_saes(sae_folder_path, cluster_map, n_layers, args, device, model_dtype):
    """
    Loads the SAE models for each layer.

    If clustering is enabled, the appropriate sub-folder (either 'baseline' or 'cluster')
    is selected based on whether the cluster_map entry for a layer contains a '-' character.
    """
    saes = {}
    for layer in range(n_layers - 1):
        submodule = f"layers.{layer}"
        if args.cluster:
            to_load = cluster_map[layer]
            if "-" not in str(to_load):
                # Load from the baseline folder if the cluster map entry is simple.
                sae_path = os.path.join(sae_folder_path, "baseline", submodule)
            else:
                sae_path = os.path.join(sae_folder_path, "cluster", str(cluster_map[layer]))
            sae = Sae.load_from_disk(sae_path, device=device).to(dtype=model_dtype)
        else:
            sae_path = os.path.join(sae_folder_path, "baseline", submodule)
            sae = Sae.load_from_disk(sae_path, device=device).to(dtype=model_dtype)
        saes[submodule] = sae
    return saes


def main():
    args = parse_args()

    # Choose device.
    device = "cuda" if torch.cuda.is_available() else "mps"
    if device != "cuda":
        print("Warning: Running on MPS, performance may be suboptimal.")

    # Processing parameters.
    batch_size = 8
    ctx_len = 256
    n_tokens = 10_000
    # The activation hook produces a fixed number of outputs per token.
    activation_factor = 128

    # Load the model and tokenizer.
    full_model_name = f"EleutherAI/{args.model_name}"
    model = AutoModel.from_pretrained(full_model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    n_layers = MODEL_MAP[args.model_name]["n_layers"]
    d_model = MODEL_MAP[args.model_name]["d_model"]

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
    saes = load_saes(args.sae_folder_path, cluster_map, n_layers, args, device, model.dtype)

    # Get a mapping from submodule names to model modules.
    name_to_module = {name: model.get_submodule(name) for name in saes.keys()}

    # Prepare cache for storing hook outputs.
    cache = {name: {"ids": [], "acts": []} for name in saes.keys()}

    # Precompute a "locations" tensor (used in the hook) based on batch and context length.
    X, Y = torch.meshgrid(torch.arange(batch_size), torch.arange(ctx_len), indexing="ij")
    locations = (
        torch.stack([X.flatten(), Y.flatten()], dim=1)
        .repeat_interleave(activation_factor, dim=0)
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
                _, top_acts, top_indices = saes[name].activation(saes[name].encode(outputs_flat))
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
    save_dir = os.path.join("latents", MODEL_MAP[args.model_name]["short_name"])
    if args.cluster:
        save_dir = os.path.join(save_dir, str(args.G))
    else:
        save_dir = os.path.join(save_dir, "baseline")

    # Save the results for each submodule.
    for submodule, data in cache.items():
        submodule_folder = os.path.join(save_dir, f".gpt_neox.{submodule}")
        os.makedirs(submodule_folder, exist_ok=True)
        if args.n_splits > 0:
            # Flatten tokens so that splitting is done on the token level.
            tokens_flat = processed_tokens.flatten()
            total_tokens = tokens_flat.numel()
            split_size = total_tokens // args.n_splits
            for i in range(args.n_splits):
                start = i * split_size
                end = total_tokens if i == args.n_splits - 1 else (i + 1) * split_size
                tokens_chunk = tokens_flat[start:end]
                start_idx, end_idx = start * activation_factor, end * activation_factor
                ids_chunk = data["ids"][start_idx:end_idx]
                acts_chunk = data["acts"][start_idx:end_idx]
                file_name = f"{start}_{end}.safetensors"
                file_path = os.path.join(submodule_folder, file_name)
                save_file(
                    {"tokens": tokens_chunk, "locations": ids_chunk, "activations": acts_chunk},
                    file_path,
                )
                print(f"Saved split: {file_path}")
        else:
            file_name = f"0_{token_counter}.safetensors"
            file_path = os.path.join(submodule_folder, file_name)
            save_file(
                {
                    "tokens": processed_tokens,
                    "locations": data["ids"],
                    "activations": data["acts"],
                },
                file_path,
            )
            print(f"Saved file: {file_path}")


if __name__ == "__main__":
    main()
