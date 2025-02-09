import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel

from group_sae import ClusterSaeTrainer, SaeConfig, TrainConfig, load_training_clusters
from group_sae.hooks import from_tokens

if __name__ == "__main__":
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-160m",
        help="Name (or path) of the model to use. For example: 'EleutherAI/pythia-160m-deduped'.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=12,
        help="number of layers",
    )
    args = parser.parse_args()
    model_name = args.model_name

    # --- Distributed data parallel setup (if applicable) ---
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl", timeout=timedelta(days=1))
        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    # --- Load clusters from JSON file ---
    training_clusters = load_training_clusters(args.model_name.split("-")[-1])

    # Convert each cluster's list of string indices to a list of ints.
    clusters_flatten = {}
    for key, value in training_clusters.items():
        try:
            clusters_flatten[key] = [int(layer_str) for layer_str in value]
        except ValueError as e:
            raise ValueError(f"Error converting cluster '{key}' values to integers: {e}")
    print(clusters_flatten)

    # --- Training hyperparameters ---
    l1_coefficient = 0.0
    max_seq_len = 1024
    target_l0 = None
    batch_size = args.batch
    lr = None
    k = 128

    # --- Load dataset ---
    # (The streaming dataset example is commented out below.)
    # dataset = load_dataset(
    #     "allenai/c4",
    #     "en",
    #     split="train",
    #     trust_remote_code=True,
    #     streaming=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # dataset = chunk_and_tokenize_streaming(dataset, tokenizer, max_seq_len=max_seq_len)

    dataset = load_dataset(
        "NeelNanda/pile-small-tokenized-2b",
        streaming=False,
        split="train",
        trust_remote_code=True,
    )
    if ddp:
        dataset = dataset.shard(dist.get_world_size(), rank)
    dataloader = DataLoader(
        dataset,
        collate_fn=from_tokens,
        batch_size=batch_size,
    )
    model = AutoModel.from_pretrained(
        "EleutherAI/" + model_name,
        device_map={"": "cuda"},
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # --- Add baseline singleton clusters ---
    # For every layer in the model we want a cluster that consists of only that layer.
    # If one already exists (even if its key is not "layers.{i}"), we rename it accordingly.
    num_layers = args.layers

    for i in range(num_layers):
        # Look for any cluster that is exactly [i]
        matching_keys = [
            key for key, layers in clusters_flatten.items() if len(layers) == 1 and layers[0] == i
        ]
        if matching_keys:
            # If one exists, rename the first one to "layers.i" if it isn’t already.
            first_key = matching_keys[0]
            if first_key != f"layers.{i}":
                clusters_flatten[f"layers.{i}"] = clusters_flatten.pop(first_key)
            # Remove any duplicates (if more than one singleton for the same layer exists)
            for extra_key in matching_keys[1:]:
                del clusters_flatten[extra_key]
        else:
            # If no singleton exists for this layer, add it.
            clusters_flatten[f"layers.{i}"] = [i]
    cfg = TrainConfig(
        SaeConfig(
            k=k,
            jumprelu=False,
            multi_topk=False,
            expansion_factor=16,
            init_b_dec_as_zeros=False,
            init_enc_as_dec_transpose=True,
        ),
        batch_size=batch_size,
        save_every=100_000,
        layers=None,
        hookpoints=None,
        lr=lr,
        lr_scheduler_name="constant",
        lr_warmup_steps=0.0,
        l1_warmup_steps=0.0,
        l1_coefficient=l1_coefficient,
        max_seq_len=max_seq_len,
        use_l2_loss=False,
        normalize_activations=1.0,
        num_training_tokens=1_000_000_000,
        num_norm_estimation_tokens=5_000_000,
        run_name="checkpoints-clusters/{}-topk".format(model_name),
        adam_epsilon=1e-8,
        adam_betas=(0.9, 0.999),
        keep_last_n_checkpoints=4,
        clusters=clusters_flatten,
        distribute_modules=ddp,
        log_to_wandb=True,
        auxk_alpha=1 / 32,
        dead_feature_threshold=10_000_000,
    )
    trainer = ClusterSaeTrainer(cfg, dataloader, model)
    trainer.fit()
