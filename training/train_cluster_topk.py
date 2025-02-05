import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel

from group_sae import ClusterSaeTrainer, SaeConfig, TrainConfig
from group_sae.hooks import from_tokens

if __name__ == "__main__":
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl", timeout=timedelta(days=1))
        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    model_name = "EleutherAI/pythia-160m-deduped"
    l1_coefficient = 0.0
    max_seq_len = 1024
    target_l0 = None
    batch_size = 4
    lr = None
    k = 128

    # Define pythia-160m-clusters
    clusters = {
        "k1": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        "k2": [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
        "k3": [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]],
        "k4": [[0, 1, 2], [3, 4, 5, 6], [7, 8], [9, 10]],
        "k5": [[0, 1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
    }
    unique_clusters = {
        "k1": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        "k2": [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
        "k3": [[0, 1, 2], [3, 4, 5, 6]],
        "k4": [[7, 8], [9, 10]],
        "k5": [[3, 4], [5, 6]],
    }
    unique_cluster_flatten = {
        "k1-c0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "k2-c0": [0, 1, 2, 3, 4, 5, 6],
        "k2-c1": [7, 8, 9, 10],
        "k3-c0": [0, 1, 2],
        "k3-c1": [3, 4, 5, 6],
        "k4-c2": [7, 8],
        "k4-c3": [9, 10],
        "k5-c1": [3, 4],
        "k5-c2": [5, 6],
    }

    # Streaming dataset example
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
        model_name,
        device_map={"": "cuda"},
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
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
        run_name="checkpoints-clusters/{}-{}-topk-{}-lambda-{}-target-L0-{}-lr-{}".format(
            model_name, max_seq_len, k, l1_coefficient, target_l0, lr
        ),
        adam_epsilon=1e-8,
        adam_betas=(0.9, 0.999),
        keep_last_n_checkpoints=4,
        clusters=unique_cluster_flatten,
        distribute_modules=ddp,
        auxk_alpha=1 / 32,
        dead_feature_threshold=10_000_000,
    )
    trainer = ClusterSaeTrainer(cfg, dataloader, model)
    trainer.fit()
