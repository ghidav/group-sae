import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel

from group_sae import SaeConfig, SaeTrainer, TrainConfig
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
    batch_size = 16
    # lr = 12e-4
    lr = None
    k = 128

    # Streaming dataset example
    # dataset = load_dataset(
    #     "togethercomputer/RedPajama-Data-1T-Sample",
    #     split="train",
    #     trust_remote_code=True,
    #     streaming=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # dataset = chunk_and_tokenize_streaming(dataset, tokenizer, max_seq_len=max_seq_len)
    # dataloader = DataLoader(dataset, batch_size=batch_size)

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
        lr=lr,
        lr_scheduler_name="constant",
        lr_warmup_steps=0.0,
        l1_coefficient=l1_coefficient,
        l1_warmup_steps=0.0,
        max_seq_len=max_seq_len,
        use_l2_loss=False,
        num_training_tokens=1_000_000_000,
        normalize_activations=1.0,
        num_norm_estimation_tokens=5_000_000,
        run_name="checkpoints/{}-{}-topk-{}-lambda-{}-target-L0-{}-lr-{}".format(
            model_name, max_seq_len, k, l1_coefficient, target_l0, lr
        ),
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        keep_last_n_checkpoints=4,
        distribute_modules=ddp,
        auxk_alpha=1 / 32,
        dead_feature_threshold=10_000_000,
    )
    trainer = SaeTrainer(cfg, dataloader, model)
    trainer.fit()
