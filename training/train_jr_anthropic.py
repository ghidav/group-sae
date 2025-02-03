import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel

from group_sae import SaeConfig, SaeTrainer, TrainConfig
from group_sae.hooks import from_tokens

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-160m-deduped"
    l1_coefficient = 10
    max_seq_len = 1024
    target_l0 = None
    batch_size = 4
    lr = 2e-4
    k = -1

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
        streaming=True,
        split="train",
        trust_remote_code=True,
    )
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
    sae_cfg = SaeConfig(
        expansion_factor=16,
        k=k,
        jumprelu=True,
        init_b_dec_as_zeros=True,
        jumprelu_target_l0=target_l0,
        init_enc_as_dec_transpose=True,
        jumprelu_bandwidth=2,
        pre_act_loss=True,
    )
    cfg = TrainConfig(
        sae=sae_cfg,
        batch_size=batch_size,
        save_every=10000,
        lr=lr,
        lr_scheduler_name="constant",
        lr_warmup_steps=0.01,
        l1_coefficient=l1_coefficient,
        l1_warmup_steps=1,
        max_seq_len=max_seq_len,
        use_l2_loss=True,
        layers=[3],
        num_training_tokens=1_000_000_000,
        normalize_activations=1,
        num_norm_estimation_tokens=1_000_000,
        run_name="checkpoints/{}-1024-jr-lambda-{}-target-L0-{}-lr-{}".format(
            model_name,
            l1_coefficient,
            target_l0,
            lr,
        ),
        adam_betas=(0.0, 0.999),
        adam_epsilon=1e-8,
        distribute_modules=False,
        keep_last_n_checkpoints=2,
    )
    trainer = SaeTrainer(cfg, dataloader, model)
    trainer.fit()
