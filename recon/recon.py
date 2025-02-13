import argparse
import logging
import os

import pandas as pd
import torch
import transformer_lens
from datasets import load_dataset
from sae_lens import ActivationsStore
from sae_lens.evals import EvalConfig, run_evals
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from group_sae.utils import get_device_for_block, load_saes_by_training_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--component", type=str, default="resid_post")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="NeelNanda/pile-small-tokenized-2b",
        help="The dataset to use for evaluation.",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="eval",
        help="The directory to save the evaluation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pythia-160m",
        help="The Huggingface ID of the model you wish to test.",
    )
    parser.add_argument(
        "--sae_root_folder",
        type=str,
        default="/home/fbelotti/group-sae/saes/pythia_160m-topk",
        help="Path to all dictionaries for your language model.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Whether to run evals on clusters or baselines.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1024,
        help="The number of steps to use for testing. "
        "The total number of tokens will be model.ctx_len * eval_steps.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="The batch size to use for testing."
    )
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.eval_dir, exist_ok=True)

    if args.n_devices > 1:
        transformer_lens.utilities.devices.get_device_for_block_index = get_device_for_block

    model = HookedTransformer.from_pretrained(args.model, device="cuda", n_devices=args.n_devices)
    device = model.cfg.device
    if device is None:
        device = "cuda"
        model.cfg.device = device
    cluster = args.cluster

    eval_batches = args.eval_steps // args.batch_size  # 5M samples
    eval_cfg = EvalConfig(
        batch_size_prompts=args.batch_size,
        # Reconstruction metrics
        n_eval_reconstruction_batches=eval_batches,
        compute_kl=True,
        compute_ce_loss=True,
        # Sparsity and variance metrics
        n_eval_sparsity_variance_batches=eval_batches,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
    )

    # Load SAEs by cluster-id
    dictionaries = load_saes_by_training_clusters(
        args.sae_root_folder,
        device=device,
        debug=True,
        cluster=args.cluster,
        load_from_sae_lens=False,
        dtype="float32",
        model_name=args.model,
    )

    # Split clusters across devices
    dictionaries = {
        k: {
            "sae": v["sae"].to(get_device_for_block(i, model.cfg, device=device)),
            "layers": v["layers"],
        }
        for i, (k, v) in enumerate(dictionaries.items())
    }

    # Load dataset
    dataset = load_dataset(
        "NeelNanda/pile-small-tokenized-2b", streaming=False, split="train"
    ).shuffle(seed=42)
    dataset = dataset.select(range(len(dataset) // 2, len(dataset)))

    with torch.no_grad():
        df = []
        for cid, sae_dict in dictionaries.items():
            for layer in tqdm(sae_dict["layers"]):
                # Prepare SAE
                sae = sae_dict["sae"]
                sae.cfg.hook_layer = int(layer.split(".")[1])
                sae.cfg.hook_name = get_act_name(args.component, sae.cfg.hook_layer)
                activations_store = ActivationsStore.from_sae(
                    model,
                    sae,
                    dataset=dataset,
                    streaming=False,
                    store_batch_size_prompts=1,
                    n_batches_in_buffer=1,
                    device=device,
                )

                # Run evals
                metrics = run_evals(sae, activations_store, model, eval_cfg)

                # Exclude feature_density stats (i.e. metrics[1])
                metrics = {
                    k_inner: v_inner
                    for k in metrics[0].keys()
                    for k_inner, v_inner in metrics[0][k].items()
                }
                metrics["layer"] = sae.cfg.hook_layer
                metrics["G"] = cid if args.cluster else "baseline"
                df.append(metrics)
        df = pd.DataFrame(df)
        df.to_csv(f"{args.eval_dir}/{args.model}_{'cluster' if cluster else 'baseline'}.csv")
