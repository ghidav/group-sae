import argparse
import logging
import os
import re

import pandas as pd
import torch
import transformer_lens
from datasets import load_dataset
from sae_lens import ActivationsStore
from sae_lens.evals import EvalConfig, run_evals
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from group_sae.utils import get_device_for_block, load_saes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--K", "-k", type=int, default=-1, help="The number of cluster to be used."
    )
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
        "--sae_folder_path",
        type=str,
        default="/home/fbelotti/group-sae/saes/pythia_160-topk",
        help="Path to all dictionaries for your language model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The batch size to use for testing."
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
    layers = list(range(model.cfg.n_layers))
    modules = [get_act_name(args.component, layer) for layer in layers]
    cluster = args.K != -1
    dictionaries = load_saes(
        args.sae_folder_path,
        device=device,
        debug=True,
        layer=None,
        cluster=None if args.K == -1 else args.K,
        load_from_sae_lens=False,
        dtype="float32",
        model_name=args.model,
    )
    dictionaries = {
        k: v.to(get_device_for_block(int(re.findall(r"\d+", k)[0]), model.cfg, device=device))
        for k, v in dictionaries.items()
    }
    logger.info(f"{len(dictionaries)} dictionaries loaded.")
    if len(dictionaries) == 0:
        raise ValueError("No dictionaries were loaded. Check the path to the dictionaries.")
    elif len(dictionaries) != len(modules):
        logging.warning(
            f"Loaded {len(dictionaries)} dictionaries, but expected {len(modules)}. "
            "Some modules may not have been loaded."
        )
        modules = [k for k in modules if k in dictionaries.keys()]

    eval_cfg = EvalConfig(
        batch_size_prompts=args.batch_size,
        # Reconstruction metrics
        n_eval_reconstruction_batches=128,
        compute_kl=True,
        compute_ce_loss=True,
        # Sparsity and variance metrics
        n_eval_sparsity_variance_batches=128,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
    )

    # Load dataset
    dataset = load_dataset(
        "NeelNanda/pile-small-tokenized-2b", streaming=False, split="train"
    ).shuffle(seed=42)
    # elapsed_tokens = 0
    # for index, batch in tqdm(enumerate(dataset), total=1_000_000_000 // 1024):
    #     elapsed_tokens += len(batch["tokens"])
    #     if elapsed_tokens >= 1_000_000_000:
    #         break
    # dataset = dataset.select(range(index, len(dataset)))

    with torch.no_grad():
        print(f"Cluster: {args.K if cluster else 'Baseline'}")
        cluster_df = []

        # Run evals for every act layer in the cluster
        for layer in tqdm(layers):
            # Select SAE
            cluster_sae = None
            for k, v in dictionaries.items():
                if str(layer) == re.findall(r"\d+", k)[0]:
                    cluster_sae = v
                    break
            if cluster_sae is None:
                raise ValueError(f"No SAE found for layer {layer}")

            activations_store = ActivationsStore.from_sae(
                model,
                cluster_sae,
                dataset=dataset,
                streaming=False,
                store_batch_size_prompts=args.batch_size,
                n_batches_in_buffer=args.batch_size,
                device=device,
            )

            # Run evals
            cluster_metrics = run_evals(cluster_sae, activations_store, model, eval_cfg)

            # Exclude feature_density stats (i.e. cluster_metrics[1])
            cluster_metrics = {
                k_inner: v_inner
                for k in cluster_metrics[0].keys()
                for k_inner, v_inner in cluster_metrics[0][k].items()
            }
            cluster_metrics["layer"] = layer
            if cluster:
                cluster_metrics["cluster"] = args.K if cluster else 0
            cluster_df.append(pd.Series(cluster_metrics))
            cluster_df = pd.concat(cluster_df, axis=1).T
            cluster_df.to_csv(
                f"{args.eval_dir}/{args.model}_{'cluster' if cluster else 'baseline'}.csv"
            )

        # if do_baseline:
        #     baseline_df = []
        #     for layer in tqdm(range(n_layers - 1)):
        #         BASELINE_PATH = baseline_folder.format(layer=layer)
        #         baseline_state_dict = load_file(f"{BASELINE_PATH}/sae.safetensors", device=device)
        #         baseline_sae_cfg = json.load(open(f"{BASELINE_PATH}/cfg.json", "r"))
        #         baseline_sae_cfg = SaeConfig.from_dict(baseline_sae_cfg)
        #         baseline_sae_cfg_training = json.load(
        #             open(f"{baseline_root_folder}/config.json", "r")
        #         )
        #         baseline_sae_cfg_training = TrainConfig.from_dict(baseline_sae_cfg_training)
        #         eleuther_sae = Sae(1024, baseline_sae_cfg)
        #         eleuther_sae.load_state_dict(
        #             {k: v.squeeze() for k, v in baseline_state_dict.items()}
        #         )
        #         baseline_sae = to_sae_lens(
        #             eleuther_sae.cfg,
        #             baseline_sae_cfg_training,
        #             eleuther_sae,
        #             model_name,
        #             "NeelNanda/pile-small-tokenized-2b",
        #             scale_factors[f"layers.{layer}"] if scale_factors is not None else None,
        #             hook_name=f"blocks.{layer}.hook_resid_post",
        #             hook_layer=layer,
        #         )

        #         activations_store = ActivationsStore.from_config(
        #             model, default_cfg, override_dataset=dataset
        #         )

        #         baseline_metrics = run_evals(baseline_sae, activations_store, model, eval_cfg)
        #         # Exclude feature_density stats (i.e. baseline_metrics[1])
        #         baseline_metrics = {
        #             k_inner: v_inner
        #             for k in baseline_metrics[0].keys()
        #             for k_inner, v_inner in baseline_metrics[0][k].items()
        #         }
        #         print(baseline_metrics)
        #         baseline_df.append(pd.Series(baseline_metrics, name=f"{layer}"))

        #     activation_fn = "relu" if baseline_sae_cfg.k <= 0 else "topk"
        #     if baseline_sae_cfg.jumprelu:
        #         activation_fn = "jumprelu"
        #     baseline_df = pd.concat(baseline_df, axis=1).T
        #     baseline_df.to_csv(f"eval/pythia_410m_{activation_fn}_{component}_baseline.csv")
