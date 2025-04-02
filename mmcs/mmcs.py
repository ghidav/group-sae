import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from group_sae.utils import MODEL_MAP
from group_sae.utils import load_saes


DTYPE_MAP = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to load (default: %(default)s)",
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Number of groups to use (default: %(default)s)",
    )
    parser.add_argument(
        "--sae_root_folder",
        type=str,
        default="saes",
        help="Path to the folder containing the SAE dictionaries (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",  # use "cuda" or "cpu"
        help="Device to use for computation (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        help="Data type to use for computation (default: %(default)s)",
    )
    args = parser.parse_args()

    root_folder = os.path.dirname(os.path.abspath(__file__))

    sae_root_folder = os.path.join(
        args.sae_root_folder, args.model_name.replace("-", "_") + "-topk"
    )
    mmcs_folder = os.path.join(root_folder, "mmcs_results", args.model_name)

    os.makedirs(mmcs_folder, exist_ok=True)

    nl = MODEL_MAP[args.model_name]["n_layers"]

    # Load SAEs
    baseline_dict = load_saes(
        sae_root_folder,
        device="cpu",
        debug=True,
        layer=None,
        cluster=None,
        load_from_sae_lens=False,
        dtype=DTYPE_MAP[args.dtype],
        model_name=args.model_name,
    )

    group_dict = load_saes(
        sae_root_folder,
        device="cpu",
        debug=True,
        layer=None,
        cluster=str(args.K),
        load_from_sae_lens=False,
        dtype=DTYPE_MAP[args.dtype],
        model_name=args.model_name,
    )

    hook_name = "blocks.{layer}.hook_resid_post"

    # Compute MMCS for each pair of layers
    def mmcs(W_baseline, W_group):
        """
        Compute the MMCS between two weight matrices.
        """
        # Compute the MMCS
        W_baseline_norm = W_baseline / torch.norm(W_baseline, dim=1, keepdim=True)
        W_group_norm = W_group / torch.norm(W_group, dim=1, keepdim=True)

        mmcs = W_baseline_norm.to(args.device) @ W_group_norm.T.to(args.device)  # [M, M]

        mmcs_a = mmcs.max(dim=0).values.mean()  # w.r.t groups features
        mmcs_b = mmcs.max(dim=1).values.mean()  # w.r.t baseline features

        return mmcs_a.cpu(), mmcs_b.cpu()

    mmcs_baseline_groups = np.zeros((nl - 1, nl - 1))
    mmcs_groups_baseline = np.zeros((nl - 1, nl - 1))

    for i in tqdm(range(nl - 1)):
        # Load i-th baseline SAE
        baseline_sae = baseline_dict[hook_name.format(layer=i)]
        W_baseline = baseline_sae.W_dec.clone().detach()  # [M, N]

        for j in range(nl - 1):
            # Load j-th Group SAE
            group_sae = group_dict[hook_name.format(layer=j)]
            W_group = group_sae.W_dec.clone().detach()  # [M, N]

            # Compute MMCS
            mmcs_a, mmcs_b = mmcs(W_baseline, W_group)
            mmcs_groups_baseline[i, j] = mmcs_a.item()
            mmcs_baseline_groups[i, j] = mmcs_b.item()

    # Save the MMCS values to a file
    save_path = os.path.join(mmcs_folder, f"baseline_K{args.K}.npy")
    np.save(save_path, mmcs_baseline_groups)

    save_path = os.path.join(mmcs_folder, f"groups_K{args.K}.npy")
    np.save(save_path, mmcs_groups_baseline)


if __name__ == "__main__":
    main()
