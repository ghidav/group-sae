import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from group_sae.utils import MODEL_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-160m",
        help="Name of the model to load (default: %(default)s)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=4,
        help="Number of groups to use (default: %(default)s)",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="features",
        help="Directory to load features from (default: %(default)s)",
    )
    parser.add_argument(
        "--concordance_dir",
        type=str,
        default="concordance",
        help="Directory to load features from (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (default: %(default)s)",
    )
    args = parser.parse_args()
    M = MODEL_MAP[args.model_name]["d_model"] * 16
    nl = MODEL_MAP[args.model_name]["n_layers"]

    pbar = tqdm(desc="Loading SAEs", total=nl - 1)
    concordance = np.zeros((nl - 1, nl - 1), dtype=np.float32)
    for baseline_layer in range(nl - 1):
        for group_layer in range(baseline_layer + 1, nl - 1):
            baseline_act = np.load(
                os.path.join(
                    args.feature_dir,
                    args.model_name,
                    str(args.K) if args.K != -1 else "baseline",
                    f"blocks.{baseline_layer}.hook_resid_post.npy",
                )
            )
            baseline_act = torch.from_numpy(baseline_act).to(args.device)

            group_act = np.load(
                os.path.join(
                    args.feature_dir,
                    args.model_name,
                    str(args.K) if args.K != -1 else "baseline",
                    f"blocks.{group_layer}.hook_resid_post.npy",
                )
            )
            group_act = torch.from_numpy(group_act).to(args.device)

            # Flatten the tensors (now each is of shape (N*K,))
            a_flat = baseline_act.reshape(-1).to(torch.int64)
            b_flat = group_act.reshape(-1).to(torch.int64)

            # Build the AND matrix (i.e. co-occurrence count matrix)
            and_matrix = torch.zeros(M, M, dtype=torch.int64, device=a_flat.device)
            and_matrix.index_put_(
                (a_flat, b_flat), torch.ones_like(a_flat, dtype=torch.int64), accumulate=True
            )

            # Count total occurrences per class in a and b (summing over N and K)
            count_a = torch.bincount(a_flat, minlength=M)
            count_b = torch.bincount(b_flat, minlength=M)

            # Compute the OR (union) matrix:
            or_matrix = count_a.view(M, 1) + count_b.view(1, M) - and_matrix

            # Compute the Jaccard similarity
            jaccard_similarity = and_matrix.float() / or_matrix.float()

            amiou = jaccard_similarity.max(dim=1).values.mean(dim=0)
            concordance[baseline_layer, group_layer] = amiou.item()

        pbar.update(1)
    pbar.close()

    # Save concordance matrix
    os.makedirs(os.path.join(args.concordance_dir, args.model_name, str(args.K)), exist_ok=True)
    np.save(
        os.path.join(
            args.concordance_dir,
            args.model_name,
            str(args.K),
            f"concordance_{args.K}.npy",
        ),
        concordance,
    )
