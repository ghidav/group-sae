import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from group_sae.utils import MODEL_MAP


def main():
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
        help="Directory to save the concordance matrix (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",  # use "cuda" or "cpu"
        help="Device to use for computation (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for processing tokens (default: %(default)s)",
    )
    args = parser.parse_args()

    # M: total number of possible features.
    M = MODEL_MAP[args.model_name]["d_model"] * 16
    nl = MODEL_MAP[args.model_name]["n_layers"]

    concordance = np.zeros((nl - 1, nl - 1), dtype=np.float32)
    pbar = tqdm(desc="Computing concordance", total=nl - 1)

    for baseline_layer in range(nl - 1):
        # Load baseline activations; expected shape (N, K)
        baseline_path = os.path.join(
            args.feature_dir,
            args.model_name,
            "baseline",
            f"blocks.{baseline_layer}.hook_resid_post.npy",
        )
        baseline_act = (
            torch.from_numpy(np.load(baseline_path)).to(args.device).reshape(-1, 128).int()
        )

        for group_layer in range(nl - 1):
            # Load group activations; expected shape (N, K)
            group_path = os.path.join(
                args.feature_dir,
                args.model_name,
                str(args.K) if args.K != -1 else "baseline",
                f"blocks.{group_layer}.hook_resid_post.npy",
            )
            group_act = (
                torch.from_numpy(np.load(group_path)).to(args.device).reshape(-1, 128).int()
            )

            # Both baseline_act and group_act now have shape (N, K)
            N_tokens = baseline_act.shape[0]
            # Allocate a global histogram vector for the AND matrix (flattened)
            global_hist = torch.zeros(M * M, device=args.device, dtype=torch.int32)

            # Process tokens in batches to avoid huge memory allocations.
            for i in range(0, N_tokens, args.batch_size):
                # Get a batch of tokens, shape: (B, K)
                baseline_batch = baseline_act[i : i + args.batch_size]
                group_batch = group_act[i : i + args.batch_size]
                # Compute the outer (Cartesian) product for each token in the batch.
                # baseline_batch.unsqueeze(2): shape (B, K, 1)
                # group_batch.unsqueeze(1): shape (B, 1, K)
                # Their broadcast yields shape: (B, K, K)
                batch_linear_idx = baseline_batch.unsqueeze(2) * M + group_batch.unsqueeze(1)
                # Flatten the (B, K, K) tensor to a 1D tensor.
                batch_linear_idx = batch_linear_idx.reshape(-1)
                # Count co-occurrences in this batch.
                batch_hist = torch.bincount(batch_linear_idx, minlength=M * M)
                global_hist += batch_hist

            # Reshape global histogram into the AND matrix of shape (M, M)
            and_matrix = global_hist.reshape(M, M).to(torch.int32)

            # Compute total counts per feature (across all tokens)
            count_a = torch.bincount(baseline_act.reshape(-1), minlength=M)
            count_b = torch.bincount(group_act.reshape(-1), minlength=M)
            # OR matrix: OR(i,j) = count_a[i] + count_b[j] - and_matrix[i,j]
            or_matrix = count_a.view(M, 1) + count_b.view(1, M) - and_matrix

            # Compute Jaccard similarity matrix.
            jaccard_similarity = and_matrix.float() / (or_matrix.float() + 1e-8)
            # Derive a concordance score (e.g., averaging the max similarity per feature)
            amiou = jaccard_similarity.max(dim=1).values.mean().item()
            concordance[baseline_layer, group_layer] = amiou

        pbar.update(1)
    pbar.close()

    # Save the concordance matrix.
    out_dir = os.path.join(args.concordance_dir, args.model_name, str(args.K))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "concordance.npy"), concordance)


if __name__ == "__main__":
    main()
