# Group-SAE: Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups

This repository contains the code that accompanies the EMNLP 2025 paper *Group-SAE: Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups*. It provides the full training, evaluation, and analysis pipeline for reproducing the results on the Pythia family of language models.

## Overview
- **Group-SAE** trains a single sparse autoencoder (SAE) for groups of contiguous transformer layers that exhibit similar residual stream representations.
- **AMAD (Average Maximum Angular Distance)** quantifies layer similarity and guides the choice of the number of groups, trading off reconstruction quality against training cost.
- The codebase covers baseline per-layer SAEs, grouped training, reconstruction benchmarking, downstream task faithfulness, feature-level analyses, and interpretability studies.
- Pre-computed layer groupings and trained SAEs for Pythia-160M/410M/1B are bundled to enable plug-and-play evaluation.

## Repository Layout
- `group_sae/`: Core training library (`Sae`, `SaeTrainer`, `ClusterSaeTrainer`, configuration dataclasses, hooks, utilities, AMAD helpers).
- `training/`: Entry points for baseline (`train_topk.py`) and Group-SAE (`train_cluster_topk.py`) training, plus multi-GPU launch scripts.
- `groups/`: Scripts and JSON artefacts for distance computation and AMAD-driven layer grouping.
- `recon/`, `downstream/`, `feature_concordance/`, `feature_spreading/`, `effects/`, `faithfulness/`, `mmcs/`, `interp/`: Evaluation and interpretability pipelines used in the paper.
- `scripts/`: Convenience shell scripts to run the full suite of experiments reported in the paper.
- `eval/`, `faithfulness/`, `effects/`, etc.: Cached results that match the tables and figures from the paper.

## Scripts Reference
- `scripts/auto_interp.sh`: Runs the end-to-end interpretability workflow (cache activations, generate feature explanations, score outputs) for baseline and grouped SAEs across Pythia-160M/410M/1B.
- `scripts/cache_features.sh`: Caches latent activations for each model, both baseline and clustered SAEs, to support downstream interpretability analyses.
- `scripts/distances.sh`: Computes inter-layer similarity matrices (angular distance and CKA) used by AMAD when deriving layer groups.
- `scripts/download.sh`: Clones the released SAE checkpoints from Hugging Face into `saes/` and preps baseline vs. cluster config files.
- `scripts/download_features.sh`: Retrieves pre-computed latent feature dumps required by the interpretability notebooks (git-lfs access needed).
- `scripts/explain_features.sh`: Generates GPT-based textual explanations for baseline and grouped SAEs for each model size.
- `scripts/feature_caching.sh`: Populates feature caches for concordance analysis by iterating over K values and saving top-activating features.
- `scripts/feature_concordance.sh`: Sweeps K for every model to quantify feature-sharing via `feature_concordance/concordance.py` (expects cached features and uses CUDA paths hard-coded for the cluster runs).
- `scripts/feature_spreading.sh`: Measures feature spreading statistics (unique feature usage per token) across K values for all models.
- `scripts/get_effects_pythia_160m.sh`: Runs the circuit-effects attribution benchmark on all tasks and group sizes for Pythia-160M.
- `scripts/get_effects_pythia_410m.sh`: Same as above but for Pythia-410M.
- `scripts/get_effects_pythia_1b.sh`: Same as above but for Pythia-1B.
- `scripts/get_faith_pythia_160m.sh`: Sweeps faithfulness/completeness metrics across tasks and K values for Pythia-160M.
- `scripts/get_faith_pythia_410m.sh`: Faithfulness sweep for Pythia-410M.
- `scripts/get_faith_pythia_1b.sh`: Faithfulness sweep for Pythia-1B.
- `scripts/mmcs.sh`: Launches mean maximum causal score computations for each model across all reported K values.
- `scripts/run_evals_recon.sh`: Recomputes reconstruction metrics (baseline vs. cluster) for every model via `recon/recon.py`.

## Installation
Requirements: Python 3.10+, PyTorch with GPU support, and access to the Hugging Face Hub (weights and datasets). We recommend creating a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Optional extras:
- `pip install bitsandbytes` to enable 8-bit Adam.
- `pip install -r requirements-dev.txt` (if you maintain your own dev extras) for linting and tests.
- The repository ships with a `uv.lock`; you can alternatively run `uv pip sync uv.lock`.

## Data, Weights, and Layer Groups
- **Datasets:** Training and evaluation use tokenised versions of *The Pile*. The scripts default to `NeelNanda/pile-small-tokenized-2b`; Hugging Face will download shards on demand. For raw text runs, utilities in `group_sae/data.py` chunk and tokenise arbitrary datasets.
- **Layer groups:** Ready-to-use assignments are stored under `group_sae/groups/pythia-*.json`. They include training clusters and AMAD curves. Load them programmatically via `group_sae.utils.load_training_clusters` or `group_sae.utils.load_amds`.
- **Pretrained SAEs:** Download the published checkpoints and configs with:

  ```bash
  bash scripts/download.sh
  ```

  This clones the Hugging Face repositories into `saes/pythia_{size}-topk/{baseline,cluster}` and copies the configuration files required by the trainers and evaluation code.
- **Latent caches:** Feature activation caches used for interpretability figures can be fetched with `bash scripts/download_features.sh` (requires git-lfs access to private repositories).

## Selecting Layer Groups with AMAD
1. **Collect layer-wise activations:** Compute similarity matrices for each model/metric via:

   ```bash
   python groups/distance.py --model pythia-160m-deduped --method angular --num_tokens 10_000_000
   python groups/distance.py --model pythia-160m-deduped --method cka --num_tokens 10_000_000
   ```

   Supported metrics are `angular`, `cka`, and `svcca`. Results are saved to `groups/dist/`.
2. **Score candidate partitions:** The notebook `groups/plots.ipynb` (or the helpers in `group_sae.utils`) converts the distance matrices into AMAD curves across possible group counts.
3. **Pick the operating point:** AMAD returns tuples `(G, AMAD, C)` where `G` is the number of groups and `C = A + T·G` approximates training cost (constants are model-specific). Choose the smallest `G` that satisfies your reconstruction or downstream threshold.
4. **Persist the mapping:** Store the final mapping as `{cluster_id: [layer indices]}` under `group_sae/groups/` and pass it to `ClusterSaeTrainer` via `TrainConfig(clusters=...)`.

## Training Sparse Autoencoders
### Baseline (layer-wise) SAEs
- `training/train_topk.py` reproduces the per-layer top-k SAEs used as the reference point in prior work.
- Launch with torchrun for multi-GPU training:

  ```bash
  torchrun --nproc-per-node=8 --standalone training/train_topk.py \
    --model_name EleutherAI/pythia-160m-deduped --batch 16
  ```

  Key `TrainConfig` knobs include `num_training_tokens`, `k`, `l1_coefficient`, `normalize_activations`, and `save_every`.

### Group-SAE (clustered layers)
- `training/train_cluster_topk.py` instantiates `ClusterSaeTrainer`, which shares a single SAE across all layers in a cluster.
- Example command (mirrors the paper setup):

  ```bash
  torchrun --nproc-per-node=8 --standalone training/train_cluster_topk.py \
    --model_name pythia-410m --batch 8 --layers 24
  ```

  The script looks up the appropriate `training_clusters` entry from `group_sae/groups/pythia_{size}_sequential.json` and augments it with singleton clusters for the baseline comparison.
- `TrainConfig` accepts cluster dictionaries, automatic LR schedules, gradient accumulation, activation normalisation, and checkpoint retention (`keep_last_n_checkpoints`). Logging to Weights & Biases is enabled by default when `log_to_wandb=True`.
- The shortcut `training/run.sh` reproduces the three models in the paper (160M, 410M, 1B).

### Tips
- Resuming: set `resume_from` to a checkpoint directory.
- Mixed-precision: adjust `torch_dtype` when loading models if you want bfloat16 training.
- Token loaders: replace the default dataset shard or collator by modifying `group_sae/hooks.from_tokens` or using the streaming utilities in `group_sae/data.py`.

## Evaluation & Analysis Pipelines
### Reconstruction Quality
Run the TransformerLens-based evaluation suite to compute KL-divergence, cross-entropy, explained variance, sparsity, and norm diagnostics:

```bash
python recon/recon.py --model pythia-160m --batch_size 8
python recon/recon.py --model pythia-160m --cluster --batch_size 8
```

Outputs are written to `eval_recon_sequential/{model}_{baseline|cluster}.csv` and match the paper tables.

### Downstream Faithfulness Experiments
- **Circuit attribution effects:**

  ```bash
  python downstream/effects.py -d ioi -m pythia-160m --sae_root_folder saes/pythia_160m-topk --K 3
  ```

  The script averages indirect effects over IOI, `greater_than`, and `subject_verb` tasks (JSON definitions in `downstream/tasks/`). Convenience launchers `scripts/get_effects_pythia_*.sh` sweep all K values.
- **Faithfulness vs. sparsity thresholds:**

  ```bash
  python downstream/faith_topk.py -d ioi -m pythia-410m --sae_root_folder saes/pythia_410m-topk --K 5
  ```

  Metrics are cached under `downstream/faithfulness_topk/` and aggregated in `faithfulness/`.

### Feature Concordance & Spreading
- Cache activations and top-k features:

  ```bash
  python feature_concordance/caching.py --model_name pythia-410m --K 9 \
    --sae_root_folder saes/pythia_410m-topk --max_tokens 1_000_000
  ```

- Compute concordance statistics with `feature_concordance/concordance.py` and feature spreading with `feature_spreading/feature_spreading.py`. Batch scripts in `scripts/feature_concordance.sh` and `scripts/feature_spreading.sh` reproduce the plots from the paper.

### Automated Feature Interpretation
The `interp/` pipeline integrates cached features with GPT-based explanations. Run `bash scripts/auto_interp.sh` to cache activations, generate natural language descriptors for baseline vs. grouped SAEs, and grade the outputs.

### Additional Analyses
- `mmcs/mmcs.py` computes mean maximum causal scores across group sizes.
- `feature_analysis/` hosts scripts to select representative neurons and compare cluster vs. baseline features.
- `effects/` and `faithfulness/` directories contain the artefacts reported in the main paper and appendix.

## Utilities and API Snippets
- `group_sae.utils.load_saes` / `load_saes_by_training_clusters` load saved checkpoints into TransformerLens-compatible `SAE` objects.
- `group_sae.export.to_sae_lens` converts internal checkpoints into the `sae_lens` format for downstream tooling.
- `group_sae.normalization.estimate_norm_scaling_factor` reproduces the activation norm scaling used in the experiments.
- `group_sae.utils.chunk_almost_equal_sum` and friends help distribute clusters evenly across devices when using DDP.

## Citation
Please cite the paper if you use this codebase or the released SAEs:

```
@inproceedings{
  ghilardi2025efficient,
  title={Efficient Training of Sparse Autoencoders for Large Language Models via Layer Groups},
  author={Davide Ghilardi and Federico Belotti and Marco Molinari and Tao Ma and Matteo Palmonari},
  booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  url={https://openreview.net/forum?id=bk4PhF17cm}
}
```

## Acknowledgements
The implementation builds on:

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [sae-lens](https://github.com/jbloom/sae_lens)
- [sparsify](https://github.com/EleutherAI/sparsify)

We thank the authors of previous SAE work whose tooling we extend in this repository.
