export OPENAI_API_KEY=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

python explain.py \
  --concordance /home/fbelotti/group-sae/feature_analysis/G9_K7_group_vs_baseline_concordance.json \
  --layers 13 14 15 \
  --k 9 \
  --tokens-dir-template ~/group-sae/feature_analysis/tokens/pythia-410m/{k} \
  --features-dir-template ~/group-sae/feature_analysis/features/pythia-410m/{k} \
  --activations-dir-template ~/group-sae/feature_analysis/activations/pythia-410m/{k} \
  --hf-tokenizer EleutherAI/pythia-410m \
  --top-k-tokens 10 \
  --max-snippets 16 \
  --bins 0.2 0.4 0.6 0.8 1.01 \
  --n-per-bin 8 \
  --device cuda:1 \
  --out-dir ./explanations
