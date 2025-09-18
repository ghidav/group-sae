export OPENAI_API_KEY=REMOVED

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
