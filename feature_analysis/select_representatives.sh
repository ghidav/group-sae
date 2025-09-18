export OPENAI_API_KEY=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

python select_representatives.py \
  --explanations /home/fbelotti/group-sae/feature_analysis/explanations/pair_explanations.jsonl \
  --bins 0.2 0.4 0.6 0.8 1.01 \
  --group-by pair \
  --k-per-bin 2 \
  --model openai/gpt-5-mini \
  --max-candidates 80 \
  --out /home/fbelotti/group-sae/feature_analysis/explanations/representatives_selection.json
