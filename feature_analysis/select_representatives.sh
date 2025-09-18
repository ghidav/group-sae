export OPENAI_API_KEY=REMOVED

python select_representatives.py \
  --explanations /home/fbelotti/group-sae/feature_analysis/explanations/pair_explanations.jsonl \
  --bins 0.2 0.4 0.6 0.8 1.01 \
  --group-by pair \
  --k-per-bin 2 \
  --model openai/gpt-5-mini \
  --max-candidates 80 \
  --out /home/fbelotti/group-sae/feature_analysis/explanations/representatives_selection.json
