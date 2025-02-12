for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    python3.11 ./downstream/faith_topk.py \
    -d subject_verb \
    --model pythia-410m \
    --what faithfulness \
    --K $K
done

for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3.11 ./downstream/faith_topk.py \
    -d subject_verb \
    --model pythia-1b \
    --what faithfulness \
    --K $K
done