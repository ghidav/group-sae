
for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3.11 ./downstream/faith_topk.py \
    -d ioi \
    --model pythia-1b \
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

for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3.11 ./downstream/faith_topk.py \
    -d greater_than \
    --model pythia-1b \
    --what faithfulness \
    --K $K
done



for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3.11 ./downstream/faith_topk.py \
    -d ioi \
    --model pythia-1b \
    --what completeness \
    --K $K
done

for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3.11 ./downstream/faith_topk.py \
    -d subject_verb \
    --model pythia-1b \
    --what completeness \
    --K $K
done

for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3.11 ./downstream/faith_topk.py \
    -d greater_than \
    --model pythia-1b \
    --what completeness \
    --K $K
done