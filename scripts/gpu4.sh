for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3.11 downstream/effects.py \
    -d greater_than \
    -l 12 \
    -m pythia-1b \
    --K $K
done