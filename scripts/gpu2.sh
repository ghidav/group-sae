for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    python3.11 downstream/effects.py \
    -d subject_verb \
    -l 6 \
    -m pythia-410m \
    --K $K
done