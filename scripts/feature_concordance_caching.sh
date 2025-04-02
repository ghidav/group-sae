#!/bin/bash
# run_all.sh
# This script runs the Python experiment for multiple models and varying K values.
# For each model:
#   - pythia-160m has L=12, so K runs from 1 to 10.
#   - pythia-410m has L=24, so K runs from 1 to 22.
#   - pythia-1b   has L=16, so K runs from 1 to 14.

max_tokens=1000000

# For pythia-160m
model="pythia-160m"
batch_size=32
L=12
echo "Running experiments for ${model} with L=${L}"
for (( K=1; K<=6; K++ )); do
    echo "Running ${model} with K=${K}"
    python /root/group-sae/feature_concordance/caching.py \
        --model_name "${model}" \
        --K "${K}" \
        --max_tokens ${max_tokens} \
        --batch_size ${batch_size} \
        --sae_root_folder /root/group-sae/saes/pythia_160m-topk
done
echo "Running ${model} with K=${K}"
python /root/group-sae/feature_concordance/caching.py \
    --model_name "${model}" \
    --K -1 \
    --max_tokens ${max_tokens} \
    --batch_size ${batch_size} \
    --sae_root_folder /root/group-sae/saes/pythia_160m-topk

# For pythia-410m
model="pythia-410m"
batch_size=16
L=24
echo "Running experiments for ${model} with L=${L}"
for (( K=1; K<=9; K++ )); do
    echo "Running ${model} with K=${K}"
    python /root/group-sae/feature_concordance/caching.py --model_name "${model}" \
        --K "${K}" \
        --max_tokens ${max_tokens} \
        --batch_size ${batch_size} \
        --sae_root_folder /root/group-sae/saes/pythia_410m-topk
done
echo "Running ${model} with K=${K}"
python /root/group-sae/feature_concordance/caching.py \
    --model_name "${model}" \
    --K -1 \
    --max_tokens ${max_tokens} \
    --batch_size ${batch_size} \
    --sae_root_folder /root/group-sae/saes/pythia_410m-topk

# For pythia-1b
model="pythia-1b"
batch_size=8
L=16
echo "Running experiments for ${model} with L=${L}"
for (( K=1; K<=6; K++ )); do
    echo "Running ${model} with K=${K}"
    python /root/group-sae/feature_concordance/caching.py --model_name "${model}" \
        --K "${K}" \
        --max_tokens ${max_tokens} \
        --batch_size ${batch_size} \
        --sae_root_folder /root/group-sae/saes/pythia_1b-topk
done
echo "Running ${model} with K=${K}"
python /root/group-sae/feature_concordance/caching.py --model_name "${model}" \
    --K -1 \
    --max_tokens ${max_tokens} \
    --batch_size ${batch_size} \
    --sae_root_folder /root/group-sae/saes/pythia_1b-topk
