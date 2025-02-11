for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    CUDA_VISIBLE_DEVICES=0,1 python downstream/effects.py \
    -d ioi \
    -c resid_post \
    -n 1024 \
    -l 15 \
    -m pythia-410m \
    -mt attrib \
    --sae_root_folder /home/fbelotti/group-sae/saes/pythia_410m-topk \
    --device cuda \
    --effects_dir /home/fbelotti/group-sae/effects \
    --task_dir /home/fbelotti/group-sae/downstream/tasks \
    --batch_size 16 \
    --n_devices 2 \
    --K $K
done

for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    CUDA_VISIBLE_DEVICES=0,1 python downstream/effects.py \
    -d greater_than \
    -c resid_post \
    -n 1024 \
    -l 12 \
    -m pythia-410m \
    -mt attrib \
    --sae_root_folder /home/fbelotti/group-sae/saes/pythia_410m-topk \
    --device cuda \
    --effects_dir /home/fbelotti/group-sae/effects \
    --task_dir /home/fbelotti/group-sae/downstream/tasks \
    --batch_size 16 \
    --n_devices 2 \
    --K $K
done

for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22; do
    CUDA_VISIBLE_DEVICES=0,1 python downstream/effects.py \
    -d subject_verb \
    -c resid_post \
    -n 1024 \
    -l 6 \
    -m pythia-410m \
    -mt attrib \
    --sae_root_folder /home/fbelotti/group-sae/saes/pythia_410m-topk \
    --device cuda \
    --effects_dir /home/fbelotti/group-sae/effects \
    --task_dir /home/fbelotti/group-sae/downstream/tasks \
    --batch_size 16 \
    --n_devices 2 \
    --K $K
done
