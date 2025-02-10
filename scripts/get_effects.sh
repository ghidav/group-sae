for K in -1 1 2 3 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=0,1 python downstream/effects.py \
    -d ioi \
    -c resid_post \
    -n 1024 \
    -l 15 \
    -m pythia-160m \
    -mt attrib \
    --sae_root_folder /home/fbelotti/group-sae/saes/pythia_160m-topk \
    --device cuda \
    --effects_dir /home/fbelotti/group-sae/effects_pass_through \
    --task_dir /home/fbelotti/group-sae/downstream/tasks \
    --batch_size 16 \
    --n_devices 2 \
    --K $K
done

for K in -1 1 2 3 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=0,1 python downstream/effects.py \
    -d greater_than \
    -c resid_post \
    -n 1024 \
    -l 12 \
    -m pythia-160m \
    -mt attrib \
    --sae_root_folder /home/fbelotti/group-sae/saes/pythia_160m-topk \
    --device cuda \
    --effects_dir /home/fbelotti/group-sae/effects_pass_through \
    --task_dir /home/fbelotti/group-sae/downstream/tasks \
    --batch_size 16 \
    --n_devices 2 \
    --K $K
done

for K in -1 1 2 3 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=0,1 python downstream/effects.py \
    -d subject_verb \
    -c resid_post \
    -n 1024 \
    -l 6 \
    -m pythia-160m \
    -mt attrib \
    --sae_root_folder /home/fbelotti/group-sae/saes/pythia_160m-topk \
    --device cuda \
    --effects_dir /home/fbelotti/group-sae/effects_pass_through \
    --task_dir /home/fbelotti/group-sae/downstream/tasks \
    --batch_size 16 \
    --n_devices 2 \
    --K $K
done
