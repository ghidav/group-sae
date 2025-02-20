for W in faithfulness completeness; do
    for K in -1 1 2 3 4 5 6 7 8 9 10; do
        python ./downstream/faith_topk.py \
        -d ioi \
        -c resid_post \
        -n 1024 \
        -mt attrib \
        --model pythia-160m \
        --sae_root_folder /home/fbelotti/group-sae/saes/pythia_160m-topk \
        --effects_dir /home/fbelotti/group-sae/effects \
        --task_dir /home/fbelotti/group-sae/downstream/tasks \
        --n_devices 1 \
        --batch_size 512 \
        --what $W \
        --K $K
    done

    for K in -1 1 2 3 4 5 6 7 8 9 10; do
        python ./downstream/faith_topk.py \
        -d subject_verb \
        -c resid_post \
        -n 1024 \
        -mt attrib \
        --model pythia-160m \
        --sae_root_folder /home/fbelotti/group-sae/saes/pythia_160m-topk \
        --effects_dir /home/fbelotti/group-sae/effects \
        --task_dir /home/fbelotti/group-sae/downstream/tasks \
        --n_devices 1 \
        --batch_size 512 \
        --what $W \
        --K $K
    done

    for K in -1 1 2 3 4 5 6 7 8 9 10; do
        python ./downstream/faith_topk.py \
        -d greater_than \
        -c resid_post \
        -n 1024 \
        -mt attrib \
        --model pythia-160m \
        --sae_root_folder /home/fbelotti/group-sae/saes/pythia_160m-topk \
        --effects_dir /home/fbelotti/group-sae/effects \
        --task_dir /home/fbelotti/group-sae/downstream/tasks \
        --n_devices 1 \
        --batch_size 512 \
        --what $W \
        --K $K
    done
done
