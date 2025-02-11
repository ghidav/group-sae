for W in faithfulness completeness; do
    for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
        python ./downstream/faith_topk.py \
        -d ioi \
        -c resid_post \
        -n 1024 \
        -mt attrib \
        --model pythia-1b \
        --sae_root_folder /home/fbelotti/group-sae/saes/pythia_1b-topk \
        --effects_dir /home/fbelotti/group-sae/effects \
        --task_dir /home/fbelotti/group-sae/downstream/tasks \
        --faith_dir faithfulness_topk \
        --n_devices 2 \
        --batch_size 512 \
        --what $W \
        --K $K
    done

    for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
        python ./downstream/faith_topk.py \
        -d subject_verb \
        -c resid_post \
        -n 1024 \
        -mt attrib \
        --model pythia-1b \
        --sae_root_folder /home/fbelotti/group-sae/saes/pythia_1b-topk \
        --effects_dir /home/fbelotti/group-sae/effects \
        --task_dir /home/fbelotti/group-sae/downstream/tasks \
        --faith_dir faithfulness_topk \
        --n_devices 2 \
        --batch_size 512 \
        --what $W \
        --K $K
    done

    for K in -1 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
        python ./downstream/faith_topk.py \
        -d greater_than \
        -c resid_post \
        -n 1024 \
        -mt attrib \
        --model pythia-1b \
        --sae_root_folder /home/fbelotti/group-sae/saes/pythia_1b-topk \
        --effects_dir /home/fbelotti/group-sae/effects \
        --task_dir /home/fbelotti/group-sae/downstream/tasks \
        --faith_dir faithfulness_topk \
        --n_devices 2 \
        --batch_size 512 \
        --what $W \
        --K $K
    done
done
