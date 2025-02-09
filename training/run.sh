torchrun --nproc-per-node=8 --standalone train_cluster_topk.py --model_name pythia-160m --batch 16 --layers 12
torchrun --nproc-per-node=8 --standalone train_cluster_topk.py --model_name pythia-410m --batch 8 --layers 24
torchrun --nproc-per-node=8 --standalone train_cluster_topk.py --model_name pythia-1b --batch 4 --layers 16
