python3.11 recon/recon.py --sae_root_folder training/checkpoints-clusters/pythia-410m-topk/step_15258 --model pythia-410m --batch_size 8

python3.11 recon/recon.py --sae_root_folder training/checkpoints-clusters/pythia-1b-topk/step_30517 --model pythia-1b --batch_size 4

python3.11 recon/recon.py --sae_root_folder training/checkpoints-clusters/pythia-160m-topk/step_7629 --model pythia-160m --batch_size 16

python3.11 recon/recon.py --sae_root_folder training/checkpoints-clusters/pythia-160m-topk/step_7629 --model pythia-160m --cluster --batch_size 16