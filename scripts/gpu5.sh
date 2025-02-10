python3.11 interp/cache_activations.py --model_name pythia-410m --n_tokens 100_000 --batch_size 8 --ctx_len 512 --cluster --G 10

python3.11 interp/cache_activations.py --model_name pythia-1b --n_tokens 100_000 --batch_size 8 --ctx_len 512
python3.11 interp/cache_activations.py --model_name pythia-1b --n_tokens 100_000 --batch_size 8 --ctx_len 512 --cluster --G 1