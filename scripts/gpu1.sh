python3.11 interp/cache_activations.py --model_name pythia-160m --n_tokens 100_000 --n_splits 5 --batch_size 8 --ctx_len 512 --cluster --G 2
python3.11 interp/cache_activations.py --model_name pythia-160m --n_tokens 100_000 --n_splits 5 --batch_size 8 --ctx_len 512 --cluster --G 3
python3.11 interp/cache_activations.py --model_name pythia-410m --n_tokens 100_000 --n_splits 5 --batch_size 8 --ctx_len 512 --cluster --G 1
python3.11 interp/cache_activations.py --model_name pythia-410m --n_tokens 100_000 --n_splits 5 --batch_size 8 --ctx_len 512 --cluster --G 2
