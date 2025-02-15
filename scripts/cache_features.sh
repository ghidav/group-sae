#python interp/cache_activations.py --model_name pythia-160m --n_tokens 10_000_000 --ctx_len 256 --batch_size 16
#python interp/cache_activations.py --model_name pythia-160m --cluster --n_tokens 10_000_000 --ctx_len 256 --batch_size 16

python interp/cache_activations.py --model_name pythia-410m --n_tokens 10_000_000 --ctx_len 256 --batch_size 16
python interp/cache_activations.py --model_name pythia-410m --cluster --n_tokens 10_000_000 --ctx_len 256 --batch_size 16

python interp/cache_activations.py --model_name pythia-1b --n_tokens 10_000_000 --ctx_len 256 --batch_size 16
python interp/cache_activations.py --model_name pythia-1b --cluster --n_tokens 10_000_000 --ctx_len 256 --batch_size 16
