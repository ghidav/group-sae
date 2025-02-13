python3.11 interp/cache_activations.py --model_name pythia-160m --cluster

python3.11 interp/explain_cluster.py --model_name=pythia-160m
python3.11 interp/score_cluster.py --model_name=pythia-160m