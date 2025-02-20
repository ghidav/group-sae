
python interp/cache_activations.py --model_name pythia-160m
python interp/cache_activations.py --model_name pythia-160m --cluster
python interp/explain_baseline.py --model_name pythia-160m
python interp/score_baseline.py --model_name pythia-160m
python interp/explain_cluster.py --model_name pythia-160m
python interp/score_cluster.py --model_name pythia-160m

python interp/cache_activations.py --model_name pythia-410m
python interp/cache_activations.py --model_name pythia-410m --cluster
python interp/explain_baseline.py --model_name pythia-410m
python interp/score_baseline.py --model_name pythia-410m
python interp/explain_cluster.py --model_name pythia-410m
python interp/score_cluster.py --model_name pythia-410m

python interp/cache_activations.py --model_name pythia-1b
python interp/cache_activations.py --model_name pythia-1b --cluster
python interp/explain_baseline.py --model_name pythia-1b
python interp/score_baseline.py --model_name pythia-1b
python interp/explain_cluster.py --model_name pythia-1b
python interp/score_cluster.py --model_name pythia-1b
