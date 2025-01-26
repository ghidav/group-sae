import json
import os
from argparse import ArgumentParser
import torch
from safetensors import safe_open

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()


state_dict = safe_open("saes/pythia-160pm-deduped/jr/baseline/8/sae.safetensors", framework="torch")
with open("saes/pythia-160pm-deduped/jr/baseline/8/cfg.json", "r") as f:
    cfg = json.load(f)