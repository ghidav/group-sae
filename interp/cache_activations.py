from argparse import ArgumentParser
from functools import partial

import torch
from delphi.autoencoders.OpenAI.model import ACTIVATIONS_CLASSES, TopK
from delphi.autoencoders.wrapper import AutoencoderLatents
from delphi.config import CacheConfig
from delphi.features import FeatureCache
from delphi.utils import load_tokenized_data
from nnsight import LanguageModel

from group_sae.sae import Sae
from group_sae.utils import MODEL_MAP, load_cluster_map

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="pythia-160m-deduped")
parser.add_argument("--sae_folder_path", type=str, default="saes/pythia_160m-topk")
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--G", type=int, default=None)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
full_model_name = f"EleutherAI/{args.model_name}"
model = LanguageModel(full_model_name, device_map=DEVICE, dispatch=True, torch_dtype="float16")

nl = MODEL_MAP[args.model_name]["n_layers"]

if args.cluster:
    if args.G is None:
        raise ValueError("If clustering, G must be specified")
    G = args.G
    map_ = load_cluster_map(args.model_name.split("-")[1])[str(G)]
    unique_mapping = {val: idx for idx, val in enumerate(sorted(set(map_)))}
    data = [unique_mapping[val] for val in map_]
    unique_values = sorted(set(data), key=lambda x: data.index(x))  # Preserve order of appearance
    mapping = {
        val: f"{data.index(val)}-{len(data) - 1 - data[::-1].index(val)}" for val in unique_values
    }
    CLUSTER_MAP = [mapping[val] for val in data]

    def fix_cluster(c):
        start, end = c.split("-")
        if start == end:
            return start
        else:
            return c

    CLUSTER_MAP = [fix_cluster(c) for c in CLUSTER_MAP]
    print(f"Cluster map loaded for G={G}")
    print(CLUSTER_MAP)

print(f"Model {args.model_name} loaded")


def load_saes(path, k):
    submodules = {}
    for layer in range(nl - 1):

        submodule = f"layers.{layer}"
        to_load = CLUSTER_MAP[layer] if args.cluster else layer
        if not args.cluster or ("-" not in str(to_load)):
            sae = Sae.load_from_disk(path + f"/baseline/{layer}", device=DEVICE).to(
                dtype=model.dtype
            )
        else:
            sae = Sae.load_from_disk(path + f"/cluster/{CLUSTER_MAP[layer]}", device=DEVICE).to(
                dtype=model.dtype
            )

        def _forward(sae, k, x):
            encoded = sae.pre_acts(x)
            if k is not None:
                trained_k = k
            else:
                trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)

        submodule = model.gpt_neox.layers[layer]
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae, k), width=sae.encoder.weight.shape[0]
        )

        submodules[submodule.path] = submodule

    with model.edit("") as edited:
        for path, submodule in submodules.items():
            if "embed" not in path and "mlp" not in path:
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited


submodule_dict, model = load_saes(args.sae_folder_path, k=128)

cfg = CacheConfig(
    dataset_repo="EleutherAI/the_pile_deduplicated",
    dataset_split="train[:1%]",
    batch_size=8,
    ctx_len=256,
    n_tokens=1_000_000,
    n_splits=5,
)


tokens = load_tokenized_data(
    ctx_len=cfg.ctx_len,
    tokenizer=model.tokenizer,
    dataset_repo=cfg.dataset_repo,
    dataset_split=cfg.dataset_split,
)
# Tokens should have the shape (n_batches,ctx_len)


cache = FeatureCache(
    model,
    submodule_dict,
    batch_size=cfg.batch_size,
)

cache.run(cfg.n_tokens, tokens)

cache.save_splits(
    n_splits=cfg.n_splits,  # We split the activation and location indices into different files to make loading faster
    save_dir="latents",
)

# The config of the cache should be saved with the results such that it can be loaded later.
cache.save_config(save_dir="latents", cfg=cfg, model_name=full_model_name)
