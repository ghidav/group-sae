from functools import partial
import os
import json
import torch
import orjson
import asyncio
from delphi.clients import OpenRouter
from delphi.config import ExperimentConfig, LatentConfig
from delphi.latents import LatentDataset, LatentLoader
from delphi.latents.constructors import default_constructor
from delphi.explainers import DefaultExplainer
from delphi.latents.samplers import sample
from delphi.pipeline import Pipeline, process_wrapper

from group_sae.utils import MODEL_MAP
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description=(
            "Extract SAE activations from a model and save them as safetensors, "
            "optionally in token splits."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pythia-160m",
        help="Name of the model (e.g. 'pythia-160m').",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Layer to select feature from.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Use clustering when loading SAEs.",
    )
    parser.add_argument(
        "--G",
        type=int,
        default=None,
        help="G parameter for clustering (required if --cluster is set).",
    )
    return parser.parse_args()


# Parse command-line arguments
args = parse_args()
G = str(args.G) if args.G else "baseline"

# Load API keys
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(f"{script_dir}/../keys.json", "r") as f:
    keys = json.load(f)
API_KEY = keys["openrouter"]

explain_dir = f"{script_dir}/results/explanations/{args.model_name}/{G}"
os.makedirs(explain_dir, exist_ok=True)

# Get model parameters from the mapping
n_layers = MODEL_MAP[args.model_name]["n_layers"]
d_model = MODEL_MAP[args.model_name]["d_model"]

# Configurations
latent_cfg = LatentConfig(
    width=d_model * 16,  # The number of latents of your SAE
    min_examples=200,  # The minimum number of examples to consider for the feature to be explained
    max_examples=10000,  # The maximum number of examples to be sampled from
    n_splits=1,  # How many splits was the cache split into
)

# Define module and feature dictionary
module = f".gpt_neox.layers.{args.layer}"  # The layer to explain
feature_dict = {module: torch.arange(0, 32)}  # The latents to explain

# Create dataset
dataset = LatentDataset(
    raw_dir=f"latents/{args.model_name}/{G}",  # The folder where the cache is stored
    cfg=latent_cfg,
    modules=[module],
    latents=feature_dict,
)

# Experiment configuration
experiment_cfg = ExperimentConfig(
    n_examples_test=10,  # Number of examples to sample for testing
    n_quantiles=10,  # Number of quantiles to divide the data into
    test_type="quantiles",  # Type of sampler to use for testing.
    n_non_activating=10,  # Number of non-activating examples to sample
    example_ctx_len=32,  # Length of each example
)

# Create constructor and sampler for loading features
constructor = partial(
    default_constructor,
    token_loader=None,
    n_not_active=experiment_cfg.n_non_activating,
    ctx_len=experiment_cfg.example_ctx_len,
    max_examples=latent_cfg.max_examples,
)
sampler = partial(sample, cfg=experiment_cfg)
loader = LatentLoader(dataset, constructor=constructor, sampler=sampler)

# Initialize the client
client = OpenRouter("google/gemini-2.0-flash-001", api_key=API_KEY)


# Load the explanations already generated
def explainer_postprocess(result):
    with open(f"{explain_dir}/{result.record.latent}.txt", "wb") as f:
        f.write(orjson.dumps(result.explanation))
    del result
    return None


explainer_pipe = process_wrapper(
    DefaultExplainer(
        client,
        tokenizer=dataset.tokenizer,
    ),
    postprocess=explainer_postprocess,
)


# Final pipeline
pipeline = Pipeline(loader, explainer_pipe)
number_of_parallel_latents = 4


async def main():
    await pipeline.run(number_of_parallel_latents)


if __name__ == "__main__":
    asyncio.run(main())
