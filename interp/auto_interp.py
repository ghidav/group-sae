import asyncio
import json
import os
from argparse import ArgumentParser
from functools import partial

import orjson
import torch

from delphi.clients import OpenRouter
from delphi.config import ExperimentConfig, FeatureConfig
from delphi.explainers import DefaultExplainer
from delphi.features import FeatureDataset, FeatureLoader
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
from delphi.pipeline import Pipeline, process_wrapper
from group_sae.utils import MODEL_MAP


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

# Load API keys
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(f"{script_dir}/../keys.json", "r") as f:
    keys = json.load(f)
API_KEY = os.getenv(keys["openrouter"])

# Get model parameters from the mapping
n_layers = MODEL_MAP[args.model_name]["n_layers"]
d_model = MODEL_MAP[args.model_name]["d_model"]

# Feature configuration
feature_cfg = FeatureConfig(
    width=d_model * 16,  # The number of latents of your SAE
    min_examples=200,    # The minimum number of examples to consider for the feature to be explained
    max_examples=10000,  # The maximum number of examples to be sampled from
    n_splits=5,          # How many splits was the cache split into
)

# Define module and feature dictionary
module = ".model.layers.10"  # The layer to explain
feature_dict = {module: torch.arange(0, 100)}  # The latents to explain

# Create dataset
dataset = FeatureDataset(
    raw_dir="latents",  # The folder where the cache is stored
    cfg=feature_cfg,
    modules=[module],
    features=feature_dict,
)

# Experiment configuration
experiment_cfg = ExperimentConfig(
    n_examples_train=40,  # Number of examples to sample for training
    example_ctx_len=32,   # Length of each example
    train_type="random",  # Type of sampler to use for training.
)

# Create constructor and sampler for loading features
constructor = partial(
    default_constructor,
    n_random=experiment_cfg.n_random,
    ctx_len=experiment_cfg.example_ctx_len,
    max_examples=feature_cfg.max_examples,
)
sampler = partial(sample, cfg=experiment_cfg)
loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)

# Initialize the client
client = OpenRouter("google/gemini-2.0-flash-001", api_key=API_KEY)


def explainer_postprocess(result):
    """Post-process the explainer result."""
    output_path = f"results/explanations/{result.record.feature}.txt"
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(result.explanation))
    del result
    return None


# Create the explainer pipeline
explainer_pipe = process_wrapper(
    DefaultExplainer(
        client,
        tokenizer=dataset.tokenizer,
    ),
    postprocess=explainer_postprocess,
)

# Assemble the main pipeline
pipeline = Pipeline(loader, explainer_pipe)

# Run the pipeline asynchronously
number_of_parallel_latents = 10
asyncio.run(pipeline.run(number_of_parallel_latents))