import asyncio
import json
import os
from argparse import ArgumentParser
from functools import partial

import orjson
import torch
from delphi.clients import OpenRouter
from delphi.config import ExperimentConfig, LatentConfig
from delphi.explainers import DefaultExplainer
from delphi.latents import LatentDataset, LatentLoader
from delphi.latents.constructors import default_constructor
from delphi.latents.samplers import sample
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
        "--max_pipelines",
        type=int,
        default=2,  # Change this default as needed.
        help="Maximum number of pipelines to run concurrently.",
    )
    return parser.parse_args()


# Parse command-line arguments
args = parse_args()

# Load API keys
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(f"{script_dir}/../keys.json", "r") as f:
    keys = json.load(f)
API_KEY = keys["openrouter"]

client = OpenRouter("google/gemini-2.0-flash-001", api_key=API_KEY)

# Get model parameters from the mapping
n_layers = MODEL_MAP[args.model_name]["n_layers"]
d_model = MODEL_MAP[args.model_name]["d_model"]

# Configurations
latent_cfg = LatentConfig(
    width=d_model * 16,  # The number of latents of your SAE
    min_examples=100,  # The minimum number of examples to consider for the feature to be explained
    max_examples=1000,  # The maximum number of examples to be sampled from
    n_splits=1,  # How many splits was the cache split into
)

# Explanation loop
number_of_parallel_latents = 4

experiment_cfg = ExperimentConfig(
    n_examples_test=10,  # Number of examples to sample for testing
    n_quantiles=10,  # Number of quantiles to divide the data into
    test_type="quantiles",  # Type of sampler to use for testing.
    n_non_activating=10,  # Number of non-activating examples to sample
    example_ctx_len=32,  # Length of each example
)

constructor = partial(
    default_constructor,
    token_loader=None,
    n_not_active=experiment_cfg.n_non_activating,
    ctx_len=experiment_cfg.example_ctx_len,
    max_examples=latent_cfg.max_examples,
)

sampler = partial(sample, cfg=experiment_cfg)

# Baselines
explain_dir = f"{script_dir}/results/explanations/{args.model_name}/baseline"
os.makedirs(explain_dir, exist_ok=True)  # Ensure directory exists


def explainer_postprocess(result):
    """Post-processes and saves explanations"""
    output_file = os.path.join(explain_dir, f"{result.record.latent}.txt")
    with open(output_file, "wb") as f:
        f.write(orjson.dumps(result.explanation))
    del result
    return None


async def run_pipeline_for_layer(layer, pipeline, semaphore):
    """Runs the pipeline for a given layer asynchronously with concurrency control."""
    async with semaphore:
        print(f"Starting pipeline for layer {layer}...")
        await pipeline.run(number_of_parallel_latents)
        print(f"Finished pipeline for layer {layer}")


async def main():
    tasks = []  # Store all pipeline tasks

    # Create a semaphore to limit concurrent pipelines.
    semaphore = asyncio.Semaphore(args.max_pipelines)

    for layer in range(n_layers - 1):  # Create a pipeline for each layer
        module = f".gpt_neox.layers.{layer}"
        feature_dict = {module: torch.arange(0, 128)}

        dataset = LatentDataset(
            raw_dir=f"interp/latents/{args.model_name}/baseline/layers.{layer}",
            cfg=latent_cfg,
            modules=[module],
            latents=feature_dict,
        )

        loader = LatentLoader(dataset, constructor=constructor, sampler=sampler)

        explainer_pipe = process_wrapper(
            DefaultExplainer(
                client,
                tokenizer=dataset.tokenizer,
            ),
            postprocess=explainer_postprocess,
        )

        pipeline = Pipeline(loader, explainer_pipe)

        # Add pipeline task with semaphore control to the async task list
        tasks.append(run_pipeline_for_layer(layer, pipeline, semaphore))

    # Run all pipelines concurrently (with at most args.max_pipelines at a time)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
