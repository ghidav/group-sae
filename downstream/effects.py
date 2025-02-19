import argparse
import gc
import logging
import math
import os
import re
from functools import partial

import torch as t
import transformer_lens
from hooks import sae_hook_pass_through, sae_ig_patching_hook
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from utils import load_examples

from group_sae.utils import MODEL_MAP, get_device_for_block, load_saes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def patching_effect_attrib(
    clean, patch, model, modules, dictionaries, metric_fn, metric_kwargs=dict()
):
    grads = {}
    hidden_states_clean = {}

    sae_hooks = []
    for i, module in enumerate(modules):
        dictionary = dictionaries[module]
        sae_hooks.append(
            (module, partial(sae_hook_pass_through, sae=dictionary, cache=hidden_states_clean))
        )

    # Forward pass with hooks
    logits = model.run_with_hooks(clean, fwd_hooks=sae_hooks)
    metric_clean = metric_fn(logits, **metric_kwargs)

    # Backward pass
    metric_clean.sum().backward()

    # Collect gradients
    for module in modules:
        if module in hidden_states_clean:
            grads[module] = hidden_states_clean[module].grad

    if patch is None:
        hidden_states_patch = {k: t.zeros_like(v) for k, v in hidden_states_clean.items()}
    else:
        hidden_states_patch = {}
        sae_hooks = []
        for i, module in enumerate(modules):
            dictionary = dictionaries[module]
            sae_hooks.append(
                (module, partial(sae_hook_pass_through, sae=dictionary, cache=hidden_states_patch))
            )

        with t.no_grad():
            model.run_with_hooks(patch, fwd_hooks=sae_hooks)

    effects = {}
    deltas = {}
    for module in modules:
        patch_state, clean_state, grad = (
            hidden_states_patch[module],
            hidden_states_clean[module],
            grads[module],
        )
        delta = patch_state - clean_state.detach()
        effect = delta * grad
        effects[module] = effect
        deltas[module] = delta
        grads[module] = grad

    del hidden_states_clean, hidden_states_patch
    gc.collect()

    return effects


def patching_effect_ig(
    clean, patch, model, modules, dictionaries, metric_fn, steps=10, metric_kwargs=dict()
):

    hidden_states_clean = {}
    sae_hooks = []

    # Forward pass through the clean input with hooks to capture hidden states
    for i, module in enumerate(modules):
        dictionary = dictionaries[module]
        sae_hooks.append(
            (module, partial(sae_hook_pass_through, sae=dictionary, cache=hidden_states_clean))
        )

    # First pass to get clean logits and metric
    model.run_with_hooks(clean, fwd_hooks=sae_hooks)

    hidden_states_patch = {}
    sae_hooks_patch = []
    for i, module in enumerate(modules):
        dictionary = dictionaries[module]
        sae_hooks_patch.append(
            (module, partial(sae_hook_pass_through, sae=dictionary, cache=hidden_states_patch))
        )

    with t.no_grad():
        model.run_with_hooks(patch, fwd_hooks=sae_hooks_patch)

    # Integrated gradients computation
    grads = {}
    effects = {}
    deltas = {}

    for module in modules:
        dictionary = dictionaries[module]
        clean_state = hidden_states_clean[module].detach()
        patch_state = hidden_states_patch[module].detach() if patch is not None else None
        delta = (
            (patch_state - clean_state.detach())
            if patch_state is not None
            else -clean_state.detach()
        )

        for step in range(steps + 1):
            interpolated_state_cache = {}
            alpha = step / steps
            interpolated_state = (
                clean_state * (1 - alpha) + patch_state * alpha
                if patch is not None
                else clean_state * (1 - alpha)
            )

            interpolated_state.requires_grad_(True)
            interpolated_state.retain_grad()

            sae_hook_ = [
                (
                    module,
                    partial(
                        sae_ig_patching_hook,
                        sae=dictionary,
                        patch=interpolated_state,
                        cache=interpolated_state_cache,
                    ),
                )
            ]

            # Forward pass with hooks
            logits_interpolated = model.run_with_hooks(clean, fwd_hooks=sae_hook_)
            metric = metric_fn(logits_interpolated, **metric_kwargs)

            # Sum the metrics and backpropagate
            metric.sum().backward(retain_graph=True)

            if module not in grads:
                grads[module] = interpolated_state_cache[module].grad.clone()
            else:
                grads[module] += interpolated_state_cache[module].grad

            if step % (steps // 5) == 0:  # Print every 20% of steps
                del interpolated_state_cache
                t.cuda.empty_cache()

            model.zero_grad(set_to_none=True)

        # Calculate gradients
        grads[module] /= steps

        # Compute effects
        effect = grads[module] * delta
        effects[module] = effect
        deltas[module] = delta

    del hidden_states_clean, hidden_states_patch
    gc.collect()

    return effects


def metric_fn(logits: t.Tensor, patch_answer_idxs: t.Tensor, clean_answer_idxs: t.Tensor):
    patch_answer_idxs = patch_answer_idxs.to(logits.device)
    clean_answer_idxs = clean_answer_idxs.to(logits.device)
    return t.gather(logits[:, -1, :], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(
        -1
    ) - t.gather(logits[:, -1, :], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--K", "-k", type=int, default=-1, help="The number of cluster to be used."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ioi",
        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.",
    )
    parser.add_argument(
        "--component",
        "-c",
        type=str,
        default="resid_post",
        help="The component to test for downstream effects.",
    )
    parser.add_argument(
        "--num_examples",
        "-n",
        type=int,
        default=1024,
        help="The number of examples from the --dataset over which to average indirect effects.",
    )
    parser.add_argument(
        "--example_length",
        "-l",
        type=int,
        default=15,
        help=(
            "The max length (if using sum aggregation) or exact length "
            "(if not aggregating) of examples."
        ),
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="pythia-160m",
        help="The Huggingface ID of the model you wish to test.",
    )
    parser.add_argument(
        "--sae_root_folder",
        type=str,
        default="saes",
        help="Path to all dictionaries for your language model.",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default="downstream/tasks",
        help="The directory to load the task dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of examples to process at once when running circuit discovery.",
    )
    parser.add_argument(
        "--method",
        "-mt",
        type=str,
        default="attrib",
        help="Method to use to compute effects ('attrib' or 'ig').",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--effects_dir", type=str, default="downstream/effects")
    parser.add_argument("--n_devices", type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.effects_dir, exist_ok=True)

    if MODEL_MAP[args.model]["short_name"] not in args.sae_root_folder:
        raise ValueError(
            f"Model name ({args.model_name}) does not match "
            f"the SAE root folder ({args.sae_root_folder})."
        )

    if args.n_devices > 1:
        transformer_lens.utilities.devices.get_device_for_block_index = get_device_for_block

    device = args.device
    patching_effect = patching_effect_attrib if args.method == "attrib" else patching_effect_ig

    model = HookedTransformer.from_pretrained(args.model, device=device, n_devices=args.n_devices)
    model.tokenizer.padding_side = "left"

    nl = model.cfg.n_layers
    nh = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    if args.layer is None:
        layers = list(range(nl - 1))
    else:
        layers = [args.layer]
    modules = [get_act_name(args.component, layer) for layer in layers]
    cluster = args.K != -1

    # loading saes
    dictionaries = load_saes(
        args.sae_root_folder,
        device=device,
        debug=True,
        layer=args.layer,
        cluster=None if args.K == -1 else str(args.K),
        load_from_sae_lens=False,
        dtype="float32",
        model_name=args.model,
    )
    dictionaries = {
        k: v.to(get_device_for_block(int(re.findall(r"\d+", k)[0]), model.cfg, device=device))
        for k, v in dictionaries.items()
    }
    logger.info(f"{len(dictionaries)} dictionaries loaded.")
    if len(dictionaries) == 0:
        raise ValueError("No dictionaries were loaded. Check the path to the dictionaries.")
    elif len(dictionaries) != len(modules):
        logging.warning(
            f"Loaded {len(dictionaries)} dictionaries, but expected {len(modules)}. "
            "Some modules may not have been loaded."
        )
        modules = [k for k in modules if k in dictionaries.keys()]
        dictionaries = {k: v for k, v in dictionaries.items() if k in modules}

    data_path = f"{args.task_dir}/{args.dataset}.json"
    examples = load_examples(data_path, args.num_examples, model, length=args.example_length)
    print(f"Loaded {len(examples)} examples from dataset {args.dataset}.")

    batch_size = args.batch_size
    num_examples = min([args.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size] for batch in range(n_batches)
    ]

    if num_examples < args.num_examples:  # warn the user
        logger.warning(
            f"Total number of examples is less than {args.num_examples}. "
            f"Using {num_examples} examples instead."
        )
    print(
        "Collecting effects for",
        args.component,
        "on",
        args.dataset,
        "with",
        args.method,
        "method.",
    )
    print("A total of", num_examples, "examples will be used.")

    running_nodes = None
    for batch in tqdm(batches, desc="Batches", total=n_batches):
        clean_inputs = t.cat([e["clean_prefix"] for e in batch], dim=0).to(device)
        clean_answer_idxs = t.tensor(
            [e["clean_answer"] for e in batch], dtype=t.long, device=device
        )

        if not args.example_length:
            args.example_length = clean_inputs.shape[1]

        patch_inputs = t.cat([e["patch_prefix"] for e in batch], dim=0).to(device)
        patch_answer_idxs = t.tensor(
            [e["patch_answer"] for e in batch], dtype=t.long, device=device
        )
        effects = patching_effect(
            clean_inputs,
            patch_inputs,
            model,
            modules,
            dictionaries,
            partial(
                metric_fn, patch_answer_idxs=patch_answer_idxs, clean_answer_idxs=clean_answer_idxs
            ),
        )

        nodes = {}
        for module in modules:
            nodes[module] = effects[module]
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}

        if running_nodes is None:
            running_nodes = {k: len(batch) * nodes[k].to("cpu") for k in nodes.keys() if k != "y"}
        else:
            for k in nodes.keys():
                if k != "y":
                    running_nodes[k] += len(batch) * nodes[k].to("cpu")

        del nodes
        gc.collect()

    nodes = {k: v.to(device) / num_examples for k, v in running_nodes.items()}
    save_dict = {"examples": examples, "nodes": nodes}
    save_path = f"{args.effects_dir}/{args.model}_{args.dataset}_"
    if cluster:
        save_path += f"K{args.K}"
    else:
        save_path += "Baseline"
    save_path += f"_n{num_examples}_{args.method}"
    if args.layer is not None:
        save_path += f"_layer{args.layer}"
    save_path += ".pt"
    with open(save_path, "wb") as outfile:
        t.save(save_dict, outfile)
