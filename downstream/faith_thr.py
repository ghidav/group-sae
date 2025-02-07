import argparse
import logging
import os
import re
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformer_lens
from hooks import sae_features_hook, sae_hook_pass_through
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from utils import load_examples, load_saes

from group_sae.utils import get_device_for_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_circuit(
    tokens,
    clean_answers,
    patch_answers,
    nodes,
    saes,
    feature_avg,
    feature_mask,
    use_resid=False,
    what="faithfulness",
    device="cuda",
):

    hooks = []
    for hook_name in nodes.keys():
        hooks.append(
            (
                hook_name,
                partial(
                    sae_features_hook,
                    sae=saes[hook_name],
                    feature_mask=feature_mask[hook_name],
                    feature_avg=feature_avg[hook_name],
                    resid=use_resid,
                    ablation=what,
                ),
            )
        )

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens.to(device),
            fwd_hooks=hooks,
        ).cpu()
        logits = logits[:, -1]

    clean_ans_logits = torch.gather(logits, 1, clean_answers.unsqueeze(1))
    patch_ans_logits = torch.gather(logits, 1, patch_answers.unsqueeze(1))
    return (clean_ans_logits - patch_ans_logits).squeeze()


def faithfulness(
    tokens,
    clean_answers,
    patch_answers,
    nodes,
    dictionaries,
    feature_avg,
    feature_mask,
    use_resid=False,
    device="cuda",
):

    # Get the model's logit diff - m(M)
    with torch.no_grad():
        logits = model(tokens.to(device)).cpu()
        logits = logits[:, -1]

    clean_ans_logits = torch.gather(logits, 1, clean_answers.unsqueeze(1))
    patch_ans_logits = torch.gather(logits, 1, patch_answers.unsqueeze(1))

    M = (clean_ans_logits - patch_ans_logits).squeeze().mean().item()

    # Get the circuit's logit diff - m(C)
    C = test_circuit(
        tokens,
        clean_answers,
        patch_answers,
        nodes,
        dictionaries,
        feature_avg,
        feature_mask=feature_mask,
        use_resid=use_resid,
        what=args.what,
        device=device,
    )

    # Get the ablated circuit's logit diff - m(zero)
    zero = test_circuit(
        tokens,
        clean_answers,
        patch_answers,
        nodes,
        dictionaries,
        feature_avg,
        feature_mask=feature_mask,
        use_resid=use_resid,
        what="empty",
        device=device,
    )

    return (C.mean().item() - zero.mean().item()) / (M - zero.mean().item() + 1e-7)


##########
## Main ##
##########

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--K", "-k", type=int, default=-1, help="The number of cluster to be used."
    )
    parser.add_argument("--active_features", nargs="+", default=[1, 61, 123, 246, 368, 492])
    parser.add_argument("-c", "--component", type=str, default="resid_post")
    parser.add_argument("-n", "--n", type=int, default=1024)
    parser.add_argument("-mt", "--method", type=str, default="attrib")
    parser.add_argument("-w", "--what", type=str, default="faithfulness")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ioi",
        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.",
    )
    parser.add_argument(
        "--effects_dir",
        type=str,
        default="effects",
        help="The directory to load the effects.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pythia-160m-deduped",
        help="The Huggingface ID of the model you wish to test.",
    )
    parser.add_argument(
        "--faith_dir", type=str, default="faithfulness", help="The directory to save faithfulness."
    )
    parser.add_argument(
        "--task_dir", type=str, default="tasks", help="The directory to load the task dataset."
    )
    parser.add_argument(
        "--sae_folder_path",
        type=str,
        default="saes",
        help="Path to all dictionaries for your language model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The batch size to use for testing."
    )
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()

    if args.n_devices > 1:
        transformer_lens.utilities.devices.get_device_for_block_index = get_device_for_block

    n = args.n
    task = args.dataset
    lengths = {"ioi": 15, "greater_than": 12, "subject_verb": 6}
    args.active_features = [int(k) for k in args.active_features]

    model = HookedTransformer.from_pretrained(args.model, device="cuda", n_devices=args.n_devices)
    device = model.cfg.device
    if device is None:
        device = "cuda"
        model.cfg.device = device
    if args.layer is None:
        layers = list(range(model.cfg.n_layers))
    else:
        layers = [args.layer]
    modules = [get_act_name(args.component, layer) for layer in layers]
    cluster = args.K != -1
    dictionaries = load_saes(
        args.sae_folder_path,
        device=device,
        debug=True,
        layer=args.layer,
        cluster=None if args.K == -1 else args.K,
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

    train_examples = load_examples(
        f"{args.task_dir}/{task}.json", 2 * n, model, length=lengths[task]
    )[:n]
    test_examples = load_examples(
        f"{args.task_dir}/{task}.json", 2 * n, model, length=lengths[task]
    )[n : 2 * n]
    if len(test_examples) < 64:
        test_examples = load_examples(
            f"{args.task_dir}/{task}.json", 2 * n, model, length=lengths[task]
        )

    assert len(test_examples) > 64, (
        f"Not enough examples can be loaded, train length: ({len(train_examples)}), "
        f"test length: ({len(test_examples)})"
    )

    train_tokens = torch.cat([e["clean_prefix"] for e in train_examples])
    feature_cache = {}
    feature_avg = {}
    hooks = [
        (
            hook_name,
            partial(
                sae_hook_pass_through,
                sae=sae,
                cache=feature_cache,
            ),
        )
        for hook_name, sae in dictionaries.items()
    ]
    with torch.no_grad():
        for i in range(0, len(train_tokens), args.batch_size):
            model.run_with_hooks(
                train_tokens[i : i + args.batch_size].to(device),
                fwd_hooks=hooks,
            )
            if len(feature_avg) == 0:
                for hook_name in feature_cache.keys():
                    feature_avg[hook_name] = feature_cache[hook_name].sum(0)
            else:
                for hook_name in feature_cache.keys():
                    feature_avg[hook_name] += feature_cache[hook_name].sum(0)
            feature_cache.clear()
        for hook_name in feature_avg.keys():
            feature_avg[hook_name] /= len(train_tokens)

    cluster = args.K != -1
    effects_path = f"{args.effects_dir}/"
    if cluster:
        effects_path += f"K{args.K}"
    else:
        effects_path += "Baseline"
    effects_path += f"_{args.dataset}_n{n}_{args.method}"
    if args.layer is not None:
        effects_path += f"_layer{args.layer}"
    effects_path += ".pt"
    effects = torch.load(effects_path)["nodes"]

    test_tokens = torch.cat([e["clean_prefix"] for e in test_examples])
    clean_answers = torch.tensor([e["clean_answer"] for e in test_examples])
    patch_answers = torch.tensor([e["patch_answer"] for e in test_examples])

    scores = []
    Ns = []
    for T in tqdm(np.exp(np.linspace(-10, np.log(100), 64))):
        feature_mask = {}
        for hook_name in effects.keys():
            feature_mask[hook_name] = (effects[hook_name].abs() > T).sum(0) > 0
        N = np.mean([feature_mask[hook_name].sum().item() for hook_name in feature_mask.keys()])

        score = 0
        for i in range(0, len(test_tokens), args.batch_size):
            batch_score = faithfulness(
                test_tokens[i : i + args.batch_size],
                clean_answers[i : i + args.batch_size],
                patch_answers[i : i + args.batch_size],
                effects,
                dictionaries,
                feature_avg=feature_avg,
                feature_mask=feature_mask,
                use_resid=True,
                device=device,
            )
            # batch_score and batch_N are the mean faithfulness score and
            # the number of active features in the current batch
            # Update them to the total score and the total number of active features
            current_batch_len = len(test_tokens[i : i + args.batch_size])
            score += batch_score * current_batch_len
        score /= len(test_tokens)
        print(f"T: {T:.4f}, score: {score:.4f}, N: {N:.4f}")
        scores.append(score)
        Ns.append(N)

    score_df = pd.DataFrame({"score": scores, "N": Ns})
    faith_result_path = f"{args.faith_dir}/"
    os.makedirs(faith_result_path, exist_ok=True)
    if cluster:
        faith_result_path += f"K{args.K}"
    else:
        faith_result_path += "Baseline"
    faith_result_path += f"_{args.dataset}_{args.method}_{args.what}"
    if args.layer is not None:
        faith_result_path += f"_layer{args.layer}"
    faith_result_path += ".csv"
    score_df.to_csv(faith_result_path, index=False)
