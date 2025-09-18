import argparse
import contextlib
import io
import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoTokenizer


# =========================
# Your renderer (verbatim)
# =========================
def get_top_activating_tokens(
    tokens: torch.Tensor,
    features: torch.Tensor,
    activations: torch.Tensor,
    feature_idx,
    top_k=5,
    context_length=16,
    print_highlighted_only: bool = True,
):
    """
    Find the top-k most activating tokens for a given feature and decode them with context.
    Assumes a global `tokenizer`.
    """
    print(f"=== TOP {top_k} ACTIVATING TOKENS FOR FEATURE {feature_idx} ===")

    # Find all positions where this feature appears
    feature_positions = torch.where(features == feature_idx)
    if len(feature_positions[0]) == 0:
        print(f"Feature {feature_idx} never activates!")
        return

    # Get the activation values for this feature at all positions where it appears
    activation_values = activations[feature_positions]

    # Get top-k positions with highest activations
    top_k_indices = torch.topk(activation_values, min(top_k, len(activation_values))).indices

    print(f"Feature {feature_idx} activates {len(feature_positions[0])} times")
    print(f"Activation range: {activation_values.min():.4f} to {activation_values.max():.4f}")
    print()

    for rank, idx in enumerate(top_k_indices):
        token_idx = feature_positions[0][idx].item()
        activation_value = activation_values[idx].item()

        # Calculate the original token position in the sequence
        batch_size = tokens.shape[1] if len(tokens.shape) > 1 else len(tokens)
        original_token_pos = int(token_idx % batch_size)
        sequence_idx = int(token_idx // batch_size)

        # Get the context around this token
        start_pos = max(0, original_token_pos - context_length)
        end_pos = min(
            len(tokens) if len(tokens.shape) == 1 else tokens.shape[1],
            original_token_pos + context_length + 1,
        )

        if len(tokens.shape) == 1:
            context_tokens = tokens[start_pos:end_pos]
            target_token = tokens[original_token_pos]
        else:
            context_tokens = tokens[sequence_idx, start_pos:end_pos]
            target_token = tokens[sequence_idx, original_token_pos]

        # Highlight the target token in context
        target_pos_in_context = original_token_pos - start_pos
        context_tokens_list = context_tokens.cpu().numpy().tolist()
        decoded_context_tokens = [
            tokenizer.decode([t], skip_special_tokens=False) for t in context_tokens_list
        ]

        highlighted_context = ""
        for i, token_text in enumerate(decoded_context_tokens):
            if i == target_pos_in_context:
                highlighted_context += f">>>{token_text}<<<"
            else:
                highlighted_context += token_text

        print(f"Highlighted: {repr(highlighted_context)}")
        print()


# =========================
# Data types
# =========================
@dataclass
class PairRecord:
    group_layer: str
    baseline_layer: str
    group_feat: int
    baseline_feat: int
    similarity: float


@dataclass
class FeatureCard:
    feature_id: int
    top_tokens: List[str]
    snippets: List[Dict[str, str]]  # {"left": "...", "token": "X", "right": "..."}


@dataclass
class PairExplanation:
    pair: PairRecord
    title: str
    tags: List[str]
    explanation_md: str


# =========================
# Concordance helpers
# =========================
def load_concordance(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_pairs(concordance: Dict[str, Dict[str, Dict[str, Any]]]) -> Iterable[PairRecord]:
    """
    Expects: concordance[group_layer][baseline_layer]["shared_features"] = [(group_feat, baseline_feat, similarity), ...]
    """
    for g_layer, inner in concordance.items():
        for b_layer, stats in inner.items():
            for g_feat, b_feat, sim in stats.get("shared_features", []):
                yield PairRecord(
                    group_layer=str(g_layer),
                    baseline_layer=str(b_layer),
                    group_feat=int(g_feat),
                    baseline_feat=int(b_feat),
                    similarity=float(sim),
                )


def build_samples_per_pair(
    concordance: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    bin_edges: List[float],
    n_per_bin: int,
    layers_filter: Optional[set] = None,  # set of layer names/ids to include; None = all
    rng: random.Random = random.Random(),
) -> List[PairRecord]:
    """
    For EACH (group_layer, baseline_layer) pair:
      - bin that pair's shared_features by similarity
      - sample up to n_per_bin from each bin
    Returns the concatenated samples for all pairs.
    """

    def _bin_index(sim: float) -> Optional[int]:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= sim < bin_edges[i + 1]:
                return i
        return None

    all_samples: List[PairRecord] = []

    for g_layer, inner in concordance.items():
        if layers_filter is not None and str(g_layer) not in layers_filter:
            continue

        for b_layer, stats in inner.items():
            if layers_filter is not None and str(b_layer) not in layers_filter:
                continue

            shared = stats.get("shared_features", [])
            # buckets for this pair only
            buckets: Dict[int, List[Tuple[int, int, float]]] = {
                i: [] for i in range(len(bin_edges) - 1)
            }
            for g_feat, b_feat, sim in shared:
                idx = _bin_index(float(sim))
                if idx is not None:
                    buckets[idx].append((int(g_feat), int(b_feat), float(sim)))

            # debug print per-pair (optional)
            # print(f"[pair {g_layer}-{b_layer}] counts per bin:",
            #       [len(buckets[i]) for i in range(len(bin_edges)-1)])

            # sample n_per_bin within this pair
            for i in range(len(bin_edges) - 1):
                bucket = buckets[i]
                if not bucket:
                    continue
                chosen = bucket if len(bucket) <= n_per_bin else rng.sample(bucket, n_per_bin)
                for g_feat, b_feat, sim in chosen:
                    all_samples.append(
                        PairRecord(
                            group_layer=str(g_layer),
                            baseline_layer=str(b_layer),
                            group_feat=g_feat,
                            baseline_feat=b_feat,
                            similarity=sim,
                        )
                    )

    return all_samples


def sample_per_bin(
    bins: Dict[int, List[PairRecord]], n_per_bin: int, rng: random.Random
) -> List[PairRecord]:
    out = []
    for _, bucket in bins.items():
        if not bucket:
            continue
        out.extend(bucket if len(bucket) <= n_per_bin else rng.sample(bucket, n_per_bin))
    return out


# =========================
# Parse renderer output -> FeatureCard
# =========================
HIGHLIGHT_RE = re.compile(r"""^Highlighted:\s*['"](?P<txt>.*?)['"]\s*$""", re.DOTALL)


def _extract_token_and_context(s: str, context_chars: int = 60):
    start = s.find(">>>")
    end = s.find("<<<", start + 3) if start >= 0 else -1
    if start >= 0 and end > start:
        token = s[start + 3 : end].strip()
        left = s[max(0, start - context_chars) : start].strip()
        right = s[end + 3 : end + 3 + context_chars].strip()
        return token, left, right

    # Fallback: center-ish window
    words = re.findall(r"\w+|\S", s)
    if not words:
        return "", s[:context_chars], s[-context_chars:]
    mid = len(words) // 2
    token = words[mid]
    full = "".join(words)
    pos = full.find(token)
    left = full[max(0, pos - context_chars) : pos].strip()
    right = full[pos + len(token) : pos + len(token) + context_chars].strip()
    return token, left, right


def parse_feature_dump_to_card(
    text: str, feature_id: int, top_k: int, max_snippets: int
) -> FeatureCard:
    highlights: List[str] = []
    for line in text.splitlines():
        m = HIGHLIGHT_RE.match(line.strip())
        if m:
            highlights.append(m.group("txt"))

    tokens_list: List[str] = []
    snippets: List[Dict[str, str]] = []
    for h in highlights:
        tok, left, right = _extract_token_and_context(h, context_chars=60)
        if tok:
            tokens_list.append(tok)
        snippets.append({"left": left, "token": tok, "right": right})

    from collections import Counter

    freq = Counter(tokens_list)
    top_tokens = [t for t, _ in freq.most_common(top_k)]

    if len(snippets) > max_snippets:
        snippets = snippets[:max_snippets]

    return FeatureCard(feature_id=feature_id, top_tokens=top_tokens, snippets=snippets)


# =========================
# Disk loaders (match your notebook)
# =========================
class DiskEnv:
    """
    Loads tokens once from the GROUP path (all_tokens.npy),
    and per-layer features/activations for both group and baseline.
    """

    def __init__(
        self,
        *,
        k: str,
        layers: List[int],
        tokens_dir_template: str,
        features_dir_template: str,
        activations_dir_template: str,
        device: str = "cpu",
        feat_width: int = 128,
    ):
        self.k = str(k)
        self.layers = [str(x) for x in layers]
        self.tokens_dir_template = tokens_dir_template
        self.features_dir_template = features_dir_template
        self.activations_dir_template = activations_dir_template
        self.device = device
        self.feat_width = feat_width

        # Resolve templates
        self.group_tokens_dir = os.path.expanduser(tokens_dir_template.format(k=self.k))
        self.group_features_dir = os.path.expanduser(features_dir_template.format(k=self.k))
        self.group_activations_dir = os.path.expanduser(activations_dir_template.format(k=self.k))

        self.base_features_dir = os.path.expanduser(features_dir_template.format(k="baseline"))
        self.base_activations_dir = os.path.expanduser(
            activations_dir_template.format(k="baseline")
        )

        # Load tokens (once)
        tokens_path = os.path.join(self.group_tokens_dir, "all_tokens.npy")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Tokens file not found: {tokens_path}")
        tokens_np = np.load(tokens_path)
        self.tokens = torch.from_numpy(tokens_np).to(self.device)

        # Caches per layer
        self._group_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._base_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        # Optional: set CUDA device
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)

    def _load_layer(self, base_or_group: str, layer: str) -> Dict[str, torch.Tensor]:
        assert base_or_group in ("group", "baseline")
        if base_or_group == "group" and layer in self._group_cache:
            return self._group_cache[layer]
        if base_or_group == "baseline" and layer in self._base_cache:
            return self._base_cache[layer]

        if base_or_group == "group":
            fpath = os.path.join(self.group_features_dir, f"blocks.{layer}.hook_resid_post.npy")
            apath = os.path.join(self.group_activations_dir, f"blocks.{layer}.hook_resid_post.npy")
        else:
            fpath = os.path.join(self.base_features_dir, f"blocks.{layer}.hook_resid_post.npy")
            apath = os.path.join(self.base_activations_dir, f"blocks.{layer}.hook_resid_post.npy")

        if not (os.path.exists(fpath) and os.path.exists(apath)):
            raise FileNotFoundError(
                f"Missing files for {base_or_group} layer {layer}:\n  {fpath}\n  {apath}"
            )

        features = torch.from_numpy(np.load(fpath)).reshape(-1, self.feat_width).to(self.device)
        activations = torch.from_numpy(np.load(apath)).reshape(-1, self.feat_width).to(self.device)

        entry = {"features": features, "activations": activations}
        if base_or_group == "group":
            self._group_cache[layer] = entry
        else:
            self._base_cache[layer] = entry
        return entry

    def get_group_tensors(self, layer: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self._load_layer("group", layer)
        return self.tokens, entry["features"], entry["activations"]

    def get_baseline_tensors(self, layer: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self._load_layer("baseline", layer)
        return self.tokens, entry["features"], entry["activations"]


# =========================
# Rendering wrappers
# =========================
def render_with_capture(
    tokens, features, activations, feature_id: int, top_k_tokens: int, context_len: int
) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        get_top_activating_tokens(
            tokens=tokens,
            features=features,
            activations=activations,
            feature_idx=feature_id,
            top_k=top_k_tokens,
            context_length=context_len,
            print_highlighted_only=True,
        )
    return buf.getvalue()


def make_group_renderer(env: DiskEnv, top_k_tokens: int, context_len: int):
    def _render(layer: str, feature_id: int) -> str:
        tokens, feats, acts = env.get_group_tensors(layer)
        return render_with_capture(tokens, feats, acts, feature_id, top_k_tokens, context_len)

    return _render


def make_baseline_renderer(env: DiskEnv, top_k_tokens: int, context_len: int):
    def _render(layer: str, feature_id: int) -> str:
        tokens, feats, acts = env.get_baseline_tensors(layer)
        return render_with_capture(tokens, feats, acts, feature_id, top_k_tokens, context_len)

    return _render


# =========================
# OpenAI helpers
# =========================
class TransientOpenAIError(Exception):
    pass


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("Please set OPENAI_API_KEY in your environment.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="REMOVED",
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type(TransientOpenAIError),
)
def chat(
    client: OpenAI,
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict] = None,
) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,  # {"type":"json_object"} for strict JSON
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["rate", "timeout", "overloaded", "temporar", "503"]):
            raise TransientOpenAIError(e)
        raise


def feature_card_md(side_name: str, layer: str, card: FeatureCard) -> str:
    toks = ", ".join(f"`{t}`" for t in card.top_tokens)
    lines = [
        f"- …{s.get('left','')} **{s.get('token','')}** {s.get('right','')}…"
        for s in card.snippets
    ]
    snips = "\n".join(lines) if lines else "_(no snippets)_"
    return (
        f"### {side_name} feature {card.feature_id} (layer `{layer}`)\n"
        f"- Top tokens: {toks}\n"
        f"- Snippets:\n{snips}\n"
    )


def make_explain_prompt(
    pair: PairRecord, g_card: FeatureCard, b_card: FeatureCard
) -> List[Dict[str, str]]:
    system = (
        "You are an expert in sparse autoencoders and interpretability. "
        "Given two features (Group-SAE vs Baseline-SAE) with top-activating tokens and short contexts, "
        "explain what each encodes and how/why they overlap or differ. "
        "Write crisp, specific, falsifiable text suitable for a camera-ready caption."
    )
    user = (
        f"Pair:\n"
        f"- Jaccard similarity: {pair.similarity:.3f}\n\n"
        f"{feature_card_md('GROUP', pair.group_layer, g_card)}\n"
        f"{feature_card_md('BASELINE', pair.baseline_layer, b_card)}\n\n"
        "Please return:\n"
        "Title: <<=8 words>\n"
        "Explanation: <<=150 words>\n"
        "Tags: tag1, tag2, tag3"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_explain_prompt_json(pair: PairRecord, g_card: FeatureCard, b_card: FeatureCard):
    system = (
        "You are an expert in sparse autoencoders and interpretability. "
        "Given two features (Group-SAE vs Baseline-SAE) with top-activating tokens and short contexts, "
        "return STRICT JSON with keys: title, explanation, tags (array of 2-4 short strings)."
    )
    user = (
        f"Pair:\n"
        f"- Jaccard similarity: {pair.similarity:.3f}\n\n"
        f"{feature_card_md('GROUP', pair.group_layer, g_card)}\n"
        f"{feature_card_md('BASELINE', pair.baseline_layer, b_card)}\n\n"
        "Respond as JSON only, e.g.:\n"
        "{\n"
        '  "title": "Concise title",\n'
        '  "explanation": "One short paragraph.",\n'
        '  "tags": ["morphology","punctuation"]\n'
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_explain_output(text: str) -> Tuple[str, str, List[str]]:
    title, tags = "", []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if l.lower().startswith("title:"):
            title = l.split(":", 1)[1].strip()
        if l.lower().startswith("tags:"):
            tags = [t.strip().strip("#") for t in l.split(":", 1)[1].split(",") if t.strip()]
    explanation = "\n".join([l for l in lines if not l.lower().startswith(("title:", "tags:"))])
    return title, explanation.strip(), tags


def make_selection_prompt(exps: List[PairExplanation], k: int) -> List[Dict[str, str]]:
    system = (
        "You are preparing figures for a camera-ready research paper. "
        "Select the most representative and diverse feature-pairs across phenomena and layers."
    )
    bullets = []
    for i, e in enumerate(exps, 1):
        expl_inline = e.explanation_md.replace("\n", " ")
        bullets.append(
            f"{i}. sim={e.pair.similarity:.3f} | {e.title or '(no title)'} | "
            f"G:{e.pair.group_layer}#{e.pair.group_feat} vs B:{e.pair.baseline_layer}#{e.pair.baseline_feat} | "
            f"tags: {', '.join(e.tags) if e.tags else ''}\n"
            f"   {expl_inline}"
        )
    user = (
        f"Choose the **top {k}** pairs that are maximally informative and diverse. "
        "Prefer crisp semantics and clear overlap/contrast. "
        "Return STRICT JSON only:\n"
        '{ "selected": [ {"rank": 1, "index": <int-from-1..N>, "reason": "..."}, ... ] }\n\n'
        "Candidates:\n" + "\n".join(bullets)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# =========================
# Pipeline
# =========================
def explain_pairs(
    samples: List[PairRecord],
    render_group_feature,
    render_baseline_feature,
    *,
    top_k_tokens: int,
    max_snippets: int,
    model_explain: str,
    jsonl_path: str,
    client: OpenAI,
    sleep_s: float = 0.0,
) -> List[PairExplanation]:
    out: List[PairExplanation] = []
    with open(jsonl_path, "a", encoding="utf-8") as fout:
        for idx, p in tqdm(enumerate(samples, 1), desc="Explaining samples", total=len(samples)):
            g_blob = render_group_feature(p.group_layer, p.group_feat)
            b_blob = render_baseline_feature(p.baseline_layer, p.baseline_feat)

            g_card = parse_feature_dump_to_card(
                g_blob, p.group_feat, top_k=top_k_tokens, max_snippets=max_snippets
            )
            b_card = parse_feature_dump_to_card(
                b_blob, p.baseline_feat, top_k=top_k_tokens, max_snippets=max_snippets
            )

            # msgs = make_explain_prompt(p, g_card, b_card)
            msgs = make_explain_prompt_json(p, g_card, b_card)

            # Strict JSON response
            raw = chat(
                client,
                msgs,
                model=model_explain,
                temperature=0.7,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw_fallback = ""

            # Parse JSON robustly
            title, explanation, tags = "", "", []
            try:
                payload = json.loads(raw)
                title = (payload.get("title") or "").strip()
                explanation = (payload.get("explanation") or "").strip()
                tags = payload.get("tags") or []
            except Exception:
                # Fallback: try once more with an even simpler instruction
                msgs[-1][
                    "content"
                ] += "\n\nReturn ONLY JSON with keys exactly: title, explanation, tags."
                raw_fallback = chat(
                    client,
                    msgs,
                    model=model_explain,
                    temperature=0.0,
                    max_tokens=700,
                    response_format={"type": "json_object"},
                )
                try:
                    payload = json.loads(raw_fallback)
                    title = (payload.get("title") or "").strip()
                    explanation = (payload.get("explanation") or "").strip()
                    tags = payload.get("tags") or []
                except Exception:
                    # Last-resort stub to avoid empty lines
                    title = "Auto-stub: needs review"
                    explanation = (
                        "The model did not return valid JSON twice. Keep this pair in the pool, "
                        "but the explanation needs manual review."
                    )
                    tags = ["stub"]

            rec = PairExplanation(pair=p, title=title, tags=tags, explanation_md=explanation)
            out.append(rec)

            json_out = json.dumps(
                {
                    "pair": {
                        "group_layer": p.group_layer,
                        "baseline_layer": p.baseline_layer,
                        "group_feat": p.group_feat,
                        "baseline_feat": p.baseline_feat,
                        "similarity": p.similarity,
                    },
                    "title": title,
                    "tags": tags,
                    "explanation_markdown": explanation,
                    "raw": raw,
                    "raw_fallback": raw_fallback,
                    "g_blob": g_blob,
                    "b_blob": b_blob,
                    "group_card": {
                        "top_tokens": g_card.top_tokens,
                        "snippets": g_card.snippets[:3],  # keep it small
                    },
                    "baseline_card": {
                        "top_tokens": b_card.top_tokens,
                        "snippets": b_card.snippets[:3],
                    },
                },
                ensure_ascii=False,
            )
            fout.write(json_out + "\n")

            if sleep_s > 0:
                import time

                time.sleep(sleep_s)
            if idx % 10 == 0:
                print(f"[explain_pairs] {idx}/{len(samples)} done")
    return out


def choose_representatives(
    explanations: List[PairExplanation],
    *,
    k: int,
    model_select: str,
    client: OpenAI,
    out_json: str,
) -> Dict[str, Any]:
    msgs = make_selection_prompt(explanations, k)
    raw = chat(
        client,
        msgs,
        model=model_select,
        temperature=0.0,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    try:
        payload = json.loads(raw)
    except Exception:
        print("[choose_representatives] JSON parse failed; falling back to similarity.")
        ranked = sorted(
            enumerate(explanations, 1), key=lambda t: t[1].pair.similarity, reverse=True
        )[:k]
        payload = {
            "selected": [
                {"rank": r + 1, "index": i, "reason": "fallback by similarity"}
                for r, (i, _) in enumerate(ranked)
            ]
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Explain & select Group-SAE vs Baseline-SAE feature pairs (disk-backed, no manifests)."
    )
    ap.add_argument(
        "--concordance",
        default="/home/fbelotti/group-sae/feature_analysis/G9_K7_group_vs_baseline_concordance.json",
        help="Path to group_vs_baseline_concordance (.pkl or .json)",
    )
    ap.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[13, 14, 15],
        help="Layers to consider (e.g., 13 14 15)",
    )
    ap.add_argument("--k", default=9, help="Run id string used in path templates (e.g., 9)")

    ap.add_argument(
        "--tokens-dir-template",
        default="~/group-sae/feature_analysis/tokens/pythia-410m/{k}",
        help="e.g., ~/group-sae/feature_analysis/tokens/pythia-410m/{k}",
    )
    ap.add_argument(
        "--features-dir-template",
        default="~/group-sae/feature_analysis/features/pythia-410m/{k}",
        help="e.g., ~/group-sae/feature_analysis/features/pythia-410m/{k}",
    )
    ap.add_argument(
        "--activations-dir-template",
        default="~/group-sae/feature_analysis/activations/pythia-410m/{k}",
        help="e.g., ~/group-sae/feature_analysis/activations/pythia-410m/{k}",
    )

    ap.add_argument(
        "--hf-tokenizer",
        default="EleutherAI/pythia-410m",
        help="HF tokenizer name/path for decoding",
    )
    ap.add_argument(
        "--feat-width", type=int, default=128, help="Feature width for reshape (default 128)"
    )

    ap.add_argument(
        "--bins",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.01],
        help="Similarity bin edges",
    )
    ap.add_argument("--n-per-bin", type=int, default=4, help="Samples per bin")

    ap.add_argument("--top-k-tokens", type=int, default=5, help="Top tokens per feature card")
    ap.add_argument("--max-snippets", type=int, default=6, help="Max snippets per feature")
    ap.add_argument(
        "--context-len", type=int, default=16, help="Context tokens (each side) in renderer"
    )

    ap.add_argument(
        "--model-explain", default="openai/gpt-5-mini", help="Model for explanation pass"
    )
    ap.add_argument("--model-select", default="openai/gpt-5-mini", help="Model for selection pass")

    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--sleep-s", type=float, default=0.0, help="Sleep between OpenAI calls")
    ap.add_argument("--device", default="cpu", help="cpu or cuda[:idx], e.g., cuda:1")
    ap.add_argument("--out-dir", default=".", help="Where to write outputs")

    args = ap.parse_args()

    # Device
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    # RNG and outputs
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "pair_explanations.jsonl")
    selection_path = os.path.join(args.out_dir, "representative_selection.json")

    # Tokenizer (global for renderer)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer, use_fast=True)

    # Disk environment (matches your notebook file layout)
    env = DiskEnv(
        k=args.k,
        layers=args.layers,
        tokens_dir_template=args.tokens_dir_template,
        features_dir_template=args.features_dir_template,
        activations_dir_template=args.activations_dir_template,
        device=args.device,
        feat_width=args.feat_width,
    )
    render_group_feature = make_group_renderer(
        env, top_k_tokens=args.top_k_tokens, context_len=args.context_len
    )
    render_baseline_feature = make_baseline_renderer(
        env, top_k_tokens=args.top_k_tokens, context_len=args.context_len
    )

    # Load concordance
    concordance = load_concordance(args.concordance)

    # Build per-pair samples (bin & sample INSIDE each (group_layer, baseline_layer))
    rng = random.Random(args.seed)
    layers_filter = set(map(str, args.layers))
    samples = build_samples_per_pair(
        concordance,
        bin_edges=args.bins,
        n_per_bin=args.n_per_bin,
        layers_filter=layers_filter,
        rng=rng,
    )

    # Optional: quick summary by bin across all pairs (purely informational)
    print(f"[main] sampled pairs (per-pair binning): {len(samples)}")

    # OpenAI client
    client = get_openai_client()

    # Explain
    explanations = explain_pairs(
        samples,
        render_group_feature,
        render_baseline_feature,
        top_k_tokens=args.top_k_tokens,
        max_snippets=args.max_snippets,
        model_explain=args.model_explain,
        jsonl_path=jsonl_path,
        client=client,
        sleep_s=args.sleep_s,
    )
    print(f"[main] explanations: {len(explanations)} → {jsonl_path}")


if __name__ == "__main__":
    main()
