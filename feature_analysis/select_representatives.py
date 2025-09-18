import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm


# =========================
# Data model
# =========================
@dataclass
class Pair:
    group_layer: str
    baseline_layer: str
    group_feat: int
    baseline_feat: int
    similarity: float


@dataclass
class Explanation:
    pair: Pair
    title: str
    tags: List[str]
    explanation_md: str


# =========================
# I/O
# =========================
def load_explanations(path: str) -> List[Explanation]:
    """Reads JSONL (preferred) or a JSON array file."""
    rows: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        import jsonlines as jsonl

        with jsonl.open(path, "r") as f:
            for line in f:
                if not line:
                    continue
                try:
                    rows.append(line)
                except Exception:
                    continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if not isinstance(obj, list):
                raise ValueError("JSON input must be an array of records")
            rows = obj

    out: List[Explanation] = []
    for it in rows:
        p = it.get("pair", {})
        title = (it.get("title") or "").strip()
        tags = it.get("tags") or []
        exp = (it.get("explanation_markdown") or it.get("explanation") or "").strip()
        pair = Pair(
            group_layer=str(p.get("group_layer")),
            baseline_layer=str(p.get("baseline_layer")),
            group_feat=int(p.get("group_feat")),
            baseline_feat=int(p.get("baseline_feat")),
            similarity=float(p.get("similarity", 0.0)),
        )
        out.append(Explanation(pair=pair, title=title, tags=tags, explanation_md=exp))
    return out


# =========================
# Binning / Grouping
# =========================
def bin_index(sim: float, edges: List[float]) -> Optional[int]:
    for i in range(len(edges) - 1):
        if edges[i] <= sim < edges[i + 1]:
            return i
    return None


def group_key(e: Explanation, mode: str) -> str:
    if mode == "pair":
        return f"G{e.pair.group_layer}-B{e.pair.baseline_layer}"
    if mode == "group_layer":
        return f"G{e.pair.group_layer}"
    if mode == "baseline_layer":
        return f"B{e.pair.baseline_layer}"
    # "all"
    return "ALL"


# =========================
# OpenAI helpers (JSON mode)
# =========================
class TransientOpenAIError(Exception):
    pass


def get_client():
    os.environ.get("OPENAI_API_KEY")
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
def chat_json(
    client: OpenAI, model: str, messages: List[Dict[str, str]], max_tokens: int = 1300
) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ["rate", "timeout", "overloaded", "temporar", "503"]):
            raise TransientOpenAIError(e)
        raise


def make_selection_prompt_json(exps: List[Explanation], k: int) -> List[Dict[str, str]]:
    system = (
        "You are selecting examples for a camera-ready interpretability paper. "
        "Return STRICT JSON with key 'selected' as an array. Each item MUST be: "
        '{"rank": int (1..K), "index": int (1..N), "reason": string (<= 256 words)}. '
        "Aim for diversity across phenomena (tags) and prefer crisp, representative and explanations. "
        "Avoid near-duplicates."
    )
    bullets = []
    for i, e in enumerate(exps, 1):
        tagstr = ", ".join(e.tags) if e.tags else ""
        expl_inline = e.explanation_md.replace("\n", " ")
        bullets.append(
            f"{i}. sim={e.pair.similarity:.3f} | "
            f"G:{e.pair.group_layer}#{e.pair.group_feat} vs "
            f"B:{e.pair.baseline_layer}#{e.pair.baseline_feat} | "
            f"tags: {tagstr}\n"
            f"   {e.title or '(no title)'} — {expl_inline}"
        )
    user = (
        f"N={len(exps)} candidates. Pick K={k} maximally informative & diverse items. "
        "Return ONLY JSON like:\n"
        "{\n"
        '  "selected": [\n'
        '     {"rank": 1, "index": 12, "reason": "clear overlap on numerals; different collocations"}\n'
        "  ]\n"
        "}\n\n"
        "Candidates:\n" + "\n".join(bullets)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# =========================
# Diversity-aware fallback
# =========================
def diversity_fallback(exps: List[Explanation], k: int) -> Dict[str, Any]:
    N = len(exps)
    sims = [e.pair.similarity for e in exps]
    tagsets = [set(e.tags or []) for e in exps]

    picked = []
    used_tags = set()
    lam = 0.75  # similarity vs novelty
    order = sorted(range(N), key=lambda i: sims[i], reverse=True)

    for _ in range(min(k, N)):
        best, best_score = None, -1e9
        for i in order:
            if i in picked:
                continue
            if picked:
                red = max(
                    (len(tagsets[i] & tagsets[j]) / max(1, len(tagsets[i] | tagsets[j])))
                    for j in picked
                )
            else:
                red = 0.0
            tag_pen = 1.0 / (1 + len(tagsets[i] & used_tags))
            novelty = tag_pen * (1.0 - red)
            score = lam * sims[i] + (1 - lam) * novelty
            if score > best_score:
                best, best_score = i, score
        picked.append(best)
        used_tags |= tagsets[best]

    return {
        "selected": [
            {"rank": r + 1, "index": i + 1, "reason": "diversity-aware fallback"}
            for r, i in enumerate(picked)
        ]
    }


# =========================
# Selection (per bin x per group)
# =========================
def select_for_subset(
    subset: List[Explanation], k: int, model: str, client: OpenAI
) -> List[Dict[str, Any]]:
    """Ask the model to pick up to k from subset; fallback if needed. Returns list of dicts."""
    if not subset:
        return []

    if len(subset) <= k:
        return [
            {"rank": i + 1, "index": i + 1, "reason": "all candidates"} for i in range(len(subset))
        ]

    msgs = make_selection_prompt_json(subset, k)
    payload = None
    try:
        raw = chat_json(client, model, msgs, max_tokens=2048)
        payload = json.loads(raw)
        assert isinstance(payload.get("selected"), list)
    except Exception:
        # tighten instruction & retry once
        msgs[-1][
            "content"
        ] += "\n\nIMPORTANT: Return ONLY valid JSON with 'selected' -> array of {rank,index,reason}."
        try:
            raw2 = chat_json(client, model, msgs, max_tokens=2048)
            payload = json.loads(raw2)
            assert isinstance(payload.get("selected"), list)
        except Exception:
            payload = diversity_fallback(subset, k)

    # sanitize
    N = len(subset)
    cleaned = []
    seen_ranks = set()
    for item in payload.get("selected", []):
        try:
            rank = int(item.get("rank", 0))
            idx = int(item.get("index", 0))
            reason = str(item.get("reason", ""))
        except Exception:
            continue
        if 1 <= rank <= k and rank not in seen_ranks and 1 <= idx <= N:
            seen_ranks.add(rank)
            cleaned.append({"rank": rank, "index": idx, "reason": reason})
    # fill if short
    if len(cleaned) < k:
        need = k - len(cleaned)
        remaining = sorted(
            [i for i in range(1, N + 1) if i not in [c["index"] for c in cleaned]],
            key=lambda i: subset[i - 1].pair.similarity,
            reverse=True,
        )[:need]
        base_rank = max([c["rank"] for c in cleaned], default=0)
        for j, idx in enumerate(remaining, 1):
            cleaned.append({"rank": base_rank + j, "index": idx, "reason": "filled by similarity"})
    return sorted(cleaned, key=lambda x: x["rank"])


# =========================
# Main driver
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Choose K representatives per similarity bin per group from explanations JSONL/JSON."
    )
    ap.add_argument(
        "--explanations", required=True, help="Path to explanations (JSONL or JSON array)"
    )
    ap.add_argument(
        "--bins",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8, 1.01],
        help="Similarity bin edges",
    )
    ap.add_argument(
        "--group-by",
        default="pair",
        choices=["pair", "group_layer", "baseline_layer", "all"],
        help="Grouping mode for per-bin selection",
    )
    ap.add_argument(
        "--k-per-bin", type=int, default=3, help="How many items to select per bin per group"
    )
    ap.add_argument("--model", default="gpt-5", help="OpenAI model (JSON mode)")
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=80,
        help="Max candidates fed to the model per bin/group (pre-trim by similarity)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out", required=True, help="Output JSON file")
    args = ap.parse_args()

    random.seed(args.seed)
    exps = load_explanations(args.explanations)
    if not exps:
        raise SystemExit("No explanations found.")

    # Bucket by (group, bin)
    edges = args.bins
    by_group_bin: Dict[Tuple[str, int], List[Explanation]] = defaultdict(list)
    for e in exps:
        b = bin_index(e.pair.similarity, edges)
        if b is None:
            continue
        g = group_key(e, args.group_by)
        by_group_bin[(g, b)].append(e)

    # Select per bucket
    client = get_client()
    result: Dict[str, Any] = {
        "bins": edges,
        "group_by": args.group_by,
        "k_per_bin": args.k_per_bin,
        "groups": {},
    }

    for (gkey, bidx), items in tqdm(
        sorted(by_group_bin.items()), total=len(by_group_bin), desc="Selecting"
    ):
        # optional pre-trim by similarity to keep prompts small
        if args.max_candidates and len(items) > args.max_candidates:
            items = sorted(items, key=lambda e: e.pair.similarity, reverse=True)[
                : args.max_candidates
            ]

        selected = select_for_subset(items, args.k_per_bin, args.model, client)

        # store with a tiny manifest of the candidates chosen
        group_entry = result["groups"].setdefault(gkey, {})
        bin_label = f"[{edges[bidx]:.2f},{edges[bidx+1]:.2f})"
        group_entry[bin_label] = {
            "count_candidates": len(items),
            "selected": selected,
            # also record the (group-local) candidate indices we used (1-based)
            "candidates_manifest": [
                {
                    "index": i + 1,
                    "pair": {
                        "group_layer": items[i].pair.group_layer,
                        "baseline_layer": items[i].pair.baseline_layer,
                        "group_feat": items[i].pair.group_feat,
                        "baseline_feat": items[i].pair.baseline_feat,
                        "similarity": items[i].pair.similarity,
                    },
                    "title": items[i].title,
                    "tags": items[i].tags,
                }
                for i in range(len(items))
            ],
        }

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote per-bin selections to {args.out}")


if __name__ == "__main__":
    main()
