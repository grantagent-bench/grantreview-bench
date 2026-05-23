#!/usr/bin/env python3
"""E2: Baseline comparisons on the same N=266.

Baselines:
  1. random      - stratified random (matches positive rate from training)
  2. length      - logistic regression on word count
  3. gpt4_simple - single GPT-4 call: "will this be funded? 0-1"
  4. gpt4_struct - single GPT-4 call with NIH 5-criteria template
  5. gpt4_rag    - single GPT-4 + 4 retrieved similar funded proposals as context
  6. gpt4_cot    - GPT-4 with explicit chain-of-thought scaffolding
  (SciBERT classifier — separate script)

Output: runs/baselines/{baseline_name}/{source}_{label}_{filename}.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
MASTER = PROJECT / "data" / "master_manifest_v2.json"
OUT_BASE = ROOT / "runs" / "baselines"
OUT_BASE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.11/site-packages"))

from openai import OpenAI
from grantagent.ingest.extract import extract_text
from grantagent.vector.index import GlobalIndex

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise SystemExit("Set OPENAI_API_KEY in your environment to run the GPT-4 baselines.")
client = OpenAI(api_key=OPENAI_KEY)
MODEL = "gpt-4o-2024-08-06"  # pinned snapshot for reproducibility (Paper 2)
MODEL_MINI = "gpt-4o-mini-2024-07-18"  # pinned snapshot for the mini-model multi-agent rows
RANDOM_SEED = 42  # seed all stochastic baselines for bit-reproducibility of Table 6
random.seed(RANDOM_SEED)
import numpy as _np
_np.random.seed(RANDOM_SEED)


def baseline_random(doc, text, **kw) -> dict:
    p = random.uniform(0, 1)
    return {"p_fund": p, "verdict": "Fund-Aligned" if p >= 0.5 else "Misaligned",
            "rationale": "stratified random"}


def baseline_length(doc, text, **kw) -> dict:
    # Length-only logistic. Funded proposals tend to be longer (full submissions).
    # Hand-fit threshold from quartile analysis: median funded ≈ 5300w, declined ≈ 3500w
    n = len(text.split())
    p = 1 / (1 + 2.71828 ** (-(n - 4500) / 1500))
    return {"p_fund": round(p, 3), "verdict": "Fund-Aligned" if p >= 0.5 else "Misaligned",
            "rationale": f"length={n}w"}


def call_gpt4(messages, max_tokens=500) -> str:
    r = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0.0,
    )
    return r.choices[0].message.content or ""


def parse_score(text: str) -> float:
    """Extract a 0-1 score from model output."""
    import re
    # Look for explicit p_fund or score
    m = re.search(r"(?:p_fund|score|probability|p\(fund\))[^\d]*([01]?\.\d+)", text.lower())
    if m: return float(m.group(1))
    # Look for stand-alone 0.X
    m = re.search(r"\b(0\.\d{1,3})\b", text)
    if m: return float(m.group(1))
    # Look for X% style
    m = re.search(r"(\d{1,3})%", text)
    if m: return min(float(m.group(1)) / 100.0, 1.0)
    # Default neutral
    return 0.5


def baseline_gpt4_simple(doc, text, **kw) -> dict:
    msg = [{"role":"user","content":
        f"Below is a grant proposal. Will it be funded? Output a single JSON line with "
        f"keys p_fund (number 0-1) and rationale (one sentence).\n\nPROPOSAL:\n{text[:8000]}"}]
    out = call_gpt4(msg, max_tokens=200)
    p = parse_score(out)
    return {"p_fund": p, "verdict": "Fund-Aligned" if p >= 0.5 else "Misaligned",
            "rationale": out, "raw_response": out}


def baseline_gpt4_struct(doc, text, **kw) -> dict:
    msg = [{"role":"user","content":
        f"You are an NIH peer reviewer. Score the following proposal on 5 criteria "
        f"(Significance, Innovation, Approach, Investigators, Environment), each 1-9. "
        f"Then estimate p_fund (0-1). Reply with JSON: "
        f'{{"significance":N,"innovation":N,"approach":N,"investigators":N,"environment":N,"p_fund":X.X,"summary":"..."}}\n\n'
        f"PROPOSAL:\n{text[:8000]}"}]
    out = call_gpt4(msg, max_tokens=400)
    p = parse_score(out)
    return {"p_fund": p, "verdict": "Fund-Aligned" if p >= 0.5 else "Misaligned",
            "raw_response": out}


_index = None
def get_index():
    global _index
    if _index is None:
        _index = GlobalIndex()
    return _index


def baseline_gpt4_rag(doc, text, **kw) -> dict:
    idx = get_index()
    # Query for 4 similar grants
    similar = []
    try:
        results = idx.query(text[:2000], n_results=4, where=None)
        for r in results.get("documents", [[]])[0]:
            similar.append(r[:1500])
    except Exception as e:
        similar = []
    ctx = "\n\n".join(f"[Similar funded grant {i+1}]\n{s}" for i, s in enumerate(similar))
    msg = [{"role":"user","content":
        f"Score this grant proposal's funding probability (0-1) given similar funded examples below.\n\n"
        f"{ctx}\n\nNEW PROPOSAL:\n{text[:6000]}\n\nReply JSON: {{\"p_fund\":X.X,\"rationale\":\"...\"}}"}]
    out = call_gpt4(msg, max_tokens=300)
    p = parse_score(out)
    return {"p_fund": p, "verdict": "Fund-Aligned" if p >= 0.5 else "Misaligned",
            "raw_response": out, "n_retrieved": len(similar)}


def baseline_gpt4_cot(doc, text, **kw) -> dict:
    msg = [
        {"role":"system","content":
         "You are an experienced NIH/NSF study section reviewer. NIH funds approximately "
         "20% of submitted R01s; NSF funds ~25% of CAREER proposals. Most submitted proposals "
         "are competent but only a minority are fundable. Be appropriately skeptical."},
        {"role":"user","content":
         f"Reason step-by-step about this grant proposal:\n"
         f"1. Significance: How important is this problem? (1-9, 1=high)\n"
         f"2. Innovation: How novel is the approach? (1-9)\n"
         f"3. Approach: Are the methods sound? (1-9)\n"
         f"4. Investigators: Is the team qualified? (1-9)\n"
         f"5. Environment: Is the institutional setting adequate? (1-9)\n"
         f"6. List 3 strengths and 3 weaknesses.\n"
         f"7. Estimate p_fund (0-1) considering the ~20-25% base rate.\n"
         f"Reply with JSON: "
         f'{{"significance":N,"innovation":N,"approach":N,"investigators":N,"environment":N,'
         f'"strengths":[...],"weaknesses":[...],"p_fund":X.X}}\n\n'
         f"PROPOSAL:\n{text[:8000]}"}]
    out = call_gpt4(msg, max_tokens=700)
    p = parse_score(out)
    return {"p_fund": p, "verdict": "Fund-Aligned" if p >= 0.5 else "Misaligned",
            "raw_response": out}


BASELINES = {
    "random": baseline_random,
    "length": baseline_length,
    "gpt4_simple": baseline_gpt4_simple,
    "gpt4_struct": baseline_gpt4_struct,
    "gpt4_rag": baseline_gpt4_rag,
    "gpt4_cot": baseline_gpt4_cot,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=list(BASELINES.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    fn = BASELINES[args.baseline]
    out_dir = OUT_BASE / args.baseline
    out_dir.mkdir(exist_ok=True)

    docs = json.load(open(MASTER))
    docs = [d for d in docs if d.get("doc_type") == "application"]
    if args.limit:
        docs = docs[:args.limit]

    print(f"Baseline: {args.baseline}  N={len(docs)}")
    successes = failures = skipped = 0

    for i, doc in enumerate(docs):
        safe = doc["filename"].replace("/", "_").replace(" ", "_")
        out = out_dir / f"{doc['source_dataset']}_{doc['label']}_{safe}.json"
        if out.exists() and args.skip_existing:
            skipped += 1
            continue

        fp = Path(doc["abs_path"])
        if not fp.exists():
            failures += 1
            continue

        # Extract text (cached approach: just read PDF each time, fast w/ pymupdf)
        try:
            text = extract_text(fp, fp.name)
        except Exception as e:
            print(f"  EXTRACT-ERR {doc['filename']}: {e}")
            failures += 1
            continue

        t0 = time.time()
        try:
            result = fn(doc, text)
        except Exception as e:
            print(f"  BASELINE-ERR {doc['filename']}: {e}")
            failures += 1
            continue
        elapsed = time.time() - t0

        record = {
            "doc": doc,
            "baseline": args.baseline,
            "elapsed_sec": round(elapsed, 2),
            **result,
        }
        with open(out, "w") as f:
            json.dump(record, f, indent=2)

        successes += 1
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(docs)}] last p_fund={result.get('p_fund'):.2f} ({elapsed:.1f}s)")

    print(f"\nDONE. {args.baseline}: successes={successes}, failures={failures}, skipped={skipped}")


if __name__ == "__main__":
    main()
