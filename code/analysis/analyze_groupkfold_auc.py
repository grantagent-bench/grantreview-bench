#!/usr/bin/env python3
"""GroupKFold-by-source AUC for Table 6 reference baselines.

Recomputes the **GroupKFold AUC** column of Paper 2 §6 Table 6 from the per-doc
result files written by `run_baselines.py`. GroupKFold by `source_dataset`
prevents source leakage across train/test folds — see Paper 2 §5.2 for the
methodology.

Pipeline
--------
  for each baseline in {random, length, gpt4_simple, gpt4_struct, gpt4_rag,
                        gpt4_cot, multi_agent_gpt4o_mini, multi_agent_gpt4o}:
    1. Load per-doc JSON outputs from runs/{baseline}/*.json
    2. Build a (p_fund, label, source) DataFrame
    3. Run 5-fold GroupKFold (groups = source_dataset)
    4. For each fold, compute roc_auc_score(test_labels, test_p_fund)
    5. Report mean ± std over folds

Reproducibility
---------------
- numpy seed 42 (only affects fold assignment via shuffle)
- sklearn GroupKFold has no `random_state` parameter; the fold assignment is
  deterministic given the input order — `RUN_ORDER` is sorted alphabetically
  by (source, label, filename) so the result is bit-reproducible.

Usage
-----
    python backend/scripts/analyze_groupkfold_auc.py
        Prints the GroupKFold AUC column for all baselines that have run/*/*.json files.

    python backend/scripts/analyze_groupkfold_auc.py --baseline length
        Single baseline.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
# Per-baseline result dirs live under backend/runs/baselines/{baseline}/.
# Override with --runs-dir if your layout differs.
RUNS = ROOT / "backend" / "runs" / "baselines"
DEFAULT_BASELINES = (
    "random", "length",
    "gpt4_simple", "gpt4_struct", "gpt4_rag", "gpt4_cot",
    "multi_agent_gpt4o_mini", "multi_agent_gpt4o",
)
SEED = 42


def load_baseline(baseline_dir: Path) -> list[dict]:
    """Read every per-doc JSON and return a list of dicts with the fields we need."""
    rows = []
    for fp in sorted(baseline_dir.glob("*.json")):
        try:
            r = json.loads(fp.read_text())
        except json.JSONDecodeError:
            continue
        doc = r.get("doc") or {}
        if doc.get("doc_type") and doc.get("doc_type") != "application":
            continue
        if "p_fund" not in r or doc.get("label") not in {"awarded", "declined"}:
            continue
        rows.append({
            "filename": doc.get("filename"),
            "source": doc.get("source_dataset"),
            "label": 1 if doc["label"] == "awarded" else 0,
            "p_fund": float(r["p_fund"]),
        })
    rows.sort(key=lambda r: (r["source"] or "", r["label"], r["filename"] or ""))
    return rows


def groupkfold_auc(rows: list[dict], n_splits: int = 5) -> tuple[float, float, int]:
    """Return (mean_auc, std_auc, n_folds_used)."""
    if not rows:
        return float("nan"), float("nan"), 0
    p = np.array([r["p_fund"] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    g = np.array([r["source"] or "_unknown" for r in rows])
    n_groups = len(set(g))
    n_splits_eff = min(n_splits, n_groups)
    if n_splits_eff < 2:
        return float("nan"), float("nan"), 0
    gkf = GroupKFold(n_splits=n_splits_eff)
    aucs = []
    for tr, te in gkf.split(p, y, groups=g):
        y_te = y[te]
        if len(set(y_te)) < 2:
            continue  # AUC undefined on single-class fold
        aucs.append(roc_auc_score(y_te, p[te]))
    if not aucs:
        return float("nan"), float("nan"), 0
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)), len(aucs)


def main() -> None:
    np.random.seed(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="append",
                    help="Baseline name (repeatable). Defaults to all known.")
    ap.add_argument("--runs-dir", type=Path, default=RUNS,
                    help=f"Directory containing per-baseline run subdirs (default: {RUNS})")
    args = ap.parse_args()
    baselines = tuple(args.baseline) if args.baseline else DEFAULT_BASELINES

    print(f"GroupKFold-by-source AUC — Paper 2 §6 Table 6 column")
    print(f"  runs_dir = {args.runs_dir}")
    print(f"  baselines = {list(baselines)}\n")
    print(f"  {'baseline':<28} {'mean AUC':>9}  {'std':>5}  {'folds':>5}  {'N docs':>7}")
    print(f"  {'-'*28} {'-'*9}  {'-'*5}  {'-'*5}  {'-'*7}")
    for b in baselines:
        bdir = args.runs_dir / b
        if not bdir.is_dir():
            print(f"  {b:<28} {'(no runs/'+b+' dir)':>40}")
            continue
        rows = load_baseline(bdir)
        mean, std, n_folds = groupkfold_auc(rows)
        print(f"  {b:<28} {mean:>9.3f}  {std:>5.3f}  {n_folds:>5d}  {len(rows):>7d}")


if __name__ == "__main__":
    main()
