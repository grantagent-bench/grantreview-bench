#!/usr/bin/env python3
"""Inter-source label-transfer matrix for Paper 2.

Codex's "highest-leverage new analysis": for each pair of sub-corpora
(both-class), train logistic regression on TF-IDF features in source A,
test on source B. The resulting transfer matrix answers:

  "Are declined proposals declined for the same reasons across sub-corpora,
   or is the label signal source-specific?"

If diagonal (within-source train+test) is high but off-diagonal (transfer)
is low, the labels do not encode a portable quality concept — they encode
source-specific conventions. That finding directly justifies why Paper 2's
two-confound evaluation protocol is necessary.

Output:
  - stdout: NxN transfer matrix
  - runs/analysis/inter_source_transfer.json
  - paper/figures/v2_transfer_heatmap.png (for Paper 2)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import cast
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", ConvergenceWarning)

import os
import argparse

# Resolve ROOT relative to script location so the analysis is portable.
# Override with GRANTREVIEW_ROOT env var.
ROOT = Path(os.environ.get("GRANTREVIEW_ROOT", Path(__file__).resolve().parents[2]))
MANIFEST = ROOT / "data" / "master_manifest_v2.json"
OUT_JSON = ROOT / "backend" / "runs" / "analysis" / "inter_source_transfer.json"
OUT_PNG = ROOT / "paper" / "figures" / "v2_transfer_heatmap.png"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_ITERS = 2000
PERMUTATION_ITERS = 1000
SEED = 42


def funder_family(funder):
    f = (funder or "").lower()
    if "nsf" in f or "nserc" in f or "national science" in f: return "NSF/NSERC"
    if "nih" in f or "national institutes of health" in f: return "NIH"
    if "wellcome" in f: return "Wellcome"
    if "erc" in f or "european" in f: return "EU/ERC"
    if "sloan" in f or "moore" in f: return "Private"
    return "Other"


def load_text(d):
    """Load proposal text from disk."""
    p = Path(d["abs_path"])
    if not p.exists(): return ""
    if d.get("format") == "txt":
        try: return p.read_text(errors="ignore")
        except: return ""
    # PDF — extract via pymupdf
    import fitz  # type: ignore
    try:
        doc = fitz.open(str(p))  # type: ignore
        txt = "".join(page.get_text() for page in doc)
        doc.close()
        return txt
    except Exception:
        return ""


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray,
                     n_iters: int = BOOTSTRAP_ITERS, seed: int = SEED,
                     ci: float = 0.95) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI on AUC. Returns (mean_auc, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(n_iters):
        idx = rng.randint(0, n, size=n)
        y_boot = y_true[idx]
        if len(set(y_boot)) < 2:
            continue
        aucs.append(roc_auc_score(y_boot, scores[idx]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    aucs_arr = np.array(aucs)
    alpha = (1.0 - ci) / 2.0
    lo, hi = float(np.quantile(aucs_arr, alpha)), float(np.quantile(aucs_arr, 1.0 - alpha))
    return float(np.mean(aucs_arr)), lo, hi


def permutation_p_one_sided(y_true: np.ndarray, scores: np.ndarray,
                            observed_auc: float,
                            n_iters: int = PERMUTATION_ITERS,
                            seed: int = SEED) -> float:
    """One-sided permutation null p-value: P(AUC <= observed | random labels).

    Holds the score vector fixed and permutes labels n_iters times. The
    one-sided p reports the fraction of permuted AUCs at or below the observed
    AUC — the right test for an "anti-correlated below chance" claim.
    """
    rng = np.random.RandomState(seed + 1)  # different seed from bootstrap
    n = len(y_true)
    n_le = 0
    valid = 0
    for _ in range(n_iters):
        perm = rng.permutation(y_true)
        if len(set(perm)) < 2:
            continue
        valid += 1
        if roc_auc_score(perm, scores) <= observed_auc:
            n_le += 1
    return (n_le + 1) / (valid + 1) if valid > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true", default=True,
                    help="Compute 2000-iter bootstrap CIs on each off-diagonal cell.")
    ap.add_argument("--no-bootstrap", dest="bootstrap", action="store_false")
    ap.add_argument("--permutation-null", action="store_true", default=True,
                    help="Compute one-sided permutation p-value for cells whose CI excludes 0.5.")
    ap.add_argument("--no-permutation", dest="permutation_null", action="store_false")
    args = ap.parse_args()

    docs = json.load(open(MANIFEST))
    apps = [d for d in docs if d.get("doc_type") == "application"]
    print(f"Total apps: {len(apps)}")

    # Build sub-corpus assignments. For ogrants, split by funder family.
    # For wellcome, keep as single sub-corpus.
    def sub_corpus(d):
        if d["source_dataset"] == "ogrants":
            return f"OG-{funder_family(d.get('funder'))}"
        if d["source_dataset"] == "wellcome":
            return "Wellcome"
        if d["source_dataset"] == "declined_extra":
            return "Decl-extras"
        return d["source_dataset"]

    # Group apps by sub_corpus and select those with both classes ≥3
    from collections import defaultdict
    groups = defaultdict(list)
    for d in apps:
        groups[sub_corpus(d)].append(d)
    keep = {}
    for k, ds in groups.items():
        aw = sum(1 for d in ds if d["label"] == "awarded")
        de = sum(1 for d in ds if d["label"] == "declined")
        if aw >= 3 and de >= 3:
            keep[k] = ds
    print(f"Sub-corpora with both classes ≥3: {sorted(keep.keys())}")
    for k, ds in keep.items():
        aw = sum(1 for d in ds if d["label"] == "awarded")
        de = len(ds) - aw
        print(f"  {k}: N={len(ds)} (aw={aw}, dec={de})")

    # Load text for each kept app
    print("\nLoading text...")
    text_data = {}  # (sub_corpus, idx) -> (text, label)
    for k, ds in keep.items():
        for i, d in enumerate(ds):
            txt = load_text(d)
            if len(txt.split()) >= 50:  # require min 50 words
                text_data[(k, i)] = (txt, 1 if d["label"] == "awarded" else 0)

    # Re-collect only sub-corpora that still have both classes after text-load
    final_groups = defaultdict(list)
    for (k, i), (t, lbl) in text_data.items():
        final_groups[k].append((t, lbl))
    final_keep = {}
    for k, items in final_groups.items():
        aw = sum(1 for _, lbl in items if lbl == 1)
        de = len(items) - aw
        if aw >= 3 and de >= 3:
            final_keep[k] = items
    print(f"\nAfter text-load filter, kept: {sorted(final_keep.keys())}")
    for k, items in final_keep.items():
        aw = sum(1 for _, lbl in items if lbl == 1)
        de = len(items) - aw
        print(f"  {k}: N={len(items)} (aw={aw}, dec={de})")

    # Build NxN transfer matrix with bootstrap CIs and (optional) permutation p-values.
    sources = sorted(final_keep.keys())
    n_sources = len(sources)
    M = np.full((n_sources, n_sources), np.nan)
    M_LO = np.full((n_sources, n_sources), np.nan)
    M_HI = np.full((n_sources, n_sources), np.nan)
    M_P  = np.full((n_sources, n_sources), np.nan)  # one-sided permutation p

    print(f"\n=== Inter-source transfer matrix (rows=train, cols=test) ===")
    print(f"Bootstrap iterations: {BOOTSTRAP_ITERS}; permutation iterations: {PERMUTATION_ITERS}; seed: {SEED}")
    print(f"Per-source N: " + ", ".join(f"{s}={len(final_keep[s])}" for s in sources))

    for i, train_src in enumerate(sources):
        train_items = final_keep[train_src]
        train_texts = [t for t, _ in train_items]
        train_labels = np.array([lbl for _, lbl in train_items])
        if len(set(train_labels)) < 2:
            continue
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=2)
        X_train = vec.fit_transform(train_texts)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, train_labels)
        for j, test_src in enumerate(sources):
            test_items = final_keep[test_src]
            test_texts = [t for t, _ in test_items]
            test_labels = np.array([lbl for _, lbl in test_items])
            if len(set(test_labels)) < 2:
                continue
            X_test = vec.transform(test_texts)
            scores = clf.predict_proba(X_test)[:, 1]
            if i == j:
                # Diagonal: 5-fold CV (no bootstrap CI on diagonals — see paper §5.3)
                from sklearn.model_selection import StratifiedKFold
                from sklearn.linear_model import LogisticRegression as LR2
                aucs: list[float] = []
                if len(test_labels) >= 10:
                    skf = StratifiedKFold(
                        n_splits=min(5, int(sum(test_labels)),
                                     int(len(test_labels) - sum(test_labels))),
                        shuffle=True, random_state=SEED)
                    for tr, te in skf.split(test_texts, test_labels):
                        if len(set(test_labels[te])) < 2: continue
                        v = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=2)
                        Xt = v.fit_transform([test_texts[k] for k in tr])
                        c = LR2(max_iter=1000, C=1.0)
                        c.fit(Xt, test_labels[tr])
                        Xv = v.transform([test_texts[k] for k in te])
                        sc = c.predict_proba(Xv)[:, 1]
                        aucs.append(float(roc_auc_score(test_labels[te], sc)))
                    M[i, j] = float(np.mean(aucs)) if aucs else np.nan
            else:
                # Off-diagonal: point AUC + bootstrap CI + (optional) permutation p
                point_auc = float(roc_auc_score(test_labels, scores))
                M[i, j] = point_auc
                if args.bootstrap:
                    _, lo, hi = bootstrap_auc_ci(test_labels, scores)
                    M_LO[i, j] = lo
                    M_HI[i, j] = hi
                if args.permutation_null:
                    M_P[i, j] = permutation_p_one_sided(test_labels, scores, point_auc)

        # Pretty-print row with CIs
        row = f"  {train_src:<22}"
        for j in range(n_sources):
            v = M[i, j]
            if np.isnan(v):
                row += f"  {'—':>15}"
            else:
                lo, hi = M_LO[i, j], M_HI[i, j]
                p = M_P[i, j]
                if np.isnan(lo):
                    row += f"  {v:>5.2f}{'':>10}"
                else:
                    sig = "*" if (not np.isnan(p) and p < 0.05) else " "
                    row += f"  {v:>4.2f}[{lo:.2f},{hi:.2f}]{sig}"
        print(row)

    # Save JSON with CIs and p-values
    out = {
        "sources": sources,
        "matrix":           [[None if np.isnan(M[i, j])    else float(M[i, j])    for j in range(n_sources)] for i in range(n_sources)],
        "ci_lo":            [[None if np.isnan(M_LO[i, j]) else float(M_LO[i, j]) for j in range(n_sources)] for i in range(n_sources)],
        "ci_hi":            [[None if np.isnan(M_HI[i, j]) else float(M_HI[i, j]) for j in range(n_sources)] for i in range(n_sources)],
        "permutation_p":    [[None if np.isnan(M_P[i, j])  else float(M_P[i, j])  for j in range(n_sources)] for i in range(n_sources)],
        "n_per_source":     {s: len(final_keep[s]) for s in sources},
        "bootstrap_iters":  BOOTSTRAP_ITERS,
        "permutation_iters": PERMUTATION_ITERS,
        "seed":             SEED,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # Heatmap figure
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(M, cmap=cmap, vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(n_sources)); ax.set_yticks(range(n_sources))
    ax.set_xticklabels(sources, rotation=35, ha="right")
    ax.set_yticklabels(sources)
    ax.set_xlabel("Test sub-corpus")
    ax.set_ylabel("Train sub-corpus")
    ax.set_title("Inter-source label-transfer matrix\n"
                 "Diagonal = within-source 5-fold CV; off-diagonal = train→test transfer\n"
                 "AUC > 0.5 means labels transfer; ≈ 0.5 means source-specific signal")
    for i in range(n_sources):
        for j in range(n_sources):
            v = M[i, j]
            if not np.isnan(v):
                color = "white" if abs(v - 0.6) > 0.2 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=10)
    fig.colorbar(im, ax=ax, label="ROC-AUC")
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
