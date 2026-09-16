"""Build the paper's cross-model grid (§6.6) from the ogrants20_* run tags.

Re-aggregates each tag with the current metrics code (pure cache read), then emits
a compact per-model table: same 40 source-controlled proposals, same systems.

Usage:  python -m scripts.ablation.crossmodel
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
RES = BACKEND / "scripts" / "ablation" / "results"

# tag -> (paper label, exact snapshot)
MODELS = [
    ("ogrants20",         "gpt-4o-mini",        "gpt-4o-mini-2024-07-18"),
    ("ogrants20_gpt4o",   "gpt-4o",             "gpt-4o-2024-11-20"),
    ("ogrants20_gpt41",   "gpt-4.1",            "gpt-4.1-2025-04-14"),
    ("ogrants20_gpt5mini","gpt-5-mini",         "gpt-5-mini-2025-08-07"),
    ("ogrants20_gpt5",    "gpt-5",              "gpt-5-2025-08-07"),
    ("ogrants20_gpt54",   "gpt-5.4",            "gpt-5.4-2026-03-05"),
    ("ogrants20_o3mini",  "o3-mini",            "o3-mini-2025-01-31"),
    ("ogrants20_haiku",   "claude-haiku-4-5",   "claude-haiku-4-5-20251001"),
    ("ogrants20_claude",  "claude-sonnet-4-5",  "claude-sonnet-4-5-20250929"),
    ("ogrants20_opus",    "claude-opus-4-8",    "claude-opus-4-8"),
]

SYSTEMS_ARG = "retrieval_knn,single_call,multi_criteria_single_call,agents_only,full"


def reaggregate(tag: str) -> None:
    """Regenerate the tag's report from its checkpoint (no LLM calls)."""
    subprocess.run(
        [sys.executable, "-m", "scripts.ablation.run",
         "--systems", SYSTEMS_ARG, "--source", "ogrants", "--balance", "20",
         "--tag", tag],
        cwd=BACKEND, capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "TOKENIZERS_PARALLELISM": "false"},
    )


def main():
    rows = []
    for tag, label, snapshot in MODELS:
        fp = RES / tag / "ablation_report.json"
        reaggregate(tag)
        if not fp.exists():
            print(f"!! missing report for {tag}")
            continue
        rep = json.load(open(fp))
        sysmap = {s["system"]: s for s in rep["systems"]}
        sc, fu = sysmap.get("single_call"), sysmap.get("full")
        ao = sysmap.get("agents_only")
        if not (sc and fu):
            print(f"!! incomplete {tag}: {list(sysmap)}")
            continue
        mc_sc = sc.get("mcnemar_vs_full", {})
        rows.append({
            "model": label, "snapshot": snapshot,
            "sc_acc": sc["accuracy"] * 100, "sc_auc": sc["auc"],
            "ao_acc": (ao["accuracy"] * 100) if ao else None,
            "full_acc": fu["accuracy"] * 100, "full_auc": fu["auc"],
            "full_gap_p": fu["score_gap"].get("p"),
            "mcnemar_sc_vs_full_p": mc_sc.get("p"),
            "mcnemar_bc": (mc_sc.get("b"), mc_sc.get("c")),
        })

    lines = ["| Model (snapshot) | Single call | Agents only | Full pipeline | Full AUC | McNemar single-vs-full |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        bc = f" (b={r['mcnemar_bc'][0]},c={r['mcnemar_bc'][1]})" if r["mcnemar_bc"][0] is not None else ""
        lines.append(
            f"| {r['model']} (`{r['snapshot']}`) | {r['sc_acc']:.1f}% | "
            f"{('%.1f%%' % r['ao_acc']) if r['ao_acc'] is not None else '—'} | "
            f"{r['full_acc']:.1f}% | {r['full_auc']:.2f} | "
            f"{r['mcnemar_sc_vs_full_p']}{bc} |")
    md = "\n".join(lines)
    out = RES / "crossmodel_grid.md"
    out.write_text(md + "\n")
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
