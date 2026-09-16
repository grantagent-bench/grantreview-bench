"""Turn an ablation_report.json into the paper's headline claims.

Reads results/<tag>/ablation_report.json and emits:
  * component-contribution deltas (full − no_X) with paired McNemar p
  * the budget-matched comparison: full vs self_consistency@N at similar token cost
  * a plain-English findings block for "When Multi-Agent LLM Grant Review Fails"

Usage:  python -m scripts.ablation.analyze --tag paired20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BACKEND_DIR / "scripts" / "ablation" / "results"


def _sig_label(p) -> str:
    if p is None:
        return ", McNemar n/a"
    return ", n.s." if p > 0.05 else ", sig."


def _by_system(report: dict) -> dict:
    return {s["system"]: s for s in report["systems"]}


def _acc(s: dict) -> float:
    return s["accuracy"] * 100


def component_contributions(sysmap: dict) -> list[dict]:
    """full − no_X : the marginal value of each component, with significance."""
    if "full" not in sysmap:
        return []
    full = sysmap["full"]
    rows = []
    labels = {
        "no_debate": "Debate", "no_meta": "Meta-review",
        "no_guardrails": "Guardrails", "no_retrieval": "Retrieval",
        "agents_only": "Debate+Meta+Guardrails (all extras)",
    }
    for key, name in labels.items():
        if key not in sysmap:
            continue
        abl = sysmap[key]
        delta = _acc(full) - _acc(abl)
        # McNemar of the ablated system was computed vs full during summarize
        p = abl.get("mcnemar_vs_full", {}).get("p")
        rows.append({"component": name, "removed_system": key,
                     "acc_full": round(_acc(full), 1), "acc_ablated": round(_acc(abl), 1),
                     "delta_pp": round(delta, 1), "mcnemar_p": p})
    return rows


def budget_matched(sysmap: dict) -> dict | None:
    """Compare full pipeline against the self-consistency baseline at matched compute."""
    full = sysmap.get("full")
    sc = next((s for k, s in sysmap.items() if k.startswith("self_consistency")), None)
    if not full or not sc:
        return None
    ft = full["compute"]["total_tokens"]
    st = sc["compute"]["total_tokens"]
    return {
        "full_acc": round(_acc(full), 1), "sc_system": sc["system"],
        "sc_acc": round(_acc(sc), 1),
        "full_tokens_per_prop": full["compute"]["mean_tokens"],
        "sc_tokens_per_prop": sc["compute"]["mean_tokens"],
        "token_ratio_full_over_sc": round(ft / st, 2) if st else None,
        "delta_pp": round(_acc(full) - _acc(sc), 1),
        "mcnemar_p": sc.get("mcnemar_vs_full", {}).get("p"),
    }


def cheap_baseline_gap(sysmap: dict) -> dict:
    """How far does the whole pipeline beat the cheapest sensible baselines?"""
    out = {}
    ref = sysmap.get("full") or sysmap.get("full_stored")
    if not ref:
        return out
    out["reference"] = ref["system"]
    out["reference_acc"] = round(_acc(ref), 1)
    for k in ("single_call", "retrieval_knn", "multi_criteria_single_call"):
        if k in sysmap:
            out[k] = {"acc": round(_acc(sysmap[k]), 1),
                      "delta_pp_vs_ref": round(_acc(ref) - _acc(sysmap[k]), 1),
                      "mcnemar_p": sysmap[k].get("mcnemar_vs_full", {}).get("p")}
    return out


def narrative(report: dict, comps, budget, cheap) -> str:
    L = []
    n = report["n_proposals"]
    L.append(f"## Findings ({report['tag']}, N={n}, model {report['llm']['model']})\n")
    if cheap:
        ref = cheap.get("reference"); racc = cheap.get("reference_acc")
        L.append(f"- **Reference system** `{ref}`: {racc}% accuracy.")
        if "single_call" in cheap:
            sc = cheap["single_call"]
            sig = _sig_label(sc["mcnemar_p"])
            L.append(f"- A **single LLM call** reaches {sc['acc']}% "
                     f"({sc['delta_pp_vs_ref']:+.1f}pp vs {ref}, McNemar p={sc['mcnemar_p']}{sig}).")
        if "retrieval_knn" in cheap:
            rk = cheap["retrieval_knn"]
            L.append(f"- **Retrieval-only k-NN** (no generation) reaches {rk['acc']}% "
                     f"({rk['delta_pp_vs_ref']:+.1f}pp vs {ref}).")
    if budget:
        sig = "not distinguishable" if (budget["mcnemar_p"] is not None and budget["mcnemar_p"] > 0.05) else "distinguishable"
        L.append(f"- **Is it just more compute?** The full pipeline spends "
                 f"~{budget['token_ratio_full_over_sc']}× the tokens of "
                 f"{budget['sc_system']}, yet scores {budget['full_acc']}% vs "
                 f"{budget['sc_acc']}% ({budget['delta_pp']:+.1f}pp, McNemar "
                 f"p={budget['mcnemar_p']} — {sig} at matched budget).")
    if comps:
        L.append("- **Component contributions (full − ablated):**")
        for c in comps:
            sig = "n.s." if (c["mcnemar_p"] is not None and c["mcnemar_p"] > 0.05) else "sig."
            L.append(f"    - {c['component']}: {c['delta_pp']:+.1f}pp "
                     f"(McNemar p={c['mcnemar_p']}, {sig})")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="paired20")
    args = ap.parse_args()
    path = OUT_DIR / args.tag / "ablation_report.json"
    if not path.exists():
        raise SystemExit(f"no report at {path} — run the ablation first")
    report = json.load(open(path))
    sysmap = _by_system(report)

    comps = component_contributions(sysmap)
    budget = budget_matched(sysmap)
    cheap = cheap_baseline_gap(sysmap)
    text = narrative(report, comps, budget, cheap)

    out = {"component_contributions": comps, "budget_matched": budget,
           "cheap_baseline_gap": cheap, "narrative": text}
    (OUT_DIR / args.tag / "analysis.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / args.tag / "analysis.md").write_text(text + "\n")
    print(text)
    print(f"\nwrote {OUT_DIR/args.tag/'analysis.json'}")


if __name__ == "__main__":
    main()
