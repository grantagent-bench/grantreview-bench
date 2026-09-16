"""CLI orchestrator for the GrantAgent controlled ablation study.

Run from the code/ directory, e.g.:

    # smoke test (no key needed; exercises plumbing in heuristic mode)
    python -m ablation.run --systems retrieval_knn,single_call --limit 8

    # real run on 40 proposals, cheap baselines + full pipeline
    python -m ablation.run \
        --systems single_call,multi_criteria_single_call,self_consistency,retrieval_knn,agents_only,no_debate,no_meta,no_guardrails,no_retrieval,full \
        --limit 40 --sc-n 10

    # stronger base model
    LLM_MODEL=gpt-4o-2024-11-20 python -m ablation.run --systems ... --tag gpt4o

    # offline re-score: rebuild the report from an existing checkpoint (zero LLM calls)
    python -m ablation.run --tag ogrants166 --aggregate-only

Per-proposal predictions stream to a JSONL checkpoint so runs are resumable:
re-running skips (system,id) pairs already present.  Aggregate metrics and a
Markdown table are written at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# --- env must load before any grantagent import (llm singleton at import time) ---
REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))
from . import envload  # noqa: E402
envload.load()

from . import dataset, metrics as M  # noqa: E402
from .systems import SYSTEMS, BATCH_FIT, Context  # noqa: E402
from .budget import METER, meter_llm  # noqa: E402

OUT_DIR = REPO_DIR / "results"


def _sc_name(base: str, ctx: Context) -> str:
    return f"self_consistency@{ctx.sc_n}" if base == "self_consistency" else base


def _id_to_source() -> dict:
    """Map eval record id -> source_dataset (from the original run files)."""
    import glob
    m = {}
    for fp in glob.glob(str(REPO_DIR / "runs" / "paper_eval" / "*.json")):
        stem = Path(fp).stem
        try:
            m[stem] = json.load(open(fp)).get("doc", {}).get("source_dataset", "?")
        except Exception:
            continue
    return m


def _balanced_subset(records, n_per_class: int):
    """Deterministic class-balanced subset, evenly strided across each class.

    Striding (rather than taking the first N by id) avoids clustering on
    filename-prefix batches (e.g. ogrants funded_/unfunded_/v2_/v3_), giving a
    representative sample within each label.
    """
    by_label: dict[str, list] = {}
    for r in sorted(records, key=lambda r: r.id):
        by_label.setdefault(r.label, []).append(r)
    picked = []
    for lab in sorted(by_label):
        pool = by_label[lab]
        if len(pool) <= n_per_class:
            picked.extend(pool)
        else:
            step = len(pool) / n_per_class
            picked.extend(pool[int(i * step)] for i in range(n_per_class))
    return sorted(picked, key=lambda r: r.id)


def load_checkpoint(path: Path) -> dict:
    """Return {(system,id): prediction} already computed."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done[(rec["system"], rec["id"])] = rec
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="retrieval_knn,single_call",
                    help="comma-separated subset of: " + ",".join(SYSTEMS))
    ap.add_argument("--limit", type=int, default=None, help="max proposals (first N)")
    ap.add_argument("--balance", type=int, default=0,
                    help="use a class-balanced subset of N per class (deterministic)")
    ap.add_argument("--source", default="",
                    help="restrict to one source_dataset (e.g. ogrants) to remove the "
                         "provenance confound before balancing")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--sc-n", type=int, default=10, help="self-consistency samples")
    ap.add_argument("--k", type=int, default=8, help="retrieval k for agentic systems")
    ap.add_argument("--knn-k", type=int, default=5, help="k for retrieval_knn baseline")
    ap.add_argument("--tag", default="default", help="output subfolder tag (e.g. model name)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild the report from the existing checkpoint only — no "
                         "dataset build, no batch fits, zero LLM/API calls")
    args = ap.parse_args()

    print("env:", json.dumps(envload.status()))

    if args.aggregate_only:
        # Offline re-score: recompute every aggregate metric from the stored
        # per-prediction checkpoint.  Nothing is predicted, fitted or fetched.
        if args.fresh:
            sys.exit("--aggregate-only re-scores the existing checkpoint; drop --fresh")
        out_dir = OUT_DIR / args.tag
        ckpt_path = out_dir / "predictions.jsonl"
        done = load_checkpoint(ckpt_path)
        if not done:
            sys.exit(f"no checkpoint at {ckpt_path} — nothing to aggregate")
        keep = sorted({s for (s, _) in done})
        n_props = len({i for (_, i) in done})
        # The llm block must describe the run that PRODUCED the checkpoint, not
        # the current environment, so reuse it from the prior report if present.
        try:
            llm_info = json.loads((out_dir / "ablation_report.json").read_text())["llm"]
        except Exception:
            llm_info = {"available": None, "provider": None, "model": None}
        print(f"aggregate-only: {len(done)} checkpoint rows, {len(keep)} systems, "
              f"{n_props} proposals — no LLM will be invoked")
        aggregate_and_write(args.tag, done, keep, n_props, llm_info, {},
                            out_dir, ckpt_path)
        return

    from grantagent.service.llm import llm_service
    live = llm_service.available()
    print(f"LLM available: {live}  provider={llm_service.provider}  model={llm_service.model}")
    if not live:
        print("!! No LLM — generative systems will use heuristic fallback (plumbing test only).")

    records = dataset.get_records(rebuild=args.rebuild_cache, limit=args.limit)
    if args.source:
        id2src = _id_to_source()
        records = [r for r in records if id2src.get(r.id) == args.source]
        print(f"source filter '{args.source}': {len(records)} records")
    if args.balance:
        records = _balanced_subset(records, args.balance)
    print("dataset:", json.dumps(dataset.label_summary(records)))

    ctx = Context(k=args.k, sc_n=args.sc_n, knn_k=args.knn_k)
    requested = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SYSTEMS]
    if unknown:
        sys.exit(f"unknown systems: {unknown}\navailable: {list(SYSTEMS)}")

    out_dir = OUT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "predictions.jsonl"
    if args.fresh and ckpt_path.exists():
        ckpt_path.unlink()
    done = load_checkpoint(ckpt_path)
    ckpt = open(ckpt_path, "a")

    labels_by_id = {r.id: r.label for r in records}

    # ── run each requested system over all records ──
    for base in requested:
        display = _sc_name(base, ctx)
        fn = SYSTEMS[base]

        if base in BATCH_FIT:
            print(f"\n[{display}] batch-fit ({len(records)} proposals)...")
            BATCH_FIT[base](records, ctx)

        n_new = 0
        t0 = time.time()
        for i, rec in enumerate(records):
            key = (display, rec.id)
            if key in done:
                continue
            METER.reset()
            t_rec = time.time()
            with meter_llm():
                try:
                    pred = fn(rec, ctx)
                except Exception as exc:
                    pred = {"system": display, "score": None, "verdict": None,
                            "pred": "declined", "meta": {"error": str(exc)}}
            calls, tokens = METER.snapshot()
            row = {"system": display, "id": rec.id, "label": rec.label,
                   "profile": rec.profile, "pred": pred.get("pred"),
                   "score": pred.get("score"), "verdict": pred.get("verdict"),
                   "n_calls": calls, "n_tokens": tokens,
                   "wall_s": round(time.time() - t_rec, 3),
                   "meta": pred.get("meta", {})}
            ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()
            done[key] = row
            n_new += 1
            if (i + 1) % 10 == 0 or live:
                dt = time.time() - t0
                print(f"  [{display}] {i+1}/{len(records)}  new={n_new}  "
                      f"{dt:.0f}s  last:{rec.label}->{row['pred']} "
                      f"(calls={calls},tok={tokens})")
        print(f"[{display}] done: {n_new} new predictions in {time.time()-t0:.0f}s")

    ckpt.close()

    # ── aggregate metrics ──
    keep = [_sc_name(b, ctx) for b in requested]
    llm_info = {"available": live, "provider": llm_service.provider,
                "model": llm_service.model}
    aggregate_and_write(args.tag, done, keep, len(records), llm_info,
                        labels_by_id, out_dir, ckpt_path)


def aggregate_and_write(tag: str, done: dict, keep: list[str], n_proposals: int,
                        llm_info: dict, labels_by_id: dict,
                        out_dir: Path, ckpt_path: Path) -> None:
    """Summarize checkpointed predictions into the report + markdown table.

    Pure aggregation over `done` ({(system,id): row}); shared by the live run
    path and --aggregate-only, so offline re-scores use the identical code.
    """
    by_system: dict[str, list[dict]] = {}
    for (system, _id), row in done.items():
        if system not in keep:
            continue
        by_system.setdefault(system, []).append(row)

    # full pipeline correctness vector for paired McNemar (align on id)
    full_name = "full"
    full_correct_by_id = None
    if full_name in by_system:
        full_correct_by_id = {r["id"]: int(r["pred"] == r["label"]) for r in by_system[full_name]}

    report = {"tag": tag, "n_proposals": n_proposals, "llm": llm_info,
              "systems": []}

    for system, rows in by_system.items():
        rows_sorted = sorted(rows, key=lambda r: r["id"])
        ids = [r["id"] for r in rows_sorted]
        preds = [r["pred"] for r in rows_sorted]
        labels = [labels_by_id.get(i, r["label"]) for i, r in zip(ids, rows_sorted)]
        scores = [r["score"] for r in rows_sorted]
        calls = [r["n_calls"] for r in rows_sorted]
        tokens = [r["n_tokens"] for r in rows_sorted]
        s = M.summarize(system, preds, labels, scores, calls, tokens)
        s.pop("_correct", None)
        # Wall-clock: rows written before per-record timing existed have no
        # "wall_s" — report null rather than a fabricated value.  The total is
        # only stated when EVERY row was timed; the mean covers the timed rows.
        walls = [r.get("wall_s") for r in rows_sorted]
        timed = [w for w in walls if w is not None]
        s["compute"]["mean_wall_s"] = round(sum(timed) / len(timed), 2) if timed else None
        s["compute"]["total_wall_s"] = (round(sum(timed), 1)
                                        if timed and len(timed) == len(walls) else None)
        # Paired McNemar strictly over ids BOTH systems predicted (a partial full-
        # pipeline run must not have its missing ids silently scored as wrong).
        if full_correct_by_id and system != full_name:
            pairs = [(int(p == l), full_correct_by_id[i])
                     for i, p, l in zip(ids, preds, labels) if i in full_correct_by_id]
            if pairs:
                own, fc = (list(t) for t in zip(*pairs))
                mc = M.mcnemar(own, fc)
                mc["p"] = round(mc["p"], 6)
                mc["n_shared"] = len(pairs)
                s["mcnemar_vs_full"] = mc
        report["systems"].append(s)

    # order rows by accuracy descending for the table
    report["systems"].sort(key=lambda s: s["accuracy"], reverse=True)

    json_path = out_dir / "ablation_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    md = render_markdown(report)
    md_path = out_dir / "ablation_table.md"
    md_path.write_text(md)
    print("\n" + md)
    print(f"\nwrote {json_path}\nwrote {md_path}\nwrote {ckpt_path}")


def render_markdown(report: dict) -> str:
    lines = []
    lines.append(f"# Ablation results — {report['tag']}")
    llm = report["llm"]
    lines.append(f"\nModel: `{llm['model']}` (provider `{llm['provider']}`, "
                 f"live={llm['available']}) · N={report['n_proposals']} proposals\n")
    lines.append("| System | Acc | 95% CI | BalAcc | MCC [95% CI] | P | R | F1 | "
                 "AUC [95% CI] | ScoreGap (p) | ECE | "
                 "calls/prop | tok/prop | wall s/prop | McNemar vs full (p) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in report["systems"]:
        ci = s["acc_ci95"]
        sg = s.get("score_gap", {})
        gap = sg.get("gap")
        gp = sg.get("p")
        mc = s.get("mcnemar_vs_full", {})
        def _fmt_p(p):
            return "<1e-4" if p < 1e-4 else (f"{p:.4f}" if p < 0.001 else f"{p}")
        mc_cell = (f"{_fmt_p(mc['p'])} (b={mc['b']},c={mc['c']})" if "p" in mc else "—")
        ba = s.get("balanced_accuracy")
        ba_cell = f"{ba*100:.1f}%" if ba is not None else "—"
        mci = s.get("mcc_ci95")
        mcc_cell = ("—" if s.get("mcc") is None else
                    f"{s['mcc']:.3f} [{mci[0]:.2f},{mci[1]:.2f}]" if mci
                    else f"{s['mcc']:.3f}")
        aci = s.get("auc_ci95")
        auc_cell = ("—" if s["auc"] is None else
                    f"{s['auc']:.3f} [{aci[0]:.2f},{aci[1]:.2f}]" if aci
                    else f"{s['auc']:.3f}")
        wall = s["compute"].get("mean_wall_s")
        wall_cell = f"{wall:.1f}" if wall is not None else "—"
        lines.append(
            f"| {s['system']} | {s['accuracy']*100:.1f}% | "
            f"[{ci[0]*100:.0f},{ci[1]*100:.0f}] | {ba_cell} | {mcc_cell} | "
            f"{s['precision']:.2f} | "
            f"{s['recall']:.2f} | {s['f1']:.2f} | "
            f"{auc_cell} | "
            f"{(f'{gap:.2f}' if gap is not None else '—')} "
            f"({(f'{gp:.3f}' if gp is not None else '—')}) | "
            f"{s['ece'] if s['ece'] is not None else '—'} | "
            f"{s['compute']['mean_calls']} | {s['compute']['mean_tokens']:.0f} | "
            f"{wall_cell} | "
            f"{mc_cell} |")
    lines.append("\n*ScoreGap = mean(awarded score) − mean(declined score); "
                 "p from Mann–Whitney U. McNemar p tests paired difference vs the full "
                 "pipeline (p>0.05 ⇒ not distinguishable from full). MCC/AUC CIs are "
                 "2000-resample percentile bootstraps. wall s/prop = mean measured "
                 "wall-clock per proposal; — means the checkpoint predates per-record "
                 "timing (never estimated).*")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
