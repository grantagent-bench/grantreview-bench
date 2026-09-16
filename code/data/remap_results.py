#!/usr/bin/env python3
"""Rewrite ablation result files to the release's opaque document IDs.

The harness keys per-item predictions by source filename, prefixed with the source and
label: ``ogrants_awarded_funded_<name>_2020_1.pdf``. Those filenames identify applicants,
so results cannot be published as produced. This maps each one onto the ``grb1_``
identifier the manifest uses, leaving every score, verdict and metric untouched.

Requires ``data/anon_mapping.PRIVATE.json`` (produced by anonymize_release.py) and so can
only be run by a maintainer holding the mapping.

    python code/data/remap_results.py --from ../grant-review/backend/scripts/ablation/results/ogrants166
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAPPING = REPO / "data" / "anon_mapping.PRIVATE.json"
DEST = REPO / "results"

# Files safe to publish once IDs are remapped. Anything else in a results directory
# (raw logs especially) is skipped rather than guessed at.
REPORTS = ("ablation_report.json", "ablation_table.md", "analysis.json", "analysis.md")


def build_lookup() -> dict[str, str]:
    """original filename -> grb1_ id (with extension)."""
    mapping = json.loads(MAPPING.read_text())["mapping"]
    return {v["filename"]: f"{k}{Path(v['filename']).suffix}"
            for k, v in mapping.items()}


def resolve(item_id: str, lookup: dict[str, str]) -> str | None:
    """Strip the harness's <source>_<label>_ prefix, then look the filename up."""
    parts = item_id.split("_", 2)
    if len(parts) == 3 and (hit := lookup.get(parts[2])):
        return hit
    return lookup.get(item_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="harness results directory to import")
    ap.add_argument("--tag", help="destination name under results/ (default: source dir name)")
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = ap.parse_args()

    if not MAPPING.exists():
        print(f"missing {MAPPING.relative_to(REPO)} — run anonymize_release.py first")
        return 1

    lookup = build_lookup()
    out = DEST / (args.tag or args.src.name)

    preds = args.src / "predictions.jsonl"
    rows, unresolved = [], []
    if preds.exists():
        for line in preds.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if (new_id := resolve(rec.get("id", ""), lookup)) is None:
                unresolved.append(rec.get("id", ""))
                continue
            rec["id"] = new_id
            rows.append(rec)

    reports = [f for f in REPORTS if (args.src / f).exists()]
    print(f"source        : {args.src}")
    print(f"predictions   : {len(rows)} rows remapped, {len(unresolved)} unresolved")
    for u in unresolved[:5]:
        print(f"    UNRESOLVED: {u[:60]}")
    print(f"report files  : {', '.join(reports) or 'none'}")
    print(f"destination   : {out.relative_to(REPO)}")

    if unresolved:
        print("\nABORT: an ID did not resolve, so a real filename would be published.")
        return 1
    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    if rows:
        (out / "predictions.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
    for name in reports:
        shutil.copy2(args.src / name, out / name)
    print(f"\nwrote {len(rows)} predictions + {len(reports)} report files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
