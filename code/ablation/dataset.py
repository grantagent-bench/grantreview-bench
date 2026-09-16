"""Evaluation dataset for the ablation study.

Source of truth: the 455 real full-system runs in ``backend/runs/paper_eval/``.
Each run file records the source document (``doc.abs_path``), the ground-truth
funding outcome (``doc.label`` in {awarded, declined}), the routed profile, and the
full-system verdict that was actually produced.

We re-extract and sectionize the source text ONCE into ``eval_cache.json`` so that
every ablation system runs on identical inputs, decoupled from PDF parsing.  The
stored full-system verdict is carried along so the "full" condition can be scored
without re-spending compute (and so our re-implementation can be validated against it).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = BACKEND_DIR / "runs" / "paper_eval"
CACHE_PATH = BACKEND_DIR / "scripts" / "ablation" / "eval_cache.json"

VALID_LABELS = {"awarded", "declined"}


@dataclass
class EvalRecord:
    id: str
    profile: str
    label: str                      # ground truth: "awarded" | "declined"
    sections: dict
    text_len: int                   # characters, for short-doc logic
    words: int
    full_verdict: dict = field(default_factory=dict)  # stored full-system output

    def to_json(self) -> dict:
        return {
            "id": self.id, "profile": self.profile, "label": self.label,
            "sections": self.sections, "text_len": self.text_len,
            "words": self.words, "full_verdict": self.full_verdict,
        }

    @staticmethod
    def from_json(d: dict) -> "EvalRecord":
        return EvalRecord(
            id=d["id"], profile=d["profile"], label=d["label"],
            sections=d["sections"], text_len=d["text_len"],
            words=d.get("words", 0), full_verdict=d.get("full_verdict", {}),
        )


def build_cache(runs_dir: Path = RUNS_DIR, limit: Optional[int] = None,
                verbose: bool = True) -> list[EvalRecord]:
    """Re-extract + sectionize every run's source document into EvalRecords."""
    from grantagent.ingest.extract import extract_text
    from grantagent.ingest.sectionize import sectionize

    records: list[EvalRecord] = []
    skipped_missing = 0
    skipped_label = 0
    files = sorted(glob.glob(str(runs_dir / "*.json")))
    if limit:
        files = files[:limit]

    for fp in files:
        try:
            run = json.load(open(fp))
        except Exception:
            continue
        doc = run.get("doc", {})
        label = (doc.get("label") or "").strip().lower()
        abs_path = doc.get("abs_path", "")
        profile = run.get("profile") or doc.get("grant_type") or "NSF_CORE"

        if label not in VALID_LABELS:
            skipped_label += 1
            continue
        if not abs_path or not os.path.exists(abs_path):
            skipped_missing += 1
            continue

        try:
            text = extract_text(abs_path, filename=os.path.basename(abs_path))
            sections = sectionize(text)
        except Exception as exc:
            if verbose:
                print(f"  parse-fail {os.path.basename(abs_path)}: {exc}")
            skipped_missing += 1
            continue

        rid = Path(fp).stem  # unique per run
        records.append(EvalRecord(
            id=rid, profile=profile, label=label, sections=sections,
            text_len=len(text), words=int(doc.get("words", 0)),
            full_verdict=run.get("verdict", {}),
        ))

    if verbose:
        print(f"built {len(records)} records "
              f"(skipped {skipped_missing} missing/parse-fail, {skipped_label} bad-label)")
    return records


def save_cache(records: list[EvalRecord], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump([r.to_json() for r in records], open(path, "w"))


def load_cache(path: Path = CACHE_PATH) -> list[EvalRecord]:
    return [EvalRecord.from_json(d) for d in json.load(open(path))]


def get_records(rebuild: bool = False, limit: Optional[int] = None) -> list[EvalRecord]:
    """Load the cache, building it if absent or if rebuild=True."""
    if CACHE_PATH.exists() and not rebuild:
        recs = load_cache()
    else:
        recs = build_cache(limit=limit)
        save_cache(recs)
    if limit:
        recs = recs[:limit]
    return recs


def label_summary(records: list[EvalRecord]) -> dict:
    from collections import Counter
    labels = Counter(r.label for r in records)
    profiles = Counter(r.profile for r in records)
    return {"n": len(records), "labels": dict(labels), "profiles": dict(profiles)}


if __name__ == "__main__":
    # Standalone cache builder:  python -m scripts.ablation.dataset
    import sys
    sys.path.insert(0, str(BACKEND_DIR))
    from . import envload
    envload.load()
    recs = build_cache()
    save_cache(recs)
    print(json.dumps(label_summary(recs), indent=2))
