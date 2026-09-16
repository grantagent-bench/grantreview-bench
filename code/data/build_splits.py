#!/usr/bin/env python3
"""Build canonical 70/15/15 train/dev/test splits for GrantReview-Bench v1.0.

Strategy:
- Stratify by label WITHIN source — so train/dev/test all see the right
  awarded/declined balance per source.
- NSF/SERC held out entirely as a distribution-shift probe (not in train/dev/test).
- Random seed 42.

Output: data/splits/canonical_v1.json with structure:
  {
    "train": [filename, ...],
    "dev":   [filename, ...],
    "test":  [filename, ...],
    "shift_holdout_serc": [filename, ...],
    "metadata": {seed, totals_per_split, ...}
  }

Reproducible: re-running this script always produces the same splits.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "master_manifest_v2.json"
OUT_DIR = ROOT / "splits"
OUT_FILE = OUT_DIR / "canonical_v1.json"
SEED = 42
TRAIN_FRAC = 0.70
DEV_FRAC = 0.15
# remainder = 0.15 → test


def main():
    docs = json.load(open(MANIFEST))
    apps = [d for d in docs if d.get("doc_type") == "application"]
    print(f"Total apps: {len(apps)}")

    rng = random.Random(SEED)

    # Hold out SERC (NSF Geo, all-awarded, single-source) as distribution-shift probe
    shift = [d for d in apps if d["source_dataset"] == "serc"]
    rest = [d for d in apps if d["source_dataset"] != "serc"]
    print(f"Shift holdout (SERC): {len(shift)}")
    print(f"Remaining for train/dev/test: {len(rest)}")

    # For the rest, stratify by (source, label) — assign each (source, label) bucket
    # randomly into train/dev/test in a way that approximates the global ratios
    # while ensuring each bucket appears in train (so models see all source-label
    # combos during training).
    buckets = defaultdict(list)
    for d in rest:
        buckets[(d["source_dataset"], d["label"])].append(d)
    for k in buckets:
        rng.shuffle(buckets[k])

    train, dev, test = [], [], []
    for (src, lbl), ds in sorted(buckets.items()):
        n = len(ds)
        n_train = max(1, int(round(n * TRAIN_FRAC)))
        n_dev = max(1, int(round(n * DEV_FRAC))) if n >= 5 else 0
        # If a tiny bucket would leave 0 or 1 in test, give the leftover to test
        n_test = n - n_train - n_dev
        if n_test < 0:
            # shrink train by 1 if dev squeezes test out
            n_train += n_test
            n_test = 0
        train += ds[:n_train]
        dev += ds[n_train:n_train + n_dev]
        test += ds[n_train + n_dev:]

    # Sanity: no overlap; covers everything
    train_set = {d["abs_path"] for d in train}
    dev_set = {d["abs_path"] for d in dev}
    test_set = {d["abs_path"] for d in test}
    shift_set = {d["abs_path"] for d in shift}
    assert len(train_set & dev_set) == 0
    assert len(train_set & test_set) == 0
    assert len(dev_set & test_set) == 0
    assert (train_set | dev_set | test_set | shift_set) == {d["abs_path"] for d in apps}, "split coverage mismatch"

    # Stats
    def stats(name, items):
        c = Counter(d["label"] for d in items)
        return {
            "name": name, "N": len(items),
            "awarded": c.get("awarded", 0),
            "declined": c.get("declined", 0),
            "by_source": dict(Counter(d["source_dataset"] for d in items)),
        }

    summary = [
        stats("train", train),
        stats("dev", dev),
        stats("test", test),
        stats("shift_holdout_serc", shift),
    ]
    print("\n=== Split summary ===")
    for s in summary:
        print(f"  {s['name']:<25} N={s['N']:3d}  aw={s['awarded']:3d}  dec={s['declined']:3d}  sources={list(s['by_source'].keys())}")

    out = {
        "version": "v1.0",
        "seed": SEED,
        "fractions": {"train": TRAIN_FRAC, "dev": DEV_FRAC, "test": 1 - TRAIN_FRAC - DEV_FRAC},
        "train": [d["filename"] for d in train],
        "dev": [d["filename"] for d in dev],
        "test": [d["filename"] for d in test],
        "shift_holdout_serc": [d["filename"] for d in shift],
        "metadata": {"summary": summary},
    }
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
