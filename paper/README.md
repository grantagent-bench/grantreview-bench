# GrantReview-Bench v1.0 — Submission Package

NeurIPS 2026 Datasets & Benchmarks. Companion paper to *When Open Grant Corpora Cannot Teach LLMs to Discriminate* (NLP4Science 2026, same author).

## Contents

| File | Purpose |
|---|---|
| `paper.pdf` | Compiled paper (19 pages) |
| `paper.tex` | LaTeX source |
| `paper.md` | Markdown source (canonical) |
| `DATASHEET.md` | Full Gebru-format Datasheet for Datasets |
| `HF_DATASET_CARD.md` | HuggingFace dataset card with YAML frontmatter |
| `LICENSE.md` | Tiered per-source license summary (non-commercial academic research use (US fair use + UK fair dealing) framing for redistributable sources) |
| `croissant.json` | Croissant 1.0 metadata (NeurIPS D&B 2024+ standard) |
| `zenodo_deposit.json` | Pre-formatted Zenodo deposit metadata for DOI minting |
| `figures/` | 3 PNG figures: corpus breakdown, length distribution, transfer-matrix heatmap |

## Reproducibility quick-start

All numbers in the paper are produced by the released code. From the repo root:

```bash
# Pin dependencies
pip install -r backend/requirements.txt

# Build the master manifest (portable; uses sys.executable + Path-relative DATA)
python data/build_master_v2.py

# Build the canonical 70/15/15 splits (seed=42)
python data/build_splits.py

# Reproduce §3.4 Wellcome labeling-rule audit
python data/wellcome_label_audit.py --audit
# Expected: Rule-vs-manifest 50/50 (Wilson 95% CI [0.929, 1.000])

python data/wellcome_label_audit.py --score data/wellcome_audit_gold.jsonl
# Expected: Rule-vs-human 50/50 (Wilson 95% CI [0.929, 1.000])

# Reproduce §5.3 inter-source label-transfer matrix (bootstrap CIs + permutation null)
python backend/scripts/analyze_inter_source_transfer.py
# Output: backend/runs/analysis/inter_source_transfer.json

# Reproduce Table 6 GroupKFold AUC column from per-baseline run files
python backend/scripts/analyze_groupkfold_auc.py
```

## Pinned model snapshots (used to produce Table 6)

- `gpt-4o-2024-08-06`
- `gpt-4o-mini-2024-07-18`

Random seed 42 throughout: `random.seed(42)`, `np.random.seed(42)`, `StratifiedKFold(random_state=42)`, `numpy.random.RandomState(42)` for bootstrap and permutation.

## License

Mixed per-source — see `LICENSE.md` and Paper 2 §8.1 for the full per-source review. CC BY 4.0 sources (Open Grants) and Wellcome's own public 2018/19 ORF release are redistributed with attribution under non-commercial academic research use (US fair use + UK fair dealing); NIH/NSF/SERC are link-only via `build_master.py`. Opt-out / takedown SLA: ≤4 weeks via a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues.

## Contact

the GrantReview-Bench maintainers, a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues. Issue tracker: https://github.com/grantagent-bench/grantreview-bench (private until paper acceptance; reviewers may request access). Opt-out / takedown: 4-week SLA via the same email.
