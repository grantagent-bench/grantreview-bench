# Code — Reproducibility

All numbers in the paper are produced by scripts in this directory.

## Pinned dependencies

```bash
pip install -r requirements.txt
```

## Pinned model snapshots

- `gpt-4o-2024-08-06`
- `gpt-4o-mini-2024-07-18`

## Pinned seeds

- `random.seed(42)` and `np.random.seed(42)` set at module load in `analysis/run_baselines.py`
- `StratifiedKFold(shuffle=True, random_state=42)` in `analysis/analyze_inter_source_transfer.py`
- Bootstrap RNG: `numpy.random.RandomState(42)` (2,000 iterations)
- Permutation-null RNG: `numpy.random.RandomState(43)` (1,000 iterations, decorrelated from bootstrap)

## Reproduce each paper section

| Paper section | Script | Expected output |
|---|---|---|
| §3.4 Wellcome rule-vs-manifest audit | `data/wellcome_label_audit.py --audit` | 50/50, Wilson [0.929, 1.000], κ=1.000 |
| §3.4 Wellcome rule-vs-human audit | `data/wellcome_label_audit.py --score ../data/wellcome_audit_gold.jsonl` | 50/50, same CI/κ, inter-rater 50/50 |
| §5.3 Inter-source transfer matrix | `analysis/analyze_inter_source_transfer.py` | 5×5 matrix with bootstrap CIs + permutation p; NSF→EU/ERC = 0.08 [0.00, 0.32], p=0.024 |
| §5.4 Canonical splits | `data/build_splits.py` | `splits/canonical_v1.json` with seed=42 |
| Table 6 GroupKFold AUC | `analysis/analyze_groupkfold_auc.py --runs-dir <dir>` | per-baseline mean ± std over 5 folds |
| Table 6 per-doc baselines | `analysis/run_baselines.py --baseline {random,length,gpt4_*}` | one JSON per doc per baseline |

## Portability

All scripts resolve paths relative to `Path(__file__).resolve().parents[N]` (or read from `GRANTREVIEW_ROOT` / `GRANTREVIEW_DATA` env overrides). No `/Users/` strings remain in any script.

The PDF-extraction subprocess in `data/build_master_v2.py` uses `sys.executable` so it works on any Python install.

## What's NOT in this repo

The document text itself. The corpus is distributed in three tiers (per the top-level `LICENSE.md`):

- **Redistributed via HuggingFace:** Open Grants (CC BY 4.0) + Wellcome ORF (fair-use academic research with attribution) — `huggingface.co/datasets/torkian/grantreview-bench`
- **Link-only:** NIH/NSF/SERC — fetched by `data/build_master_v2.py`
- **Per-file:** declined extras — see `../data/declined_extras_provenance.json`
