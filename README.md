# GrantReview-Bench v1.0

**The largest open evaluation corpus we know of for LLM-based grant proposal review** — N=455 real proposals across 8 sources and 10 funder families, with 194 confirmed declined examples (4.1× the prior largest open corpus). Paired with 61 NIH peer-reviewer summary statements for per-criterion alignment.

> If you know of a larger open corpus with real declined proposals at scale, please contact us — we will run our recommended evaluation protocol on it and publish a comparison.

**Maintainer:** 

---

## Quickstart — reproduce every paper number from scratch

```bash

git clone https://github.com/grantagent-bench/grantreview-bench   # private until paper acceptance; reviewers may request access
# If git does not work, download .zip from this repo
cd grantreview-bench
pip install -r code/requirements.txt

# §3.4 Wellcome labeling-rule audit
python code/data/wellcome_label_audit.py --audit
# Expected: 50/50 agreement, Wilson CI [0.929, 1.000], κ=1.000

python code/data/wellcome_label_audit.py --score data/wellcome_audit_gold.jsonl
# Expected: 50/50, same Wilson CI, κ=1.000, inter-rater 50/50

# §5.3 inter-source label-transfer matrix (5×5; bootstrap CIs + permutation null)
python code/analysis/analyze_inter_source_transfer.py
# Expected: 5 sub-corpora kept (N=11/22/56/59/155); NSF→EU/ERC = 0.08 [0.00,0.32], permutation p=0.024

# Table 6 GroupKFold AUC column (requires per-baseline run files; see code/README)
python code/analysis/analyze_groupkfold_auc.py --runs-dir <your-runs-dir>
```

All numbers are produced by these scripts with `random.seed(42)`, `np.random.seed(42)`, `StratifiedKFold(random_state=42)`, and `numpy.random.RandomState(42)` for the bootstrap (43 for the permutation null, decorrelated). LLM model snapshots pinned to `gpt-4o-2024-08-06` and `gpt-4o-mini-2024-07-18`.

## Headline finding

A 5-sub-corpus inter-source label-transfer matrix shows that **awarded/declined labels do not transfer across funder sub-corpora** — across 20 off-diagonal cells, AUCs concentrate around chance and **no cell survives Bonferroni or BH multiple-comparison correction at α=0.05**. The most extreme cell (OG-NSF/NSERC → OG-EU/ERC: AUC 0.08, 95% CI [0.00, 0.32], uncorrected permutation p=0.024) suggests *anti-correlation* — labels predicting the opposite outcome across funders. Direct empirical evidence that open multi-funder grant data does not yet encode a portable concept of quality.

## Recommended evaluation protocol (Paper §5)

Future system papers using this corpus should report:

1. **Length-quartile-stratified AUC** with Q4 (≥17,313 words) as the cleanest length-controlled comparison.
2. **Per-source AUC** with bootstrap 95% CIs for the two sources with both classes ≥10 (Open Grants 119/47; Wellcome 17/138).
3. **Cross-source GroupKFold CV** (5-fold by source) as the headline metric. StratifiedKFold leaks source.
4. **Inter-source transfer matrix** for any cross-funder generalization claim.
5. **Multiple-comparison correction** (Bonferroni or BH) across pairwise tests.
6. **Bootstrap CIs** on every reported AUC.

Reference baseline numbers for 8 systems are in Table 6 of the paper.

## Repository layout

```
grantreview-bench/
├── README.md                    (this file)
├── LICENSE.md                   (per-source license summary)
├── CITATION.cff                 (GitHub-recognized citation file)
├── paper/                       (paper PDF + LaTeX + datasheet + HF card + Croissant + Zenodo)
│   ├── paper.pdf                (19 pages)
│   ├── paper.tex
│   ├── paper.md                 (canonical markdown source)
│   ├── DATASHEET.md             (full Gebru-format datasheet)
│   ├── HF_DATASET_CARD.md       (HuggingFace dataset card)
│   ├── LICENSE.md
│   ├── croissant.json           (Croissant 1.0 metadata)
│   ├── zenodo_deposit.json      (pre-formatted Zenodo deposit metadata)
│   └── figures/                 (3 PNG figures)
├── code/                        (all reproducibility code)
│   ├── requirements.txt         (== pinned dependencies)
│   ├── analysis/
│   │   ├── analyze_inter_source_transfer.py   (§5.3 transfer matrix + bootstrap + permutation)
│   │   ├── analyze_groupkfold_auc.py          (Table 6 GroupKFold AUC column)
│   │   └── run_baselines.py                   (per-doc score generation; pinned snapshots)
│   └── data/
│       ├── build_master_v2.py                 (master manifest builder; portable)
│       ├── build_splits.py                    (canonical 70/15/15 splits; seed=42)
│       └── wellcome_label_audit.py            (§3.4 audit; --audit and --score modes)
├── data/                        (manifests + splits + audit artifacts; NOT the documents)
│   ├── master_manifest_v2.json                (455 application records; rel_path inside)
│   ├── splits/canonical_v1.json               (70/15/15 + SERC holdout)
│   ├── wellcome_audit_sample.jsonl            (seeded 50-entry sample)
│   ├── wellcome_audit_gold.jsonl              (2-rater human gold)
│   ├── declined_extras_provenance.json        (per-file provenance for the 10 declined extras)
│   └── inter_source_transfer.json             (live output of §5.3 analysis)
```

## Where the document text lives

This repo ships the **manifest, splits, audit artifacts, and code** — but not the document text itself. Document text is distributed under three tiers (per `LICENSE.md`):

- **Open Grants** (166 docs, CC BY 4.0) and **Wellcome ORF 2018/19** (155 docs, redistributed under non-commercial academic research use (US fair use + UK fair dealing) with attribution) — distributed via HuggingFace at `huggingface.co/datasets/torkian/grantreview-bench` (URL pending upload at paper acceptance).
- **NIH/NSF/SERC** (124 docs) — link-only; `code/data/build_master_v2.py` fetches from per-IC URLs.
- **Declined extras** (10 docs, mixed) — per-file decision in `data/declined_extras_provenance.json`.

## License

Mixed per-source — see [`LICENSE.md`](LICENSE.md). Released for **non-commercial academic research**. Opt-out / takedown SLA: ≤4 weeks.

## Citation


## Contributing

PRs welcome for:
- New sources (with per-source license documentation in `data/declined_extras_provenance.json`-style provenance)
- Errata in existing entries
- Extensions to the evaluation protocol
- Reproducible system results to add to the leaderboard

PRs that add documents must include a per-document license statement and provenance link.


Ben Torkian, University of South Carolina, `torkian@sc.edu`. GitHub issues welcome. Response time: ≤2 weeks for license / takedown questions, ≤4 weeks for general issues.
