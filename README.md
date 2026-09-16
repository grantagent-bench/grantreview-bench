# GrantReview-Bench v1.0

**The largest open evaluation corpus we know of for LLM-based grant proposal review** — N=455 real proposals across 8 sources and 10 funder families, with 194 confirmed declined examples (4.1× the prior largest open corpus). Paired with 61 NIH peer-reviewer summary statements for per-criterion alignment.

> If you know of a larger open corpus with real declined proposals at scale, please open a
> GitHub issue — we will run our recommended evaluation protocol on it and publish a comparison.

**Companion paper (NeurIPS 2026 Datasets & Benchmarks):** [`paper/paper.pdf`](paper/paper.pdf) — *GrantReview-Bench: A Multi-Funder Open Benchmark for Evaluating LLM Grant Review Systems with a Recommended Two-Confound Evaluation Protocol*

**Companion system paper (NLP4Science 2026):** *When Open Grant Corpora Cannot Teach LLMs to Discriminate*

**Questions, corrections, and removal requests:** please
[open a GitHub issue](../../issues). We do not list a personal contact address; the issue
tracker is the maintained channel for everything, including data-removal requests.

---

## Running the benchmark

```bash
git clone https://github.com/grantagent-bench/grantreview-bench
cd grantreview-bench
pip install -r code/requirements.txt
```

**What you get without any setup.** The corpus metadata, splits, audit artifacts, and the
measured reference results are all in this repository. To read the headline system results,
open `results/ogrants166/ablation_table.md`, or re-derive every aggregate from the stored
per-item predictions — no API key, no network, no cost:

```bash
python -m code.ablation.run --tag ogrants166 --aggregate-only
```

**To evaluate your own system.** Score each proposal in `data/master_manifest_v2.json`,
write one JSON object per line keyed by the manifest's `filename` (the `grb1_…` id), and
place it at `results/<your-tag>/predictions.jsonl`:

```json
{"system": "my-system", "id": "grb1_<id>.pdf", "label": "awarded", "pred": "awarded", "score": 3.5}
```

Then run the same `--aggregate-only` command with your tag to get accuracy, balanced
accuracy, MCC, AUC with bootstrap CIs, ECE, and McNemar tests against the other arms.
Report the metrics listed under *Recommended evaluation protocol* below.

**To re-run the LLM arms** (costs money — the full pipeline is roughly 18× a single call):

```bash
export LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini-2024-07-18 OPENAI_API_KEY=...
python -m code.ablation.run --systems single_call,full --source ogrants --tag my-run
```

The document text is not distributed here — see *Where the document text lives* below for
how to obtain it.

## Reproducing the paper's analysis

```bash

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

- **Open Grants** (166 docs, CC BY 4.0) and **Wellcome ORF 2018/19** (155 docs, redistributed under non-commercial academic research use (US fair use + UK fair dealing) with attribution) — distributed via HuggingFace (dataset URL pending upload at paper acceptance).
- **NIH/NSF/SERC** (124 docs) — link-only; `code/data/build_master_v2.py` fetches from per-IC URLs.
- **Declined extras** (10 docs, mixed) — per-file decision in `data/declined_extras_provenance.json`.

## Reference system results

`code/ablation/` is the multi-agent evaluation harness, and `results/ogrants166/` holds its
measured output on the source-controlled Open Grants population (N=166,
`gpt-4o-mini-2024-07-18`): **13 architecture arms** — single call, multi-criteria single
call, self-consistency@30, retrieval k-NN, agents-only, debate-only, the four
single-component ablations, the full pipeline, a debate-connected variant, and a
long-context control.

`ablation_report.json` carries per-arm accuracy, balanced accuracy, MCC, AUC (all with
bootstrap 95% CIs), ECE, McNemar tests against the full pipeline, and per-proposal call and
token counts. `predictions.jsonl` holds per-item predictions keyed by the same `grb1_`
identifiers as the manifest, so results join directly to corpus metadata.

Re-aggregate the reports offline, without any API calls:

```bash
python -m code.ablation.run --tag ogrants166 --aggregate-only
```

Token counts are exact for the single-shot and self-consistency arms and `chars/4`
estimates for the agentic arms, so the comparison is **approximately budget-matched**
rather than compute-matched. No dollar figures are reported, since estimated tokens do not
support a defensible price.

## Data provenance, privacy, and removal

Every document in this corpus was **already published** — by the applicants themselves, by
the funder, or through statutory disclosure. We are not the original publisher of any
proposal or funding decision here.

Even so, applicant identity is not needed to evaluate a grant-review system, so it is not
published: documents are keyed by opaque `grb1_…` identifiers, and applicant names,
proposal titles, and verbatim funder decision wording are withheld from the manifest.
Only the derived `awarded` / `declined` label ships there. The two Wellcome label-audit
files keep `decision_raw`, since checking the decision-text-to-label rule is what they
exist for. Identifiers are stable, so every published number stays reproducible and
per-item results stay joinable.

**If your proposal is in this corpus and you would rather it were not, we will remove it —
no reason needed.** [Open a GitHub issue](../../issues) titled `Removal request`. You do not
have to post any details publicly — naming the funder and year is enough for a maintainer to
find the record and follow up. See [`PRIVACY_AND_REMOVAL.md`](PRIVACY_AND_REMOVAL.md) for
what helps, response times, and an honest statement of what removal here can and cannot undo.

## License

Mixed per-source — see [`LICENSE.md`](LICENSE.md). Released for **non-commercial academic research**. Opt-out / takedown: ≤4 weeks via a [GitHub issue](../../issues).

## Citation

Author and venue details will be added once the companion paper is through review.

```bibtex
@inproceedings{grantreviewbench2026,
  title     = {GrantReview-Bench: A Multi-Funder Open Benchmark for Evaluating LLM Grant Review Systems with a Recommended Two-Confound Evaluation Protocol},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026}
}
```

## Contributing

PRs welcome for:
- New sources (with per-source license documentation in `data/declined_extras_provenance.json`-style provenance)
- Errata in existing entries
- Extensions to the evaluation protocol
- Reproducible system results to add to the leaderboard

PRs that add documents must include a per-document license statement and provenance link.

## Contact

All correspondence goes through the [GitHub issue tracker](../../issues) — questions,
errata, licensing, and removal requests alike. No personal contact address is published.
Response time: ≤2 weeks for licensing and removal requests, ≤4 weeks for general issues.
