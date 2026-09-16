---
license: other
license_name: grantreview-bench-mixed-per-source
license_link: LICENSE.md
language:
- en
pretty_name: GrantReview-Bench v1.0
size_categories:
- n<1K
task_categories:
- text-classification
task_ids:
- multi-class-classification
tags:
- grant-review
- peer-review
- evaluation
- llm-evaluation
- science-of-science
- bibliometrics
- negative-results
- recommended-evaluation-protocol
annotations_creators:
- expert-generated
- found
language_creators:
- found
source_datasets:
- original
multilinguality:
- monolingual
configs:
- config_name: default
  data_files:
    - split: train
      path: splits/train.jsonl
    - split: dev
      path: splits/dev.jsonl
    - split: test
      path: splits/test.jsonl
    - split: shift_holdout_serc
      path: splits/shift_holdout_serc.jsonl
- config_name: paired_summary_statements
  data_files:
    - split: pairs
      path: paired/summary_statements.jsonl
dataset_info:
  features:
  - name: filename
    dtype: string
  - name: source_dataset
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          0: declined
          1: awarded
  - name: funder
    dtype: string
  - name: funder_family
    dtype: string
  - name: format
    dtype: string
  - name: words
    dtype: int32
  - name: text
    dtype: string
  - name: extra
    dtype: string
  splits:
  - name: train
    num_examples: 302
  - name: dev
    num_examples: 65
  - name: test
    num_examples: 66
  - name: shift_holdout_serc
    num_examples: 22
---

# GrantReview-Bench v1.0

> The largest open evaluation corpus we know of for LLM-based grant proposal review (N=455 + 61 paired summary statements; 194 confirmed declined examples; 8 sources; 10 funder families). If a larger one exists, please contact the maintainer — we will run our recommended evaluation protocol on it and publish a comparison.

**Maintainer:** the GrantReview-Bench maintainers(USC) — a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues
**Companion papers:**
- *When Open Grant Corpora Cannot Teach LLMs to Discriminate* (NLP4Science 2026, Paper 1)
- *GrantReview-Bench: A Multi-Funder Open Benchmark...* (NeurIPS D&B 2026 submission, Paper 2)

**Headline empirical finding:** a 5-sub-corpus inter-source label-transfer matrix shows that awarded/declined labels do not transfer across funder sub-corpora — across 20 off-diagonal cells, AUCs concentrate around chance and no cell survives Bonferroni/BH correction at α=0.05; the most extreme cell (OG-NSF/NSERC→OG-EU/ERC: AUC 0.08, 95% CI [0.00, 0.32], uncorrected permutation p=0.024) suggests anti-correlation. The corpus is therefore released *not* as a benchmark where higher AUC is the goal, but as a resource for studying *why* open multi-funder grant data resists generalization, and a falsifiability invitation for extending the corpus with non-voluntary-share declined examples (which would tighten every CI in this matrix).

## Quick start

```python
from datasets import load_dataset

# Default splits (train/dev/test/shift_holdout)
ds = load_dataset("grantagent-bench/grantreview-bench", "default")
print(ds)
# DatasetDict({
#   train: Dataset({features: [...], num_rows: 302}),
#   dev: Dataset({features: [...], num_rows: 65}),
#   test: Dataset({features: [...], num_rows: 66}),
#   shift_holdout_serc: Dataset({features: [...], num_rows: 22}),
# })

# Paired NIH peer-reviewer summary statements
pairs = load_dataset("grantagent-bench/grantreview-bench", "paired_summary_statements")
```

## Per-record schema

Each record contains:

| field | type | description |
|---|---|---|
| `filename` | string | unique filename within source |
| `source_dataset` | string | one of: ogrants, wellcome, niaid, nci, nidcd, nhgri, serc, declined_extra |
| `label` | string | "awarded" or "declined" |
| `funder` | string | funder name as published (e.g., "U.S. National Science Foundation (NSF)") |
| `funder_family` | string | normalized: NSF/NSERC, NIH, Wellcome, EU/ERC, Private, Other |
| `format` | string | "pdf" or "txt" |
| `words` | int | extracted word count |
| `text` | string | extracted plain text (may be empty for link-only files; see License below) |
| `extra` | string | optional source-specific metadata, JSON-encoded since fields vary per source (e.g., Wellcome `decision_raw`/`pi`/`title`, Open Grants `attribution_string`, NIH biosketch fields). Empty string for sources with no source-specific metadata. |

For the `paired_summary_statements` config: each record additionally has a `summary_statement_text` field with the parsed NIH summary statement (NIAID + NIDCD only; N=37 with parsable per-criterion scores out of 61 total pairs).

## Recommended evaluation protocol

The corpus has two confounds that affect naive aggregate metrics: **length** (declined median 992 vs awarded 10,229 words) and **source** (Simpson's-paradox effects across sub-corpora). We strongly recommend any future system paper using this corpus follow §5 of the companion Paper 2:

1. **Cross-source GroupKFold CV** as the headline metric (StratifiedKFold leaks source).
2. **Per-quartile AUC** with Q4 (≥17,313 words) as the cleanest length-controlled comparison.
3. **Per-source AUC** with bootstrap 95% CIs for the two sources with both classes ≥10 (Open Grants 119/47; Wellcome 17/138).
4. **Inter-source transfer matrix** for any cross-funder generalization claim.
5. **Multiple-comparison correction** (Bonferroni or BH) across pairwise tests.

Reference baseline numbers (Paper 2 §6) are provided so future systems are directly comparable.

## Known biases

The corpus has documented sampling biases that users must understand before drawing conclusions from results computed on it:

1. **Voluntary-share bias on declined examples.** 47 of 194 declined are from Open Grants (voluntary author sharing) and 138 from Wellcome's 2018/19 Open Research Fund release. Authors who voluntarily share rejected proposals are likely systematically different — more confident, more career-secure, more invested in open-science norms — than the population whose proposals are declined.
2. **Awarded examples come from agency exemplar repositories.** NIAID, NCI, NIDCD, NHGRI, and SERC release these as model proposals — selected by agency staff to exemplify good practice. They are not representative of all funded proposals.
3. **Length confound.** Awarded median 10,229 words vs. declined median 992 words; a length-only logistic regression achieves AUC 0.87 under StratifiedKFold and 0.65 under cross-source GroupKFold. The length-quartile-stratified protocol controls for this, but reports must include it.
4. **Source confound.** 6 of 8 sources contain only awarded examples. Aggregate AUC measures source-distinguishing capacity as much as content discrimination. Per-source AUC reporting is mandatory.
5. **Format heterogeneity.** Documents range from 1-page Wellcome summaries to 200-page NIH applications. Format conventions encode signal that cross-funder generalization should not depend on.
6. **Inter-source label non-transfer.** Empirically demonstrated in Paper 2 §5.3 — labels do not transfer across funder sub-corpora; across 20 off-diagonal cells of the 5-sub-corpus matrix, two have uncorrected permutation p<0.05 (OG-NSF/NSERC→OG-EU/ERC at AUC 0.08; OG-EU/ERC→Wellcome at 0.36) but neither survives multiple-comparison correction. Treat the awarded/declined label as funder-conditional, not as ground-truth quality.

## License

**Mixed per-source.** The corpus aggregates 8 sources with different redistribution rights:

| Source | License | Distribution in this package |
|---|---|---|
| Open Grants (CC BY 4.0) | 166 docs, redistributed with attribution per entry | ✓ included |
| Wellcome ORF 2018/19 | 155 docs, Wellcome's own public 2018/19 ORF release | ✓ included with attribution to Wellcome on every record (redistributed under non-commercial academic research use (US 17 U.S.C. § 107 fair use + UK CDPA 1988 §29 fair dealing); opt-out via §A.9 of Paper 2) |
| NIAID/NCI/NIDCD/NHGRI | 102 docs, NIH educational use | ✗ link-only (download via `build_master.py`) |
| SERC Carleton (NSF Geo) | 22 docs, PI permission per program | ✗ link-only |
| FOIA + ERC anonymized + author-share | 10 docs | mixed; per-file decisions |

Run `build_master.py` after downloading this dataset to fetch the link-only files locally.

## Citation

```bibtex
@misc{grantreviewbench2026,
  author    = {GrantReview-Bench maintainers},
  title     = {GrantReview-Bench v1.0: A Multi-Funder Open Benchmark for LLM Grant Review Systems},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/grantreview-bench}
}
```

If you use the recommended evaluation protocol or the inter-source transfer matrix (during the review period — change `@unpublished` to `@inproceedings` after acceptance):

```bibtex
@unpublished{companionpaper2026,
  author = {GrantReview-Bench maintainers},
  title  = {GrantReview-Bench: A Multi-Funder Open Benchmark for Evaluating LLM Grant Review Systems with a Recommended Two-Confound Evaluation Protocol},
  note   = {Under review, NeurIPS 2026 Datasets and Benchmarks},
  year   = {2026}
}
```

## Contact

GitHub issues: https://github.com/grantagent-bench/grantreview-bench (private until paper acceptance; reviewers may request access via email). Email maintainer: a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues. Response time: ~2 weeks for license/takedown questions, ~4 weeks for general issues.
