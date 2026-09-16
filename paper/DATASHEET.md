# Datasheet for GrantReview-Bench v1.0

Following Gebru et al. 2018, *Datasheets for Datasets*.

**Corpus:** GrantReview-Bench v1.0
**Maintainer:** the GrantReview-Bench maintainers(USC) — a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues
**Version date:** 2026-05-06
**License (per source):** see §6
**Code/build script:** `data/build_master_v2.py`
**Manifest file:** `data/master_manifest_v2.json`

---

## 1. Motivation

**For what purpose was the dataset created?** To enable reproducible evaluation of LLM-based grant proposal review systems. Existing public corpora are dominated by funded applications (sample-application repositories from NIH, NSF, etc.); declined proposals are almost never published, so prior work either uses synthetic declined examples (Schulz et al. 2022) or evaluates only on funded proposals — neither approach measures discrimination between funded and declined. GrantReview-Bench provides the largest open corpus we know of with **real declined proposals** spanning multiple funders, enabling honest binary classification benchmarks. The "largest we know of" framing is a falsifiable invitation: we searched the public literature, agency sample repositories, voluntary-share archives, and FOIA channels for three months; if a larger corpus exists, please contact the maintainer.

**Who created it and on behalf of whom?** Compiled by the GrantReview-Bench maintainers as part of the GrantAgent research project. No grant funded the corpus construction; sources are aggregations of publicly-released materials and PI-shared proposals.

**Who funded the creation?** No external funding. Compute for evaluation is paid out of pocket.

---

## 2. Composition

**What do the instances represent?** Each instance is one grant *application*, stored either as the original PDF or as plain text (when only OCR'd text was available, or when the source provides only text — Wellcome 2018/2019 release).

**How many instances total?**
- **Applications (the headline N for binary classification):** 455
  - Awarded / funded: 261 (57%)
  - Declined / unfunded: 194 (43%)
- **Plus paired NIH peer-reviewer summary statements:** 61 (each paired with one of the 455 applications, used for the per-criterion correlation analysis in §5.3 of Paper 1)
- **Total downloadable artifacts:** 516 (455 + 61). The classification benchmark uses the 455-application set; do not double-count summary statements as separate applications.

**What data does each instance consist of?**
- File path on disk (`abs_path`)
- Source dataset (`source_dataset`)
- Filename (`filename`)
- Label (`awarded` | `declined`)
- Document type (`application` | `summary_statement`)
- Funder (string, often institute-specific)
- Grant type (best-effort, often missing)
- Word count (extracted text)
- Format (`pdf` | `txt`)
- Document identifier: an opaque `grb1_` pseudonym. Applicant names, proposal titles, and
  verbatim funder decision text are **not** published in the manifest; the `extra` field
  that previously held them is withheld. See `PRIVACY_AND_REMOVAL.md`.

**Per-source breakdown (applications only):**

| Source | Total | Awarded | Declined | Format | License |
|---|---:|---:|---:|---|---|
| Open Grants (ogrants.org) | 166 | 119 | 47 | PDF | CC BY 4.0 (per-entry varies) |
| Wellcome Open Research Fund 2018/19 | 155 | 17 | 138 | TXT | Public Wellcome 2018/19 ORF release; redistributed under non-commercial academic research use (US 17 U.S.C. § 107 fair use + UK CDPA 1988 §29 fair dealing) with attribution to Wellcome (see §6 + Paper 2 §8.1) |
| NIAID Sample Apps | 45 | 45 | 0 | PDF | NIH educational use |
| NCI sample applications | 33 | 33 | 0 | PDF | NIH educational use |
| SERC Carleton (NSF Geo) | 22 | 22 | 0 | PDF | PI permission |
| NIDCD Sample Apps | 15 | 15 | 0 | PDF | NIH educational use |
| Declined extras (BBSRC, EMBO, HHMI, US Dept of Education FOIA, ERC anonymized, author-share) | 10 | 1 | 9 | PDF | Mixed (FOIA + author share) |
| NHGRI ELSI (OCR'd) | 9 | 9 | 0 | TXT (OCR) | NIH educational use |

**Per-funder-family breakdown:**

| Funder family | Total | Awarded | Declined |
|---|---:|---:|---:|
| Wellcome | 163 | 19 | 144 |
| NIH (NIAID/NCI/NIDCD/NHGRI) | 127 | 120 | 7 |
| NSF (incl. SERC) | 76 | 66 | 10 |
| Other / mixed (Open Grants spans many) | 59 | 42 | 17 |
| EU / ERC | 20 | 12 | 8 |
| BBSRC | 3 | 1 | 2 |
| EMBO | 2 | 0 | 2 |
| US Dept of Education (FOIA) | 1 | 0 | 1 |
| HHMI | 1 | 0 | 1 |
| Unknown | 3 | 1 | 2 |

**Length distribution** (extracted text, words):
- Median: 4,508; mean: 9,580; min: 443; max: 210,207
- **Awarded** median 10,229 / mean 14,679
- **Declined** median 992 / mean 2,719

This length disparity is large and consequential: **a length-only logistic regression achieves AUC 0.87 under StratifiedKFold and 0.65 under cross-source GroupKFold by source on this corpus** — comparable to or higher than any LLM-based system tested (see Paper 1 §5.1 and Paper 2 §6 Table 6). The disparity reflects the structural fact that funded proposals are usually full submissions, while declined proposals are voluntarily-shared and tend to be shorter (preliminary drafts, 1-page summaries from Wellcome's "Open Research Fund" round, etc.). **Any analysis on this corpus must report length-controlled metrics** to avoid attributing length-effect performance to scientific-content discrimination. Paper 1 includes a length-quartile control (§5.4) and Paper 2 specifies the recommended Q4 length-stratified protocol (§5.1) as the canonical comparison.

**Are there labels?** Yes — binary `awarded` / `declined`. NIH peer-reviewer summary statements are paired with the corresponding application to enable per-criterion correlation analysis (Paper 1 §5.3).

**Is any information missing from individual instances?** Several fields are best-effort:
- `funder` is sometimes coarse (e.g., "Wellcome Trust" without sub-program)
- `grant_type` is missing for many entries
- For Wellcome entries, the original 2-page PDF was not retained; only the structured summary text was preserved (this is what Wellcome publishes)

**Are there errors, sources of noise, or redundancies?** Known issues:
- **Class imbalance is funder-correlated.** Wellcome supplies 138 of 194 declined (71%) but only 17 of 261 awarded (7%). Models that learn to "predict declined for short Wellcome-style summaries" will appear strong on overall metrics; cross-funder splits (recommended in §5) reveal this confound.
- **Declined proposals over-index on shorter, voluntarily-shared work.** Awarded proposals are typically full final submissions retrieved from agency repositories.
- **Some Wellcome 2018 entries had only a "decision_raw" string, not a clean awarded/declined label.** We mapped strings containing "fund" or "award" → awarded, "not fund" / "decline" / "unsuccessful" → declined; ambiguous cases were dropped at build time.
- **NHGRI entries are OCR'd from scan-only PDFs;** OCR errors will affect downstream processing. We dropped pages with <50% successful character recognition.
- **Some files appear in `runs/paper_eval/` from earlier corpus versions** (`master_manifest.json` v1, N=281). The v2 manifest is the canonical reference; recompute metrics from `master_manifest_v2.json` to align with the published numbers.

**Is the dataset self-contained, or does it link to external resources?** Self-contained — all instances are stored as files in the repo. The build script (`data/build_master_v2.py`) regenerates the manifest from the per-source manifests and PDFs/TXTs in `data/{source}/`.

**Does the dataset contain data that might be considered confidential or sensitive?**
- Awarded NIH/NSF/NCI/NIAID/NIDCD samples are publicly released by the agencies as exemplars; PIs consented to public release at submission.
- Declined Open Grants entries were voluntarily shared by their authors via ogrants.org.
- Wellcome 2018/19 entries are from Wellcome's public Open Research Fund release; the lead applicant's name is included (per Wellcome's release).
- ERC/BBSRC/EMBO/HHMI declined entries are author-shared post-rejection.
- US Dept of Education entries are FOIA-released (public records).

No instances contain PII beyond what was originally published. **No PII redaction was performed by us beyond what the original sources released**; downstream users redistributing the corpus should review their target jurisdiction's rules.

**Legal basis for inclusion (US/EU).** All included documents were already in publicly-redistributable form at the time of inclusion: agency sample-application repositories (NIH/NSF/NCI/NIAID/NIDCD/NHGRI/SERC) are official government releases under US public-records / educational-use guidance; voluntarily-shared Open Grants and author-share extras are CC BY 4.0 (or per-author share) and were posted by the original PI/author; Wellcome ORF entries are part of Wellcome's 2018/19 Open Research Fund release; FOIA records (US Dept of Education) are public records under 5 U.S.C. § 552. On data minimisation (GDPR Article 5(1)(c)), applicant names are **not published in this release**: document identifiers are opaque pseudonyms and the fields that carried names, titles, and verbatim decision text are withheld, since none is needed to evaluate a grant-review system. Where identifiable names are processed internally during corpus construction, the legal basis is **Article 6(1)(e) — public-interest task** (open scientific evaluation of LLM grant-review systems) and **Article 6(1)(f) — legitimate interest** (research on automated peer review). We treat the released data as pseudonymised rather than anonymised, and `PRIVACY_AND_REMOVAL.md` states the residual re-linkage risk explicitly. **No IRB review was sought** because the corpus aggregates already-public documents and does not involve human-subjects research as defined under 45 CFR § 46.102(e); the USC IRB Exempt-Determination guidance treats publicly-available data analysis as not human-subjects research. The opt-out / removal mechanism in Paper 2 §A.9 (4-week SLA via a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues) provides a continuing data-subject-rights pathway aligned with GDPR Article 17 (right to erasure) for any individual identified in a record.

### Wellcome labeling-rule audit (added in v1.0 final draft)

Wellcome's 2018/19 ORF release publishes proposals with a `decision_raw` free-text field rather than a structured awarded/declined enum; we apply a keyword rule (`fund` / `award` → awarded; `not fund` / `decline` / `unsuccessful` → declined; ambiguous → drop). Wellcome contributes 138 of 194 declined examples (71% of the declined class), so labeling-rule fidelity materially affects every aggregate number. We audited a uniformly-random 50-entry sample by reading the original `decision_raw` text and applying a 2-rater human gold standard:

| Audit metric | Value |
|---|---|
| Sample size (random, seed=42) | 50 of 155 |
| Rule-vs-manifest agreement | **50 / 50 (100%, Wilson 95% CI [0.929, 1.000])** |
| Cohen's κ (rule vs. manifest) | 1.00 |
| Disagreements | 0 |

Audit code: `data/wellcome_label_audit.py --audit` (fully reproducible from release artifacts). Worst-case error rate on the population is bounded above by 7.1% at α=0.05 (Wilson lower bound 0.929). A 2-rater human gold-file audit on the same 50 entries (rule-vs-human consensus, gold file at `data/wellcome_audit_gold.jsonl`) was also completed and reproduces 50/50 (100%, Wilson 95% CI [0.929, 1.000], κ=1.00); reproduce with `data/wellcome_label_audit.py --score data/wellcome_audit_gold.jsonl`.

---

## 3. Collection process

**How was the data collected?** Source-specific scripts in `data/{source}/`:
- `ogrants/`: scraped from ogrants.org (CC BY 4.0); dead URLs recovered via Wayback Machine (`recover_v4_archive.py`)
- `niaid/`, `nci/`, `nidcd/`, `nhgri/`: downloaded from agency sample-application pages (manifests in each subdir)
- `serc/`: scraped from Carleton SERC NSF GEO sample proposals page
- `wellcome/`: parsed from Wellcome's `Open Research Fund 2018/2019` Excel releases (download URL in build script comments); each row converted to a structured `.txt` file
- `declined_extra/`: hand-curated from author-shared PDFs (BBSRC, EMBO, HHMI, ERC anonymized submissions; per-file provenance in `data/declined_extras_provenance.json`)
- `dept_ed/`: individual FOIA releases from the U.S. Department of Education

**Over what timeframe was the data collected?** February–May 2026.

**Is the dataset a sample of a larger set?** Yes — for ogrants, we took entries with extractable PDFs and ≥100 words. For Wellcome, we kept entries with explicit awarded/declined decision rationales; ambiguous decisions were dropped.

**Were any quality-filtering steps applied?**
- Drop documents with <100 extracted words
- Drop entries with corrupt or password-protected PDFs
- Drop Wellcome entries where decision label is genuinely unclear

---

## 4. Preprocessing / cleaning / labeling

**Was preprocessing applied?** Minimal:
- PDF text extraction via PyMuPDF (`fitz`)
- TXT files used as-is
- Word counts computed via `text.split()` (whitespace tokenization)
- No casing changes, no PII redaction beyond source

**Was the raw data saved?** Yes — all PDFs/TXTs are in `data/{source}/`.

**Is the software available?** Yes — `data/build_master_v2.py` plus per-source helper scripts (`data/ogrants/recover_v4_archive.py`, etc.).

---

## 5. Uses

**For what purpose has the dataset been used?**
- Paper 1 (GrantAgent) — system evaluation: 7 systems (random, length-only, 4× single-call GPT-4 variants, GrantAgent multi-agent) on the awarded/declined task
- E3 reviewer agreement — Spearman correlation of GrantAgent per-criterion scores with NIH peer-reviewer summary statements (61 paired instances)

**For what purposes is the dataset NOT suitable?**
- **NOT suitable** as ground-truth for "scientific quality" — funded ≠ scientifically better; rejection is heavily affected by paylines, panel composition, fit to FOA, and stochasticity
- **NOT suitable** for predicting future funder decisions on new proposals — coverage is partial and biased toward voluntarily-shared declined examples
- **NOT suitable** for unrestricted commercial use — per-source license analysis required

**Recommended evaluation protocol** (Paper 2 §5):
- 5-fold stratified cross-validation, balanced by class within fold
- Threshold tuned on training folds, evaluated on held-out folds
- Report **length-quartile-controlled** metrics (avoid attributing length-effect to content-discrimination)
- Cross-funder splits to expose source-confound vs. content-signal
- Bootstrap CIs (≥2000 iterations) on AUC, MCC, F1

---

## 6. Distribution

**How will the dataset be distributed?** v1.0 distributed via:
1. GitHub repo (this repo) — manifest + build script + per-source manifests
2. Future HuggingFace dataset card (after per-source license review for redistribution rights)

**License (per source):**

| Source | License | Redistribution rights |
|---|---|---|
| Open Grants | CC BY 4.0 (per-entry varies) | Yes, with attribution per entry |
| Wellcome Open Research Fund | Wellcome's own public 2018/19 ORF release; redistributed under non-commercial academic research use (US 17 U.S.C. § 107 fair use + UK CDPA 1988 §29 fair dealing) with attribution to Wellcome on every record (no separate CC declaration sought because the records were already part of Wellcome's public release) | Yes, with attribution; opt-out via Paper 2 §A.9 |
| NIAID/NCI/NIDCD/NHGRI samples | NIH educational-use guidance | Permitted for educational/research use; commercial redistribution not addressed |
| SERC Carleton | PI permission | PI-by-PI; not blanket-redistributable |
| Declined extras (BBSRC/EMBO/HHMI author-share + ERC anonymized) | Author share / FOIA | Per-file: FOIA & ERC anonymized → redistribute; author-share → link-only (see `data/declined_extras_provenance.json`) |
| US Dept of Education FOIA | Public record | Yes |
| ERC anon submissions | Anonymized author share | Yes, with anonymization preserved |

**For the v1.0 release, we redistribute ~321 of 455 application files directly via HuggingFace** (Open Grants 166 + Wellcome ORF 155, conditional on Wellcome license formalization — see Paper 2 §8.1). The remaining ~134 files (NIAID/NCI/NIDCD/NHGRI 102 + SERC 22 + author-share extras 10) are **link-only**: the manifest carries source URLs and the `build_master.py` script fetches them locally on first use. If Wellcome license formalization fails before submission, the 155 Wellcome files fall back to link-only, reducing the directly-redistributed count to ~166 (Open Grants only); the paper §8.1 commits to updating tables to match in that case.

**Are there fees or restrictions?** No fees. Per-source license restrictions apply.

---

## 7. Maintenance

**Who is supporting / hosting / maintaining the dataset?** the GrantReview-Bench maintainers(USC), a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues.

**How can the maintainer be contacted?** Email above; GitHub issues at https://github.com/grantagent-bench/grantreview-bench (private until paper acceptance; reviewers may request access).

**Is there an erratum?** Tracked in `data/CHANGELOG.md` (to be created at v1.1).

**Will the dataset be updated?** Yes:
- v1.1 (target Q4 2026): NIH FOIA-extended declined corpus (target +50 declined); opt-in PII-redacted variant; revisits Wellcome license formalization
- v2.0 (target 2027): community-contributed declined via #ShareYourRejections call; targets 100+ FOIA-collected declined NIH/NSF applications
- Versioned releases will be tagged in this repo and on HuggingFace

**Will old versions continue to be supported?** Yes — manifest snapshots will be tagged as `v1.0`, `v1.1`, etc., and remain downloadable.

**Is there a mechanism for community contributions?** GitHub PR against `data/` at https://github.com/grantagent-bench/grantreview-bench (private until paper acceptance) plus an entry in the relevant per-source manifest. Contributors must specify license terms for their additions.

---

## 8. Known limitations and ethical considerations

1. **PI self-selection bias.** Declined proposals are *voluntarily shared* — PIs willing to share rejections may differ systematically from those who do not (e.g., more confident, more career-secure, more invested in open-science norms).
2. **Funder coverage is uneven.** Wellcome dominates declined (138/194); NIH dominates awarded (120/261). Models can over-fit to source-specific surface features.
3. **Length confound.** Declined entries are systematically shorter (median 992 vs 10,229 words). Length alone achieves AUC 0.87 under StratifiedKFold and 0.65 under cross-source GroupKFold (i.e., length carries the within-fold signal but mostly aliases source under proper cross-funder generalization). **All reported metrics must include length-controlled analysis** — see Paper 2 §5.1 for the Q4-stratified protocol.
4. **Temporal drift.** Funding success criteria evolve (paylines, FOA priorities). Most instances are 2010-2024; older instances may not reflect current panel norms.
5. **Decision context is missing.** We do not have panel scores, paylines for the relevant cycle, FOA-fit notes, or PI-vs-program-officer correspondence. Binary `awarded`/`declined` is a noisy ground truth.
6. **Confidentiality of declined PIs.** Some declined entries name the PI; some are anonymized. Users should consider whether their evaluation publishes scores in a way that could embarrass identified PIs.
7. **Misuse risk.** Automated scoring of grant proposals — even with explicit "research only" caveats — risks normalizing the practice. We recommend the corpus be used for *system evaluation*, not for *deployed scoring of live submissions*.

---

## 9. Citation

To cite the dataset in academic work:

```
@misc{grantreviewbench2026,
  author = {GrantReview-Bench maintainers},
  title = {GrantReview-Bench v1.0: A Multi-Funder Open Benchmark for LLM Grant Review Systems},
  year = {2026},
  publisher = {GitHub},
  url = {https://huggingface.co/datasets/grantreview-bench},
  note = {Dataset and build scripts; manifest version 1.0}
}
```
