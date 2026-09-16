# GrantReview-Bench v1.0 — License

This corpus aggregates documents from 8 sources under different licenses. Per-source
distribution decisions and license URLs are documented in Paper 2 §8.1.

## Per-source summary

| Source | License | Distribution |
|---|---|---|
| Open Grants (166 docs) | CC BY 4.0 (per-entry; default) — https://creativecommons.org/licenses/by/4.0/ | Redistributed via HuggingFace with per-entry attribution. |
| Wellcome ORF 2018/19 (155 docs) | Wellcome's own public 2018/19 Open Research Fund release. Wellcome itself made the records public; we redistribute under non-commercial academic research use (US 17 U.S.C. § 107 fair use + UK CDPA 1988 §29 fair dealing) with attribution to Wellcome on every record. No separate CC declaration was sought because the records were already part of Wellcome's public release. | Redistributed via HuggingFace with attribution. |
| NIAID / NCI / NIDCD / NHGRI (102 docs) | NIH educational-use guidance per IC | Link-only. `build_master.py` fetches from the per-IC URLs in §8.1. |
| SERC Carleton NSF Geo (22 docs) | PI permission per program | Link-only. URLs in `data/source_urls.json`. |
| Declined extras (10 docs) | Mixed: BBSRC / EMBO / HHMI author-share + ERC anonymized + US Dept of Education FOIA | Per-file: FOIA & ERC anonymized → redistribute; author-share → link-only. Provenance in `data/declined_extras_provenance.json`. |

## Attribution requirement

For Open Grants (CC BY 4.0) and Wellcome ORF entries, downstream users must preserve
the per-record `extra.attribution_string` field, which carries the required
attribution form (author + URL + source-of-record).

## Opt-out / removal

Any individual identified in a record may email a GitHub issue at https://github.com/grantagent-bench/grantreview-bench/issues to request removal.
SLA: ≤4 weeks for the next versioned HuggingFace release. Removed records are
replaced with a per-record stub citing "removed at PI request" so cross-version
reproducibility scripts do not silently fail. All removals logged in `CHANGELOG.md`.
NIH/NSF agency-released records are not subject to opt-out (they are official agency
releases) but the same email path is available for any concern.

## Personal data and GDPR

The voluntarily-shared declined records and Wellcome ORF entries name PIs and
institutions in the document text, consistent with each source's published policy.
Legal basis under EU GDPR Article 6: Article 6(1)(e) (public-interest task — open
scientific evaluation of LLM grant-review systems) and Article 6(1)(f) (legitimate
interest — research on automated peer review). Article 17 (right to erasure) is
honored via the opt-out path above.

No IRB review was sought because the corpus aggregates already-public documents and
does not constitute human-subjects research as defined under 45 CFR § 46.102(e).

## Use scope

This corpus is released for **non-commercial academic research** on LLM grant-review
evaluation. Commercial redistribution, commercial training data, or any use that
would re-monetize agency or PI-shared materials beyond their original public-release
intent is **out of scope**. Downstream commercial users must contact each original
source for their own license terms.
