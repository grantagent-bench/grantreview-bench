# Data Provenance, Privacy, and Removal Requests

## Where this data comes from

GrantReview-Bench is built from grant proposals that were **already published** — by the
applicants themselves, by the funding bodies, or through statutory disclosure. We did not
obtain any document through a confidential channel, and we are not the original publisher
of any proposal or funding decision in this corpus.

| Source | How the documents became available |
|---|---|
| Open Grants | Researchers voluntarily published their own proposals, including unfunded ones, under CC BY 4.0 |
| Wellcome Open Research Fund 2018/19 | Published by the funder as part of an open-data release |
| NIH / NSF / SERC | Public award records and agency-published application materials; distributed here as links only |
| Declined extras | Per-document basis recorded in [`data/declined_extras_provenance.json`](data/declined_extras_provenance.json) — author-shared with permission, anonymized at source, or released under FOIA |

Per-source licensing terms are in [`LICENSE.md`](LICENSE.md).

## What we removed anyway

Being permitted to republish something is not the same as needing to. Applicant identity is
not required to evaluate a grant-review system, so we removed it:

- **Document identifiers are pseudonymous.** Every document is keyed by an opaque
  `grb1_…` identifier. The original filenames — which in several sources embedded the
  applicant's name — do not appear in this repository.
- **Applicant names are not published.** The `pi` field present in our internal records is
  withheld from the public release.
- **Proposal titles are not published**, because a title alone is often enough to identify
  the applicant.
- **Verbatim funder decision wording is removed from the manifest.** Only the derived
  binary label (`awarded` / `declined`) ships there, which is what the benchmark needs.
  The one exception is the two Wellcome label-audit files, which retain `decision_raw`
  ("Not shortlisted", "Shortlisted, not funded", "Funded") because auditing the
  decision-text-to-label rule is precisely what those files are for — remove the decision
  text and the audit cannot be checked. Attached to an opaque `grb1_…` identifier, that
  wording does not identify anyone.

Identifiers are stable across releases, so per-item results remain joinable and every
published number remains reproducible. The re-identification mapping is held privately by
the maintainer and is not distributed.

We describe this as **pseudonymization, not guaranteed anonymization**, and one limit is
worth stating precisely rather than leaving implied. Each record still publishes its
source, funder, grant type, outcome label, format, and **exact word count**. Word count
is close to a fingerprint: anyone already holding the original documents can match most
records back to them on those fields alone, without attacking the identifiers. We keep the
exact counts because the length-stratified analysis the benchmark recommends depends on
them, and because this linkage only helps someone who already has the documents — and
therefore already has the names. Users who need stronger guarantees should coarsen `words`
into bands before redistributing any derivative.

## Requesting removal

If your proposal is represented in this corpus and you would prefer it were not, we will
remove it. You do not need to give a reason, and you do not need to be the copyright
holder — being the applicant is sufficient.

**[Open a GitHub issue](../../issues)** titled `Removal request`. Include whichever of these
you can — and no more than you are comfortable posting, since issues are public:

- the funder and approximate year of the application
- the proposal title, or the original filename if you know it
- the `grb1_…` identifier, if you happen to have it

You do not need to identify the record yourself, and you should not post your name in a
public issue if you would rather not. Saying only "removal request, Wellcome 2018" is
enough for a maintainer to open a private channel with you and take it from there.

**What we will do:** acknowledge within 2 weeks; remove the record from the manifest,
splits, and all per-item result files, and reissue the affected release within 4 weeks.
Removal requests are honoured for corrections and withdrawals as well as deletions.

**What we cannot do, stated plainly:** this repository has been publicly accessible, and we
cannot retract copies that others have already cloned, forked, mirrored, cached, or
indexed, nor can we alter the original sources, which we do not control. Removal here
prevents further distribution from this repository; it is not erasure from the internet.

## Reporting a problem

If you believe this release discloses something it should not, or that a licensing basis is
stated incorrectly, open an issue saying so **without repeating the sensitive details** —
a maintainer will follow up privately so that nothing is further republished in the process
of reporting it.
