#!/usr/bin/env python3
"""Strip identifying information from the public release artifacts.

The manifest identifies researchers in three separate places, and stripping only the
obvious one leaves the disclosure intact:

1. ``filename`` / ``rel_path`` -- embed applicant names, in the shape
   ``wellcome_declined_wellcome_2018_Prof_<First>_<Last>_4.txt``.
2. ``extra.pi`` -- an explicit full name on 165 records, 147 of them declined, alongside
   ``extra.decision_raw`` holding the funder's verbatim wording ("Not shortlisted").
   Together these publish "this named person was rejected".
3. ``extra.title`` -- the full proposal title, which re-identifies the applicant on its
   own even with every name removed.

The audit split files key on the same names and are rewritten too.

Construction notes:

* IDs are ``HMAC-SHA256(key, "<source>|<filename>")`` truncated to 128 bits. A plain
  digest is not enough -- the candidate space is small enough to enumerate, so an
  attacker can hash guessed filenames and match. The key must stay secret; it is NOT
  a published salt.
* IDs carry no source or outcome prefix. ``wellcome_declined_...`` would leak the
  target label into the identifier, which both re-identifies and lets a careless
  loader cheat. Source and label stay as separate structured fields.
* ``grb1_`` is a fresh namespace. Old record numbers are deliberately not preserved,
  because the previous identifier space is already public.
* Verification does not trust a name regex. It asserts that no original filename stem
  and no known PI name survives anywhere in the rewritten output.

Dry run (default) reports what would change; --apply writes the files.

    python code/data/anonymize_release.py
    GRB_ANON_KEY=<hex> python code/data/anonymize_release.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
MANIFEST = DATA / "master_manifest_v2.json"
SPLITS = DATA / "splits" / "canonical_v1.json"
AUDIT_FILES = [DATA / "splits" / "wellcome_audit_sample.jsonl",
               DATA / "splits" / "wellcome_audit_gold.jsonl",
               DATA / "wellcome_audit_sample.jsonl",
               DATA / "wellcome_audit_gold.jsonl"]
MAPPING = DATA / "anon_mapping.PRIVATE.json"

NAMESPACE = "grb1"
ID_HEX_CHARS = 32  # 128 bits

# Dropped outright: each re-identifies the applicant, and none is needed to reproduce
# a benchmark score. The binary `label` already carries the outcome.
DROP_EXTRA_KEYS = ("pi", "title", "decision_raw")


def nfc(text: str) -> str:
    """Compose accented characters so the same name compares equal across files."""
    return unicodedata.normalize("NFC", text)


def public_id(key: bytes, source: str, filename: str) -> str:
    """Opaque, stable pseudonym for one document."""
    msg = f"{source}|{filename}".encode()
    return f"{NAMESPACE}_{hmac.new(key, msg, hashlib.sha256).hexdigest()[:ID_HEX_CHARS]}"


def load_key(explicit: str | None) -> tuple[bytes, bool]:
    """Reuse an existing key so IDs stay stable across releases; else mint one."""
    for candidate in (explicit, os.environ.get("GRB_ANON_KEY")):
        if candidate:
            return bytes.fromhex(candidate), False
    if MAPPING.exists():
        if prior := json.loads(MAPPING.read_text()).get("key"):
            return bytes.fromhex(prior), False
    return secrets.token_bytes(32), True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    ap.add_argument("--key", help="hex HMAC key to reuse instead of generating one")
    args = ap.parse_args()

    key, minted = load_key(args.key)

    records = json.loads(MANIFEST.read_text())
    if not isinstance(records, list):
        records = next(v for v in records.values() if isinstance(v, list))

    mapping: dict[str, dict[str, object]] = {}
    assigned = 0                     # counted separately: two records colliding on one
                                     # ID would silently overwrite in `mapping`
    secrets_seen: set[str] = set()   # every string that must not survive
    dropped = {k: 0 for k in DROP_EXTRA_KEYS}

    for rec in records:
        original = rec.get("filename") or ""
        if not original:
            continue
        source = rec.get("source_dataset", "unknown")
        new_id = public_id(key, source, original)
        suffix = Path(original).suffix

        secrets_seen.add(nfc(Path(original).stem))
        withheld: dict[str, str] = {}
        if isinstance(extra := rec.get("extra"), dict):
            for field in DROP_EXTRA_KEYS:
                if (value := extra.pop(field, None)) is not None:
                    dropped[field] += 1
                    withheld[field] = value
                    if field in ("pi", "title") and isinstance(value, str):
                        secrets_seen.add(nfc(value.strip()))

        assigned += 1
        mapping[new_id] = {"filename": original,
                           "rel_path": rec.get("rel_path", ""),
                           "source_dataset": source,
                           "label": rec.get("label", ""),
                           "withheld": withheld}

        rec["filename"] = f"{new_id}{suffix}"
        if old_path := rec.get("rel_path"):
            rec["rel_path"] = str(Path(old_path).parent / f"{new_id}{suffix}")

    # Splits and audit files reference the same documents by filename; keep them joinable.
    # Remap structurally rather than by text substitution. These are JSON, and filenames
    # carrying accented characters are stored escaped -- "\\u00e4" as six literal
    # characters -- so a raw-text replace silently fails to match them.
    lookup = {nfc(str(meta["filename"])): f"{new_id}{Path(str(meta['filename'])).suffix}"
              for new_id, meta in mapping.items()}
    rewrites = 0

    def remap(node):
        """Rewrite any filename string anywhere in a decoded JSON structure."""
        nonlocal rewrites
        if isinstance(node, str):
            if (hit := lookup.get(nfc(node))) is not None:
                rewrites += 1
                return hit
            return node
        if isinstance(node, list):
            return [remap(x) for x in node]
        if isinstance(node, dict):
            return {k: remap(v) for k, v in node.items()}
        return node

    side_files: dict[Path, str] = {}
    for path in [SPLITS, *AUDIT_FILES]:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            lines = [json.dumps(remap(json.loads(line)))
                     for line in path.read_text().splitlines() if line.strip()]
            side_files[path] = "\n".join(lines) + "\n"
        else:
            side_files[path] = json.dumps(remap(json.loads(path.read_text())), indent=1) + "\n"

    # Verification: assert nothing identifying survives, rather than trusting a regex.
    # ensure_ascii=False so accented names appear as themselves, not as \uXXXX escapes.
    blob = nfc(json.dumps(records, ensure_ascii=False) + "".join(side_files.values()))
    survivors = sorted(s for s in secrets_seen if s and s in blob)

    # Referential integrity: every document a side file names must exist in the manifest.
    # An unresolved reference means a rewrite was missed and a real filename survived.
    ids = {r.get("filename") for r in records}
    referenced = {tok.strip('"\', ') for text in side_files.values()
                  for tok in text.replace(",", " ").replace('"', " ").split()
                  if tok.strip('"\', ').endswith((".pdf", ".txt"))}
    unresolved = sorted(referenced - ids)

    print(f"manifest records         : {len(records)}")
    print(f"  opaque IDs assigned    : {len(mapping)}  (collisions: "
          f"{assigned - len(mapping)})")
    for field, count in dropped.items():
        print(f"  extra.{field:<14s} dropped from {count} records")
    print(f"side files rewritten     : {len(side_files)} files, {rewrites} references")
    print(f"hmac key                 : {'MINTED - save it' if minted else 'reused'}")
    print(f"identifying strings left : {len(survivors)}")
    for leftover in survivors[:5]:
        print(f"    STILL PRESENT: {leftover[:60]}")
    print(f"unresolved side-file refs: {len(unresolved)}")
    for ref in unresolved[:5]:
        print(f"    UNRESOLVED: {ref[:60]}")

    if survivors or unresolved:
        print("\nABORT: identifying data survived, or a reference was missed. "
              "Nothing written.")
        return 1

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to commit the changes.")
        return 0

    MANIFEST.write_text(json.dumps(records, indent=1) + "\n")
    for path, text in side_files.items():
        path.write_text(text)
    MAPPING.write_text(json.dumps(
        {"key": key.hex(), "namespace": NAMESPACE, "mapping": mapping}, indent=1) + "\n")

    print(f"\nwrote {MANIFEST.relative_to(REPO)}")
    for path in side_files:
        print(f"wrote {path.relative_to(REPO)}")
    print(f"wrote {MAPPING.relative_to(REPO)}  <-- PRIVATE: holds the key, keep out of git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
