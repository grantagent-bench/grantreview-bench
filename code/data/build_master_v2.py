#!/usr/bin/env python3
"""Build master_manifest_v2.json including ALL sources (mining session results).

Sources:
- ogrants (PDFs, verified_manifest.json) — 166 entries
- niaid (PDFs, manifest.json) — 82 entries
- nidcd (PDFs, manifest.json) — 32 entries
- nci (PDFs, manifest.json) — 33 entries
- serc (PDFs, manifest.json) — 22 entries
- nhgri (PDFs, manifest.json) — 24 entries (with OCR'd text)
- wellcome (text files, manifest.json) — 155 entries
- dept_ed FOIA (PDFs, no manifest yet) — 21 entries
- declined_extra (PDFs of Bunce/Deville/ERC/BIOCHANGE/Leek) — ~10 entries
"""
import json
import os
import sys
import subprocess
from pathlib import Path

# Resolve DATA relative to this script so the build is portable across machines.
# Override with GRANTREVIEW_DATA env var if running from a non-default layout.
DATA = Path(os.environ.get("GRANTREVIEW_DATA", Path(__file__).resolve().parent))

def extract_words_pdf(pdf_path: Path) -> tuple[int, str]:
    """Extract via subprocess (using current sys.executable) to isolate segfaults."""
    sub = subprocess.run(
        [sys.executable, "-c",
         f"import fitz; d=fitz.open(r'{pdf_path}'); t=''.join(p.get_text() for p in d); d.close(); "
         f"import sys; sys.stdout.write(t)"],
        capture_output=True, text=True, timeout=60,
    )
    if sub.returncode == 0:
        return len(sub.stdout.split()), sub.stdout
    return 0, ""

def extract_words_txt(txt_path: Path) -> tuple[int, str]:
    text = txt_path.read_text(errors='ignore')
    return len(text.split()), text

master = []

# 1. ogrants (PDF)
print("Loading ogrants...")
ogrants = json.load(open(DATA/"ogrants/verified_manifest.json"))
for e in ogrants:
    pdf = DATA/"ogrants/pdfs"/e["filename"]
    if not pdf.exists(): continue
    master.append({
        "source_dataset": "ogrants",
        "filename": e["filename"], "abs_path": str(pdf),
        "label": e.get("label","unknown"), "doc_type": "application",
        "funder": e.get("funder"), "grant_type": None,
        "words": e.get("words", 0), "format": "pdf",
    })
print(f"  +{len([m for m in master if m['source_dataset']=='ogrants'])}")

# 2. niaid
print("Loading niaid...")
for e in json.load(open(DATA/"niaid/manifest.json")):
    pdf = DATA/"niaid/pdfs"/e["filename"]
    if not pdf.exists(): continue
    master.append({
        "source_dataset": "niaid", "filename": e["filename"], "abs_path": str(pdf),
        "label": e.get("label","unknown"), "doc_type": e.get("doc_type","application"),
        "funder": e.get("funder","NIH/NIAID"), "grant_type": e.get("grant_type"),
        "words": 0, "format": "pdf",
    })

# 3. nidcd
print("Loading nidcd...")
for e in json.load(open(DATA/"nidcd/manifest.json")):
    pdf = DATA/"nidcd/pdfs"/e["filename"]
    if not pdf.exists(): continue
    master.append({
        "source_dataset": "nidcd", "filename": e["filename"], "abs_path": str(pdf),
        "label": e.get("label","unknown"), "doc_type": e.get("doc_type","application"),
        "funder": "NIH/NIDCD", "grant_type": e.get("grant_type"),
        "words": 0, "format": "pdf",
    })

# 4. nci
print("Loading nci...")
for e in json.load(open(DATA/"nci/manifest.json")):
    pdf = DATA/"nci/pdfs"/e["filename"]
    if not pdf.exists(): continue
    master.append({
        "source_dataset": "nci", "filename": e["filename"], "abs_path": str(pdf),
        "label": e.get("label","unknown"), "doc_type": "application",
        "funder": "NIH/NCI", "grant_type": e.get("grant_type"),
        "words": 0, "format": "pdf",
    })

# 5. serc
print("Loading serc...")
for e in json.load(open(DATA/"serc/manifest.json")):
    pdf = DATA/"serc/pdfs"/e["filename"]
    if not pdf.exists(): continue
    master.append({
        "source_dataset": "serc", "filename": e["filename"], "abs_path": str(pdf),
        "label": e.get("label","unknown"), "doc_type": "application",
        "funder": "NSF", "grant_type": e.get("program"),
        "words": 0, "format": "pdf",
    })

# 6. nhgri (use OCR'd text files instead of original scan PDFs)
print("Loading nhgri (OCR'd)...")
for e in json.load(open(DATA/"nhgri/manifest.json")):
    stem = e["filename"].replace(".pdf","")
    txt = DATA/"nhgri/text"/f"{stem}.txt"
    if not txt.exists(): continue
    master.append({
        "source_dataset": "nhgri", "filename": txt.name, "abs_path": str(txt),
        "label": e.get("label","awarded"), "doc_type": e.get("doc_type","application"),
        "funder": "NIH/NHGRI", "grant_type": e.get("grant_type"),
        "words": 0, "format": "txt",
    })

# 7. wellcome (text files)
print("Loading wellcome...")
for e in json.load(open(DATA/"wellcome/manifest.json")):
    txt = DATA/"wellcome/text"/e["filename"]
    if not txt.exists(): continue
    master.append({
        "source_dataset": "wellcome", "filename": e["filename"], "abs_path": str(txt),
        "label": e.get("label","unknown"), "doc_type": "application",
        "funder": e.get("funder","Wellcome Trust"),
        "grant_type": e.get("round"),
        "words": e.get("words",0), "format": "txt",
        "extra": {"decision_raw": e.get("decision_raw"), "pi": e.get("pi"), "title": e.get("title")},
    })

# 8. dept_ed FOIA (build manifest on the fly)
print("Loading dept_ed FOIA...")
DEPT_ED = DATA/"declined_extra/pdfs/dept_ed"
for pdf in sorted(DEPT_ED.glob("*.pdf")):
    master.append({
        "source_dataset": "dept_ed", "filename": pdf.name, "abs_path": str(pdf),
        "label": "awarded", "doc_type": "application",
        "funder": "U.S. Department of Education (FOIA)",
        "grant_type": "FIPSE/Strengthening" if "fipse" in pdf.name.lower() or "strength" in pdf.name.lower() else "DoE",
        "words": 0, "format": "pdf",
    })

# 9. declined_extra individual proposals
print("Loading declined_extra...")
DEXTRA = DATA/"declined_extra/pdfs"
DECLINED_FILES = {
    "bunce_nsf_2011_rejected.pdf": ("declined", "NSF Cultural Anthropology", "Bunce 2011"),
    "deville_erc_2009_rejected.pdf": ("declined", "ERC Starting Grant", "Deville 2009"),
    "anon_erc_2020_B1_rejected.pdf": ("declined", "ERC Starting Grant", "Anon ERC 2020 B1"),
    "anon_erc_2020_B2_rejected.pdf": ("declined", "ERC Starting Grant", "Anon ERC 2020 B2"),
    "bunce_nsf_2012_funded.pdf": ("awarded", "NSF Cultural Anthropology", "Bunce 2012"),
    # Bergman Lab — mix; 2 are funded per Jabberwocky list, 2 unspecified.
    # Conservative: label all as declined for now (need manual labeling later)
    "bergman_lab_3082400.pdf": ("declined", "EMBO YIP 2008", "Bergman EMBO 2008"),
    "bergman_lab_3082409.pdf": ("declined", "BBSRC", "Bergman pubmed2ensembl"),
}
for fname, (lbl, gtype, pi) in DECLINED_FILES.items():
    pdf = DEXTRA/fname
    if not pdf.exists(): continue
    master.append({
        "source_dataset": "declined_extra", "filename": fname, "abs_path": str(pdf),
        "label": lbl, "doc_type": "application",
        "funder": gtype, "grant_type": gtype,
        "words": 0, "format": "pdf", "extra": {"pi": pi},
    })

# BIOCHANGE
for fname in ["BIOCHANGE_Stage1.pdf", "BIOCHANGE_Stage2.pdf"]:
    pdf = DEXTRA/"biochange"/fname
    if not pdf.exists(): continue
    master.append({
        "source_dataset": "declined_extra", "filename": fname, "abs_path": str(pdf),
        "label": "declined", "doc_type": "application",
        "funder": "ERC Starting Grant", "grant_type": "ERC StG",
        "words": 0, "format": "pdf", "extra": {"pi": "Anon BIOCHANGE"},
    })

# Leek HHMI
leek = DEXTRA/"round2/leek_hhmi_unfunded_2017.pdf"
if leek.exists():
    master.append({
        "source_dataset": "declined_extra", "filename": "leek_hhmi_unfunded_2017.pdf",
        "abs_path": str(leek), "label": "declined", "doc_type": "application",
        "funder": "HHMI", "grant_type": "Teaching Professors",
        "words": 0, "format": "pdf", "extra": {"pi": "Jeff Leek 2017"},
    })

# Now extract word counts for entries that don't have them
print(f"\nTotal entries before word extraction: {len(master)}")
print("Extracting word counts (this may take a while for new PDFs)...")
extracted = 0
for entry in master:
    if entry["words"] > 0: continue
    p = Path(entry["abs_path"])
    if entry["format"] == "txt":
        words, _ = extract_words_txt(p)
    else:
        words, _ = extract_words_pdf(p)
    entry["words"] = words
    extracted += 1
    if extracted % 25 == 0:
        print(f"  ...extracted {extracted} so far")

# Filter: drop entries with <100 words
before = len(master)
master = [m for m in master if m["words"] >= 100]
print(f"Filtered: {before} → {len(master)} (dropped {before-len(master)} short docs)")

# Save
with open(DATA/"master_manifest_v2.json","w") as f:
    json.dump(master, f, indent=2, default=str)

# Stats
from collections import Counter
print(f"\n=== MASTER MANIFEST V2 STATS ===")
print(f"Total: {len(master)}")
print(f"\nBy label:")
for k,v in Counter(m["label"] for m in master).items():
    print(f"  {k}: {v}")
print(f"\nBy source:")
for k,v in Counter(m["source_dataset"] for m in master).items():
    print(f"  {k}: {v}")
print(f"\nBy doc_type:")
for k,v in Counter(m["doc_type"] for m in master).items():
    print(f"  {k}: {v}")

# Apps only
apps = [m for m in master if m["doc_type"] == "application"]
print(f"\n=== APPLICATIONS ONLY ===")
print(f"Total: {len(apps)}")
print(f"By label: {dict(Counter(m['label'] for m in apps))}")
