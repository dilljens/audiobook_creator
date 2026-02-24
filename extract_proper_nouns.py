"""
extract_proper_nouns.py
───────────────────────
Scan 'Audio Master Nem Full.txt' and extract all proper nouns into
'proper_nouns.txt', grouped by type and sorted alphabetically.

Uses spaCy for:
  • NER  (PERSON, GPE, LOC, ORG, …)   – named entity recognition
  • POS  (PROPN)                        – catches names spaCy's NER misses
    because they are not in its training vocabulary (e.g. Hagoth, Meninta)

Run:
    .venv/bin/python extract_proper_nouns.py
"""

import re
from collections import defaultdict
from pathlib import Path

import spacy

SOURCE = Path("Audio Master Nem Full.txt")
OUTPUT = Path("proper_nouns.txt")

# ── spaCy setup ────────────────────────────────────────────────────────────────
print("Loading spaCy model …")
nlp = spacy.load("en_core_web_sm")
# Increase max length for the large source file
nlp.max_length = 2_000_000

# ── NER label groups ───────────────────────────────────────────────────────────
PERSON_LABELS = {"PERSON"}
PLACE_LABELS  = {"GPE", "LOC", "FAC"}
ORG_LABELS    = {"ORG", "NORP"}
OTHER_LABELS  = {"EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "LANGUAGE"}

# ── Noise filters ──────────────────────────────────────────────────────────────
# All-caps lines are section headers, not spoken names — skip them.
# Also skip very short tokens that are likely artefacts.
SKIP_PATTERNS = re.compile(
    r"^(THE|A|AN|AND|OF|IN|TO|FOR|BY|AT|IS|WAS|BE|HE|SHE|IT|"
    r"CHAPTER|VERSE|YEA|BEHOLD|LORD|GOD|CHRIST|HOLY|GHOST)$"
)

def is_noise(text: str) -> bool:
    t = text.strip()
    if len(t) <= 1:
        return True
    if t.isupper() and len(t) > 4:      # all-caps section header word
        return True
    if SKIP_PATTERNS.match(t.upper()):
        return True
    if re.search(r"[^a-zA-Z\-' ]", t):  # contains digits or symbols
        return True
    return False


def canonical(text: str) -> str:
    """Normalise whitespace and title-case."""
    return " ".join(text.split()).title()


# ── Read and process ───────────────────────────────────────────────────────────
print(f"Reading '{SOURCE}' …")
raw_text = SOURCE.read_text(encoding="utf-8")

print("Running spaCy pipeline (this may take a minute) …")
doc = nlp(raw_text)

# Buckets: keyed by display-group name → set of canonical strings
buckets: dict[str, set[str]] = defaultdict(set)

# 1. NER pass — trust spaCy's entity labels
for ent in doc.ents:
    name = canonical(ent.text)
    if is_noise(name):
        continue
    if ent.label_ in PERSON_LABELS:
        buckets["People & Characters"].add(name)
    elif ent.label_ in PLACE_LABELS:
        buckets["Places & Lands"].add(name)
    elif ent.label_ in ORG_LABELS:
        buckets["Groups & Nations"].add(name)
    elif ent.label_ in OTHER_LABELS:
        buckets["Other Named Things"].add(name)
    else:
        buckets["Other Named Things"].add(name)

# 2. PROPN pass — catch names spaCy didn't recognise as entities
#    Only include tokens that are inside a sentence (not at position 0)
#    and are title-cased (filters out all-caps headers).
for token in doc:
    if token.pos_ != "PROPN":
        continue
    text = token.text.strip()
    if not text[0].isupper() or text.isupper():
        continue                          # skip all-caps
    if token.i == token.sent.start:
        continue                          # skip sentence-initial (could be any word)
    name = canonical(text)
    if is_noise(name):
        continue
    # Only add if not already captured by NER
    already_captured = any(name in s for s in buckets.values())
    if not already_captured:
        buckets["Unclassified Proper Nouns"].add(name)

# ── Write output ───────────────────────────────────────────────────────────────
GROUP_ORDER = [
    "People & Characters",
    "Places & Lands",
    "Groups & Nations",
    "Other Named Things",
    "Unclassified Proper Nouns",
]

lines: list[str] = []
lines.append("PROPER NOUNS — Book of the Nem")
lines.append("=" * 50)
lines.append(
    "Review this list for TTS mispronunciations.\n"
    "Each entry is the form that appears in the text.\n"
)

total = 0
for group in GROUP_ORDER:
    names = sorted(buckets.get(group, set()), key=str.casefold)
    if not names:
        continue
    lines.append(f"\n{'─' * 50}")
    lines.append(f"{group.upper()}  ({len(names)})")
    lines.append(f"{'─' * 50}")
    for name in names:
        lines.append(f"  {name}")
    total += len(names)

lines.append(f"\n{'=' * 50}")
lines.append(f"TOTAL: {total} unique proper nouns")

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\n✓  Written '{OUTPUT}'  ({total} unique proper nouns)")
