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
from wordfreq import top_n_list

# ── Top 10 000 most-frequent English words ──────────────────────────
TOP_10K_ENGLISH: frozenset[str] = frozenset(top_n_list("en", 10_000))

# Words in the top-10k list that are genuine proper nouns in this text —
# keep them despite the frequency filter.
PROPER_NOUN_WHITELIST: frozenset[str] = frozenset({
    # Biblical names
    "aaron", "abel", "abraham", "adam", "cain", "eden", "egypt",
    "elijah", "ephraim", "eve", "gad", "ham", "isaac", "israel",
    "jacob", "james", "jehovah", "john", "joseph", "judah",
    "laban", "lehi", "levi", "micah", "michael", "moses", "noah",
    "peter", "pharaoh", "samuel", "sarah", "sarai", "seth", "simeon",
    "timothy", "zion",
    # Book-specific names that happen to match English words
    "alma", "ether", "gideon", "limhi", "mormon", "moroni", "mulek",
    "mosiah", "nephi", "satan", "sidon",
})

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
# Common English words that should be dropped when splitting multi-word entities.
STOP_WORDS: set[str] = {
    "A", "AN", "AND", "AS", "AT", "BE", "BUT", "BY",
    "DO", "DID", "DOTH",
    "EVEN", "FOR", "FROM",
    "HAD", "HAS", "HAVE", "HATH", "HE", "HER", "HIS", "HOW",
    "I", "IN", "IS", "IT", "ITS",
    "MAY", "ME", "MORE", "MY",
    "NAY", "NO", "NOT", "NOW",
    "OF", "OR", "OUR",
    "SHALL", "SHE", "SO", "SOME",
    "THAT", "THE", "THEE", "THEIR", "THEN", "THERE", "THESE", "THEY",
    "THIS", "THOSE", "THOU", "THUS", "THY", "TO",
    "UP", "UPON", "US",
    "WAS", "WE", "WHEN", "WHERE", "WHICH", "WHO", "WILL", "WITH",
    "YE", "YEA", "YET", "YOU", "YOUR",
    # Book-specific common words not worth flagging
    "BEHOLD", "CHAPTER", "CHRIST", "GOD", "GHOST", "HOLY", "LORD", "VERSE",
    # Generic nouns that slip through NER
    "CITY", "DAYS", "DAY", "GREAT", "LAND", "MAN", "MEN", "NEW",
    "PEOPLE", "SON", "TIME",
}


def is_noise(text: str) -> bool:
    t = text.strip()
    if len(t) <= 1:
        return True
    if t.isupper() and len(t) > 4:      # all-caps section header word
        return True
    if t.upper() in STOP_WORDS:
        return True
    if re.search(r"[^a-zA-Z\-']", t):   # contains digits, spaces, or symbols
        return True
    # Drop common English words (no hyphens) unless whitelisted as proper nouns.
    if "-" not in t and t.lower() in TOP_10K_ENGLISH and t.lower() not in PROPER_NOUN_WHITELIST:
        return True
    return False


def canonical(text: str) -> str:
    """Normalise whitespace and title-case."""
    return " ".join(text.split()).title()


def split_words(phrase: str) -> list[str]:
    """Split a phrase on spaces; hyphenated words are kept as one token."""
    return phrase.split()


# ── Read and process ───────────────────────────────────────────────────────────
print(f"Reading '{SOURCE}' …")
raw_text = SOURCE.read_text(encoding="utf-8")

print("Running spaCy pipeline (this may take a minute) …")
doc = nlp(raw_text)

# Buckets: keyed by display-group name → set of canonical strings
buckets: dict[str, set[str]] = defaultdict(set)

# 1. NER pass — trust spaCy's entity labels
#    Multi-word entities (e.g. "Peter James John") are split into individual
#    words; hyphenated words (e.g. "Anti-Nephi-Lehi") stay as one token.
for ent in doc.ents:
    phrase = canonical(ent.text)
    for word in split_words(phrase):
        if is_noise(word):
            continue
        if ent.label_ in PERSON_LABELS:
            buckets["People & Characters"].add(word)
        elif ent.label_ in PLACE_LABELS:
            buckets["Places & Lands"].add(word)
        elif ent.label_ in ORG_LABELS:
            buckets["Groups & Nations"].add(word)
        elif ent.label_ in OTHER_LABELS:
            buckets["Other Named Things"].add(word)
        else:
            buckets["Other Named Things"].add(word)

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
    word = canonical(text)
    if is_noise(word):
        continue
    # Only add if not already captured by NER
    already_captured = any(word in s for s in buckets.values())
    if not already_captured:
        buckets["Unclassified Proper Nouns"].add(word)

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
