"""
generate_proper_noun_audio.py
──────────────────────────────
Read proper_nouns.txt, generate a short TTS audio clip for every entry
using am_michael, and save a JSON manifest for the GUI.

Outputs:
    output_proper_nouns/<slug>.wav   – one wav per entry
    output_proper_nouns/manifest.json – { "Word" : "slug.wav", … }

Already-generated files are skipped, so re-runs are fast.

Run:
    .venv/bin/python generate_proper_noun_audio.py
"""

import json
import re
import sys
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from kokoro import KPipeline

PROPER_NOUNS_FILE = Path("proper_nouns.txt")
OUTPUT_DIR        = Path("output_proper_nouns")
MANIFEST_FILE     = OUTPUT_DIR / "manifest.json"
VOICE             = "am_michael"
SAMPLE_RATE       = 24000
SPEED             = 1.0

# ── Parse proper_nouns.txt ─────────────────────────────────────────────────────

def parse_entries(path: Path) -> list[tuple[str, str]]:
    """Return list of (category, entry) pairs."""
    entries: list[tuple[str, str]] = []
    current_cat = "Uncategorised"
    header_re = re.compile(r"^[A-Z &]+\s+\(\d+\)$")

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("=") or stripped.startswith("─"):
            continue
        if header_re.match(stripped):
            # e.g.  "PEOPLE & CHARACTERS  (301)"
            current_cat = stripped.rsplit("(", 1)[0].strip().title()
            continue
        if stripped.startswith("TOTAL:"):
            continue
        if stripped.startswith("Review this") or stripped.startswith("Each entry"):
            continue
        if stripped.startswith("PROPER NOUNS"):
            continue
        # Regular entry — indented two spaces in the file
        if line.startswith("  "):
            entries.append((current_cat, stripped))

    return entries


def slugify(text: str) -> str:
    """Convert 'Hagoth-II foo' → 'hagoth_ii_foo'."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ── TTS generation ─────────────────────────────────────────────────────────────

def generate(pipeline: KPipeline, text: str, out_path: Path) -> bool:
    chunks = []
    # Speak the word in a short carrier phrase so the TTS pronounces it
    # naturally (isolated tokens sometimes get clipped prosody).
    spoken = text
    for _, _, chunk in pipeline(spoken, voice=VOICE, speed=SPEED):
        if hasattr(chunk, "numpy"):
            chunk = chunk.cpu().numpy()
        chunk = np.atleast_1d(chunk.squeeze())
        if chunk.size > 0:
            chunks.append(chunk)
    if chunks:
        audio = np.concatenate(chunks)
        sf.write(str(out_path), audio, SAMPLE_RATE)
        return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Parsing '{PROPER_NOUNS_FILE}' …")
    entries = parse_entries(PROPER_NOUNS_FILE)
    print(f"  {len(entries)} entries found.\n")

    # Load existing manifest so we can skip already-done words
    if MANIFEST_FILE.exists():
        manifest: dict = json.loads(MANIFEST_FILE.read_text())
    else:
        manifest = {}

    print("Initialising Kokoro pipeline …")
    pipeline = KPipeline(lang_code="a")

    skipped = 0
    generated = 0
    failed = 0

    for i, (cat, entry) in enumerate(entries):
        slug = slugify(entry)
        wav_name = f"{slug}.wav"
        wav_path = OUTPUT_DIR / wav_name

        if entry in manifest and wav_path.exists():
            skipped += 1
            continue

        sys.stdout.write(f"\r[{i+1}/{len(entries)}] {entry[:55]:<55}")
        sys.stdout.flush()

        ok = generate(pipeline, entry, wav_path)
        if ok:
            manifest[entry] = wav_name
            generated += 1
        else:
            print(f"\n  ✗  Failed: {entry}")
            failed += 1

    print(f"\n\nDone.  generated={generated}  skipped={skipped}  failed={failed}")

    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Manifest saved → '{MANIFEST_FILE}'")


if __name__ == "__main__":
    main()
