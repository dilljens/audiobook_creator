"""
audiobook_nem.py
────────────────
Generate the Book of the Nem audiobook — one unique voice per book/section.

Usage:
    python audiobook_nem.py

To skip a section, comment out its entry in BOOKS below.
Output .wav files are written to OUTPUT_DIR (created automatically).
"""

import re
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from kokoro import KPipeline

# ── Config ─────────────────────────────────────────────────────────────────────
SOURCE_FILE = Path("Audio Master Nem Full.txt")
OUTPUT_DIR  = Path("output_audiobook")
SAMPLE_RATE = 24000
SPEED       = 1.0
LANG_CODE   = "a"   # 'a' = American English

# ── Available Kokoro voices (American English, lang_code='a') ──────────────────
#   af_heart   – warm American female      [downloaded]
#   af_nicole  – American female             [downloaded]
#   am_adam    – American male (deep)        [downloaded]
#   am_echo    – American male               [downloaded]
#   am_eric    – American male               [downloaded]
#   am_fenrir  – American male               [downloaded]
#   am_liam    – American male               [downloaded]
#   am_michael – American male (clear)       [downloaded]
#   am_onyx    – American male               [downloaded]
#   am_puck    – American male               [downloaded]
#   am_santa   – American male               [downloaded] (not used)

# ── Book definitions ───────────────────────────────────────────────────────────
# Format: (label, start_marker, voice, output_wav)
#   start_marker – exact text of the FIRST line of the section header in the source
#                  (leading/trailing whitespace is ignored when matching)
#   voice        – Kokoro voice name
#   output_wav   – filename saved inside OUTPUT_DIR
#
# Comment out any line to skip that section entirely.
BOOKS = [
    # label                       start_marker                       voice         output_wav
    ("Introduction",              "Introduction",                    "af_heart",   "00_introduction.wav"),
    ("Book of Hagoth",            "THE BOOK OF HAGOTH",              "am_fenrir",  "01_hagoth.wav"),
    ("Shi-Tugo I",                "THE FIRST BOOK OF SHI-TUGO",      "am_eric",    "02_shi_tugo_1.wav"),
    ("Sanempet",                  "THE BOOK OF SANEMPET",            "am_liam",    "03_sanempet.wav"),
    ("Oug",                       "THE BOOK OF OUG",                 "am_michael", "04_oug.wav"),
    ("Temple Writings of Oug",    "THE BOOK OF",                     "am_michael", "05_temple_writings_oug.wav"),
    ("Sacred Temple Writings",    "THE SACRED",                      "am_michael", "06_sacred_temple_writings.wav"),
    ("Samuel the Lamanite I",     "THE FIRST BOOK",                  "am_echo",    "07_samuel_lamanite_1.wav"),
    ("Samuel the Lamanite II",    "THE SECOND BOOK",                 "am_echo",    "08_samuel_lamanite_2.wav"),
    ("Manti",                     "THE BOOK OF MANTI",               "am_onyx",    "09_manti.wav"),
    ("Pa Nat I",                  "THE FIRST BOOK OF PA NAT",        "af_nicole",  "10_pa_nat_1.wav"),
    ("Moroni I",                  "THE FIRST BOOK OF MORONI",        "am_adam",    "11_moroni_1.wav"),
    ("Moroni II",                 "THE SECOND BOOK OF MORONI",       "am_adam",    "12_moroni_2.wav"),
    ("Moroni III",                "THE THIRD BOOK OF MORONI",        "am_adam",    "13_moroni_3.wav"),
    ("Shioni",                    "THE BOOK OF SHIONI",              "am_puck",    "14_shioni.wav"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_split(source: Path, books: list) -> dict[str, str]:
    """
    Read the source file and split it into sections keyed by label.
    Each section starts at its start_marker line and ends just before the
    next section's start_marker.
    """
    raw_lines = source.read_text(encoding="utf-8").splitlines()

    # Build a mapping: marker_text → index in BOOKS
    markers = [(label, marker.strip()) for label, marker, _, _ in books]

    # Find the line index of each marker's first occurrence
    marker_positions: list[tuple[int, int]] = []   # (line_idx, books_idx)
    for book_idx, (label, marker) in enumerate(markers):
        for line_idx, line in enumerate(raw_lines):
            if line.strip() == marker:
                marker_positions.append((line_idx, book_idx))
                break
        else:
            print(f"  ⚠  Marker not found for '{label}': '{marker}' — skipping")

    marker_positions.sort(key=lambda x: x[0])

    sections: dict[str, str] = {}
    for rank, (line_idx, book_idx) in enumerate(marker_positions):
        label = markers[book_idx][0]
        if rank + 1 < len(marker_positions):
            end_line = marker_positions[rank + 1][0]
        else:
            end_line = len(raw_lines)
        text = "\n".join(raw_lines[line_idx:end_line]).strip()
        sections[label] = text

    return sections


def clean_text(text: str) -> str:
    """
    Strip formatting artifacts, underscores, and normalise whitespace
    so the TTS receives clean prose.
    """
    # Remove lines that are pure underscores (horizontal rules)
    text = re.sub(r"^_{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove leading chapter headers that are all-caps lines
    # (keep them as natural spoken title for context)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_audio(pipeline: KPipeline, text: str, voice: str,
                   output_path: Path) -> None:
    chunks = []
    for _, _, chunk_audio in pipeline(text, voice=voice, speed=SPEED):
        if hasattr(chunk_audio, "numpy"):
            chunk_audio = chunk_audio.cpu().numpy()
        chunk_audio = np.atleast_1d(chunk_audio.squeeze())
        if chunk_audio.size > 0:
            chunks.append(chunk_audio)

    if chunks:
        audio = np.concatenate(chunks, axis=0)
        sf.write(str(output_path), audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        print(f"  ✓  Saved '{output_path.name}'  ({duration:.1f}s)")
    else:
        print(f"  ✗  No audio produced for voice='{voice}'")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\nParsing '{SOURCE_FILE}' …")
    sections = load_and_split(SOURCE_FILE, BOOKS)
    print(f"  Found {len(sections)} sections.\n")

    print("Initialising Kokoro pipeline …")
    pipeline = KPipeline(lang_code=LANG_CODE)

    for label, marker, voice, wav_name in BOOKS:
        if label not in sections:
            continue  # marker was not found; warning already printed

        print(f"\n[{label}]  voice={voice}  →  {wav_name}")
        text = clean_text(sections[label])
        if not text:
            print("  ⚠  Empty text — skipping")
            continue

        out_path = OUTPUT_DIR / wav_name
        generate_audio(pipeline, text, voice, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
