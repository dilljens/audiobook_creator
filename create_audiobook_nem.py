"""
audiobook_nem.py
────────────────
Generate the Book of the Nem audiobook — one unique voice per book/section.

Usage:
    python create_audiobook_nem.py                   # all enabled books
    python create_audiobook_nem.py --list            # list available book labels
    python create_audiobook_nem.py Introduction
    python create_audiobook_nem.py "Book of Hagoth"
    python create_audiobook_nem.py Introduction "Book of Hagoth"

To permanently skip a section, comment out its entry in BOOKS below.
Output .wav files are written to OUTPUT_DIR (created automatically).
"""

import argparse
import re
import time
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from kokoro import KPipeline

# ── Config ─────────────────────────────────────────────────────────────────────
_FIXED_FILE   = Path("Audio Master Nem Full (TTS Fixed).txt")
_ORIG_FILE    = Path("Audio Master Nem Full.txt")
SOURCE_FILE   = _FIXED_FILE if _FIXED_FILE.exists() else _ORIG_FILE
OUTPUT_DIR    = Path("output_audiobook")
SAMPLE_RATE   = 24000
SPEED         = 1.0
LANG_CODE     = "a"   # 'a' = American English

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
# Format: (label, (start_line1, start_line2), voice, output_wav)
#   start_line1 – exact text of the FIRST line of the section header
#   start_line2 – prefix of the SECOND line (used together for unambiguous matching)
#   voice        – Kokoro voice name
#   output_wav   – filename saved inside OUTPUT_DIR
#
# Comment out any line to skip that section entirely.
BOOKS = [
    # label                       (start_line1,                    start_line2)                           voice         output_wav
    ("Introduction",              ("Introduction",                 "The Book of the Nem"),                "af_heart",   "00_introduction.wav"),
    ("Book of Hagoth",            ("THE BOOK OF HAGOTH",           "THE SON OF HAGMENI,"),                 "am_fenrir",  "01_hagoth.wav"),
    ("Shi-Tugo I",                ("THE FIRST BOOK OF SHI-TUGO",  "FORMER WARRIOR, AMMONITE"),            "am_eric",    "02_shi_tugo_1.wav"),
    ("Sanempet",                  ("THE BOOK OF SANEMPET",        "THE SON OF HAGMENI,"),                 "am_liam",    "03_sanempet.wav"),
    ("Oug",                       ("THE BOOK OF OUG",             "THE SON OF SANEMPET"),                 "am_michael", "04_oug.wav"),
    ("Temple Writings of Oug",    ("THE BOOK OF",                 "THE TEMPLE WRITINGS"),                "am_michael", "05_temple_writings_oug.wav"),
    ("Sacred Temple Writings",    ("THE SACRED",                  "TEMPLE WRITINGS"),                     "am_michael", "06_sacred_temple_writings.wav"),
    ("Samuel the Lamanite I",     ("THE FIRST BOOK",              "OF SAMUEL THE LAMANITE"),             "am_echo",    "07_samuel_lamanite_1.wav"),
    ("Samuel the Lamanite II",    ("THE SECOND BOOK",             "OF SAMUEL THE LAMANITE"),             "am_echo",    "08_samuel_lamanite_2.wav"),
    ("Manti",                     ("THE BOOK OF MANTI",           "THE SON OF OUG"),                      "am_onyx",    "09_manti.wav"),
    ("Pa Nat I",                  ("THE FIRST BOOK OF PA NAT",    "THE DAUGHTER OF SHIMLEI"),             "af_nicole",  "10_pa_nat_1.wav"),
    ("Moroni I",                  ("THE FIRST BOOK OF MORONI",    "THE SON OF MORMON,"),                  "am_adam",    "11_moroni_1.wav"),
    ("Moroni II",                 ("THE SECOND BOOK OF MORONI",   "THE SON OF MORMON,"),                  "am_adam",    "12_moroni_2.wav"),
    ("Moroni III",                ("THE THIRD BOOK OF MORONI",    "THE SON OF MORMON,"),                  "am_adam",    "13_moroni_3.wav"),
    ("Shioni",                    ("THE BOOK OF SHIONI",          "THE SON OF MORONI"),                   "am_puck",    "14_shioni.wav"),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_split(source: Path, books: list) -> dict[str, str]:
    """
    Read the source file and split it into sections keyed by label.
    Each section starts at its (start_line1, start_line2) marker pair and
    ends just before the next section's marker.
    """
    raw_lines = source.read_text(encoding="utf-8").splitlines()

    # Build a mapping: (label, line1, line2) for each book
    markers = [(label, m[0].strip(), m[1].strip()) for label, m, _, _ in books]

    # Find the line index of each marker's first occurrence (two-line match)
    marker_positions: list[tuple[int, int]] = []   # (line_idx, books_idx)
    for book_idx, (label, m1, m2) in enumerate(markers):
        for line_idx, line in enumerate(raw_lines[:-1]):
            if (line.strip() == m1 and
                    raw_lines[line_idx + 1].strip().startswith(m2)):
                marker_positions.append((line_idx, book_idx))
                break
        else:
            print(f"  ⚠  Marker not found for '{label}': '{m1}' / '{m2}' — skipping")

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


def _fmt_duration(seconds: float) -> str:
    """Format seconds as 'Xm Ys' or 'Xs'."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.0f}s"


def generate_audio(pipeline: KPipeline, text: str, voice: str,
                   output_path: Path) -> float:
    """Generate audio and return wall-clock seconds elapsed."""
    t0 = time.monotonic()
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
        elapsed = time.monotonic() - t0
        duration = len(audio) / SAMPLE_RATE
        print(f"  ✓  Saved '{output_path.name}'  ({duration:.1f}s audio  |  {elapsed:.1f}s wall-clock)")
    else:
        elapsed = time.monotonic() - t0
        print(f"  ✗  No audio produced for voice='{voice}'")
    return elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── CLI ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Generate Nem audiobook sections.")
    parser.add_argument(
        "books", nargs="*",
        help="Labels of sections to generate (default: all enabled books). "
             "Use --list to see available labels."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print all enabled book labels and exit."
    )
    args = parser.parse_args()

    enabled_labels = [label for label, _, _, _ in BOOKS]

    if args.list:
        print("Enabled books:")
        for label in enabled_labels:
            print(f"  {label}")
        return

    # Filter to requested subset, preserving BOOKS order
    if args.books:
        unknown = [b for b in args.books if b not in enabled_labels]
        if unknown:
            print(f"Unknown book label(s): {', '.join(unknown)}")
            print(f"Run with --list to see available labels.")
            return
        run_books = [b for b in BOOKS if b[0] in args.books]
    else:
        run_books = list(BOOKS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\nSource: '{SOURCE_FILE}'"
          + (" ✓ (TTS fixed)" if SOURCE_FILE == _FIXED_FILE else
             " ⚠ (original — run 'Apply Fixes to Text' in the GUI to use phonetic fixes)"))
    # Always split using ALL books for correct section boundaries,
    # but only generate for run_books.
    sections = load_and_split(SOURCE_FILE, BOOKS)
    print(f"  Found {len(sections)} sections ({len(run_books)} selected).\n")

    print("Initialising Kokoro pipeline …")
    pipeline = KPipeline(lang_code=LANG_CODE)

    # Pre-compute char counts for all sections so we can estimate ETAs
    section_chars: dict[str, int] = {
        label: len(clean_text(sections[label]))
        for label, _, _, _ in run_books
        if label in sections
    }

    # Print char count summary before starting
    print(f"\n{'─' * 52}")
    print(f"  {'Section':<30}  {'Chars':>8}")
    print(f"{'─' * 52}")
    for label, _, _, wav_name in run_books:
        if label in section_chars:
            print(f"  {label:<30}  {section_chars[label]:>8,}")
    print(f"{'─' * 52}")
    total_chars = sum(section_chars.values())
    print(f"  {'TOTAL':<30}  {total_chars:>8,}")
    print()

    chars_per_sec: float | None = None   # derived from the first book that finishes
    timing_rows: list[tuple[str, int, float]] = []  # (label, chars, elapsed)

    for label, _marker, voice, wav_name in run_books:
        if label not in sections:
            continue

        text = clean_text(sections[label])
        if not text:
            print(f"\n[{label}]  ⚠  Empty text — skipping")
            continue

        chars = section_chars[label]

        # Print ETA once we have a calibration rate
        if chars_per_sec is not None:
            eta_sec = chars / chars_per_sec
            eta_str = _fmt_duration(eta_sec)
            print(f"\n[{label}]  voice={voice}  →  {wav_name}  (est. {eta_str})")
        else:
            print(f"\n[{label}]  voice={voice}  →  {wav_name}  (timing calibration run)")

        stem, ext = wav_name.rsplit(".", 1)
        out_path = OUTPUT_DIR / f"{stem}_{voice}.{ext}"
        elapsed = generate_audio(pipeline, text, voice, out_path)
        timing_rows.append((label, chars, elapsed))

        # Update calibration as a cumulative average after every book
        total_chars_done = sum(c for _, c, _ in timing_rows)
        total_elapsed_done = sum(e for _, _, e in timing_rows)
        if total_elapsed_done > 0:
            chars_per_sec = total_chars_done / total_elapsed_done
            print(f"  ⏱  Calibration: {chars_per_sec:.0f} chars/sec")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  {'Section':<30}  {'Chars':>7}  {'Actual':>8}  {'Est':>8}")
    print("─" * 60)
    for i, (label, chars, elapsed) in enumerate(timing_rows):
        actual_str = _fmt_duration(elapsed)
        # Estimate using the cumulative rate *before* this book was added
        prior_chars = sum(c for _, c, _ in timing_rows[:i])
        prior_elapsed = sum(e for _, _, e in timing_rows[:i])
        if prior_elapsed > 0:
            est_str = _fmt_duration(chars / (prior_chars / prior_elapsed))
        else:
            est_str = "(first run)"
        print(f"  {label:<30}  {chars:>7,}  {actual_str:>8}  {est_str:>8}")
    total_elapsed = sum(e for _, _, e in timing_rows)
    print("─" * 60)
    print(f"  {'TOTAL':<30}  {sum(c for _,c,_ in timing_rows):>7,}  {_fmt_duration(total_elapsed):>8}")
    print("\nDone.")


if __name__ == "__main__":
    main()
