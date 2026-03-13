"""
create_audiobook_lightbringer.py
─────────────────────────────────
Generate the "A Darkness Rising" audiobook — one file per chapter/prologue.

Reads all .txt files from NOVEL_DIR, detects Prologue + Chapter headings,
and writes one .wav per chapter into OUTPUT_DIR.

Usage:
    python create_audiobook_lightbringer.py            # all chapters
    python create_audiobook_lightbringer.py --list     # list detected chapters
    python create_audiobook_lightbringer.py 0 1 2      # prologue + ch1 + ch2
    python create_audiobook_lightbringer.py --preview  # short preview clips

Output filenames:
    chapter_00_prologue.wav
    chapter_01_homecoming.wav
    chapter_02_the_anhuil_ehlar.wav
    ...
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
NOVEL_DIR   = Path("Audio Text for Novel Lightbringer")
OUTPUT_DIR  = Path("output_audiobook_lightbringer")
SAMPLE_RATE = 24000
SPEED       = 1.0
LANG_CODE   = "a"     # American English
VOICE       = "am_onyx"      # default narrator voice

# Regex that matches a chapter/prologue heading line (case-insensitive).
# Group 1 captures the chapter number (or None for Prologue).
# Group 2 captures the optional subtitle after " - ".
_HEADING_RE = re.compile(
    r"^(?:Chapter\s+(\d+)\s*(?:-\s*(.+))?|(Prologue))\s*$",
    re.IGNORECASE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Convert title text to a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_all_chapters(novel_dir: Path) -> list[dict]:
    """
    Read all .txt files in *novel_dir* in sorted order, detect Prologue /
    Chapter headings, and return a list of chapter dicts:
        {
            "num":   int,          # 0 = Prologue
            "title": str,          # subtitle portion, e.g. "Homecoming"
            "label": str,          # human label, e.g. "Chapter 1 - Homecoming"
            "slug":  str,          # e.g. "chapter_01_homecoming"
            "text":  str,          # full body text of the chapter
        }
    Chapters from multiple files are concatenated in sorted-filename order.
    """
    txt_files = sorted(novel_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{novel_dir}'")

    # Collect (chapter_num, title_line, body_lines) across all files
    raw: list[tuple[int, str, list[str]]] = []  # (num, heading_text, body)
    current_num: int | None = None
    current_heading: str = ""
    current_body: list[str] = []

    def _flush():
        if current_num is not None:
            raw.append((current_num, current_heading, list(current_body)))

    for fpath in txt_files:
        lines = fpath.read_text(encoding="utf-8").splitlines()
        for line in lines:
            m = _HEADING_RE.match(line.strip())
            if m:
                _flush()
                if m.group(3):               # Prologue
                    current_num = 0
                    current_heading = "Prologue"
                else:                        # Chapter N
                    current_num = int(m.group(1))
                    subtitle = (m.group(2) or "").strip()
                    current_heading = f"Chapter {current_num}" + (f" - {subtitle}" if subtitle else "")
                current_body = [line]        # keep heading inside text
            else:
                if current_num is not None:
                    current_body.append(line)
    _flush()

    # Build chapter dicts, deduplicated and sorted by number
    seen: set[int] = set()
    chapters: list[dict] = []
    for num, heading, body in sorted(raw, key=lambda x: x[0]):
        if num in seen:
            continue
        seen.add(num)
        # Derive subtitle / slug
        subtitle = ""
        sm = re.match(r"Chapter\s+\d+\s*-\s*(.+)", heading, re.IGNORECASE)
        if sm:
            subtitle = sm.group(1).strip()
        elif heading.lower() == "prologue":
            subtitle = "Prologue"

        num_str = f"{num:02d}"
        if subtitle:
            slug = f"chapter_{num_str}_{_slug(subtitle)}"
        else:
            slug = f"chapter_{num_str}"

        chapters.append({
            "num":   num,
            "title": subtitle or heading,
            "label": heading,
            "slug":  slug,
            "text":  "\n".join(body),
        })

    return chapters


def clean_text(text: str) -> str:
    """Strip formatting artifacts and normalise whitespace for TTS."""
    # Remove horizontal-rule lines (underscores / asterisks / dashes)
    text = re.sub(r"^[_\-\*\s]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


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

    elapsed = time.monotonic() - t0
    if chunks:
        audio = np.concatenate(chunks, axis=0)
        sf.write(str(output_path), audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        print(f"  ✓  Saved '{output_path.name}'  "
              f"({_fmt_duration(duration)} audio  |  {_fmt_duration(elapsed)} wall-clock)")
    else:
        print(f"  ✗  No audio produced for voice='{voice}'")
    return elapsed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 'A Darkness Rising' audiobook, one file per chapter."
    )
    parser.add_argument(
        "chapters", nargs="*", type=int,
        help="Chapter numbers to generate (0 = Prologue). Default: all.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print detected chapters and exit.",
    )
    parser.add_argument(
        "--voice", default=VOICE,
        help=f"Kokoro voice to use (default: {VOICE}).",
    )
    parser.add_argument(
        "--preview", nargs="?", const=3000, type=int, metavar="CHARS",
        help="Generate short preview clips (default: 3000 chars). "
             "Output filenames get a _preview suffix.",
    )
    args = parser.parse_args()

    print("Loading chapters …")
    all_chapters = load_all_chapters(NOVEL_DIR)

    if args.list:
        print(f"\nDetected {len(all_chapters)} chapters:\n")
        print(f"  {'#':>4}  {'Label':<45}  {'Chars':>8}  {'Output filename'}")
        print(f"  {'─'*4}  {'─'*45}  {'─'*8}  {'─'*30}")
        for ch in all_chapters:
            chars = len(clean_text(ch["text"]))
            print(f"  {ch['num']:>4}  {ch['label']:<45}  {chars:>8,}  {ch['slug']}.wav")
        return

    # Filter to requested subset
    if args.chapters:
        requested = set(args.chapters)
        run_chapters = [ch for ch in all_chapters if ch["num"] in requested]
        missing = requested - {ch["num"] for ch in run_chapters}
        if missing:
            print(f"⚠  Chapter(s) not found: {sorted(missing)}")
    else:
        run_chapters = all_chapters

    if not run_chapters:
        print("No chapters selected. Use --list to see available chapters.")
        return

    voice = args.voice
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Voice:  {voice}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Pre-compute char counts
    chapter_chars = {ch["num"]: len(clean_text(ch["text"])) for ch in run_chapters}

    preview_note = (f"  ⚡ PREVIEW MODE — capped at {args.preview:,} chars/chapter\n"
                    if args.preview else "")
    print(f"\n{preview_note}{'─'*65}")
    print(f"  {'#':>4}  {'Label':<40}  {'Chars':>8}")
    print(f"  {'─'*4}  {'─'*40}  {'─'*8}")
    for ch in run_chapters:
        print(f"  {ch['num']:>4}  {ch['label']:<40}  {chapter_chars[ch['num']]:>8,}")
    print(f"  {'─'*55}")
    total_chars = sum(chapter_chars.values())
    print(f"  {'TOTAL':<45}  {total_chars:>8,}\n")

    print("Initialising Kokoro pipeline …")
    pipeline = KPipeline(lang_code=LANG_CODE)

    chars_per_sec: float | None = None
    timing_rows: list[tuple[str, int, float]] = []

    for ch in run_chapters:
        text = clean_text(ch["text"])
        if not text:
            print(f"\n[{ch['label']}]  ⚠  Empty text — skipping")
            continue

        preview_chars = args.preview
        if preview_chars and len(text) > preview_chars:
            cut = text.rfind(" ", 0, preview_chars)
            text = text[: cut if cut > 0 else preview_chars]

        chars = len(text)
        preview_tag = "_preview" if args.preview else ""
        out_path = OUTPUT_DIR / f"{ch['slug']}{preview_tag}.wav"

        if chars_per_sec is not None:
            eta_str = _fmt_duration(chars / chars_per_sec)
            print(f"\n[{ch['label']}]  voice={voice}  →  {out_path.name}  (est. {eta_str})")
        else:
            print(f"\n[{ch['label']}]  voice={voice}  →  {out_path.name}  (calibration run)")

        elapsed = generate_audio(pipeline, text, voice, out_path)
        timing_rows.append((ch["label"], chars, elapsed))

        total_done = sum(c for _, c, _ in timing_rows)
        total_elapsed_done = sum(e for _, _, e in timing_rows)
        if total_elapsed_done > 0:
            chars_per_sec = total_done / total_elapsed_done
            remaining = total_chars - total_done
            eta_overall = _fmt_duration(remaining / chars_per_sec) if remaining > 0 else "0s"
            print(f"  ⏱  Speed: {chars_per_sec:.0f} chars/sec  |  Est. overall remaining: {eta_overall}")

    # Summary
    print("\n" + "─" * 65)
    print(f"  {'Chapter':<35}  {'Chars':>7}  {'Actual':>8}  {'Est':>8}")
    print("─" * 65)
    for i, (label, chars, elapsed) in enumerate(timing_rows):
        actual_str = _fmt_duration(elapsed)
        prior_chars = sum(c for _, c, _ in timing_rows[:i])
        prior_elapsed = sum(e for _, _, e in timing_rows[:i])
        if prior_elapsed > 0:
            est_str = _fmt_duration(chars / (prior_chars / prior_elapsed))
        else:
            est_str = "(first)"
        print(f"  {label:<35}  {chars:>7,}  {actual_str:>8}  {est_str:>8}")
    total_elapsed = sum(e for _, _, e in timing_rows)
    print("─" * 65)
    print(f"  {'TOTAL':<35}  {sum(c for _,c,_ in timing_rows):>7,}  "
          f"{_fmt_duration(total_elapsed):>8}")
    print("\nDone.")


if __name__ == "__main__":
    main()
