"""
create_temple_voices.py
────────────────────────
Generate the "Sacred Temple Writings" section of the Nem audiobook using one
distinct Microsoft Edge neural TTS voice per character (NOT Kokoro).

Uses the free edge-tts library which streams Microsoft Azure neural voices.
Audio is stitched into a single WAV and saved to OUTPUT_DIR.

Usage:
    python create_temple_voices.py                    # full render
    python create_temple_voices.py --preview 40       # first 40 segments only
    python create_temple_voices.py --print-segments   # inspect parsed segments
    python create_temple_voices.py --list-voices      # list available en voices

Voice assignments live in CHARACTER_VOICES below — easy to customise.
Run  --list-voices  to discover all available edge-tts voice names.
"""

import argparse
import asyncio
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import edge_tts

# ── File / output config ───────────────────────────────────────────────────────
_FIXED_FILE  = Path("Audio Master Nem Full (TTS Fixed).txt")
_ORIG_FILE   = Path("Audio Master Nem Full.txt")
SOURCE_FILE  = _FIXED_FILE if _FIXED_FILE.exists() else _ORIG_FILE

OUTPUT_DIR   = Path("output_temple_voices")
OUTPUT_FILE  = "sacred_temple_writings_multivoice.wav"

SAMPLE_RATE  = 24_000   # Hz — final WAV sample rate
PAUSE_SAME   = 350      # ms silence between same-speaker segments
PAUSE_CHANGE = 650      # ms silence between different-speaker segments

# ── Section boundary markers (match create_audiobook_nem.py BOOKS order) ──────
#   Sacred Temple Writings starts at "THE SACRED" / "TEMPLE WRITINGS"
#   and ends just before "THE FIRST BOOK" / "OF SAMUEL THE LAMANITE"
_SEC_START_L1 = "THE SACRED"
_SEC_START_L2 = "TEMPLE WRITINGS"
_SEC_END_L1   = "THE FIRST BOOK"
_SEC_END_L2   = "OF SAMUEL THE LAMANITE"

# ── Character → edge-tts voice ────────────────────────────────────────────────
# Run  python create_temple_voices.py --list-voices  to see all available voices.
# Keys must match the speaker labels exactly as they appear in the source file.
CHARACTER_VOICES: dict[str, str] = {
    # ── Celestial beings ───────────────────────────────────────────────────────
    "Narrator":               "en-US-GuyNeural",         # calm neutral narrator
    "Elohim Heavenly Mother": "en-US-JennyNeural",       # warm, wise matriarch
    "Elohim Heavenly Father": "en-US-AndrewMultilingualNeural",  # expressive, authoritative
    "Jehovah":                "en-US-AndrewNeural",      # clear, gentle divine
    "Angel of the Lord":      "en-US-BrianNeural",       # ethereal divine messenger
    "Holy Ghost":             "en-US-EricNeural",        # quiet, inward, spiritual
    "Holy Ghost Elders":      "en-US-BrianNeural",       # measured elder council

    # ── Dark beings ────────────────────────────────────────────────────────────
    "Lucifer":                "en-CA-LiamNeural",        # smooth, persuasive tempter
    "Satan":                  "en-US-SteffanNeural",     # cold, commanding adversary

    # ── Mortal / earth characters ──────────────────────────────────────────────
    "Michael":                "en-US-RogerNeural",        # noble warrior archangel
    "Adam":                   "en-US-ChristopherNeural",  # earnest first man
    "Eve":                    "en-US-AriaNeural",        # curious, warm first woman

    # ── Apostles ───────────────────────────────────────────────────────────────
    "Peter":                  "en-GB-RyanNeural",        # firm British apostle
    "James":                  "en-AU-WilliamMultilingualNeural",  # steady Australian voice
    "John":                   "en-IE-ConnorNeural",      # gentle Irish apostle

    # ── Other roles ────────────────────────────────────────────────────────────
    "Preacher":               "en-US-AvaNeural",         # bold emphatic preacher
    "Mob":                    "en-US-MichelleNeural",    # crowd / multitude voice
    "The Voice of the Mob":   "en-US-MichelleNeural",   # alias used in some editions
}

# Voice used when a speaker label isn't found in CHARACTER_VOICES
FALLBACK_VOICE = "en-US-GuyNeural"

# Lines/patterns that are ceremony stage-directions → read by Narrator
_STAGE_NARRATOR = re.compile(
    r"^(Break for Instruction|Resume Session|All\s+arise|"
    r"CHAPTER\s*\d*|________________+|────+)",
    re.IGNORECASE,
)

# Lines to skip entirely (decorative / empty)
_SKIP_RE = re.compile(r"^[—\-_\s\u2014\u2013]*$")


# ── Section extraction ─────────────────────────────────────────────────────────

def extract_section(source: Path) -> str:
    """Return text of the Sacred Temple Writings section."""
    lines = source.read_text(encoding="utf-8").splitlines()
    in_sec = False
    out: list[str] = []

    for i, line in enumerate(lines):
        s = line.strip()
        if not in_sec:
            if (s.upper() == _SEC_START_L1 and
                    i + 1 < len(lines) and
                    lines[i + 1].strip().upper().startswith(_SEC_START_L2)):
                in_sec = True
        else:
            # End just before the next section
            if (s.upper() == _SEC_END_L1 and
                    i + 1 < len(lines) and
                    lines[i + 1].strip().upper().startswith(_SEC_END_L2)):
                break
            out.append(line)

    if not out:
        raise RuntimeError(
            f"Could not locate 'Sacred Temple Writings' in '{source}'.\n"
            "Ensure the source file has a line exactly matching "
            f"'{_SEC_START_L1}' followed by '{_SEC_START_L2}'."
        )
    return "\n".join(out)


# ── Segment parser ─────────────────────────────────────────────────────────────

def _speaker_regex(characters: list[str]) -> re.Pattern:
    """Regex matching  [optional-number]  CharacterName:  text"""
    # Sort longest-first so "Holy Ghost Elders" matches before "Holy Ghost"
    names = sorted(characters, key=len, reverse=True)
    pat = "|".join(re.escape(n) for n in names)
    return re.compile(r"^\d*\s*(" + pat + r")\s*:\s*(.*)", re.IGNORECASE)


def parse_segments(text: str) -> list[tuple[str, str]]:
    """
    Convert section text into a list of (normalised_speaker, spoken_text) tuples.
    Non-attributed prose becomes Narrator lines.
    """
    char_re = _speaker_regex(list(CHARACTER_VOICES.keys()))

    # Build a quick lowercase→canonical lookup for speaker name normalisation
    canon: dict[str, str] = {k.lower(): k for k in CHARACTER_VOICES}

    segments: list[tuple[str, str]] = []
    cur_speaker = "Narrator"
    buf: list[str] = []

    def flush() -> None:
        combined = " ".join(l.strip() for l in buf if l.strip())
        if combined:
            segments.append((cur_speaker, combined))
        buf.clear()

    for raw in text.splitlines():
        line = raw.strip()

        if not line or _SKIP_RE.match(line):
            continue

        # Stage direction → Narrator reads it
        if _STAGE_NARRATOR.match(line):
            flush()
            cur_speaker = "Narrator"
            buf.append(line)
            continue

        # "The words of Jehovah … are in blue." — formatting note, skip
        if re.search(r"are in blue|words of jehovah", line, re.IGNORECASE):
            continue

        m = char_re.match(line)
        if m:
            flush()
            raw_name = m.group(1)
            cur_speaker = canon.get(raw_name.lower(), raw_name)
            spoken = m.group(2).strip()
            if spoken:
                buf.append(spoken)
        else:
            # Continuation of current speaker (or unattributed narrator prose)
            buf.append(line)

    flush()
    return segments


# ── Audio generation ───────────────────────────────────────────────────────────

async def _tts_bytes(text: str, voice: str) -> bytes:
    """Stream edge-tts and return raw MP3 bytes."""
    communicate = edge_tts.Communicate(text, voice)
    data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data.extend(chunk["data"])
    return bytes(data)


def _mp3_to_numpy(mp3: bytes) -> np.ndarray:
    """Decode MP3 bytes → mono float32 numpy array at SAMPLE_RATE using ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",                    # read MP3 from stdin
        "-f", "f32le",                      # raw 32-bit little-endian float PCM
        "-acodec", "pcm_f32le",
        "-ac", "1",                          # mono
        "-ar", str(SAMPLE_RATE),            # resample to target rate
        "pipe:1",                           # write PCM to stdout
    ]
    result = subprocess.run(cmd, input=mp3, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)


async def render(
    segments: list[tuple[str, str]],
    preview: int | None = None,
) -> np.ndarray:
    """Generate and stitch all segment audio; return concatenated float32 array."""
    if preview is not None:
        segments = segments[:preview]

    parts: list[np.ndarray] = []
    last_speaker: str | None = None
    t0 = time.monotonic()

    for idx, (speaker, text) in enumerate(segments, 1):
        voice = CHARACTER_VOICES.get(speaker, FALLBACK_VOICE)
        marker = "⚠" if speaker not in CHARACTER_VOICES else " "
        print(f"  {marker}[{idx:>4}/{len(segments)}]  {speaker:<28}  {voice}")

        try:
            mp3 = await _tts_bytes(text, voice)
        except Exception as exc:
            print(f"       ↳ ERROR with '{voice}': {exc}  — falling back to {FALLBACK_VOICE}")
            mp3 = await _tts_bytes(text, FALLBACK_VOICE)

        audio = _mp3_to_numpy(mp3)

        if parts:
            gap = PAUSE_SAME if speaker == last_speaker else PAUSE_CHANGE
            parts.append(_silence(gap))
        parts.append(audio)
        last_speaker = speaker

    elapsed = time.monotonic() - t0
    print(f"\n  ✓  {len(segments)} segments in {elapsed:.0f}s")
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


# ── Voice listing ──────────────────────────────────────────────────────────────

async def _list_voices_async() -> None:
    voices = await edge_tts.list_voices()
    english = sorted(
        (v for v in voices if v["Locale"].startswith("en-")),
        key=lambda v: (v["Locale"], v["ShortName"]),
    )
    print(f"\n  {'Locale':<12}  {'Short Name':<45}  Gender")
    print("  " + "─" * 68)
    for v in english:
        print(f"  {v['Locale']:<12}  {v['ShortName']:<45}  {v['Gender']}")
    print(f"\n  {len(english)} English voices total.")


# ── CLI / main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render Sacred Temple Writings with per-character edge-tts voices."
    )
    ap.add_argument("--list-voices", action="store_true",
                    help="Print all available English edge-tts voices and exit.")
    ap.add_argument("--print-segments", action="store_true",
                    help="Print parsed (speaker, text) segments and exit.")
    ap.add_argument("--preview", type=int, metavar="N",
                    help="Render only the first N segments (quick test).")
    args = ap.parse_args()

    if args.list_voices:
        asyncio.run(_list_voices_async())
        return

    # ── Extract & parse ────────────────────────────────────────────────────────
    print(f"Source : {SOURCE_FILE}")
    text = extract_section(SOURCE_FILE)
    print(f"Section: {len(text):,} chars extracted\n")

    segments = parse_segments(text)

    if args.print_segments:
        print(f"Parsed {len(segments)} segments:\n")
        for i, (spkr, txt) in enumerate(segments, 1):
            snippet = txt[:90] + ("…" if len(txt) > 90 else "")
            voice = CHARACTER_VOICES.get(spkr, f"{FALLBACK_VOICE} ⚠")
            print(f"  {i:>4}.  [{spkr}]  ({voice})\n        {snippet}\n")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    counts = Counter(s for s, _ in segments)
    unrecognised = {s for s in counts if s not in CHARACTER_VOICES}

    print(f"Parsed {len(segments)} segments across {len(counts)} speakers:\n")
    print(f"  {'Speaker':<28}  {'Segs':>5}  {'Voice'}")
    print(f"  {'─'*28}  {'─'*5}  {'─'*45}")
    for spkr, voice in CHARACTER_VOICES.items():
        if counts[spkr]:
            print(f"  {spkr:<28}  {counts[spkr]:>5}  {voice}")
    for spkr in sorted(unrecognised):
        print(f"  {spkr:<28}  {counts[spkr]:>5}  {FALLBACK_VOICE}  ⚠ unrecognised")

    total_chars = sum(len(t) for _, t in segments)
    print(f"\n  Total chars: {total_chars:,}")
    if args.preview:
        print(f"  ⚡ PREVIEW MODE — rendering first {args.preview} segments only")

    # ── GPU note ───────────────────────────────────────────────────────────────
    # edge-tts is cloud-based (Microsoft Azure neural, free) — GPU not used.
    print("\nNote: edge-tts uses Microsoft's servers (free, no API key needed).\n"
          "      Render speed depends on your internet connection.\n")

    # ── Render ─────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / (
        f"sacred_temple_writings_preview{args.preview}.wav"
        if args.preview else OUTPUT_FILE
    )

    print("Rendering segments …\n")
    audio = asyncio.run(render(segments, args.preview))

    if audio.size > 0:
        sf.write(str(out_path), audio, SAMPLE_RATE)
        dur = len(audio) / SAMPLE_RATE
        m, s = divmod(int(dur), 60)
        print(f"\n✓  Saved '{out_path}'  ({m}m {s:02d}s audio  |  {SAMPLE_RATE} Hz)")
    else:
        print("✗  No audio produced — check parsing with --print-segments")


if __name__ == "__main__":
    main()
