import torch
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from text_input import TEXT

# ── Device setup ──────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SAMPLE_RATE = 24000
SPEED = 1.0
VOICES = [
    ("af_heart",   "output_af_heart.wav"),    # warm American female
    ("am_michael", "output_am_michael.wav"),   # best American male
]

pipeline = KPipeline(lang_code="a")


def generate(voice: str, output_file: str) -> None:
    print(f"\nGenerating '{voice}' → {output_file} …")
    chunks = []
    for _, _, chunk_audio in pipeline(TEXT, voice=voice, speed=SPEED):
        if hasattr(chunk_audio, "numpy"):
            chunk_audio = chunk_audio.cpu().numpy()
        chunk_audio = np.atleast_1d(chunk_audio.squeeze())
        if chunk_audio.size > 0:
            chunks.append(chunk_audio)

    if chunks:
        audio = np.concatenate(chunks, axis=0)
        sf.write(output_file, audio, SAMPLE_RATE)
        print(f"  ✓ Saved '{output_file}'  ({len(audio) / SAMPLE_RATE:.1f}s, {SAMPLE_RATE} Hz)")
    else:
        print(f"  ✗ No audio produced for '{voice}'")


for voice, path in VOICES:
    generate(voice, path)

print("\nDone.")
