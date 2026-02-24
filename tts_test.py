import torch
import soundfile as sf
from kokoro import KPipeline

# ── Device setup ──────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Test paragraph ─────────────────────────────────────────────────────────────
TEXT = (
    "The world of artificial intelligence is evolving at a remarkable pace. "
    "Modern language models can now read, write, and even speak with surprising "
    "clarity and nuance. This audio was generated entirely on a local machine "
    "using the Kokoro text-to-speech model, running on an NVIDIA RTX 3060 GPU. "
    "No cloud, no API keys — just raw local compute turning words into sound."
)

# ── Build pipeline ─────────────────────────────────────────────────────────────
# lang_code: 'a' = American English, 'b' = British English
# voices: af_heart, af_bella, af_nova, am_adam, am_michael, bf_emma, bm_george …
pipeline = KPipeline(lang_code="a")

OUTPUT_FILE = "output.wav"
VOICE = "af_heart"          # warm American female voice
SPEED = 1.0                 # 1.0 = normal speed

# ── Generate audio ─────────────────────────────────────────────────────────────
print(f"Generating speech with voice '{VOICE}' …")

import numpy as np

audio_chunks = []
for _, _, chunk_audio in pipeline(TEXT, voice=VOICE, speed=SPEED):
    # chunk_audio is a torch.Tensor of shape [N], dtype float32
    if hasattr(chunk_audio, "numpy"):
        chunk_audio = chunk_audio.cpu().numpy()
    chunk_audio = np.atleast_1d(chunk_audio.squeeze())
    if chunk_audio.size > 0:
        audio_chunks.append(chunk_audio)

if audio_chunks:
    audio = np.concatenate(audio_chunks, axis=0)
    sf.write(OUTPUT_FILE, audio, 24000)
    duration = len(audio) / 24000
    print(f"✓ Saved '{OUTPUT_FILE}'  ({duration:.1f}s, 24 kHz)")
else:
    print("No audio generated — check input text.")
