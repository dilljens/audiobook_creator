import torch, numpy as np, soundfile as sf
from kokoro import KPipeline
from text_input import TEXT

pipeline = KPipeline(lang_code="a")
print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
print("Generating am_michael ...")

chunks = []
for _, _, chunk_audio in pipeline(TEXT, voice="am_michael", speed=1.0):
    if hasattr(chunk_audio, "numpy"):
        chunk_audio = chunk_audio.cpu().numpy()
    chunk_audio = np.atleast_1d(chunk_audio.squeeze())
    if chunk_audio.size > 0:
        chunks.append(chunk_audio)

audio = np.concatenate(chunks)
sf.write("output_am_michael.wav", audio, 24000)
print(f"Saved output_am_michael.wav  ({len(audio)/24000:.1f}s)")
