# Audiobook Creator

AI-powered audiobook generator using the [Kokoro TTS](https://github.com/hexgrad/kokoro) model.
Generates high-quality narrated `.wav` files from plain-text novels, with a GUI tool for auditing and fixing proper noun pronunciations per book.

---

## Features

- **Multi-book support** — each book's proper nouns, fixes, and audio are fully isolated
- **Proper Noun GUI** — hear every extracted name, mark it correct or type a phonetic fix
- **Audiobook generation** — one `.wav` per chapter, GPU-accelerated via CUDA
- **In-GUI extraction** — click one button to run NLP extraction and generate audio, no separate scripts needed
- **Apply Fixes** — writes a TTS-ready copy of the source text with all phonetic substitutions applied

---

## Project structure

```
Audio Text for Novel Lightbringer/   ← multi-file book (chapters as .txt)
Audio Master Nem Full.txt            ← single-file book

gui_proper_noun_player.py            ← proper noun auditing GUI
create_audiobook_lightbringer.py     ← generate Lightbringer audiobook chapters
create_audiobook_nem.py              ← generate Nem audiobook chapters

output_audiobook_lightbringer/       ← chapter WAV output
output_audiobook/                    ← Nem WAV output
output_proper_nouns/<book>/          ← manifest + JSON fix data per book
proper_nouns_audio/<book>/           ← word audio + replacements cache per book

requirements.txt
setup_windows.bat                    ← one-click Windows setup
run_gui.bat                          ← launch GUI on Windows
run_audiobook.bat                    ← generate audiobook on Windows
```

---

## Setup (Linux / Mac)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> For CPU-only: replace the torch line with `pip install torch`

---

## Setup (Windows)

See [SETUP_WINDOWS.md](SETUP_WINDOWS.md) for a step-by-step guide aimed at non-technical users.

---

## Usage

### Proper Noun GUI

```bash
.venv/bin/python gui_proper_noun_player.py
```

1. Select a book from the dropdown
2. Click **⚙ Extract & Generate Audio** — extracts proper nouns via spaCy and generates a TTS clip for each one
3. Click words in the Review list to hear them; press Enter to mark correct or type a phonetic replacement first
4. Click **⇄ Apply Fixes to Text** to write a pronunciation-corrected copy of the source file

### Generate Audiobook

```bash
# All chapters
.venv/bin/python create_audiobook_lightbringer.py

# List chapters only
.venv/bin/python create_audiobook_lightbringer.py --list

# Preview clips
.venv/bin/python create_audiobook_lightbringer.py --preview

# Specific chapters
.venv/bin/python create_audiobook_lightbringer.py 0 1 2
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `kokoro` | Kokoro-82M TTS model |
| `torch` | GPU inference |
| `soundfile` / `sounddevice` | Audio I/O |
| `numpy` | Audio array operations |
| `spacy` + `en_core_web_sm` | Proper noun extraction (NER + PROPN) |
| `wordfreq` | Common-word filter during extraction |

---

## Output

| Path | Contents |
|---|---|
| `output_audiobook_lightbringer/` | `chapter_01_homecoming.wav`, … |
| `output_proper_nouns/<book>/manifest.json` | Word → WAV filename map |
| `output_proper_nouns/<book>/pronunciation_fixes.json` | `{"Nephi": "Kneephi", …}` |
| `output_proper_nouns/<book>/correct_words.json` | Words confirmed correct |
| `proper_nouns_audio/<book>/` | Per-word audio clips |
| `proper_nouns_audio/<book>/replacements_cache/` | Cached phonetic fix clips |

