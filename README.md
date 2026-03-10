# Audiobook Generator — Windows 11 Setup Guide

This guide is written for someone who has never used Python or the command line.
Follow the steps in order and you'll be generating audiobook chapters with a gaming GPU.

---

## What you'll need

| Requirement | Why |
|---|---|
| Windows 11 PC with a modern NVIDIA GPU | Fast audio generation using CUDA |
| ~5 GB free disk space | Python, PyTorch, and the TTS model |
| Internet connection (first-time only) | Downloads packages and the AI voice model |

---

## Step 1 — Install Python

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **"Download Python 3.11.x"** button
3. Run the installer
4. **IMPORTANT:** On the first screen, tick the box that says **"Add Python to PATH"** before you click Install Now

If you skipped that checkbox, uninstall Python and reinstall with the box ticked.

---

## Step 2 — Get the project files

You should have a folder (e.g. `voice_model`) containing the project. Make sure it contains:

```
setup_windows.bat
run_gui.bat
run_audiobook.bat
requirements.txt
gui_proper_noun_player.py
create_audiobook_lightbringer.py
Audio Text for Novel Lightbringer\   ← your text files go here
```

---

## Step 3 — Run Setup (one time only)

1. Open the `voice_model` folder in File Explorer
2. Double-click **`setup_windows.bat`**
3. A black terminal window will open and run through 5 steps:
   - Checks Python is installed
   - Creates a private Python environment
   - Downloads PyTorch with GPU (CUDA) support — **~2.5 GB, be patient**
   - Installs the remaining packages
   - Downloads the Kokoro AI voice model — **~330 MB**
4. When it says **"Setup complete!"**, press any key to close

You only need to do this once.

---

## Step 4 — Launch the GUI (Proper Noun Player)

1. Double-click **`run_gui.bat`**
2. The Proper Noun Player window opens
3. Use it to review and fix how proper nouns are pronounced before generating audio

**Controls:**
- Click a word in the Review list to hear it
- Type a phonetic spelling in the box at the bottom and press Enter to save a fix
- Press Enter without changing anything to mark the word as Correct
- Press Space to replay the current word
- Click "Apply Fixes to Text" when done to save a pronunciation-corrected text file

---

## Step 5 — Create the Audiobook

1. Double-click **`run_audiobook.bat`**
2. A menu appears:
   - **1** — Generate ALL chapters (this can take many hours — leave it running overnight)
   - **2** — Just list what chapters were detected (safe, instant)
   - **3** — Generate a short preview clip of each chapter (quick test)
   - **4** — Generate specific chapter numbers only
3. Choose an option and press Enter
4. When finished, the `.wav` files will be in the `output_audiobook_lightbringer` folder

---

## Troubleshooting

**"Python was not found"**
→ Python is not installed, or you forgot to tick "Add Python to PATH". Reinstall Python.

**The window opens and immediately closes**
→ Right-click the `.bat` file → "Run as administrator", or open a new terminal window first:
press `Win + R`, type `cmd`, press Enter, then drag the `.bat` file into that window and press Enter.

**Audio generation is very slow**
→ The GPU (CUDA) version of PyTorch may not have installed correctly. Re-run `setup_windows.bat`.

**"No .txt files found in Audio Text for Novel Lightbringer"**
→ Make sure your chapter text files are placed in the `Audio Text for Novel Lightbringer` subfolder.

---

## Output files

| Folder | Contents |
|---|---|
| `output_audiobook_lightbringer\` | One `.wav` file per chapter |
| `output_proper_nouns\` | Pronunciation fix data (JSON) |
| `proper_nouns_audio\` | Cached audio for each proper noun |
