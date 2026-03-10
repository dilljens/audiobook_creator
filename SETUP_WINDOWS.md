# Audiobook Creator — Windows 11 Setup Guide

This guide is written for someone who has never used Python or the command line.
Follow the steps in order and you will be generating audiobook chapters with your gaming GPU.

---

## What you will need

| Requirement | Why |
|---|---|
| Windows 11 PC with a modern NVIDIA GPU | Fast audio generation using CUDA |
| ~5 GB free disk space | Python, PyTorch, and the AI voice model |
| Internet connection (first-time only) | Downloads packages and the Kokoro voice model |

---

## Step 1 — Install Python

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow **"Download Python 3.11.x"** button
3. Run the installer
4. **IMPORTANT:** On the very first screen of the installer, tick the checkbox that says **"Add Python to PATH"** before clicking Install Now

> If you missed that checkbox, uninstall Python from Windows Settings and reinstall it with the box ticked.

---

## Step 2 — Get the project files

You should have a folder called `audiobook_creator` (or similar) containing the project files. Make sure it includes these files:

```
setup_windows.bat
run_gui.bat
run_audiobook.bat
requirements.txt
gui_proper_noun_player.py
create_audiobook_lightbringer.py
Audio Text for Novel Lightbringer\    ← your chapter text files go here
```

If you received a ZIP file, extract it first so the folder is not inside another folder.

---

## Step 3 — Run Setup (one time only)

1. Open the project folder in File Explorer
2. Double-click **`setup_windows.bat`**
3. A black terminal window opens and runs through these steps automatically:
   - Checks Python is installed
   - Creates a private Python environment (`.venv` folder)
   - Downloads PyTorch with GPU (CUDA) support — **about 2.5 GB, this takes several minutes**
   - Installs the remaining packages (kokoro, spaCy, etc.)
   - Downloads the spaCy English language model
   - Downloads the Kokoro AI voice model — **about 330 MB**
4. When it says **"Setup complete!"**, press any key to close the window

You only need to do this once. If you run it again it will safely skip anything already installed.

---

## Step 4 — Review Proper Noun Pronunciations (GUI)

Before generating the audiobook, it helps to check how unusual names are pronounced.

1. Double-click **`run_gui.bat`**
2. The Proper Noun Pronunciation Auditor window opens
3. Select your book from the dropdown at the top
4. Click **⚙ Extract & Generate Audio** — this scans the text and creates a short audio clip for every proper noun found (takes a few minutes the first time)
5. Click any word in the **To Review** list to hear how it sounds
6. If it sounds wrong, type the phonetic spelling in the box at the bottom and press **Enter** to save a fix
   - Example: type `Kneephi` instead of `Nephi`
7. If it sounds correct, just press **Enter** without changing anything
8. When you are done reviewing, click **⇄ Apply Fixes to Text** to save a corrected copy of the source text

**Keyboard shortcuts:**
| Key | Action |
|---|---|
| Space | Replay current word |
| Enter | Mark correct (or save fix if text was changed) |
| Escape | Reset the fix box, go back to word list |
| s | Stop audio |
| ↑ / ↓ | Navigate the word list from the fix box |
| Delete | Move a word back to Review from Correct or Fixes |

---

## Step 5 — Generate the Audiobook

1. Double-click **`run_audiobook.bat`**
2. A menu appears — type the number of your choice and press Enter:

| Option | What it does |
|---|---|
| 1 | Generate **all chapters** — can take many hours, safe to leave running overnight |
| 2 | **List** detected chapters only — instant, nothing is generated |
| 3 | Generate a short **preview clip** of each chapter — quick sanity check |
| 4 | Generate **specific chapters** — enter chapter numbers separated by spaces |

3. When finished, `.wav` files will be in the `output_audiobook_lightbringer` folder

---

## Troubleshooting

**"Python was not found"**
→ Python is not installed, or you forgot to tick "Add Python to PATH" during installation. Uninstall and reinstall Python from https://www.python.org/downloads/ making sure to tick that box.

**The black window opens and immediately closes**
→ There was an error. To see it: press `Win + R`, type `cmd`, press Enter, then drag the `.bat` file into that black window and press Enter. The error message will stay visible.

**Audio generation is very slow (taking hours per chapter)**
→ The GPU version of PyTorch may not have installed correctly. Re-run `setup_windows.bat` — it will reinstall just that part.

**"No .txt files found in Audio Text for Novel Lightbringer"**
→ Make sure your chapter `.txt` files are inside the `Audio Text for Novel Lightbringer` subfolder, not loose in the main project folder.

**The GUI says "No manifest yet"**
→ You need to click **⚙ Extract & Generate Audio** first for that book.

**Antivirus blocks the .bat files**
→ Right-click the `.bat` file, choose Properties, and click "Unblock" at the bottom. Then try again.

---

## Output files

| Folder | Contents |
|---|---|
| `output_audiobook_lightbringer\` | One `.wav` file per chapter |
| `output_proper_nouns\<book>\` | Pronunciation data (JSON files) |
| `proper_nouns_audio\<book>\` | Cached word audio clips |
