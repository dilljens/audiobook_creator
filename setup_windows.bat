@echo off
setlocal EnableDelayedExpansion
title Audiobook Setup

echo ============================================================
echo  Audiobook Setup for Windows 11
echo ============================================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────────────────────
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python was not found.
    echo.
    echo  Please install Python 3.11 from https://www.python.org/downloads/
    echo  IMPORTANT: On the installer, tick "Add Python to PATH" before clicking Install.
    echo.
    echo  After installing, close this window and double-click setup_windows.bat again.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  Found Python %PY_VER%
echo.

:: ── 2. Create virtual environment ────────────────────────────────────────────
echo [2/5] Creating virtual environment (.venv)...
if exist .venv (
    echo  .venv already exists, skipping creation.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Virtual environment created.
)
echo.

:: ── 3. Install PyTorch with CUDA (for gaming GPU) ────────────────────────────
echo [3/5] Installing PyTorch with CUDA 12.4 support (this may take a while)...
echo  Downloading ~2.5 GB — please be patient.
echo.
.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo.
    echo  WARNING: CUDA build failed. Falling back to CPU-only PyTorch.
    echo  Audio generation will be slower but will still work.
    .venv\Scripts\pip install torch
)
echo.

:: ── 4. Install remaining packages ────────────────────────────────────────────
echo [4/5] Installing remaining packages (kokoro, soundfile, sounddevice, spacy, wordfreq)...
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo  ERROR: Package installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo Downloading spaCy English language model (en_core_web_sm, ~15 MB)...
.venv\Scripts\python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo  WARNING: spaCy model download failed. Proper noun extraction will not work
    echo  until you re-run:  .venv\Scripts\python -m spacy download en_core_web_sm
)
echo.

:: ── 5. Download the Kokoro TTS model ─────────────────────────────────────────
echo [5/5] Downloading the Kokoro TTS model (hexgrad/Kokoro-82M, ~330 MB)...
echo  This only happens once.
echo.
.venv\Scripts\python -c "from kokoro import KPipeline; KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M'); print('Model ready.')"
if errorlevel 1 (
    echo.
    echo  WARNING: Model download failed. It will retry the first time you run the app.
    echo  Make sure you have an internet connection on first launch.
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To launch the GUI:          double-click  run_gui.bat
echo  To create the audiobook:   double-click  run_audiobook.bat
echo ============================================================
echo.
pause
