@echo off
title Proper Noun GUI

:: Change to the folder this .bat file lives in
cd /d "%~dp0"

:: Check setup has been run
if not exist .venv\Scripts\python.exe (
    echo ERROR: Setup has not been run yet.
    echo Please double-click setup_windows.bat first.
    pause
    exit /b 1
)

echo Starting Proper Noun Player GUI...
.venv\Scripts\python gui_proper_noun_player.py
if errorlevel 1 (
    echo.
    echo The application closed with an error. See message above.
    pause
)
