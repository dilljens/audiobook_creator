@echo off
title Create Audiobook

:: Change to the folder this .bat file lives in
cd /d "%~dp0"

:: Check setup has been run
if not exist .venv\Scripts\python.exe (
    echo ERROR: Setup has not been run yet.
    echo Please double-click setup_windows.bat first.
    pause
    exit /b 1
)

echo ============================================================
echo  Audiobook Creator
echo ============================================================
echo.
echo  Options:
echo    1 - Generate ALL chapters  (may take many hours)
echo    2 - List detected chapters only
echo    3 - Generate a short PREVIEW of each chapter
echo    4 - Generate specific chapters (enter numbers next)
echo.
set /p CHOICE="Enter choice (1/2/3/4): "

if "%CHOICE%"=="1" (
    .venv\Scripts\python create_audiobook_lightbringer.py
) else if "%CHOICE%"=="2" (
    .venv\Scripts\python create_audiobook_lightbringer.py --list
) else if "%CHOICE%"=="3" (
    .venv\Scripts\python create_audiobook_lightbringer.py --preview
) else if "%CHOICE%"=="4" (
    set /p CHAPTERS="Enter chapter numbers separated by spaces (e.g. 0 1 2): "
    .venv\Scripts\python create_audiobook_lightbringer.py %CHAPTERS%
) else (
    echo Invalid choice.
)

echo.
echo Done. Output files are in the output_audiobook_lightbringer folder.
pause
