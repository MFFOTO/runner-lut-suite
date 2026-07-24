@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Runner LUT Suite - Setup

echo ================================================
echo   Runner LUT Suite - Setup
echo ================================================
echo.

where py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ first:
    echo         winget install -e --id Python.Python.3.12
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment ...
    py -3 -m venv .venv || ( echo [ERROR] venv creation failed & pause & exit /b 1 )
)

set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || ( echo [ERROR] pip install failed & pause & exit /b 1 )

echo.
echo Generating starter LUTs into luts\ ...
".venv\Scripts\python.exe" make_sample_luts.py

echo.
echo Setup complete.
echo Drop your own .cube files into the luts\ folder, then run run_lut.bat
pause
