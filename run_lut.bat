@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Runner LUT Suite - Run

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv missing. Run setup_lut.bat first.
    pause
    exit /b 1
)

if not exist "settings_lut.json" (
    if exist "settings_lut.example.json" (
        copy "settings_lut.example.json" "settings_lut.json" >nul
        echo Created settings_lut.json from the template.
        echo Please set input_folder and output_folder in settings_lut.json, then run again.
        echo.
        pause
        exit /b 0
    )
)

set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" lut_suite.py settings_lut.json

echo.
pause
