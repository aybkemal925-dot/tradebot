@echo off
REM Austrade Trading Bot Launcher
REM Start the trading bot with virtual environment

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Check if venv exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install/update requirements if needed
pip install -q -r requirements.txt 2>nul

REM Start the app
cls
echo.
echo ============================================
echo    Austrade Trading Bot - Starting...
echo ============================================
echo.

python app.py

pause
