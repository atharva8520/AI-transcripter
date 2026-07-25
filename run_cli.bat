@echo off
echo Starting AI Video Assistant CLI...
cd /d "%~dp0"
if not exist .venv (
    echo Error: .venv folder not found! Please make sure the virtual environment is set up.
    pause
    exit /b
)
call .venv\Scripts\activate.bat
python main.py
pause
