@echo off
echo Starting AI Audio Transcriber Web App...
cd /d "%~dp0"
if not exist .venv (
    echo Error: .venv folder not found! Please make sure the virtual environment is set up.
    pause
    exit /b
)
call .venv\Scripts\activate.bat
streamlit run app.py
pause
