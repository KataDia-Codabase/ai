@echo off
REM Windows setup script

echo === English Pronunciation AI Service Setup ===

REM Check Python
python --version >temp_version.txt 2>&1
python --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 📁 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Create directories
echo 📁 Creating directories...
if not exist "data\processed_english" mkdir data\processed_english
if not exist "models\wav2vec2-english-finetuned" mkdir models\wav2vec2-english-finetuned
if not exist "logs" mkdir logs
if not exist "temp_audio" mkdir temp_audio

echo.
echo ✅ Setup completed successfully!
echo.
echo Next steps:
echo 1. Copy .env.example to .env and configure your API keys
echo 2. Run: python -m uvicorn app.main:app --reload
echo 3. Open http://localhost:8000/docs to see API documentation
echo.
pause
