@echo off
echo ====================================
echo Installing Faster-Whisper Dependencies
echo ====================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)

echo Python found, proceeding with installation...

REM Install Faster-Whisper and dependencies
echo.
echo Installing Faster-Whisper...
pip install faster-whisper

echo.
echo Installing other dependencies...
pip install flask flask-cors librosa soundfile numpy transformers torch torchaudio

echo.
echo ====================================
echo Installation Complete!
echo ====================================
echo.
echo To start the Faster-Whisper audio server:
echo   python audio_redaction_server_faster_whisper.py
echo.
echo To test the installation:
echo   python test_faster_whisper_api.py
echo.
pause