@echo off
echo Starting Video Filter API...
cd /d "%~dp0"

echo Activating Python environment...
REM Adjust this path to your Python environment if needed
python --version

echo Starting video filter API on port 5001...
cd backend
python video_filter_api.py

pause