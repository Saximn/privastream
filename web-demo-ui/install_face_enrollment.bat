@echo off
echo ====================================
echo Installing Face Enrollment Dependencies
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

REM Install Face Enrollment dependencies
echo.
echo Installing Face Enrollment Server dependencies...
pip install -r face_enrollment_requirements.txt

echo.
echo ====================================
echo Installation Complete!
echo ====================================
echo.
echo To start the Face Enrollment server:
echo   python face_enrollment_server.py
echo.
echo To test the installation:
echo   python test_face_enrollment_api.py
echo.
echo Server will run on: http://localhost:5003
echo.
pause