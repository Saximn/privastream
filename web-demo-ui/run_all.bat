@echo off
set BASE_DIR=%~dp0

:: Make sure logs folder exists
if not exist "%BASE_DIR%\logs" mkdir "%BASE_DIR%\logs"

echo Starting backend app.py...
start cmd /k "cd /d %BASE_DIR%\backend && call env\Scripts\activate && python app.py > %BASE_DIR%\logs\app.log 2>&1"

echo Starting video_filter_api.py...
start cmd /k "cd /d %BASE_DIR%\backend && call env\Scripts\activate && python video_filter_api.py > %BASE_DIR%\logs\video_filter.log 2>&1"

echo Starting frontend (npm run dev)...
start cmd /k "cd /d %BASE_DIR%\frontend && npm run dev > %BASE_DIR%\logs\frontend.log 2>&1"

echo Starting mediasoup-server (npm run dev)...
start cmd /k "cd /d %BASE_DIR%\mediasoup-server && npm run dev > %BASE_DIR%\logs\mediasoup.log 2>&1"

echo Starting audio redaction server...
start cmd /k "cd /d %BASE_DIR%\audio-william && call env\Scripts\activate && python audio_redaction_server_faster_whisper.py > %BASE_DIR%\logs\audio_redaction.log 2>&1"

echo All processes started. Logs can be found in %BASE_DIR%\logs
pause
