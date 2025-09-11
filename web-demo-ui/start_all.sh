#!/bin/bash

# 1. Start app.py with its virtual environment
echo "Starting backend app.py..."
(
    source ~/tiktok-techjam-2025/web-demo-ui/backend/env/bin/activate
    cd ~/tiktok-techjam-2025/web-demo-ui/backend
    python app.py
) &

# 2. Start video_filter_api.py with its virtual environment
echo "Starting video_filter_api.py..."
(
    source ~/tiktok-techjam-2025/web-demo-ui/backend/env/bin/activate
    cd ~/tiktok-techjam-2025/web-demo-ui/backend
    python video_filter_api.py
) &

# 3. Build and start frontend
echo "Building and starting frontend..."
(
    cd ~/tiktok-techjam-2025/web-demo-ui/frontend
    npm run build
    npm start
) &

# 4. Start mediasoup server
echo "Starting mediasoup server..."
(
    cd ~/tiktok-techjam-2025/web-demo-ui/mediasoup-server
    npm start
) &

# 5. Start audio_redaction_server_faster_whisper.py with its virtual environment
echo "Starting audio_redaction_server_faster_whisper.py..."
(
    source ~/tiktok-techjam-2025/web-demo-ui/audio-william/env/bin/activate
    cd ~/tiktok-techjam-2025/web-demo-ui/audio-william
    python audio_redaction_server_faster_whisper.py
) &

echo "All applications started!"
wait
