# TikTok TechJam 2025: AI-Powered Privacy Streaming Platform

## Overview
This project is a full-stack web application designed to enhance user privacy in live video streaming using AI. It leverages real-time detection and redaction of Personally Identifiable Information (PII), faces, and license plates in both audio and video streams. The system is scalable, supporting hundreds of viewers via a Mediasoup SFU (Selective Forwarding Unit) server.

---

## Features & Functionality
- **Real-Time Video Privacy Filtering:**
   - Detects faces, PII text, and license plates in live video streams using AI models.
   - Applies blur or pixelation to sensitive regions at 4fps detection with 30fps output.
   - GPU acceleration available for fast client-side processing.
- **Audio Privacy Protection:**
   - Detects PII in audio using Whisper-based models.
   - Redacts detected PII with beep, silence, or reverse effects.
   - Optionally applies mouth blur in sync with audio redaction.
- **Scalable Streaming:**
   - Uses Mediasoup SFU for scalable, low-latency video streaming to many viewers.
- **User Controls:**
   - Host dashboard to start/stop streaming, enable/disable privacy filters, and view detection stats.
   - Toggle GPU acceleration for video filtering.
- **Viewer Experience:**
   - Viewers join a room and watch the privacy-protected live stream.

---

## Development Tools
- **Frontend:** Next.js, React, Tailwind CSS
- **Backend:** Python (Flask), Socket.IO
- **Mediasoup Server:** Node.js
- **AI Models:** Whisper (audio), custom video detection models (face, PII, plate)

---

## APIs Used
- **Video Filter API:** Python Flask backend (`video_filter_api.py`) for frame analysis and region detection.
- **Socket.IO:** Real-time signaling between frontend, backend, and Mediasoup server.
- **Mediasoup:** WebRTC SFU for scalable video streaming.

---

## Assets Used
- No third-party copyrighted music, trademarks, or assets.
- All AI models and code are open source or custom-developed for this competition.

---

## Libraries Used
- **Frontend:**
   - React, Next.js, Tailwind CSS
   - mediasoup-client
   - socket.io-client
- **Backend:**
   - Flask, Flask-SocketIO
   - OpenAI Whisper
   - NumPy, OpenCV, other Python ML libraries
- **Mediasoup Server:**
   - mediasoup, express, socket.io

---

## Problem Statement
This project addresses the challenge of protecting user privacy in live streaming environments. It uses AI to automatically detect and redact sensitive information in both video and audio, ensuring that users can stream safely without exposing personal data. The solution demonstrates "AI for Privacy" (using AI to defend user privacy).

---

## Setup Instructions

### Backend
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Start the Flask backend:
    ```bash
    python app.py
    ```
3. Start the Video Filter API (for frame analysis):
    ```bash
    python video_filter_api.py
    ```

### Frontend
1. Install dependencies:
    ```bash
    npm install
    ```
2. Start the Next.js frontend:
    ```bash
    npm run dev
    ```

### Mediasoup Server
1. Install dependencies:
    ```bash
    npm install
    ```
2. Start the Mediasoup server:
    ```bash
    npm start
    ```

---

## Demo Video
- [YouTube Demo Link](https://www.youtube.com/your-demo-link)
   - Less than 3 minutes
   - Shows the app running on web (desktop)
   - No copyrighted music or third-party trademarks

---

## Public GitHub Repository
- [GitHub Repo Link](https://github.com/Saximn/tiktok-techjam-2025/)

---

## Ownership & Copyright
```
Copyright (c) 2025, blueberry jam
All code, models, and assets in this repository are original or used with permission.
This project was developed for the TikTok TechJam 2025 competition.
```

---

## Supplementary Notes
- All code is commented for clarity and ownership.
- No third-party copyrighted assets are included.
- All dependencies are open source.

---

## Folder Structure

## Folder Structure
 - `frontend/` - Next.js web app (host/viewer dashboards)
 - `backend/` - Python Flask backend and video filter API
 - `mediasoup-server/` - Node.js Mediasoup SFU server
 - `audio_processing/` - AI models and scripts for audio privacy

## How We Built It

We structured PrivaStream with a **modular, scalable pipeline** using web technologies:

1. **Capture & Preprocessing**: Video and audio are captured from the user's webcam and microphone, with video resized to 720p for optimal performance.
2. **Detection (sparse)**: Every N frames, the backend detects faces, license plates, and PII text using Python ML models.
3. **Tracking (continuous)**: Detected regions are tracked between frames to maintain privacy protection even when detection is not run every frame.
4. **Mask Composer**: Client-side (browser) GPU acceleration applies irreversible Gaussian blur or pixelation to sensitive regions using WebGL shaders.
5. **Retroactive Buffer**: The frontend buffers several seconds of video frames to allow retroactive masking if a detection is delayed.
6. **Encoding & Streaming**: Video and audio are streamed to viewers using Mediasoup SFU (WebRTC), supporting hundreds of concurrent viewers.

**Development stack**: Python (Flask, OpenCV, NumPy) for backend and ML, JavaScript/TypeScript (Next.js, React, mediasoup-client, WebGL) for frontend, Node.js for Mediasoup server. All AI/ML runs on Python backend, with GPU acceleration for blurring in the browser.

**Models used**:

* Custom face detection (OpenCV, DNN)
* License plate detection (OpenCV, DNN)
* PII text detection (DB/EAST or similar, OpenCV)
* Whisper (OpenAI) for audio PII detection

---
## How It Works
1. Host starts a room and begins streaming.
2. Video and audio are processed in real-time for privacy protection.
3. Viewers join the room and watch the privacy-protected stream.
4. All processing is scalable and low-latency, suitable for large audiences.
---

## How It Works
1. Host starts a room and begins streaming.
2. Video and audio are processed in real-time for privacy protection.
3. Viewers join the room and watch the privacy-protected stream.
4. All processing is scalable and low-latency, suitable for large audiences.

---

## License
MIT License