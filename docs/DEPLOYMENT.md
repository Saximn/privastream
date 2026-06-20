# PrivaStream Deployment

This deployment keeps Python as the ML-serving runtime and runs Flask services behind production process managers. Mediasoup remains the Node.js SFU boundary.

## Local Setup

Create a repository-local virtual environment before running Python commands:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r src\web\backend\requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/web/backend/requirements.txt
```

Frontend and SFU dependencies:

```bash
npm --prefix src/web/frontend ci
npm --prefix src/web/mediasoup ci
```

## Required Environment

Production must set real values for:

- `SECRET_KEY`
- `MEDIASOUP_ANNOUNCED_IP`
- `CORS_ALLOWED_ORIGINS`

Common service URLs:

- `BACKEND_URL=http://backend:5000`
- `VIDEO_API_URL=http://video-filter-api:5001`
- `SFU_URL=http://mediasoup:3001`
- `AUDIO_API_URL=http://audio-api:5002`

Frontend public URLs:

- `NEXT_PUBLIC_BACKEND_URL`
- `NEXT_PUBLIC_VIDEO_API_URL`
- `NEXT_PUBLIC_SFU_URL`
- `NEXT_PUBLIC_AUDIO_API_URL`

Security switches:

- `REQUIRE_AUTH=true` for production
- `FLASK_ENV=production`
- `NODE_ENV=production`

## Production Processes

Room backend:

```bash
SOCKETIO_ASYNC_MODE=eventlet \
gunicorn -k eventlet -w 1 --bind 0.0.0.0:5000 \
  src.web.backend.wsgi_backend:app
```

Video filter API:

```bash
gunicorn --workers "${VIDEO_API_WORKERS:-1}" \
  --threads "${VIDEO_API_THREADS:-8}" \
  --timeout 120 \
  --bind 0.0.0.0:5001 \
  src.web.backend.wsgi_video_filter:app
```

Use one video API worker per GPU ownership unit. Threads handle request concurrency around a single resident model instance.

Mediasoup:

```bash
npm --prefix src/web/mediasoup start
```

Frontend:

```bash
npm --prefix src/web/frontend run build
npm --prefix src/web/frontend start
```

## Docker Compose

`docker-compose.yml` defines:

- `backend`
- `video-filter-api`
- `mediasoup`
- `frontend`

Before starting compose, create `.env` from `.env.example` and set at least:

```env
SECRET_KEY=<random 32+ byte secret>
MEDIASOUP_ANNOUNCED_IP=<public WebRTC-reachable IP>
REQUIRE_AUTH=true
CORS_ALLOWED_ORIGINS=https://your-frontend.example
```

Start:

```bash
docker compose up --build
```

Validate config without starting services:

```bash
SECRET_KEY=test MEDIASOUP_ANNOUNCED_IP=127.0.0.1 docker compose config --quiet
```

## Health Checks

- Room backend: `GET http://localhost:5000/health`
- Video filter API: `GET http://localhost:5001/health`
- Mediasoup: `GET http://localhost:3001/health`

Compose health checks use these endpoints.

## Model Artifacts

Python remains the detector runtime. Use exported ONNX or TensorRT artifacts where available, with PyTorch weights as fallback. The plate detector checks these environment variables before falling back to `.pt` weights:

- `PLATE_DETECTOR_ENGINE_PATH`
- `PLATE_DETECTOR_ONNX_PATH`
- `PLATE_DETECTOR_EXPORTED_MODEL`

## Operational Notes

- Do not use Flask development servers in production.
- Do not default `SECRET_KEY` in production.
- Keep `CORS_ALLOWED_ORIGINS` explicit; do not use wildcard origins for production.
- The JSON/base64 video routes remain for compatibility. Prefer WebRTC media transport with Python returning detection coordinates, and use `/process-frame-binary` for incremental binary payload work.
- Debug and per-frame logs should stay at debug level in production.
