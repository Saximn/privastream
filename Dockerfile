FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY setup.py .
COPY README.md .
COPY LICENSE .
COPY src/web/backend/requirements.txt ./src/web/backend/requirements.txt

RUN pip install --no-cache-dir -r src/web/backend/requirements.txt

COPY src/ ./src/
COPY main.py .

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data /app/models /app/logs

ENV PYTHONPATH="/app"
ENV FLASK_ENV="production"
ENV SOCKETIO_ASYNC_MODE="eventlet"

EXPOSE 5000 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()"

CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "src.web.backend.wsgi_backend:app"]
