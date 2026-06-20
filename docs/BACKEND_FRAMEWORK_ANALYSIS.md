# Backend Framework Analysis

## Recommendation: keep Python for ML-serving

Do not rewrite the ML-serving backend in Rust, C#, or Go. PrivaStream's backend latency is dominated by GPU model inference and video/audio serialization, not by Flask route dispatch. A framework or language rewrite would add substantial delivery risk while preserving the same expensive model calls.

The practical path is incremental hardening of the current architecture: run the Flask services with production WSGI/Socket.IO workers, keep model code in Python where the ML ecosystem is strongest, and move only narrowly scoped infrastructure concerns to other services when they have a clear operational payoff.

## Current backend stack

PrivaStream currently has three backend-facing runtime pieces:

1. `src/web/backend/app.py` is the room orchestration service. It uses Flask, Flask-SocketIO, and Flask-CORS to create rooms, issue room tokens, track viewers, and coordinate Mediasoup stream state. This service is I/O-heavy and lightweight.
2. `src/web/backend/video_filter_api.py` is the heavy video filtering service. It accepts base64-encoded frames, decodes them with OpenCV/Numpy, calls `UnifiedBlurDetector`, optionally enrolls faces with InsightFace, and returns processed frames plus metadata.
3. `src/web/mediasoup` is the Node.js Mediasoup SFU used for WebRTC transport. It is already a separate service boundary and should remain separate from ML inference.

## Performance characteristics

The expensive work happens after a request reaches the video filtering service:

- YOLO face and license-plate detection run through the Python ML stack.
- InsightFace face embeddings are used for enrollment and whitelisting.
- Text and audio privacy detection depend on Python-native model/runtime integrations such as OpenCV, PyTorch, HuggingFace Transformers, Faster Whisper, and DeBERTa-family classifiers.
- Frame decode/encode and base64 payload handling are usually more meaningful than Flask's request routing overhead for this endpoint shape.

Because the critical path is GPU inference plus image/audio processing, replacing Flask with Rust, C#, or Go would not remove the dominant cost. It would mostly move HTTP routing overhead that is small relative to model execution.

## Migration option evaluation

| Option | Fit | Expected performance gain | Primary risk |
| --- | --- | --- | --- |
| Rust rewrite | Poor for ML-serving | Very low unless the ML stack is also replaced | Rebuilding or bridging Python/PyTorch/Ultralytics/InsightFace code through FFI adds complexity and operational risk. |
| C# rewrite | Poor for ML-serving | Very low unless the ML stack is also replaced | Model support and deployment ergonomics are weaker than Python for the current libraries. |
| Go rewrite | Poor for ML-serving | Very low unless the ML stack is also replaced | Good networking stack, but the service would still need to call Python or exported model runtimes for core inference. |
| FastAPI for room/API edges | Reasonable incremental option | Low to moderate for I/O endpoints, low for GPU inference endpoints | Requires careful Socket.IO/WebSocket behavior parity and deployment updates. |
| Keep Flask plus production server/process model | Best near-term option | Moderate operational reliability gain, low latency gain | Requires deployment discipline rather than a large code rewrite. |

## Highest-ROI improvements

1. Run production processes under gunicorn/compatible Socket.IO workers instead of Flask's development server.
2. Keep ML inference in Python and optimize model artifacts first: export YOLO-family models to ONNX/TensorRT where supported, keeping PyTorch weights as the development fallback.
3. Scale the video filter service by process and GPU ownership rather than Python threads when GPU capacity is the real limiting resource.
4. Reduce payload overhead before changing language: avoid unnecessary JPEG/base64 round trips where the client/server protocol allows binary frames.
5. Consider FastAPI only for new or clearly I/O-bound HTTP endpoints. Treat it as an incremental service boundary improvement, not as a rewrite of the detector pipeline.
6. Use a queue such as Celery or RQ for batch/offline workloads so long-running jobs do not occupy request threads.

## Decision

Keep the current Python backend for ML-serving. Do not migrate `src/web/backend/video_filter_api.py` to Rust, C#, or Go. If framework modernization is needed, start with a small FastAPI proof of concept around non-ML I/O endpoints or a new gateway service, and measure it against production-like traffic before committing to a broader migration.

This preserves access to the Python ML ecosystem that PrivaStream depends on while focusing engineering effort on the parts most likely to improve local and online runtime behavior.
