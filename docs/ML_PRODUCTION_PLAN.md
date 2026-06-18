# ML production roadmap

This document captures the productionization targets for Privastream's real-time ML stack.

## Latency budgets

| Path | Target p95 | Hard fail-safe |
| --- | ---: | --- |
| Video face/plate detection | 75 ms per sampled frame | Reuse tracked boxes, then full-frame blur if tracker confidence expires |
| Video blur/compositing | 16 ms per frame at 30 FPS | Drop or full-frame blur rather than emit unredacted video |
| Audio ASR partial result | 1.5 s after speech | Hold/delay stream segment until policy resolves |
| Text PII recognition | 50 ms per transcript chunk | Redact deterministic recognizer matches even if neural model times out |

## Production model profiles

| Profile | Use case | ASR | Text PII | Vision |
| --- | --- | --- | --- | --- |
| `low_latency` | High concurrency live rooms | faster-whisper small/distil, int8/float16 | deterministic recognizers + compact token classifier | TensorRT FP16, no TTA, tracker between detections |
| `balanced` | Default production | faster-whisper medium, float16 | deterministic recognizers + DeBERTa token classifier | TensorRT FP16, low-light enhancement only when needed |
| `quality` | Offline review or premium streams | faster-whisper large-v3, float16 | deterministic recognizers + large DeBERTa | TensorRT/ONNX with periodic high-recall TTA |

## Serving architecture

1. Keep Mediasoup focused on media routing and do not call Python with base64 frames on the hot path.
2. Run ASR, text PII, and vision detectors as independently scalable inference workers.
3. Prefer NVIDIA Triton or an equivalent model-serving layer for dynamic batching, multiple model instances, warmup, and model versioning.
4. Export YOLO-family models to ONNX/TensorRT for production GPUs; keep PyTorch weights as development fallback.
5. Track every stream with queue depth, dropped-frame count, p50/p95/p99 latency, GPU memory, and GPU utilization.

## Privacy-first degradation policy

When latency or capacity budgets are exceeded, the system must degrade toward more privacy, not less:

1. Use cached/tracked boxes for short gaps.
2. Reduce detection FPS only if tracker confidence remains above threshold.
3. Switch to smaller model profiles under load.
4. Full-frame blur if model output is stale.
5. Drop frames or reject new streams before sending unredacted content.

## Release gates

A production release should publish an evaluation report containing:

- Audio PII recall, precision, F1, and missed-redaction examples.
- Video face/plate recall and false-redaction rate.
- End-to-end p50/p95/p99 latency by profile and hardware.
- Stress-test results for concurrent stream counts and overload policies.
- Model versions, dataset versions, calibration thresholds, and commit SHA.
