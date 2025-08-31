# Contributing to PrivaStream

Thank you for your interest in contributing to the Live Privacy Filter project! This document provides the necessary details for developers, including the project architecture, setup instructions, and testing procedures.

---

## 🧱 High-Level Architecture

```
┌───────────────┐   frames@30fps    ┌──────────────────┐     ┌────────────────────────┐
│ I/O Livestream ├───────────────────►│  Video Scheduler │─────► Unified Video Analyzer │
└───────────────┘                   │   (e.g., 4fps)   │     └──────────┬─────────────┘
                                    └──────────────────┘              │
                                                                      │ merges
                             ┌────────────────────┐     ┌────────────────────┐          ▼
                             │ Face Model         │     │ License Plate Model│     [Boxes to blur]
(frame_id,frame) │ → [(x1,y1,x2,y2)]  │     │ → [(x1,y1,x2,y2)]  │──────────────► Blur Engine
───────────────► └────────────────────┘     └────────────────────┘
                             ┌────────────────────┐
                             │ PII Text Model     │  OCR + PII rules/ML
                             │ → [(x1,y1,x2,y2)]  │
                             └────────────────────┘
```

**Audio side:**

```
┌────────┐   audio stream    ┌───────────────┐   5s batch    ┌───────────────┐
│  I/O   ├──────────────────►│  Whisper STT  ├──────────────►│  DeBERTa PII  │
└────────┘                   └───────────────┘               └──────┬────────┘
                                                              PII tokens + timestamps
                                               map to frames  ┌─────▼───────────┐
                                               via scheduler  │   Mouth Blur    │
                                                              └──────────────────┘
```

---

## ⚙️ Interfaces & Contracts

All computer-vision models share a **common I/O contract**:
- **Input:** `(frame_id: int, frame: np.ndarray[BGR])`
- **Output:** `(frame_id: int, boxes: List[Tuple[x1,y1,x2,y2]])`
- Coordinates are **pixel-space integers** in the original frame size.

### Video model interface
```python
def infer(frame_id: int, frame_bgr: "np.ndarray") -> "tuple[int, list[tuple[int,int,int,int]]]":
    """
    Returns (same frame_id, list of boxes) where boxes are (x1, y1, x2, y2) in pixels.
    """
```

---

## 🛠️ Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | CUDA-capable (4GB VRAM) | RTX 3070+ (8GB+ VRAM) |
| **RAM** | 8GB | 16GB+ |
| **CPU** | 4 cores | 8+ cores |
| **Storage** | 5GB free space | 10GB+ SSD |

### Software Dependencies

- Python 3.10+
- CUDA-capable GPU recommended (PyTorch with CUDA)
- Core libraries: See `requirements.txt` for a full list (`ultralytics`, `torch`, `opencv-python`, `python-doctr[torch]`, `openai-whisper`, `transformers`).

---

## 🔧 Configuration (example `config.yaml`)

```yaml
io:
  source: 0                 # camera index or "path/to/video.mp4"
  out: outputs/blurred.mp4
  write_output: true

scheduler:
  input_fps: 30
  target_fps: 4

video:
  models:
    face:
      enabled: true
      weights: models/face_best.pt
      conf: 0.4
    license:
      enabled: true
      weights: models/best.pt
      conf: 0.25
    pii_text:
      enabled: true
      ocr: doctr              # or easyocr
      clf_path: models/pii_clf.joblib
      conf_gate: 0.35

  stabilization:
    k_confirm: 2
    k_hold: 8
    iou_thresh: 0.3

  blur:
    type: gaussian            # gaussian | mosaic
    ksize: 41
    pad: 4

audio:
  whisper_model: small        # tiny | base | small | medium | large-v3
  chunk_seconds: 5
  deberta_model: models/deberta-pii/   # local or HF path
  mouth_blur_window_ms: 500
```

---

## 🚀 Development Setup

```bash
# Clone repository
git clone [https://github.com/Saximn/tiktok-techjam-2025.git](https://github.com/Saximn/tiktok-techjam-2025.git)
cd tiktok-techjam-2025

# Create development environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (if available)
pre-commit install
```

### Code Style

- **Python**: Black, isort, flake8
- **Type Hints**: Required for all public APIs
- **Documentation**: Google-style docstrings

---

## 🧪 Testing

The project aims for >90% test coverage.

### Running Tests

```bash
# Run component-specific tests
python models/plate_blur/test_plate_detector.py
python models/face_blur/test_face_detector.py
python audio-processing/test_setup.py

# Full integration tests
python models/unified_bbox_test.py
python scripts/test_pipeline.py

# Performance benchmarks
python scripts/benchmark.py --config configs/balanced.yaml
```

### Evaluation Strategy
- **Video**: Measure PII-F1 on a labeled set of frames (faces/plates/text).
- **Audio**: Measure precision/recall of PII-tagged tokens per 5s window.
- **Latency**: Measure FPS (median, p95) and time breakdown per component.

---

## 🗺️ Roadmap

- [ ] Multi-language street/address lexicons
- [ ] ONNX export paths for mobile/embedded deployment
- [ ] Virtual camera output for OBS/VT Cam
- [ ] Per-user ignore lists and manual blur management UI

---

## 📞 Support & Community

- **GitHub Issues**: [Report bugs and request features](https://github.com/Saximn/tiktok-techjam-2025/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/Saximn/tiktok-techjam-2025/discussions)
- **Email**: Contact the development team for other inquiries.
