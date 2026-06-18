# Contributing to PrivaStream

Thank you for your interest in contributing to the Live Privacy Filter project! This document provides the necessary details for developers, including the project architecture, setup instructions, and testing procedures.

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

## 🔧 Configuration (example `config.yaml`)

```yaml
io:
  source: 0 # camera index or "path/to/video.mp4"
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
      ocr: doctr # or easyocr
      clf_path: models/pii_clf.joblib
      conf_gate: 0.35

  stabilization:
    k_confirm: 2
    k_hold: 8
    iou_thresh: 0.3

  blur:
    type: gaussian # gaussian | mosaic
    ksize: 41
    pad: 4

audio:
  whisper_model: small # tiny | base | small | medium | large-v3
  chunk_seconds: 5
  deberta_model: models/deberta-pii/ # local or HF path
  mouth_blur_window_ms: 500
```

## 🚀 Development Setup

```bash
# Clone repository
git clone https://github.com/Saximn/privastream.git
cd privastream

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

## 🔒 Branch Protection

The `main` branch is protected and must always remain runnable. All changes land
through pull requests that must pass the automated CI checks (`Lint (flake8)` and
`Tests (pytest)`) before they can be merged. See
[`.github/BRANCH_PROTECTION.md`](.github/BRANCH_PROTECTION.md) for the full set of
required branch protection settings.

## 🗺️ Roadmap

- [ ] Multi-language street/address lexicons
- [ ] ONNX export paths for mobile/embedded deployment
- [ ] Virtual camera output for OBS/VT Cam
- [ ] Per-user ignore lists and manual blur management UI

## 🏷️ Versioning & Releases

PrivaStream follows **[Semantic Versioning](https://semver.org/)** (`MAJOR.MINOR.PATCH`).

### How releases are created

Releases are **tag-driven** — pushing a semver tag to `main` automatically triggers
the release pipeline (`.github/workflows/release.yml`):

```bash
# Ensure you are on main and your working tree is clean
git checkout main
git pull

# Create an annotated tag
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

The pipeline then:
1. **Creates a GitHub Release** with auto-generated release notes derived from
   merged pull-request titles and commits since the previous tag.
2. **Builds the backend Docker image** and pushes it to the
   [GitHub Container Registry (GHCR)](https://ghcr.io/saximn/privastream) with
   the following tags:
   | Tag | Example | Description |
   |-----|---------|-------------|
   | Full semver | `v1.2.3` | Exact release version |
   | Minor semver | `v1.2` | Latest patch for this minor |
   | Major semver | `v1` | Latest minor for this major |
   | `latest` | `latest` | Most recent stable release |
   | Commit SHA | `sha-abc1234` | Immutable reference for debugging |

### Version bump guidelines

| Change type | Version component to bump | Example |
|-------------|--------------------------|---------|
| Breaking API change | `MAJOR` | `v1.0.0` → `v2.0.0` |
| New backwards-compatible feature | `MINOR` | `v1.0.0` → `v1.1.0` |
| Bug fix / patch | `PATCH` | `v1.0.0` → `v1.0.1` |

### Commit message conventions

We recommend following
[Conventional Commits](https://www.conventionalcommits.org/) — this ensures
auto-generated release notes are clean and grouped:

```
feat: add license plate whitelist support
fix: resolve memory leak in video pipeline
docs: update API reference for /process endpoint
chore: bump dependency versions
```

Conventional commit prefixes like `feat!:` or `BREAKING CHANGE:` in the body
signal a `MAJOR` bump.

### Updating the changelog

After tagging, update [`CHANGELOG.md`](CHANGELOG.md):

1. Move the `[Unreleased]` section content under a new `[X.Y.Z] - YYYY-MM-DD`
   heading.
2. Open a new empty `[Unreleased]` section at the top.
3. Commit directly to `main` (no PR required for changelog-only commits).

## 📞 Support & Community

- **GitHub Issues**: [Report bugs and request features](https://github.com/Saximn/tiktok-techjam-2025/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/Saximn/tiktok-techjam-2025/discussions)
- **Email**: privastream.opensource@gmail.com for other inquiries.
