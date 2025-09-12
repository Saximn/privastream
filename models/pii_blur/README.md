# PII Blur System

Real-time personally identifiable information (PII) detection and blurring system for video streams and street imagery.

## 🚀 Features

- **Real-time Processing**: Live webcam or video file processing
- **Smart Detection**: Uses OCR + ML classifier to identify street addresses, postal codes, and unit numbers
- **Temporal Stability**: Hysteresis tracking prevents flickering blur regions
- **Configurable**: Adjustable confidence thresholds and blur parameters

## 📋 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- docTR or EasyOCR
- scikit-learn

## 🎯 Usage

### Live Webcam Processing

```bash
python run_live.py --mode live --source 0 --show-boxes \
  --classifier pii_clf.joblib --conf-thresh 0.35 --k-confirm 2 --k-hold 8 --ksize 41
```

### Video File Processing

```bash
python run_live.py --mode video --source input_video.mp4 --output blurred_output.mp4 \
  --classifier pii_clf.joblib --conf-thresh 0.35 --k-confirm 2 --k-hold 8 --ksize 41
```

## ⚙️ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source` | Video source (0 for webcam, path for file) | `0` |
| `--classifier` | Path to trained PII classifier | `pii_clf.joblib` |
| `--conf-thresh` | OCR confidence threshold (0.0-1.0) | `0.35` |
| `--k-confirm` | Frames needed to confirm PII detection | `2` |
| `--k-hold` | Frames to hold blur after last detection | `8` |
| `--ksize` | Gaussian blur kernel size (odd number) | `41` |
| `--show-boxes` | Display detection bounding boxes | `False` |

## 🧠 How It Works

1. **OCR Detection**: Uses docTR/EasyOCR to extract text from video frames
2. **PII Classification**: ML classifier identifies addresses, postal codes, unit numbers
3. **Temporal Tracking**: Hysteresis system ensures stable blur regions across frames  
4. **Gaussian Blur**: Detected PII regions are blurred to protect privacy

## 📊 Supported PII Types

- Street addresses (e.g., "123 Main Street")
- Unit/apartment numbers (e.g., "#12-345")
- Postal codes (e.g., "S123456")
- Building blocks (e.g., "Blk 123A")

## 🎮 Controls

- **ESC**: Exit live processing
- **Space**: Pause/resume (live mode)

## 📝 Example Output

The system will blur detected PII text while preserving other content:
- ✅ Street names, addresses, postal codes → **Blurred**
- ❌ Store names, general text → **Not blurred**

---

*Built for TikTok TechJam 2025 - Privacy-First Street Imagery*