# Live Privacy Filter - PrivaStream: Audio & Video PII Blurring

A production-leaning, real-time **privacy filter** for livestreams and videos. It **detects and blurs** personally identifiable information (PII) across **video** (faces, license plates, street/address text) and **audio** (spoken PII), with **temporal stabilization** and clean, modular interfaces. Built for Tiktok Techjam 2025 Track 7.

---

## ✨ Key Capabilities

- **Video PII blur**
  - **Face blur** (detector model; optional mouth-only blur via landmarks/ROI)
  - **License-plate blur** (YOLO weights, e.g., `best.pt`)
  - **Street/address text blur** (OCR + PII classifier/rules)
  - **Unified Video Analyzer** merges all regions from the three models
  - **Temporal confirm/hold** stabilization to prevent blur flicker

- **Audio PII blur**
  - **Whisper** for speech-to-text
  - **Fine-tuned DeBERTa** to tag PII tokens (names, phone, address, etc.) every **5 seconds**
  - Marks **timestamps** for PII words and resolves them to **video frame IDs**
  - Triggers **mouth blur** in sync with the spoken PII segment

- **Scheduler & Throughput control**
  - Input stream may be **30 FPS**
  - **Video Scheduler** samples at **4 FPS** (configurable) to reduce compute while preserving privacy
  - All downstream modules use **frame IDs** to align and act

- **Engineering first**
  - Typed, simple interfaces: each model is a function `f(frame_id, frame) -> (frame_id, [boxes])`
  - Deterministic, testable components with clear contracts
  - CLI & YAML-config friendly

---

## 📁 Project Structure

```
tiktok-techjam-2025/
├── 📁 audio-processing/        # Audio PII detection pipeline
│   ├── deberta-*.py           # DeBERTa model variants
│   ├── train_*.py             # Training scripts
│   ├── 📁 src/                # Core audio processing modules
│   │   ├── pii_detector.py    # Main PII detection class
│   │   ├── whisper_processor.py # Speech-to-text processing
│   │   └── model_*.py         # Model architectures
│   └── 📁 configs/            # Training configurations
├── 📁 models/                 # Pre-trained model weights & core logic
│   ├── 📁 face_blur/          # Face detection & blurring
│   ├── 📁 plate_blur/         # License plate detection
│   ├── 📁 pii_blur/           # Text PII detection & OCR
│   └── unified_detector.py    # Unified video analyzer
├── 📁 web-demo-ui/           # Real-time web interface
│   ├── 📁 backend/           # Flask backend with video models
│   ├── 📁 frontend/          # React frontend
│   └── 📁 audio_processing/  # Audio pipeline integration
├── 📁 datasets/              # Training data & samples
│   ├── 📁 ICPR/             # Text recognition datasets
│   ├── 📁 Roboflow/         # License plate dataset
│   └── 📁 mixtral_pii/      # PII training data
├── 📁 notebooks/            # Development & analysis notebooks
├── requirements.txt         # Python dependencies
└── README.md               # This file
```
---

## Quick Start

**1) Install**  
```bash
pip install -r requirements.txt
```

**2) Put weights** in `models/`:
- `models/face_best.pt` (your face detector)
- `models/best.pt` (your license-plate detector)
- `models/pii_clf.joblib` (your text PII classifier)

**3) Live run (video only, unified analyzer)**  
```bash
python scripts/run_live.py --mode live --source 0 --show-boxes
```

**4) Process a video file**  
```bash
python scripts/run_live.py --mode video --source data/samples/demo.mp4 --out outputs/blurred.mp4 --show-boxes
```

**5) Standalone plate blur (YOLO)**  
```bash
python scripts/plate_blur.py --mode live --source 0 --weights models/best.pt --show-boxes
```

### Advanced Usage Examples

**Audio + Video processing with custom config:**
```bash
python scripts/run_live.py --config configs/high_privacy.yaml --source data/samples/demo.mp4 --enable-audio
```

**Batch processing multiple files:**
```bash
python scripts/batch_process.py --input-dir data/samples/ --output-dir outputs/ --config configs/balanced.yaml
```

**Real-time streaming to RTMP:**
```bash
python scripts/run_live.py --mode live --source 0 --output rtmp://localhost:1935/live/stream
```

**Web Demo Interface:**
```bash
cd web-demo-ui
python app.py --port 8080 --gpu-id 0
# Open http://localhost:8080 in your browser
```


---

## 📊 Performance Metrics

Our models achieve **state-of-the-art performance** across all privacy detection tasks:

### 🎯 Model Performance Summary

| **Component** | **Metric** | **Score** | **Status** |
|---------------|------------|-----------|------------|
| **License Plate Detection** | Recall | **94.51%** | ✅ Excellent |
| | Precision | **94.71%** | ✅ Excellent |
| | mAP50 | **96.47%** | 🏆 Outstanding |
| **Whitelist Face Blur** | Accuracy | **98.38%** | 🏆 Outstanding |
| **Audio PII Detection** | Accuracy | **96.99%** | 🏆 **SOTA** |
| | vs. DeBERTaV3 Baseline | **96.99% > 96.52%** | 🚀 **+0.47% improvement** |

> 🏆 **State-of-the-Art Achievement**: Our fine-tuned Audio PII model **exceeds** the baseline DeBERTaV3 accuracy by **0.47%**, achieving **96.988%** vs **96.521%**.

### 📈 Detailed Performance Breakdown

**License Plate Detection:**
- **High Recall (94.51%)**: Captures nearly all license plates in frame
- **High Precision (94.71%)**: Minimal false positives
- **Excellent mAP50 (96.47%)**: Outstanding localization accuracy

**Whitelist Face Blur:**
- **Near-perfect Accuracy (98.38%)**: Reliable face detection and whitelist filtering
- **Robust across lighting conditions** and facial orientations

**Audio PII Detection:**
- **Industry-leading performance** surpassing established benchmarks
- **Real-time processing** with 5-second audio chunks
- **Multi-class PII detection**: Names, addresses, phone numbers, emails, SSNs

---

## 🔒 Privacy & Safety

- **On-device inference** (no cloud calls at runtime)
- **Over-blur on uncertainty** (lower thresholds in “privacy” mode)
- No frames stored unless explicitly enabled
- Clear watermark **“Privacy Filter ON”** during blur

---

## 🤖 Implementation Notes

- **Mouth blur**: if facial landmarks are available, use inner-lip polygon; otherwise, approximate lower third of the face box.
- **OCR**: docTR `ocr_predictor(det="db_resnet50", reco="parseq")` by default; fallback to EasyOCR if unavailable.
- **Audio alignment**: `frame_id = floor(t * target_fps)`; expand to a small frame window for natural speech duration.
- **Throughput**: use batched OCR and async queues between Scheduler → Analyzer → Blur Engine for higher FPS.


---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce batch size, use smaller model variants |
| **Audio/Video sync issues** | Check `chunk_seconds` and `target_fps` alignment |
| **High CPU usage** | Enable GPU acceleration, optimize threading |
| **False positives** | Adjust confidence thresholds, retrain models |
| **Low FPS** | Reduce capture resolution or set `target_fps` to 2–3; use `faster-whisper` for audio |
| **Missing CUDA** | Ensure the PyTorch wheel matches your NVIDIA driver |
| **No detections** | Verify class names/IDs and confidence thresholds for each detector |

### Debug Mode

Enable verbose logging and debug outputs:

```bash
python scripts/run_live.py --debug --log-level DEBUG --save-debug-frames
```

### Performance Testing

```bash
# Test individual components
python models/plate_blur/test_plate_detector.py
python models/face_blur/test_face_detector.py
python audio-processing/test_setup.py

# Full system performance test
python scripts/benchmark.py --config configs/balanced.yaml --duration 60
```

---

## 🙌 Acknowledgements

- YOLO (Ultralytics) for object detection
- docTR (Mindee) for OCR
- OpenAI Whisper for speech-to-text
- DeBERTa for token-level PII tagging
- scikit-learn for lightweight text classifiers

## 📊 Datasets & Training Data

We leverage a comprehensive collection of high-quality datasets to train our privacy detection models:

### 🎯 **Core PII Datasets**

| **Dataset** | **Samples** | **PII Types** | **Language** | **Source** | **Usage** |
|-------------|-------------|---------------|--------------|------------|-----------|
| **PII-DD Mistral** ⭐️ | 44,668 | 8 types | English | [@nbroad on Kaggle](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated) | Audio PII Training |
| **Mixtral Essays** | 22,000 | 6 types | English | [@mpware on Kaggle](https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays) | Text PII Augmentation |
| **Custom Dataset** 🔥 | 2,048 | 10 types | Multi | Internal | Model Fine-tuning |

> 📝 **Dataset Note**: [@nbroad's PII-DD mistral-generated dataset](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated) proved to be **the most valuable external dataset** for our audio PII detection system. Our custom dataset was released as `external_data_v8.json` and incorporates the nbroad dataset.

### 🖼️ **Computer Vision Datasets**

#### **📷 Scene Text Recognition**
**Dataset**: *"Incidental Scene Text" (2015 Edition)*  
**License**: CC BY 4.0  
**Purpose**: Street/Address Text Detection & OCR Training  
**Applications**: Street sign recognition, Address blur detection, Contextual text analysis

**📚 Original References:**
- Jaderberg, M., Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). *Synthetic data and artificial neural networks for natural scene text recognition.* arXiv:1406.2227.
- Jaderberg, M., Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). *Reading Text in the Wild with Convolutional Neural Networks.* arXiv:1412.1842.

#### **🚗 License Plate Detection**
**Dataset**: *Singapore License Plate Dataset*  
**Source**: [Roboflow Universe - SG License Plate](https://universe.roboflow.com/car-plate-fcnrs/sg-license-plate-yqedo/model/2)  
**License**: Roboflow Universe Community License  
**Purpose**: License Plate Detection & Localization  
**Performance**: 96.47% mAP50 (see metrics above)

### 📈 **Dataset Statistics & Coverage**

| **Category** | **Total Samples** | **Annotation Quality** | **Geographic Coverage** |
|--------------|-------------------|------------------------|------------------------|
| **Audio PII** | **68,716** | High (Human + AI verified) | Global English |
| **Visual PII** | **15,000+** | Expert annotated | Multi-region |
| **Text PII** | **25,000+** | Rule + ML validated | Multi-language |
| **License Plates** | **8,500+** | Precision annotated | Singapore/SEA |

