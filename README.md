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

## System Requirements

### Hardware Requirements
- **Minimum RAM**: 16GB (32GB+ recommended for audio processing)
- **GPU**: NVIDIA A100 GPU
- **Storage**: 10GB+ free space for models and dependencies
- **CPU**: Multi-core processor (Intel i9/AMD Ryzen 7 or better)

### Software Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **Node.js**: 18+ (for web demo)
- **CUDA**: 12.1+ (for GPU acceleration)
- **Operating Systems**: Windows 10/11, Ubuntu 20.04+, macOS 12+

## Quick Start

### 1) Environment Setup

**Python Environment:**
```bash
# Create virtual environment (recommended)
python -m venv privastream
source privastream/bin/activate  # Linux/Mac
# or
privastream\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt
```

**For Web Demo (Node.js components):**
```bash
# Install frontend dependencies
cd web-demo-ui/frontend
npm install

# Install mediasoup server dependencies
cd ../mediasoup-server
npm install

# Install backend dependencies
cd ../backend
pip install -r requirements.txt
```

### 2) Model Setup

**Download Pre-trained Models:**
- Download models from [releases](https://github.com/your-repo/releases) or train your own
- Place models in the `models/` directory:
  - `models/face_best.pt` - Face detection model (~50MB)
  - `models/best.pt` - License plate detection model (~15MB)
  - `models/pii_clf.joblib` - Text PII classifier (~5MB)

**Audio PII Models** (for audio processing):
- DeBERTa models are automatically downloaded on first run
- Or place custom trained models in `audio-processing/models/`

### 3) Basic Usage

**Live video processing:**
```bash
python scripts/run_live.py --mode live --source 0 --show-boxes
```

**Process video file:**
```bash
python scripts/run_live.py --mode video --source data/samples/demo.mp4 --out outputs/blurred.mp4 --show-boxes
```

**License plate only:**
```bash
python scripts/plate_blur.py --mode live --source 0 --weights models/best.pt --show-boxes
```

**Audio PII detection:**
```bash
python start_audio_redaction.py --input audio_sample.wav --output processed_audio.wav
```

### 4) Web Demo

**Start all services:**
```bash
# Terminal 1: Start backend API (Flask)
cd web-demo-ui/backend
python app.py
# Runs on http://localhost:5000

# Terminal 2: Start mediasoup server
cd web-demo-ui/mediasoup-server
npm run dev
# Runs on http://localhost:3001

# Terminal 3: Start frontend
cd web-demo-ui/frontend
npm run dev
# Runs on http://localhost:3000
```

**Access the demo:** Open http://localhost:3000

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
[Web Demo Folder](web-demo-ui/)


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

## 🏆 Audio PII Model Training

The solution incorporates **five DeBERTa-v3-large models** with different architectures for enhanced diversity and performance:

**Multi-Sample Dropout Model** (improves training stability):
```bash
cd audio-processing
python train_multidropout.py
```

**BiLSTM Layer Model** (enhanced feature extraction):
```bash
python train_bilstm.py
```

**Knowledge Distillation** (requires a teacher model):
```bash
python train_distil.py
```

**Experiment 073** (augmented data with name swaps):
```bash
python train_exp073.py
```

**Experiment 076** (random consequential names addition):
```bash
python train_exp076.py
```

**For detailed methodology and inference procedures**, see our [Methodology](https://www.github.com/Saximn/tiktok-techjam-2025/main/audio-processing.md) and [Inference Notebook](https://www.github.com/Saximn/tiktok-techjam-2025/main/pii-inference.ipynb).

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

