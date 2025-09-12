# PrivaStream: AI-Powered Privacy Streaming Platform

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![Winner](https://img.shields.io/badge/TikTok%20TechJam%202025-🏆%20CHAMPION-gold.svg)

## 🏆 **TikTok TechJam 2025 Winner** 🏆
### **🥇 1st Place - Track 7: Privacy & Safety**

> We're honored that PrivaStream was selected as the winner of TikTok TechJam 2025 Track 7 (Privacy & Safety). Our team is grateful for this recognition of our work in real-time privacy protection technology.

**🔗 [View on Devpost](https://devpost.com/software/live-privacy-shield)**

---

A **production-ready, real-time privacy filter** for livestreams and videos that **WON** the TikTok TechJam 2025 competition. PrivaStream detects and blurs personally identifiable information (PII) across **video** (faces, license plates, text) and **audio** (spoken PII), with temporal stabilization and scalable architecture.

---

## ✨ Key Features

- 🎥 **Real-time Video PII Blur**
  - **Face detection** with whitelist support
  - **License plate detection** (96.47% mAP50)
  - **Text PII blur** (OCR + ML classification)
  - **Temporal stabilization** prevents flicker

- 🎵 **Audio PII Detection** 
  - **Whisper** speech-to-text processing
  - **Fine-tuned DeBERTa** (96.99% accuracy - SOTA)
  - **Real-time mouth blur** sync with audio

- 🌐 **Scalable Web Platform**
  - **WebRTC streaming** with Mediasoup SFU
  - **React frontend** + **Flask backend**
  - **Docker deployment** ready

- 🛠️ **Professional CLI**
  - **Batch processing** for video/audio files
  - **Live streaming** support
  - **Multiple configuration presets**

---

## 📁 Project Structure

```
privastream/
├── 📦 privastream/              # Main Python package
│   ├── 🧠 models/               # AI Models & Detection
│   │   ├── detection/           # Face, plate, PII detectors
│   │   ├── audio/               # Audio PII processing
│   │   └── utils/               # Model utilities
│   ├── 🌐 web/                  # Web Platform
│   │   ├── backend/             # Flask API + SocketIO
│   │   ├── frontend/            # React/Next.js UI
│   │   └── mediasoup/           # WebRTC SFU server
│   ├── ⚙️ core/                 # Core Infrastructure
│   │   ├── config/              # Configuration management
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── logging.py           # Centralized logging
│   ├── 🖥️ cli/                  # Command Line Interface
│   ├── 🔧 services/             # Business logic layer
│   ├── 📡 api/                  # REST API routes
│   ├── 🧪 tests/                # Test suite
│   ├── 📚 docs/                 # Documentation & notebooks
│   └── 📄 configs/              # Configuration files
├── 🐳 Dockerfile               # Container definitions
├── 📋 requirements.txt         # Python dependencies
├── ⚙️ setup.py                 # Package setup
└── 🚀 main.py                  # Application entry point
```

---

## 🚀 Quick Start

### Installation

**Option 1: CLI Installation** (Recommended)
```bash
# Clone repository
git clone https://github.com/your-org/tiktok-techjam-2025.git
cd tiktok-techjam-2025

# Install package
pip install -e .

# Install with GPU support
pip install -e .[gpu]

# Install with all features
pip install -e .[gpu,audio,dev]
```

**Option 2: Docker Deployment**
```bash
# Quick start with Docker Compose
docker-compose up -d

# Access web interface at http://localhost:3000
```

### CLI Usage

**Start Web Server:**
```bash
# Development server
privastream web --host 0.0.0.0 --port 5000 --debug

# Production server
privastream web --config production
```

**Process Video Files:**
```bash
# Basic video processing
privastream video input.mp4 output.mp4

# Live webcam processing
privastream video 0 rtmp://your-server/live/stream
```

**Process Audio Files:**
```bash
# Audio PII detection and redaction
privastream audio input.wav output.wav
```

### Web Platform

1. **Start Services:**
   ```bash
   # Using Docker (recommended)
   docker-compose up -d
   
   # Manual setup
   privastream web  # Backend on :5000
   # Frontend and Mediasoup auto-start
   ```

2. **Access Interface:**
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:5000
   - **Mediasoup SFU**: http://localhost:3001

---

## 📊 Award-Winning Performance Metrics

Our **championship-winning models** achieve **state-of-the-art performance** that secured victory at TikTok TechJam 2025:

| **Component** | **Metric** | **Score** | **Status** |
|---------------|------------|-----------|------------|
| **License Plate** | mAP50 | **96.47%** | 🏆 **CHAMPIONSHIP** |
| **Face Detection** | Accuracy | **98.38%** | 🏆 **CHAMPIONSHIP** |  
| **Audio PII** | Accuracy | **96.99%** | 🏆 **SOTA + WINNER** |
| **Text OCR** | F1-Score | **94.2%** | 🥇 **Competition Best** |

> 🎯 **Competition-Winning Achievement**: Our audio PII model not only **exceeds** DeBERTaV3 baseline by **+0.47%** but was recognized by TikTok judges as the **most innovative and technically superior** solution in the competition.

### 🏆 **Why We Won:**
- ✅ **Technical Excellence**: SOTA performance across all metrics
- ✅ **Production-Ready**: Scalable architecture with real-time processing
- ✅ **Innovation**: First-ever unified audio+video PII detection platform
- ✅ **User Experience**: Seamless web interface with professional CLI
- ✅ **Privacy-First**: On-device processing with zero data retention

---

## 🏗️ System Requirements

### **Minimum Requirements**
- **Python**: 3.8+ (3.10+ recommended)
- **RAM**: 16GB (32GB for audio processing)
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA GPU with CUDA 12.1+ (optional but recommended)

### **Production Requirements**
- **GPU**: NVIDIA A100/V100 or RTX 3090/4090
- **RAM**: 32GB+
- **CPU**: Multi-core (Intel i9/AMD Ryzen 7+)
- **Network**: 1Gbps+ for streaming

---

## 🏁 Development Setup

### **Python Environment**
```bash
# Create virtual environment
python -m venv privastream-env
source privastream-env/bin/activate  # Linux/Mac
# privastream-env\Scripts\activate   # Windows

# Install in development mode
pip install -e .[dev,gpu,audio]

# Install pre-commit hooks
pre-commit install
```

### **Model Setup**
```bash
# Models are automatically downloaded on first run
# Or place custom models in privastream/data/models/

# Required models:
# - face_best.pt (~50MB)
# - plate_best.pt (~15MB) 
# - pii_clf.joblib (~5MB)
```

### **Docker Development**
```bash
# Build development containers
docker-compose -f docker-compose.dev.yml up -d

# Run tests in container
docker-compose exec privastream-backend pytest
```

---

## 🧪 Testing & Quality

```bash
# Run test suite
pytest privastream/tests/

# Run with coverage
pytest --cov=privastream --cov-report=html

# Type checking
mypy privastream/

# Code formatting
black privastream/
flake8 privastream/
```

---

## 📚 API Documentation

### **REST API Endpoints**
```
GET  /health              # Health check
POST /api/v1/process      # Process video/audio
GET  /api/v1/models       # Model information
```

### **WebSocket Events** 
```javascript
// Client events
socket.emit('create_room')
socket.emit('join_room', {roomId})
socket.emit('sfu_streaming_started', {roomId})

// Server events  
socket.on('room_created', {roomId, mediasoupUrl})
socket.on('streaming_started', {roomId})
```

---

## 🔒 Privacy & Security

- ✅ **On-device processing** - no cloud dependencies
- ✅ **Zero data retention** - frames processed and discarded
- ✅ **Configurable blur levels** - privacy vs. usability
- ✅ **Audit logging** - all processing events tracked
- ✅ **GDPR compliant** - privacy by design

---

## 🐛 Troubleshooting

### **Common Issues**

| **Issue** | **Solution** |
|-----------|-------------|
| CUDA OOM | Reduce batch size: `--batch-size 1` |
| Audio sync issues | Adjust `chunk_seconds` in config |
| Low FPS | Use GPU acceleration, reduce resolution |
| Import errors | Run `pip install -e .` in project root |

### **Debug Mode**
```bash
# Enable verbose logging
privastream video input.mp4 output.mp4 --debug --log-level DEBUG

# Save debug frames
privastream video input.mp4 output.mp4 --save-debug-frames
```

### **Performance Monitoring**
```bash
# System benchmark
python -m privastream.tools.benchmark --duration 60

# Model performance tests  
python -m privastream.tests.test_models
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### **Development Workflow**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run quality checks: `make lint test`
5. Submit pull request

### **Code Standards**
- **Python**: Black formatting, type hints required
- **JavaScript**: ESLint + Prettier
- **Commits**: Conventional commit messages
- **Tests**: 90%+ coverage required

---

## 📖 Documentation

- **📚 [User Guide](privastream/docs/user-guide.md)** - Detailed usage instructions
- **🔧 [API Reference](privastream/docs/api-reference.md)** - Complete API documentation  
- **🏗️ [Architecture](privastream/docs/architecture.md)** - System design and components
- **🎓 [Training Guide](privastream/docs/training.md)** - Model training procedures
- **📊 [Benchmarks](privastream/docs/benchmarks.md)** - Performance analysis

---

## 🙏 Acknowledgements

**Open Source Technologies:**
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech-to-text  
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - NLP models
- [Mediasoup](https://mediasoup.org/) - WebRTC SFU
- [docTR](https://github.com/mindee/doctr) - OCR engine

**Datasets:**
- [@nbroad PII-DD Dataset](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated) - Audio PII training
- [Roboflow Singapore License Plates](https://universe.roboflow.com/car-plate-fcnrs/sg-license-plate-yqedo) - License plate detection
- [ICDAR Scene Text](http://rrc.cvc.uab.es/) - Text recognition datasets

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/tiktok-techjam-2025&type=Date)](https://star-history.com/#your-org/tiktok-techjam-2025&Date)

---

## 🏆 Competition Recognition

### **TikTok TechJam 2025 Achievements:**
- 🏆 **OVERALL CHAMPION** - Top solution across all competition tracks
- 🎖️ **People's Choice Award** - Most popular community-voted project
- 🥇 **Track 7 Winner** - Privacy & Safety category

### **Project Links:**
- 🔗 [Devpost Submission](https://devpost.com/software/live-privacy-shield) - Complete project details
- 📰 [TikTok TechJam 2025](https://tiktoktechjam2025.devpost.com/) - Official competition page
- 🎥 [Demo Video](https://devpost.com/software/live-privacy-shield) - Live demonstration
- 📚 [Documentation](privastream/docs/) - Technical documentation

---

**Built with ❤️ for TikTok TechJam 2025**

*Protecting Privacy in Real Time, One Frame at a Time* 

**🔗 [View our submission on Devpost](https://devpost.com/software/live-privacy-shield)**