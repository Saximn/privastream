# PrivaStream Installation Guide

## System Requirements

### Minimum Requirements
- **Python**: 3.8+ (Python 3.10+ recommended)
- **RAM**: 16GB (32GB recommended for audio processing)
- **Storage**: 10GB+ free space for models and dependencies
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+

### Recommended for Production
- **GPU**: NVIDIA GPU with CUDA 12.1+ support
- **RAM**: 32GB+
- **CPU**: Multi-core processor (Intel i9/AMD Ryzen 7+)
- **Network**: 200Mbps+ for streaming applications
- **Storage**: SSD with 50GB+ free space

## Installation Methods

### Method 1: Package Installation (Recommended)

#### Basic Installation
```bash
# Clone the repository
git clone https://github.com/Saximn/privastream.git
cd privastream

# Install the package
pip install -e .
```

#### Full Installation with GPU Support
```bash
# Install with all features
pip install -e .[gpu,audio,dev]
```

#### Verify Installation
```bash
privastream --version
privastream --help
```

### Method 2: Docker Installation

#### Prerequisites
- Docker 20.10+
- Docker Compose 1.29+
- NVIDIA Docker (for GPU support)

#### Quick Start
```bash
# Clone repository
git clone https://github.com/Saximn/privastream.git
cd privastream

# Start all services
docker-compose up -d

# Access web interface at http://localhost:3000
```

#### With GPU Support
```bash
# Use GPU-enabled compose file
docker-compose -f docker-compose.gpu.yml up -d
```

### Method 3: Development Setup

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv privastream-env

# Activate (Linux/Mac)
source privastream-env/bin/activate

# Activate (Windows)
privastream-env\Scripts\activate
```

#### Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PrivaStream in development mode
pip install -e .[dev,gpu,audio]

# Install pre-commit hooks (optional)
pre-commit install
```

## Platform-Specific Instructions

### Windows

#### Prerequisites
```powershell
# Install Python 3.10+
winget install Python.Python.3.10

# Install Git
winget install Git.Git

# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools
```

#### CUDA Setup
1. Download and install [CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads)
2. Add CUDA to PATH:
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
   ```

#### Installation
```powershell
# Clone and install
git clone https://github.com/privastream/tiktok-techjam-2025.git
cd tiktok-techjam-2025
pip install -e .[gpu,audio]
```

### Ubuntu/Debian

#### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and build tools
sudo apt install python3.10 python3.10-venv python3.10-dev
sudo apt install build-essential cmake
sudo apt install git curl
```

#### CUDA Setup
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Install CUDA
sudo apt install cuda-toolkit-12-1
```

#### Installation
```bash
# Clone and install
git clone https://github.com/privastream/tiktok-techjam-2025.git
cd tiktok-techjam-2025

# Create virtual environment
python3.10 -m venv privastream-env
source privastream-env/bin/activate

# Install with GPU support
pip install -e .[gpu,audio]
```

### macOS

#### Prerequisites
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.10 git cmake
```

#### Installation
```bash
# Clone and install
git clone https://github.com/privastream/tiktok-techjam-2025.git
cd tiktok-techjam-2025

# Create virtual environment
python3.10 -m venv privastream-env
source privastream-env/bin/activate

# Install (CPU only on macOS)
pip install -e .[audio,dev]
```

## Model Setup

### Automatic Model Download
Models are automatically downloaded on first run. The following models will be downloaded:

- **Face Detection**: `face_best.pt` (~50MB)
- **License Plate Detection**: `plate_best.pt` (~15MB)
- **PII Classifier**: `pii_clf.joblib` (~5MB)
- **Whisper Models**: Downloaded as needed (varies by size)

### Manual Model Setup
```bash
# Create models directory
mkdir -p ~/.privastream/models

# Download models manually (if needed)
wget -O ~/.privastream/models/face_best.pt [MODEL_URL]
wget -O ~/.privastream/models/plate_best.pt [MODEL_URL]
wget -O ~/.privastream/models/pii_clf.joblib [MODEL_URL]
```

## Configuration

### Environment Variables
Create a `.env` file:
```bash
# Core settings
PRIVASTREAM_LOG_LEVEL=INFO
PRIVASTREAM_CACHE_DIR=~/.privastream/cache

# GPU settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# API settings
FLASK_ENV=production
FLASK_DEBUG=False
```

### Configuration File
Create `~/.privastream/config.yaml`:
```yaml
video:
  models:
    face:
      enabled: true
      confidence: 0.4
    license:
      enabled: true
      confidence: 0.25
    pii_text:
      enabled: true
      ocr: "doctr"
      confidence_gate: 0.35

audio:
  whisper_model: "small"
  chunk_seconds: 5
```

## Verification

### Test Installation
```bash
# Check version
privastream --version

# Run health check
privastream health

# Test with sample video
privastream video test_input.mp4 test_output.mp4
```

### Performance Test
```bash
# Run benchmark
python -m privastream.tools.benchmark --duration 60

# Test GPU availability
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Solution 2: Use smaller models
privastream video input.mp4 output.mp4 --model-size small
```

#### Import Errors
```bash
# Reinstall in development mode
pip install -e .

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

#### Audio Processing Issues
```bash
# Install additional audio dependencies
pip install librosa soundfile

# On Ubuntu, install system libraries
sudo apt install ffmpeg libsndfile1
```

#### Port Conflicts
```bash
# Check what's using port 5000
netstat -tulpn | grep 5000

# Use different ports
privastream web --port 5001
```

### Getting Help

1. **Check logs**: Look in `~/.privastream/logs/`
2. **Run diagnostics**: `privastream diagnose`
3. **Enable debug mode**: `privastream --debug`
4. **Check system requirements**: `privastream system-info`

### Performance Optimization

#### GPU Optimization
```bash
# Set optimal GPU memory fraction
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
```

#### CPU Optimization
```bash
# Set thread limits
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Next Steps

After installation:

1. **Quick Start**: Follow the [Quick Start Guide](../README.md#quick-start)
2. **Configuration**: Review [Configuration Guide](CONFIG.md)
3. **Development**: See [Development Guide](DEVELOPMENT.md)
4. **API Usage**: Check [API Documentation](API.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/privastream/tiktok-techjam-2025/issues)
- **Discussions**: [GitHub Discussions](https://github.com/privastream/tiktok-techjam-2025/discussions)
- **Email**: support@privastream.ai