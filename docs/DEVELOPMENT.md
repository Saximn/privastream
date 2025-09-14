# PrivaStream Development Guide

## Development Environment Setup

### Prerequisites

#### System Requirements

- **Python**: 3.8+ (3.10+ recommended for development)
- **Node.js**: 18+ for frontend development
- **Git**: Latest version
- **GPU**: NVIDIA GPU with CUDA 12.1+ (optional but recommended)

#### Development Tools

```bash
# Install essential tools
pip install pre-commit black flake8 mypy pytest
npm install -g prettier eslint

# IDE recommendations
# - VS Code with Python and React extensions
# - PyCharm Professional
# - IntelliJ IDEA with Python plugin
```

### Quick Development Setup

```bash
# Clone repository
git clone https://github.com/privastream/tiktok-techjam-2025.git
cd privastream

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .[dev,gpu,audio]

# Install pre-commit hooks
pre-commit install

# Setup frontend dependencies
cd src/web/frontend && npm install && cd -
cd src/web/mediasoup && npm install && cd -
```

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

#### VS Code Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.flake8",
    "ms-python.black-formatter",
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode"
  ]
}
```

## Project Structure

### Overview

```
tiktok-techjam-2025/
├── src/                           # Source code
│   ├── models/                    # AI Models
│   │   ├── detection/             # Video detection models
│   │   │   ├── face_blur/         # Face detection & blurring
│   │   │   ├── plate_blur/        # License plate detection
│   │   │   ├── pii_blur/          # Text PII detection
│   │   │   └── unified_detector.py # Model coordination
│   │   ├── audio/                 # Audio processing
│   │   │   ├── training/          # Model training scripts
│   │   │   └── *.py              # Audio processing pipeline
│   │   └── utils/                 # Shared utilities
│   ├── web/                       # Web application
│   │   ├── backend/               # Flask backend
│   │   ├── frontend/              # React frontend
│   │   └── mediasoup/             # WebRTC SFU server
│   ├── core/                      # Core utilities
│   │   ├── config/                # Configuration management
│   │   ├── exceptions.py          # Custom exceptions
│   │   └── logging.py             # Logging setup
│   ├── cli/                       # Command-line interface
│   ├── tests/                     # Test suite
│   └── tools/                     # Development tools
├── docs/                          # Documentation
├── docker-compose.yml             # Development containers
└── setup.py                      # Package configuration
```

### Key Modules

#### Models Module (`src/models/`)

- **detection/**: Computer vision models for video processing
- **audio/**: Speech processing and PII detection
- **utils/**: Shared utilities for model operations

#### Web Module (`src/web/`)

- **backend/**: Flask application with SocketIO
- **frontend/**: React application with Next.js
- **mediasoup/**: Node.js WebRTC SFU server

#### Core Module (`src/core/`)

- **config/**: Configuration management system
- **exceptions.py**: Custom exception classes
- **logging.py**: Centralized logging configuration

## Development Workflow

### Feature Development

#### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

#### 2. Make Changes

Follow the coding standards and write tests for new functionality.

#### 3. Run Tests

```bash
# Python tests
pytest src/tests/

# Frontend tests
cd src/web/frontend && npm test

# Integration tests
python src/models/detection/unified_bbox_test.py
```

#### 4. Code Quality Checks

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Manual quality checks
black src/
flake8 src/
mypy src/
```

#### 5. Commit Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

#### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create pull request on GitHub
```

### Code Style Guidelines

#### Python Code Style

We follow PEP 8 with Black formatting:

```python
"""Module docstring following Google style."""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from privastream.core.config import Config
from privastream.core.exceptions import ModelLoadError


class FaceDetector:
    """Face detection using YOLO model.

    Args:
        model_path: Path to the YOLO model file
        confidence: Confidence threshold for detections
    """

    def __init__(self, model_path: str, confidence: float = 0.4) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Optional[YOLO] = None

    def load_model(self) -> None:
        """Load the YOLO model."""
        try:
            self._model = YOLO(self.model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame.

        Args:
            frame: Input image as BGR numpy array

        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples

        Raises:
            ModelLoadError: If model is not loaded
        """
        if self._model is None:
            raise ModelLoadError("Model not loaded")

        results = self._model(frame, conf=self.confidence)
        boxes = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return boxes
```

#### JavaScript/TypeScript Code Style

We use Prettier and ESLint:

```javascript
import React, { useState, useEffect, useCallback } from "react";
import { io, Socket } from "socket.io-client";

interface StreamConfig {
  faceBlur: boolean;
  plateBlur: boolean;
  textBlur: boolean;
  audioProcessing: boolean;
}

const StreamingComponent: React.FC = () => {
  const [socket, setSocket] = (useState < Socket) | (null > null);
  const [roomId, setRoomId] = useState < string > "";
  const [config, setConfig] =
    useState <
    StreamConfig >
    {
      faceBlur: true,
      plateBlur: true,
      textBlur: true,
      audioProcessing: true,
    };

  const handleCreateRoom = useCallback(() => {
    if (socket) {
      socket.emit("create_room");
    }
  }, [socket]);

  useEffect(() => {
    const newSocket = io("http://localhost:5000");

    newSocket.on("room_created", (data) => {
      setRoomId(data.roomId);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  return (
    <div className="streaming-container">
      <button onClick={handleCreateRoom}>Create Room</button>
      {roomId && <div className="room-info">Room ID: {roomId}</div>}
    </div>
  );
};

export default StreamingComponent;
```

## Testing Strategy

### Test Structure

```
src/tests/
├── unit/                  # Unit tests
│   ├── test_models.py     # Model unit tests
│   ├── test_config.py     # Configuration tests
│   └── test_utils.py      # Utility function tests
├── integration/           # Integration tests
│   ├── test_api.py        # API integration tests
│   ├── test_pipeline.py   # End-to-end pipeline tests
│   └── test_web.py        # Web interface tests
├── fixtures/              # Test fixtures
│   ├── test_video.mp4     # Sample video files
│   ├── test_audio.wav     # Sample audio files
│   └── test_models/       # Mock models for testing
└── conftest.py           # Pytest configuration
```

### Running Tests

#### Unit Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest src/tests/unit/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel
pytest -n auto
```

#### Integration Tests

```bash
# Run integration tests
pytest src/tests/integration/

# Run with real models (slower)
pytest src/tests/integration/ --use-real-models
```

#### Model-Specific Tests

```bash
# Test face detection
python src/models/detection/face_blur/test_face_detector.py

# Test license plate detection
python src/models/detection/plate_blur/test_plate_detector.py

# Test audio processing
python src/models/audio/training/test_setup.py
```

### Writing Tests

#### Test Example

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

from privastream.models.detection.face_blur.face_detector import FaceDetector
from privastream.core.exceptions import ModelLoadError


class TestFaceDetector:
    """Test suite for FaceDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a FaceDetector instance for testing."""
        return FaceDetector("mock_model.pt", confidence=0.5)

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_init(self, detector):
        """Test FaceDetector initialization."""
        assert detector.model_path == "mock_model.pt"
        assert detector.confidence == 0.5
        assert detector._model is None

    @patch('privastream.models.detection.face_blur.face_detector.YOLO')
    def test_load_model_success(self, mock_yolo, detector):
        """Test successful model loading."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        detector.load_model()

        mock_yolo.assert_called_once_with("mock_model.pt")
        assert detector._model == mock_model

    @patch('privastream.models.detection.face_blur.face_detector.YOLO')
    def test_load_model_failure(self, mock_yolo, detector):
        """Test model loading failure."""
        mock_yolo.side_effect = Exception("Model not found")

        with pytest.raises(ModelLoadError, match="Failed to load model"):
            detector.load_model()

    def test_detect_without_loaded_model(self, detector, sample_frame):
        """Test detection without loaded model."""
        with pytest.raises(ModelLoadError, match="Model not loaded"):
            detector.detect(sample_frame)

    @patch('privastream.models.detection.face_blur.face_detector.YOLO')
    def test_detect_with_results(self, mock_yolo, detector, sample_frame):
        """Test detection with results."""
        # Setup mock
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        mock_box.xyxy = [np.array([10, 20, 100, 200])]
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector.load_model()
        results = detector.detect(sample_frame)

        assert results == [(10, 20, 100, 200)]
        mock_model.assert_called_once_with(sample_frame, conf=0.5)
```

## Debugging

### Debug Configuration

#### Python Debugging

```python
import logging
import pdb

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Set breakpoint
pdb.set_trace()

# Or use modern breakpoint()
breakpoint()
```

#### VS Code Launch Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Flask App",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/web/backend/app.py",
      "console": "integratedTerminal",
      "env": {
        "FLASK_ENV": "development",
        "FLASK_DEBUG": "1"
      }
    },
    {
      "name": "Python: Process Video",
      "type": "python",
      "request": "launch",
      "module": "privastream.cli.main",
      "args": ["video", "test_input.mp4", "test_output.mp4", "--debug"],
      "console": "integratedTerminal"
    }
  ]
}
```

### Performance Profiling

#### Python Profiling

```python
import cProfile
import pstats
from pstats import SortKey

# Profile function
pr = cProfile.Profile()
pr.enable()

# Your code here
result = your_function()

pr.disable()
s = pstats.Stats(pr)
s.sort_stats(SortKey.TIME)
s.print_stats()
```

#### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler your_script.py
```

#### GPU Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Your GPU code here
    model(input_tensor)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Model Development

### Adding New Models

#### 1. Create Model Module

```python
# src/models/detection/your_model/your_detector.py
from typing import List, Tuple
import numpy as np

class YourDetector:
    """Your custom detector implementation."""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        # Initialize your model

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect objects in frame.

        Args:
            frame: BGR image array

        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        # Your detection logic
        return []
```

#### 2. Add Tests

```python
# src/models/detection/your_model/test_your_detector.py
import pytest
from your_detector import YourDetector

def test_your_detector():
    detector = YourDetector("model.pt")
    # Add your tests
```

#### 3. Integrate with Pipeline

```python
# src/models/detection/unified_detector.py
from .your_model.your_detector import YourDetector

class UnifiedDetector:
    def __init__(self, config):
        # Add your detector
        if config.get('your_model', {}).get('enabled'):
            self.your_detector = YourDetector(
                config['your_model']['weights']
            )
```

### Model Training

#### Setup Training Environment

```bash
# Create training environment
python -m venv training-env
source training-env/bin/activate

# Install training dependencies
pip install torch torchvision transformers datasets wandb

# Setup Weights & Biases (optional)
wandb login
```

#### Training Script Template

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define your model architecture

    def forward(self, x):
        # Forward pass
        return x

def train_model():
    # Load data
    train_dataset = load_dataset("train")
    val_dataset = load_dataset("val")

    # Initialize model
    model = CustomModel(config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/your_model",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

if __name__ == "__main__":
    train_model()
```

## CI/CD Pipeline

### GitHub Actions

#### Test Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Lint with flake8
        run: |
          flake8 src/

      - name: Type check with mypy
        run: |
          mypy src/

      - name: Test with pytest
        run: |
          pytest --cov=src --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
```

### Pre-commit Configuration

#### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0-alpha.9-for-vscode
    hooks:
      - id: prettier
        files: \.(js|jsx|ts|tsx|json|css|md)$
```

## Documentation

### Writing Documentation

#### Docstring Standards

Follow Google-style docstrings:

```python
def process_video(
    input_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> ProcessingResult:
    """Process video with privacy filtering.

    Args:
        input_path: Path to input video file
        output_path: Path for output video file
        config: Processing configuration dictionary

    Returns:
        ProcessingResult object with processing statistics

    Raises:
        FileNotFoundError: If input file doesn't exist
        ModelLoadError: If required models can't be loaded
        ProcessingError: If processing fails

    Example:
        >>> config = {'face_blur': True, 'plate_blur': True}
        >>> result = process_video('input.mp4', 'output.mp4', config)
        >>> print(f"Processed {result.frame_count} frames")
    """
```

#### API Documentation

Use type hints and docstrings that can be processed by Sphinx:

```python
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    """Result of video processing operation.

    Attributes:
        frame_count: Number of frames processed
        processing_time: Total processing time in seconds
        detections: Dictionary of detection counts by type
    """
    frame_count: int
    processing_time: float
    detections: Dict[str, int]
```

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Build documentation
cd docs/
sphinx-build -b html . _build/html
```

## Contributing Guidelines

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes with tests
4. **Ensure** all tests pass and code is formatted
5. **Update** documentation if needed
6. **Submit** pull request with clear description

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated for changes
- [ ] Documentation updated if needed
- [ ] No breaking changes without discussion
- [ ] Performance implications considered
- [ ] Security implications reviewed

### Release Process

1. **Version Bump**: Update version in `setup.py`
2. **Changelog**: Update `CHANGELOG.md`
3. **Tag Release**: Create git tag `v1.0.0`
4. **GitHub Release**: Create release with notes
5. **Deploy**: Automated deployment via GitHub Actions

## Getting Help

### Resources

- **Documentation**: Check existing docs first
- **Issues**: Search GitHub issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Code Examples**: Check test files for usage examples

### Contact

- **General Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Security Issues**: security@privastream.ai
- **Development**: team@privastream.ai

Happy coding! 🚀
