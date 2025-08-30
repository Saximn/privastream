# Live Stream Pipeline

A real-time video processing pipeline for privacy protection with face, PII, and license plate detection and blurring.

## 🚀 Features

- **Real-time Processing**: 30fps input with optimized 4fps AI processing
- **Multi-threaded Architecture**: Separate threads for capture, processing, and output
- **Triple Detection**: Face, PII (text), and license plate detection
- **Configurable Blur**: Gaussian, pixelation, or fill blur options
- **Smart Persistence**: Detections persist across multiple frames for stability
- **Performance Monitoring**: Real-time FPS and processing time statistics
- **Flexible Configuration**: Easy customization via config file

## 🏗️ Architecture

```
Camera/Video (30fps) → VideoScheduler (4fps) → UnifiedVideoAnalyzer → BlurProcessor → Output (30fps)
                    ↓                                                                      ↑
                Input Buffer ←→ Processing Buffer ←→ Output Buffer ←→ Live Preview
```

### Components:

1. **VideoScheduler**: Selects frames for processing at reduced FPS (4fps from 30fps input)
2. **UnifiedVideoAnalyzer**: Runs AI models (face, PII, plate detection) on selected frames  
3. **BlurProcessor**: Applies blur to all frames based on detected regions with persistence
4. **FrameBuffer**: Thread-safe buffers for smooth data flow between components

## 🛠️ Installation

1. **Prerequisites**:
   ```bash
   pip install opencv-python numpy torch torchvision
   pip install doctr easyocr  # For PII detection
   pip install ultralytics    # For plate detection
   pip install insightface    # For face detection
   ```

2. **Model Files**:
   - Face detection: `face_blur/whitelist/creator_embedding.json`
   - PII detection: `pii_blur/pii_clf.joblib`
   - Plate detection: `plate_blur/best.pt`

## 🚀 Quick Start

### Run Tests
```bash
python test_pipeline.py
```

### Launch Pipeline
```bash
python launch_pipeline.py
```

### Configuration
Edit `pipeline_config.py` to customize:

```python
# Basic settings
INPUT_SOURCE = 0           # 0=webcam, or video file path
INPUT_FPS = 30.0          # Input framerate
PROCESSING_FPS = 4.0      # AI processing framerate
BLUR_TYPE = "gaussian"    # Blur type: gaussian, pixelate, fill

# Enable/disable models
ENABLE_FACE_DETECTION = True
ENABLE_PII_DETECTION = True  
ENABLE_PLATE_DETECTION = True

# Performance vs quality
BLUR_HOLD_FRAMES = 8      # Frames to hold blur after detection
```

## 🎮 Controls

- **'q'**: Quit application
- **'s'**: Save current frame as image

## 📊 Performance

**Typical Performance** (RTX 3070, 1280x720):
- Input: 30fps stable
- Processing: 4fps AI detection
- Output: 30fps smooth playback
- Latency: ~200ms end-to-end

**Resource Usage**:
- CPU: 15-25% (multi-threaded)
- GPU: 20-40% (AI models)
- Memory: ~2GB (with buffers)

## ⚙️ Advanced Configuration

### Processing FPS Optimization
```python
# High accuracy (slower)
PROCESSING_FPS = 8.0      # Process every ~4 frames

# Balanced (recommended)  
PROCESSING_FPS = 4.0      # Process every ~8 frames

# Performance mode (faster)
PROCESSING_FPS = 2.0      # Process every ~15 frames
```

### Blur Persistence Tuning
```python
# Responsive (less persistence)
BLUR_HOLD_FRAMES = 4      # 133ms at 30fps

# Stable (recommended)
BLUR_HOLD_FRAMES = 8      # 267ms at 30fps

# Persistent (more stability)
BLUR_HOLD_FRAMES = 15     # 500ms at 30fps
```

### Buffer Sizes
```python
INPUT_BUFFER_SIZE = 60     # ~2 seconds at 30fps
PROCESSING_BUFFER_SIZE = 10 # ~2.5 seconds at 4fps
OUTPUT_BUFFER_SIZE = 60     # ~2 seconds at 30fps
```

## 🔍 Monitoring

The pipeline provides real-time statistics:

```
Pipeline Stats - Input:29.8fps Processing:3.9fps Output:29.9fps Dropped:0 AvgTime:45.2ms
```

- **Input FPS**: Frames captured from camera
- **Processing FPS**: Frames processed by AI models  
- **Output FPS**: Frames displayed/saved
- **Dropped**: Frames dropped due to buffer overflow
- **AvgTime**: Average AI processing time per frame

## 🎯 Use Cases

1. **Live Streaming Privacy**: Protect privacy in live streams/video calls
2. **Content Creation**: Automatically blur sensitive information in recordings
3. **Security Monitoring**: Blur faces/plates in security footage
4. **Research**: Study real-time video processing performance

## 🧪 Testing

### Component Tests
```bash
python test_pipeline.py
```
Tests individual components:
- FrameBuffer thread safety
- PerformanceMonitor accuracy  
- UnifiedDetector model loading
- BlurUtils functionality
- Camera access

### Integration Test
```bash
python -c "from live_stream_pipeline import *; print('Import successful')"
```

## 🚧 Future Enhancements

### Audio Pipeline (Planned)
```
Microphone → Whisper STT → DeBerta PII → Mouth Blur Coordination
```

Features to be implemented:
- Real-time speech-to-text with Whisper
- PII detection in audio transcripts
- Temporal alignment with video frames
- Mouth blur when PII is spoken
- 5-second processing buffer for audio

### Network Streaming
- RTMP output for streaming platforms
- WebRTC for low-latency applications
- REST API for remote control

## 📝 API Reference

### Main Classes

#### `LiveStreamPipeline`
Main pipeline coordinator.

```python
config = PipelineConfig(input_source=0, processing_fps=4.0)
pipeline = LiveStreamPipeline(config)
pipeline.start()  # Blocks until quit
```

#### `PipelineConfig`
Configuration dataclass with all settings.

```python
config = PipelineConfig(
    input_source=0,
    input_fps=30.0,
    processing_fps=4.0,
    blur_type="gaussian",
    enable_face=True
)
```

#### `UnifiedBlurDetector`
Unified interface for all detection models.

```python
detector = UnifiedBlurDetector(config)
results = detector.process_frame(frame, frame_id)
rectangles = detector.get_all_rectangles(results)
polygons = detector.get_all_polygons(results)
```

### Utility Functions

#### `apply_blur_regions`
Apply blur to image regions.

```python
from blur_utils import apply_blur_regions

blurred = apply_blur_regions(
    image, 
    rectangles=[[x1,y1,x2,y2], ...],
    polygons=[polygon_array, ...],
    blur_type="gaussian"
)
```

## 🐛 Troubleshooting

### Common Issues

**Camera Not Found**:
```
[ERROR] Cannot open video source: 0
```
- Check camera permissions
- Try different INPUT_SOURCE values (0, 1, 2...)
- Test with video file instead

**Models Not Loading**:
```
[WARN] FaceDetector not available
```
- Check model file paths in config
- Verify all dependencies installed
- Run `test_pipeline.py` to diagnose

**Poor Performance**:
```
Pipeline Stats - Input:15.2fps Processing:1.2fps ...
```
- Lower PROCESSING_FPS in config
- Reduce INPUT_WIDTH/INPUT_HEIGHT
- Disable unused detection models
- Check GPU availability

**Buffer Overflows**:
```
[WARNING] Processing buffer full, dropping frame
```
- Increase buffer sizes in config
- Lower processing FPS
- Improve hardware specs

## 📄 License

This project is part of the TikTok TechJam 2025 submission.

## 🤝 Contributing

1. Test changes with `test_pipeline.py`
2. Update configuration examples
3. Add performance benchmarks
4. Document new features

---

**Ready to protect privacy in real-time! 🛡️**
