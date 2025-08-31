# Unified Bounding Box Test System

A comprehensive testing system for all three detection models (Face, PII, and Plate detection) that visualizes all bounding boxes and polygons simultaneously with different colors and provides detailed performance metrics.

## Features

- **Multi-Model Testing**: Test all three detection models simultaneously
- **Fixed Frame Rate Processing**: Configurable target FPS (default: 3 FPS)
- **Visual Differentiation**: Different colors for each detection type
  - 🔴 **Red**: Face detections
  - 🟢 **Green**: PII detections  
  - 🔵 **Blue**: Plate detections
- **Real-time Performance Metrics**: Processing FPS vs Display FPS, timing info
- **Multiple Test Modes**: Webcam and single image testing
- **Interactive Controls**: Save frames, toggle settings, adjust FPS on-the-fly
- **Confidence Scores**: Display confidence scores for applicable detections
- **Comprehensive Logging**: Detailed statistics and timing information

## Quick Start

### 1. Quick Test (Recommended for first run)
```bash
cd models
python test_quick.py
```
This creates a synthetic test image and verifies all models are working correctly.

### 2. Fixed FPS Functionality Test
```bash
cd models
python test_fixed_fps.py
```
Tests the fixed frame rate processing for 10 seconds with webcam.

### 3. FPS Modes Demonstration
```bash
cd models
python demo_fps_modes.py
```
Shows the difference between variable FPS and fixed FPS processing.

### 4. Interactive Batch Script
```bash
cd models
run_unified_test.bat
```
Choose from menu options for different test modes and FPS settings.

### 5. Direct Command Line Usage

#### Webcam Test with Fixed FPS (Default)
```bash
python unified_bbox_test.py --mode webcam --camera 0 --fps 3.0
```

#### Webcam Test with 2 FPS
```bash
python unified_bbox_test.py --mode webcam --camera 0 --fps 2.0
```

#### Image Test
```bash
python unified_bbox_test.py --mode image --image "path/to/your/image.jpg"
```

#### With Custom Configuration and FPS
```bash
python unified_bbox_test.py --mode webcam --config unified_test_config.json --fps 2.5
```

## Files Overview

### Main Files
- **`unified_bbox_test.py`** - Main unified testing system with fixed FPS support
- **`unified_test_config.json`** - Configuration file for all models
- **`test_quick.py`** - Quick verification test with synthetic image
- **`test_fixed_fps.py`** - 10-second test to verify fixed FPS functionality
- **`demo_fps_modes.py`** - Demonstration of variable vs fixed FPS modes
- **`run_unified_test.bat`** - Batch script for easy launching with FPS options

### Model Files (Created Earlier)
- **`face_blur/face_detector.py`** - Face detection model
- **`pii_blur/pii_detector.py`** - PII detection model
- **`plate_blur/plate_detector.py`** - Plate detection model
- **`blur_utils.py`** - Utility functions for applying blur

## Interactive Controls

### Webcam Mode Controls
- **`q`** - Quit the application
- **`s`** - Save current frame and detection results
- **`p`** - Toggle face detector panic mode (blur entire frame)
- **`r`** - Reload face detector embedding from disk
- **`i`** - Print current performance statistics to console
- **`+` or `=`** - Increase target FPS by 0.5
- **`-` or `_`** - Decrease target FPS by 0.5

### Image Mode
- **Any key** - Close the image and exit

## Fixed Frame Rate Processing

The unified tester now supports **fixed frame rate processing** which is ideal for:

### Benefits
- **⏱️ Predictable Performance**: Consistent processing intervals
- **🔄 Stable Resource Usage**: Prevents system overload
- **📊 Better Benchmarking**: Reliable performance comparisons  
- **⚡ Real-time Applications**: Suitable for live streaming/recording
- **🎯 Consistent Results**: Reproducible test outcomes

### How It Works
- **Target FPS**: Set desired processing rate (e.g., 2-3 FPS)
- **Frame Skipping**: Only processes frames at specified intervals
- **Display FPS**: Shows actual display refresh rate vs processing rate
- **Dynamic Adjustment**: Change FPS on-the-fly with +/- keys

### Usage Examples
```bash
# Process at 2 FPS
python unified_bbox_test.py --fps 2.0

# Process at 3 FPS (default)
python unified_bbox_test.py --fps 3.0

# Custom FPS with configuration
python unified_bbox_test.py --fps 2.5 --config unified_test_config.json
```

### Detection Visualization
- **Face Rectangles**: Red boxes with "FACE_N" labels
- **PII Polygons**: Green polygons with "PII_N" labels
- **Plate Rectangles**: Blue boxes with "PLATE_N" labels and confidence scores

### Information Overlay
- Current frame number and FPS
- Total detection count
- Per-model detection counts and processing times
- Overall processing time

### Performance Metrics
Real-time display of:
- Frames per second (FPS)
- Detection counts per model
- Processing time per model in milliseconds
- Total processing time

## Configuration

### Configuration File Format (`unified_test_config.json`)
```json
{
    "face": {
        "embed_path": "face_blur/whitelist/creator_embedding.json",
        "gpu_id": 0,
        "det_size": 960,
        "threshold": 0.35,
        "dilate_px": 12,
        "smooth_ms": 300
    },
    "pii": {
        "classifier_path": "pii_blur/pii_clf.joblib",
        "conf_thresh": 0.35,
        "min_area": 80,
        "K_confirm": 2,
        "K_hold": 8
    },
    "plate": {
        "weights_path": "plate_blur/best.pt",
        "imgsz": 960,
        "conf_thresh": 0.25,
        "iou_thresh": 0.5,
        "pad": 4
    }
}
```

### Command Line Arguments
```bash
python unified_bbox_test.py [OPTIONS]

Options:
  --mode {webcam,image}     Test mode (default: webcam)
  --camera INT             Camera ID for webcam mode (default: 0)
  --image PATH            Path to image file for image mode
  --width INT             Camera width for webcam mode (default: 1280)
  --height INT            Camera height for webcam mode (default: 720)
  --fps FLOAT             Target processing FPS (default: 3.0, range: 0.5-10.0)
  --config PATH           Path to JSON configuration file
```

## Output Files

### Saved Frame Data
When you press 's' during webcam mode, the system saves:
- **`unified_test_frame_<timestamp>.jpg`** - Annotated frame image
- **`unified_test_results_<timestamp>.txt`** - Detailed detection results

### Synthetic Test Files
The quick test generates:
- **`test_image_synthetic.jpg`** - Generated test image
- **`test_results_visualization.jpg`** - Annotated results

### Example Results File
```
Unified Detection Results - Frame 1234
Timestamp: 1693478400.123
Target FPS: 3.0
Display FPS: 15.2

FACE RESULTS:
  Count: 2
  Type: rectangles
  Processing time: 45.2ms

PII RESULTS:
  Count: 1
  Type: polygons
  Processing time: 123.8ms

PLATE RESULTS:
  Count: 0
  Type: rectangles_with_confidence
  Processing time: 67.1ms
```

## Performance Statistics

The system tracks and displays comprehensive statistics:

### Per-Model Statistics
- Total processed frames
- Total detections made
- Average detections per frame
- Average processing time per frame
- Model-specific FPS

### Overall Statistics
- Combined processing time
- Overall system FPS
- Total detection count across all models

### Example Statistics Output
```
==============================================================
UNIFIED DETECTION STATISTICS
==============================================================
FACE DETECTOR:
  Processed frames: 500
  Total detections: 45
  Avg detections/frame: 0.09
  Avg processing time: 42.3ms
  Model FPS: 23.6

PII DETECTOR:
  Processed frames: 500
  Total detections: 12
  Avg detections/frame: 0.02
  Avg processing time: 156.7ms
  Model FPS: 6.4

PLATE DETECTOR:
  Processed frames: 500
  Total detections: 8
  Avg detections/frame: 0.02
  Avg processing time: 89.1ms
  Model FPS: 11.2

OVERALL:
  Total frames processed: 500
  Total detections: 65
  Combined processing time: 144.05s
  Overall FPS: 3.5
==============================================================
```

## Dependencies

### Core Requirements
- `opencv-python` - Image processing and display
- `numpy` - Array operations

### Model-Specific Requirements
- **Face Detection**: `insightface` 
- **PII Detection**: `torch`, `doctr` or `easyocr`, `scikit-learn`
- **Plate Detection**: `torch`, `ultralytics`

### Installation
```bash
# Core requirements
pip install opencv-python numpy

# Face detection
pip install insightface

# PII detection
pip install torch doctr easyocr scikit-learn
# OR minimal: pip install torch easyocr scikit-learn

# Plate detection  
pip install torch ultralytics
```

## Troubleshooting

### No Models Available
If you see "No models available for testing!", check that dependencies are installed:
```bash
pip install insightface torch doctr ultralytics scikit-learn easyocr
```

### Camera Access Issues
- Try different camera IDs (0, 1, 2...)
- Check camera permissions
- Close other applications using the camera

### Model Loading Errors
- Verify model files exist (best.pt, pii_clf.joblib, creator_embedding.json)
- Check file paths in configuration
- Ensure proper model file formats

### Performance Issues
- Reduce image resolution with --width and --height
- Increase confidence thresholds in configuration
- Use GPU acceleration if available

### Import Errors
- Ensure you're running from the correct directory
- Check Python path includes model directories
- Verify all required packages are installed

## Advanced Usage

### Custom Model Configurations
Modify `unified_test_config.json` to adjust:
- Detection thresholds
- Image processing sizes
- Temporal smoothing parameters
- GPU device selection

### Batch Processing
The system can be extended for batch processing multiple images:
```python
tester = UnifiedBoundingBoxTester(config)
for image_path in image_list:
    frame = cv2.imread(image_path)
    results = tester.process_frame(frame, idx)
    # Process results...
```

### Custom Visualization
Extend the visualization methods to:
- Add custom colors or labels
- Draw additional information
- Save different output formats
- Create video outputs

## Integration

The unified testing system demonstrates how to integrate all three detection models. You can use this as a template for:
- Real-time surveillance systems
- Batch image processing pipelines
- Privacy protection applications
- Computer vision research platforms

## Performance Optimization

### GPU Acceleration
- Face detection: Automatic GPU detection
- PII detection: Uses GPU for OCR when available
- Plate detection: Automatic GPU detection with PyTorch

### Memory Management
- Efficient frame processing
- Automatic cleanup of detection results
- Optimized visualization rendering

### Processing Optimization
- Frame skipping options for face detection
- Temporal smoothing reduces redundant processing
- Configurable detection thresholds balance speed vs accuracy

## Future Enhancements

Potential improvements to the testing system:
- Video file input support
- Batch image processing mode
- Result export to JSON/CSV formats
- Performance profiling and optimization suggestions
- Model ensemble evaluation
- Custom blur application preview
- ROI (Region of Interest) selection
- Multi-camera support
