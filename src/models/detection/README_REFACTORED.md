# Refactored Blur Detection Models

This document describes the refactored blur detection models that have been changed from directly performing blur operations to extracting regions/polygons that should be blurred.

## Overview

The three models (Face Blur, PII Blur, and Plate Blur) have been refactored to follow a unified interface:

- **Input**: Frame (image) and frame ID
- **Output**: Frame ID and list of rectangles/polygons to be blurred
- **No direct blurring**: Models only identify regions, blur application is separate

## Refactored Models

### 1. Face Detector (`models/face_blur/face_detector.py`)

**Class**: `FaceDetector`

**Functionality**: 
- Detects faces in frames
- Supports creator whitelisting (enrolled face won't be blurred)
- Returns rectangles for faces that should be blurred
- Includes temporal smoothing and panic mode

**Key Methods**:
```python
detector = FaceDetector(
    embed_path="whitelist/creator_embedding.json",
    gpu_id=0,
    det_size=960,
    threshold=0.35
)

frame_id, rectangles = detector.process_frame(frame, frame_id)
# rectangles: List[List[int]] - [[x1, y1, x2, y2], ...]
```

**Features**:
- Creator embedding-based whitelisting
- Temporal voting for stable decisions
- Low-light enhancement with CLAHE
- Test-time augmentation (TTA)
- Panic mode (blur entire frame)
- GPU acceleration support

### 2. PII Detector (`models/pii_blur/pii_detector.py`)

**Class**: `PIIDetector`

**Functionality**:
- Detects personally identifiable information in text
- Uses OCR (docTR/EasyOCR) + rule-based + ML classification
- Returns polygons for text regions containing PII
- Includes temporal stabilization via hysteresis

**Key Methods**:
```python
detector = PIIDetector(
    classifier_path="pii_clf.joblib",
    conf_thresh=0.35,
    min_area=80
)

frame_id, polygons = detector.process_frame(frame, frame_id)
# polygons: List[np.ndarray] - List of polygon coordinates
```

**Features**:
- Hybrid PII detection (rules + ML)
- OCR with confidence thresholding
- Temporal stabilization with hysteresis tracking
- Configurable confirmation/hold frames
- Support for both docTR and EasyOCR

### 3. Plate Detector (`models/plate_blur/plate_detector.py`)

**Class**: `PlateDetector`

**Functionality**:
- Detects license plates using YOLO
- Returns rectangles for detected plates
- Supports confidence and IoU thresholding

**Key Methods**:
```python
detector = PlateDetector(
    weights_path="best.pt",
    imgsz=960,
    conf_thresh=0.25
)

frame_id, rectangles = detector.process_frame(frame, frame_id)
# rectangles: List[List[int]] - [[x1, y1, x2, y2], ...]

# Or with metadata:
frame_id, detection_data = detector.process_frame_with_metadata(frame, frame_id)
# detection_data: List[Dict] - [{"rectangle": [x1,y1,x2,y2], "confidence": 0.95, ...}, ...]
```

**Features**:
- YOLO-based detection
- Configurable confidence/IoU thresholds
- Automatic padding around detections
- GPU acceleration support
- Metadata extraction (confidence, class)

## Usage Examples

### Individual Model Usage

```python
# Face Detection
from face_detector import FaceDetector

detector = FaceDetector()
frame_id, face_rectangles = detector.process_frame(frame, 0)

# PII Detection  
from pii_detector import PIIDetector

detector = PIIDetector()
frame_id, pii_polygons = detector.process_frame(frame, 0)

# Plate Detection
from plate_detector import PlateDetector

detector = PlateDetector(weights_path="best.pt")
frame_id, plate_rectangles = detector.process_frame(frame, 0)
```

### Unified Interface

```python
from unified_detector import UnifiedBlurDetector

config = {
    "enable_face": True,
    "enable_pii": True, 
    "enable_plate": True,
    "face": {"threshold": 0.35},
    "pii": {"conf_thresh": 0.35},
    "plate": {"conf_thresh": 0.25}
}

detector = UnifiedBlurDetector(config)
results = detector.process_frame(frame, frame_id)

# Extract all rectangles and polygons
rectangles = detector.get_all_rectangles(results)
polygons = detector.get_all_polygons(results)
```

## Testing

Each model includes a test script:

```bash
# Test face detector
cd models/face_blur
python test_face_detector.py

# Test PII detector  
cd models/pii_blur
python test_pii_detector.py

# Test plate detector
cd models/plate_blur
python test_plate_detector.py

# Test unified detector
cd models
python unified_detector.py
```

## Migration from Original Models

### Before (Original)
```python
# Original models directly blurred frames
blurred_frame = blur_faces(frame)
blurred_frame = blur_pii(blurred_frame)
blurred_frame = blur_plates(blurred_frame)
```

### After (Refactored)
```python
# New models return regions to blur
face_rectangles = face_detector.process_frame(frame, frame_id)[1]
pii_polygons = pii_detector.process_frame(frame, frame_id)[1] 
plate_rectangles = plate_detector.process_frame(frame, frame_id)[1]

# Apply blur separately
blurred_frame = apply_blur_regions(frame, face_rectangles, pii_polygons, plate_rectangles)
```

## Benefits of Refactoring

1. **Separation of Concerns**: Detection logic separated from blur application
2. **Flexibility**: Can apply different blur types (Gaussian, pixelation, fill)
3. **Reusability**: Detection results can be used for other purposes
4. **Testability**: Easier to test detection accuracy without blur effects
5. **Performance**: Can optimize blur application separately
6. **Integration**: Easier to combine multiple detection models

## Dependencies

### Face Detector
- `insightface`
- `opencv-python`
- `numpy`

### PII Detector  
- `torch` (optional, for GPU)
- `doctr` or `easyocr`
- `scikit-learn` (for ML classifier)
- `opencv-python`
- `numpy`

### Plate Detector
- `torch`
- `ultralytics`
- `opencv-python`
- `numpy`

## Configuration

Each model supports extensive configuration through constructor parameters. See individual model documentation for detailed parameter descriptions.

## Error Handling

All models include comprehensive error handling:
- Graceful fallbacks (CPU when GPU unavailable)
- Missing file handling
- Import error handling
- Runtime exception catching

## Performance Considerations

- **GPU Acceleration**: All models support GPU when available
- **Temporal Optimization**: Face and PII detectors include frame skipping options
- **Memory Management**: Efficient handling of detection results
- **Batch Processing**: Models can be extended for batch processing

## Future Enhancements

Potential improvements:
- Batch processing support
- Model ensemble capabilities
- Advanced temporal tracking
- Custom blur application utilities
- Performance profiling tools
