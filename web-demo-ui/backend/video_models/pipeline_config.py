"""
Configuration file for the Live Stream Pipeline.
Modify these settings to customize the pipeline behavior.
"""

# ============================================================================
# Video Input Configuration
# ============================================================================

# Video input source
# 0 = Default webcam, 1 = Second camera, or path to video file
INPUT_SOURCE = 0

# Input video properties
INPUT_FPS = 30.0        # Frames per second from input
INPUT_WIDTH = 1280      # Input frame width
INPUT_HEIGHT = 720      # Input frame height

# ============================================================================
# Processing Configuration
# ============================================================================

# Processing FPS (how often to run AI models)
# Lower values improve performance but may miss some detections
# Higher values are more accurate but use more CPU/GPU
PROCESSING_FPS = 4.0    # Recommended: 2-8 fps

# Output FPS (should match input FPS for smooth output)
OUTPUT_FPS = 30.0

# Buffer sizes (in number of frames)
INPUT_BUFFER_SIZE = 60      # ~2 seconds at 30fps
PROCESSING_BUFFER_SIZE = 10 # ~2.5 seconds at 4fps processing
OUTPUT_BUFFER_SIZE = 60     # ~2 seconds at 30fps

# ============================================================================
# AI Model Configuration
# ============================================================================

# Enable/disable individual models
ENABLE_FACE_DETECTION = True
ENABLE_PII_DETECTION = True
ENABLE_PLATE_DETECTION = True

# Face detection settings
FACE_CONFIG = {
    "embed_path": "face_blur/whitelist/creator_embedding.json",
    "threshold": 0.35,      # Lower = more sensitive
    "dilate_px": 12,        # Blur padding around faces
    "smooth_ms": 300        # Smoothing window
}

# PII detection settings  
PII_CONFIG = {
    "classifier_path": "pii_blur/pii_clf.joblib",
    "conf_thresh": 0.35,    # OCR confidence threshold
    "min_area": 80,         # Minimum text area to consider
    "K_confirm": 2,         # Frames needed to confirm detection
    "K_hold": 8             # Frames to hold detection
}

# License plate detection settings
PLATE_CONFIG = {
    "weights_path": "plate_blur/best.pt",
    "imgsz": 960,           # YOLO input size
    "conf_thresh": 0.25,    # Detection confidence threshold
    "iou_thresh": 0.5       # Non-maximum suppression threshold
}

# ============================================================================
# Blur Configuration
# ============================================================================

# Blur type: "gaussian", "pixelate", or "fill"
BLUR_TYPE = "gaussian"

# Gaussian blur settings
BLUR_KERNEL_SIZE = 35       # Must be odd number, larger = more blur

# Pixelation settings
BLUR_PIXEL_SIZE = 16        # Block size for pixelation effect

# Fill settings
BLUR_FILL_COLOR = (128, 128, 128)  # Gray color for fill mode

# Blur persistence
BLUR_HOLD_FRAMES = 8        # Number of frames to keep blur after detection stops

# Blur padding
RECT_BLUR_PADDING = 0       # Extra padding for rectangular regions
POLY_BLUR_PADDING = 3       # Extra padding for polygon regions

# ============================================================================
# Output Configuration
# ============================================================================

# Display settings
SHOW_PREVIEW = True         # Show live preview window
SHOW_DETECTIONS = True      # Draw detection boxes on preview

# Video recording
SAVE_OUTPUT_VIDEO = False   # Set to True to save processed video
OUTPUT_VIDEO_PATH = "output_stream.mp4"
OUTPUT_VIDEO_CODEC = "mp4v" # Video codec

# ============================================================================
# Performance Configuration
# ============================================================================

# Performance monitoring
ENABLE_PROFILING = True     # Show performance statistics
STATS_INTERVAL = 5.0        # Print stats every N seconds

# Threading settings
THREAD_TIMEOUT = 2.0        # Timeout for thread operations

# GPU settings
USE_GPU = True              # Use GPU if available
GPU_DEVICE_ID = 0           # GPU device ID

# ============================================================================
# Audio Configuration (Future Implementation)
# ============================================================================

# Audio input
AUDIO_INPUT_DEVICE = None   # None = default microphone
AUDIO_SAMPLE_RATE = 16000   # Sample rate for Whisper
AUDIO_CHUNK_SIZE = 1024     # Audio processing chunk size

# Speech-to-text settings
WHISPER_MODEL = "base"      # Whisper model: tiny, base, small, medium, large
WHISPER_LANGUAGE = "en"     # Language code

# PII detection in audio
DEBERTA_MODEL = "microsoft/deberta-v3-base"  # DeBerta model for PII detection
AUDIO_PII_CONFIDENCE = 0.8  # Confidence threshold for audio PII
AUDIO_BUFFER_SECONDS = 5.0  # Buffer audio for this many seconds

# Audio-video synchronization
AUDIO_VIDEO_SYNC_OFFSET = 0.0  # Offset in seconds to sync audio/video

# ============================================================================
# Logging Configuration
# ============================================================================

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Log to file
LOG_TO_FILE = False
LOG_FILE_PATH = "pipeline.log"
LOG_MAX_SIZE_MB = 10        # Max log file size before rotation
LOG_BACKUP_COUNT = 3        # Number of backup log files

# ============================================================================
# Advanced Configuration
# ============================================================================

# Frame dropping strategy when buffers are full
# "oldest" = drop oldest frames, "newest" = drop newest frames
BUFFER_DROP_STRATEGY = "oldest"

# Quality vs performance trade-off
# "performance" = prioritize speed, "quality" = prioritize accuracy
PRIORITY_MODE = "performance"

# Network streaming (future feature)
ENABLE_RTMP_OUTPUT = False
RTMP_URL = "rtmp://localhost:1935/live/stream"

# API server (future feature)
ENABLE_API_SERVER = False
API_PORT = 8080
API_HOST = "localhost"
