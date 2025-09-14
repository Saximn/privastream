from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Centralized configuration for all detection models"""
    
    # Detection thresholds
    FACE_THRESHOLD: float = 0.35
    PII_CONFIDENCE_THRESHOLD: float = 0.35
    PLATE_CONFIDENCE_THRESHOLD: float = 0.25
    
    # Image processing parameters
    DETECTION_SIZE: int = 960
    GAUSSIAN_KERNEL_SIZE: int = 35
    PIXEL_SIZE: int = 16
    DILATION_PIXELS: int = 12
    
    # Temporal stabilization settings
    SMOOTH_DURATION_MS: int = 300
    CONFIRMATION_FRAMES: int = 2
    HOLD_FRAMES: int = 8
    TRACKING_TIMEOUT_SECONDS: float = 2.0
    
    # GPU/CPU settings
    GPU_ID: int = 0
    USE_GPU: bool = True
    
    # Model paths (relative to privastream/models/)
    MODEL_BASE_PATH: Path = Path("privastream/models/detection")
    FACE_MODEL_PATH: str = "face_blur/face_best.pt"
    FACE_EMBED_PATH: str = "face_blur/whitelist/creator_embedding.json" 
    PLATE_MODEL_PATH: str = "plate_blur/best.pt"
    PII_CLASSIFIER_PATH: str = "pii_blur/pii_clf.joblib"
    
    # Processing settings
    MAX_WORKERS: int = 4
    CHUNK_SIZE: int = 1
    TARGET_FPS: int = 30
    
    # Blur parameters
    FACE_BLUR_KERNEL: int = 35
    PLATE_BLUR_KERNEL: int = 35
    PII_BLUR_KERNEL: int = 35
    
    # OCR settings
    OCR_ENGINE: str = "easyocr"
    OCR_LANGUAGES: list = None
    OCR_CONFIDENCE_THRESHOLD: float = 0.25
    
    def __post_init__(self):
        """Initialize default values that depend on other settings"""
        if self.OCR_LANGUAGES is None:
            self.OCR_LANGUAGES = ['en']
    
    @property
    def face_model_full_path(self) -> Path:
        """Get full path to face detection model"""
        return self.MODEL_BASE_PATH / self.FACE_MODEL_PATH
    
    @property
    def plate_model_full_path(self) -> Path:
        """Get full path to plate detection model"""
        return self.MODEL_BASE_PATH / self.PLATE_MODEL_PATH
    
    @property
    def pii_classifier_full_path(self) -> Path:
        """Get full path to PII classifier model"""
        return self.MODEL_BASE_PATH / self.PII_CLASSIFIER_PATH
    
    @property 
    def face_embed_full_path(self) -> Path:
        """Get full path to face embedding file"""
        return self.MODEL_BASE_PATH / self.FACE_EMBED_PATH


@dataclass
class WebDemoConfig(ModelConfig):
    """Configuration specific to web demo deployment"""
    
    # Web-specific settings
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = False
    
    # WebRTC/Mediasoup settings
    MEDIASOUP_HOST: str = "0.0.0.0"
    MEDIASOUP_PORT: int = 3001
    
    # Frontend settings
    FRONTEND_HOST: str = "0.0.0.0"
    FRONTEND_PORT: int = 3000
    
    # Performance settings for web demo
    MAX_WORKERS: int = 2  # Reduced for web deployment
    TARGET_FPS: int = 4   # Reduced for real-time processing
    DETECTION_SIZE: int = 640  # Smaller for faster processing


@dataclass
class ProductionConfig(ModelConfig):
    """Configuration optimized for production deployment"""
    
    # High-performance settings
    MAX_WORKERS: int = 8
    TARGET_FPS: int = 30
    DETECTION_SIZE: int = 960
    
    # Production reliability
    USE_GPU: bool = True
    CHUNK_SIZE: int = 4
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "production.log"


# Default configuration instance
default_config = ModelConfig()
web_config = WebDemoConfig()
production_config = ProductionConfig()