"""Custom exceptions for Privastream."""


class PrivastreamError(Exception):
    """Base exception for all Privastream errors."""
    pass


class ModelError(PrivastreamError):
    """Base exception for model-related errors."""
    pass


class ModelInitializationError(ModelError):
    """Raised when model fails to initialize."""
    pass


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""
    pass


class ConfigurationError(PrivastreamError):
    """Raised when configuration is invalid."""
    pass


class StreamingError(PrivastreamError):
    """Raised when streaming operations fail."""
    pass


class AudioProcessingError(PrivastreamError):
    """Raised when audio processing fails."""
    pass


class VideoProcessingError(PrivastreamError):
    """Raised when video processing fails."""
    pass