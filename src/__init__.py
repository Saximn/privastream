"""
Privastream - AI-Powered Privacy Streaming Platform

A comprehensive privacy-focused streaming platform that provides real-time 
detection and redaction of PII, faces, and license plates in video streams.
"""

__version__ = "1.0.0"
__author__ = "Privastream Team"

from .core.config import Config
from .core.logging import setup_logging
from .core.exceptions import PrivastreamError

__all__ = ["Config", "setup_logging", "PrivastreamError"]