"""
Data structures and types for the livestream PII detection pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import time


class PIIType(Enum):
    """Enumeration of PII types that can be detected."""
    NAME_STUDENT = "NAME_STUDENT"
    EMAIL = "EMAIL"
    USERNAME = "USERNAME" 
    PHONE_NUM = "PHONE_NUM"
    URL_PERSONAL = "URL_PERSONAL"
    STREET_ADDRESS = "STREET_ADDRESS"
    ID_NUM = "ID_NUM"
    NAME_INSTRUCTOR = "NAME_INSTRUCTOR"
    OTHER = "OTHER"


@dataclass
class AudioSegment:
    """Represents a segment of audio for processing."""
    audio_data: bytes
    start_time: float
    end_time: float
    sample_rate: int
    channels: int
    segment_id: str
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass 
class TranscriptionResult:
    """Result of whisper transcription."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    language: str
    segment_id: str
    word_timestamps: List[Dict] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PIIDetection:
    """Represents a detected PII instance."""
    pii_type: PIIType
    text: str
    start_char: int
    end_char: int
    confidence: float
    start_time: float
    end_time: float
    word_indices: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'pii_type': self.pii_type.value,
            'text': self.text,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'confidence': self.confidence,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'word_indices': self.word_indices
        }


@dataclass
class RedactionResult:
    """Result of PII detection and redaction."""
    original_text: str
    redacted_text: str
    detections: List[PIIDetection]
    segment_id: str
    processing_time: float
    
    def to_dict(self) -> Dict:
        return {
            'original_text': self.original_text,
            'redacted_text': self.redacted_text,
            'detections': [d.to_dict() for d in self.detections],
            'segment_id': self.segment_id,
            'processing_time': self.processing_time
        }


@dataclass
class VideoBlurInstruction:
    """Instructions for blurring video segments."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    blur_region: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    blur_type: str = "gaussian"  # gaussian, pixelate, black_box
    blur_strength: float = 10.0
    reason: str = "PII_DETECTED"
    pii_detections: List[PIIDetection] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'blur_region': self.blur_region,
            'blur_type': self.blur_type,
            'blur_strength': self.blur_strength,
            'reason': self.reason,
            'pii_detections': [d.to_dict() for d in self.pii_detections]
        }


@dataclass
class ProcessingResult:
    """Complete processing result for a stream segment."""
    segment_id: str
    audio_segment: AudioSegment
    transcription: TranscriptionResult
    redaction: RedactionResult
    blur_instructions: List[VideoBlurInstruction]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'segment_id': self.segment_id,
            'timestamp': self.timestamp,
            'audio_segment': {
                'start_time': self.audio_segment.start_time,
                'end_time': self.audio_segment.end_time,
                'duration': self.audio_segment.duration,
                'sample_rate': self.audio_segment.sample_rate,
                'channels': self.audio_segment.channels
            },
            'transcription': {
                'text': self.transcription.text,
                'start_time': self.transcription.start_time,
                'end_time': self.transcription.end_time,
                'confidence': self.transcription.confidence,
                'language': self.transcription.language,
                'duration': self.transcription.duration
            },
            'redaction': self.redaction.to_dict(),
            'blur_instructions': [bi.to_dict() for bi in self.blur_instructions]
        }


@dataclass
class StreamConfig:
    """Configuration for a specific stream."""
    stream_id: str
    fps: float = 30.0
    audio_sample_rate: int = 16000
    video_width: int = 1920
    video_height: int = 1080
    processing_delay: float = 2.0  # Delay in seconds for real-time processing
    
    def frames_to_time(self, frame_number: int) -> float:
        """Convert frame number to time in seconds."""
        return frame_number / self.fps
        
    def time_to_frames(self, time_seconds: float) -> int:
        """Convert time in seconds to frame number."""
        return int(time_seconds * self.fps)
