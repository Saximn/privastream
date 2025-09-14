"""
Whisper-based audio transcription processor for livestream PII detection pipeline.
"""

import asyncio
import whisper
import torch
import numpy as np
import logging
import time
import queue
import threading
from typing import Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import librosa

from .pipeline_types import (
    AudioSegment, 
    TranscriptionResult, 
    StreamConfig
)


class WhisperProcessor:
    """
    Whisper-based audio transcription processor optimized for real-time livestream processing.
    """
    
    def __init__(
        self, 
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = "en",
        max_workers: int = 2,
        enable_word_timestamps: bool = True
    ):
        """
        Initialize the Whisper processor.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to run inference on (cuda, cpu)
            compute_type: Precision type (float16, float32)
            language: Language code or None for auto-detection
            max_workers: Maximum number of worker threads for parallel processing
            enable_word_timestamps: Whether to generate word-level timestamps
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.max_workers = max_workers
        self.enable_word_timestamps = enable_word_timestamps
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load the Whisper model
        self.logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(
            model_name, 
            device=device,
            download_root=None
        )
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Processing queue
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Statistics
        self.stats = {
            'processed_segments': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'errors': 0
        }
        
        self.logger.info(f"Whisper processor initialized with model {model_name} on {device}")
    
    def preprocess_audio(
        self, 
        audio_data: Union[np.ndarray, bytes], 
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Preprocess audio data for Whisper model.
        
        Args:
            audio_data: Raw audio data as numpy array or bytes
            sample_rate: Sample rate of the audio
            
        Returns:
            Preprocessed audio array suitable for Whisper
        """
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio_array = audio_data
            
            # Ensure we have the correct sample rate (16kHz for Whisper)
            if sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            
            # Normalize audio to [-1, 1] range
            if audio_array.dtype != np.float32:
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif audio_array.dtype == np.int32:
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:
                    audio_array = audio_array.astype(np.float32)
            
            # Ensure mono audio
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Apply noise reduction and normalization
            audio_array = self._apply_audio_enhancements(audio_array)
            
            return audio_array
            
        except Exception as e:
            self.logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def _apply_audio_enhancements(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio enhancements for better transcription quality."""
        # Simple normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        return audio
    
    def transcribe_segment(
        self, 
        audio_segment: AudioSegment,
        return_word_timestamps: bool = None
    ) -> TranscriptionResult:
        """
        Transcribe a single audio segment.
        
        Args:
            audio_segment: AudioSegment to transcribe
            return_word_timestamps: Override global word timestamp setting
            
        Returns:
            TranscriptionResult with transcription and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess audio
            audio_array = self.preprocess_audio(
                audio_segment.audio_data, 
                audio_segment.sample_rate
            )
            
            # Prepare transcription options
            options = {
                "language": self.language,
                "fp16": self.compute_type == "float16",
                "word_timestamps": return_word_timestamps if return_word_timestamps is not None else self.enable_word_timestamps
            }
            
            # Perform transcription
            result = self.model.transcribe(audio_array, **options)
            
            # Extract word-level timestamps if available
            word_timestamps = []
            if options["word_timestamps"] and "segments" in result:
                for segment in result["segments"]:
                    if "words" in segment:
                        for word in segment["words"]:
                            word_timestamps.append({
                                'word': word.get('word', ''),
                                'start': word.get('start', 0.0) + audio_segment.start_time,
                                'end': word.get('end', 0.0) + audio_segment.start_time,
                                'probability': word.get('probability', 1.0)
                            })
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['processed_segments'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['processed_segments']
            )
            
            transcription_result = TranscriptionResult(
                text=result["text"].strip(),
                start_time=audio_segment.start_time,
                end_time=audio_segment.end_time,
                confidence=self._calculate_confidence(result),
                language=result.get("language", "en"),
                segment_id=audio_segment.segment_id,
                word_timestamps=word_timestamps
            )
            
            self.logger.debug(
                f"Transcribed segment {audio_segment.segment_id} in {processing_time:.2f}s: "
                f"'{transcription_result.text[:50]}...'"
            )
            
            return transcription_result
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Error transcribing segment {audio_segment.segment_id}: {e}")
            
            # Return empty transcription result on error
            return TranscriptionResult(
                text="",
                start_time=audio_segment.start_time,
                end_time=audio_segment.end_time,
                confidence=0.0,
                language="en",
                segment_id=audio_segment.segment_id,
                word_timestamps=[]
            )
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate overall confidence score from Whisper result."""
        if "segments" not in whisper_result:
            return 1.0
        
        total_prob = 0.0
        total_words = 0
        
        for segment in whisper_result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    if "probability" in word:
                        total_prob += word["probability"]
                        total_words += 1
        
        return total_prob / total_words if total_words > 0 else 1.0
    
    async def transcribe_segment_async(
        self, 
        audio_segment: AudioSegment
    ) -> TranscriptionResult:
        """
        Asynchronously transcribe an audio segment.
        
        Args:
            audio_segment: AudioSegment to transcribe
            
        Returns:
            TranscriptionResult
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self.transcribe_segment, 
            audio_segment
        )
        return result
    
    def batch_transcribe(
        self, 
        audio_segments: List[AudioSegment],
        max_concurrent: int = None
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio segments in parallel.
        
        Args:
            audio_segments: List of AudioSegments to transcribe
            max_concurrent: Maximum number of concurrent transcriptions
            
        Returns:
            List of TranscriptionResults in the same order as input
        """
        if max_concurrent is None:
            max_concurrent = min(len(audio_segments), self.max_workers)
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(self.transcribe_segment, segment)
                for segment in audio_segments
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30.0)  # 30 second timeout per segment
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in batch transcription: {e}")
                    # Add empty result to maintain order
                    results.append(TranscriptionResult(
                        text="", start_time=0.0, end_time=0.0, 
                        confidence=0.0, language="en", segment_id="error",
                        word_timestamps=[]
                    ))
            
            return results
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'processed_segments': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'errors': 0
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Whisper processor cleaned up")
