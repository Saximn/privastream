"""
Ultra Low-Latency Audio PII Redaction Engine
Optimized for real-time streaming with sub-second latency
"""

import asyncio
import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
import uuid
import threading
import queue
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
import whisper
from transformers import AutoTokenizer, AutoModelForTokenClassification
import librosa
import soundfile as sf

from pipeline_types import AudioSegment, TranscriptionResult, PIIDetection, PIIType
from realtime_audio_redactor import AudioRedactionResult


@dataclass
class StreamingChunk:
    """Smaller audio chunk for streaming processing"""
    audio_data: np.ndarray
    chunk_id: str
    start_time: float
    end_time: float
    is_speech: bool = True


@dataclass 
class ProcessingMetrics:
    """Metrics for latency monitoring"""
    chunk_latency: float = 0.0
    whisper_latency: float = 0.0
    pii_latency: float = 0.0
    total_latency: float = 0.0
    queue_time: float = 0.0
    throughput: float = 0.0


class VoiceActivityDetector:
    """Lightweight voice activity detection to skip silent segments"""
    
    def __init__(self, sample_rate: int = 16000, frame_duration: float = 0.03):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_length = int(sample_rate * frame_duration)
        
    def is_speech(self, audio: np.ndarray, threshold: float = 0.02) -> bool:
        """Fast speech detection using energy and spectral features"""
        if len(audio) == 0:
            return False
            
        # Energy-based detection
        energy = np.mean(audio ** 2)
        if energy < threshold:
            return False
        
        # Spectral centroid for voice characteristics
        if len(audio) >= 512:
            stft = np.abs(librosa.stft(audio, n_fft=512, hop_length=256))
            spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=self.sample_rate)[0]
            
            # Voice typically has spectral centroid between 1000-4000 Hz
            avg_centroid = np.mean(spectral_centroid)
            if 1000 <= avg_centroid <= 4000:
                return True
                
        return energy > threshold * 2  # Higher threshold if no spectral features


class OptimizedWhisperProcessor:
    """Optimized Whisper processor with model caching and batch processing"""
    
    def __init__(
        self,
        model_name: str = "base",  # Use faster model by default
        device: str = "cuda",
        compute_type: str = "float16",  # Use FP16 for speed
        max_workers: int = 2
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load optimized model
        self.model = self._load_optimized_model()
        
        # Processing pools
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = asyncio.Queue(maxsize=10)
        
        # Cache for repeated transcriptions
        self.transcription_cache = {}
        self.cache_max_size = 100
        
    def _load_optimized_model(self):
        """Load and optimize Whisper model"""
        try:
            model = whisper.load_model(self.model_name, device=self.device)
            
            # Enable optimizations
            if hasattr(model, 'half') and self.compute_type == "float16":
                model = model.half()
                
            # Enable torch compile for newer PyTorch versions
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode="reduce-overhead")
            except:
                pass
                
            return model
        except Exception as e:
            self.logger.error(f"Failed to load optimized model: {e}")
            raise
    
    async def transcribe_chunk_async(self, chunk: StreamingChunk) -> Optional[TranscriptionResult]:
        """Asynchronously transcribe audio chunk"""
        if not chunk.is_speech:
            return None
            
        # Check cache first
        audio_hash = hash(chunk.audio_data.tobytes())
        if audio_hash in self.transcription_cache:
            return self.transcription_cache[audio_hash]
        
        start_time = time.time()
        
        try:
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._transcribe_sync,
                chunk
            )
            
            # Cache result
            if len(self.transcription_cache) < self.cache_max_size:
                self.transcription_cache[audio_hash] = result
            
            processing_time = time.time() - start_time
            self.logger.debug(f"Transcribed chunk in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return None
    
    def _transcribe_sync(self, chunk: StreamingChunk) -> TranscriptionResult:
        """Synchronous transcription for thread pool"""
        options = {
            "language": "en",
            "fp16": self.compute_type == "float16",
            "word_timestamps": True,
            "condition_on_previous_text": False,  # Faster processing
            "no_speech_threshold": 0.6
        }
        
        with torch.no_grad():
            result = self.model.transcribe(chunk.audio_data, **options)
        
        # Extract word timestamps
        word_timestamps = []
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        word_timestamps.append({
                            'word': word.get('word', ''),
                            'start': word.get('start', 0.0) + chunk.start_time,
                            'end': word.get('end', 0.0) + chunk.start_time,
                            'probability': word.get('probability', 1.0)
                        })
        
        return TranscriptionResult(
            text=result["text"].strip(),
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            confidence=self._calculate_confidence(result),
            language=result.get("language", "en"),
            segment_id=chunk.chunk_id,
            word_timestamps=word_timestamps
        )
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate confidence from Whisper result"""
        if "segments" not in result:
            return 1.0
        
        total_prob = 0.0
        total_words = 0
        
        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    if "probability" in word:
                        total_prob += word["probability"]
                        total_words += 1
        
        return total_prob / total_words if total_words > 0 else 1.0


class LightweightPIIDetector:
    """Lightweight PII detector optimized for speed"""
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",  # Use base instead of large
        device: str = "cuda",
        confidence_threshold: float = 0.8,  # Higher threshold for speed
        max_length: int = 256  # Shorter sequences for speed
    ):
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load lightweight model
        self.tokenizer, self.model = self._load_lightweight_model()
        
        # PII pattern cache
        self.pattern_cache = {}
        
        # Fast regex patterns for common PII
        import re
        self.quick_patterns = {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE_NUM: re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'),
            PIIType.USERNAME: re.compile(r'@\w+'),
        }
    
    def _load_lightweight_model(self):
        """Load optimized lightweight model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            model.to(self.device)
            model.eval()
            
            # Enable optimization
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="max-autotune")
                except:
                    pass
            
            return tokenizer, model
            
        except Exception as e:
            self.logger.error(f"Failed to load PII model: {e}")
            raise
    
    async def detect_pii_async(self, transcription: TranscriptionResult) -> List[PIIDetection]:
        """Async PII detection with fast preprocessing"""
        if not transcription.text.strip():
            return []
        
        # Quick regex-based detection first
        quick_detections = self._quick_regex_detection(transcription.text)
        if quick_detections:
            return quick_detections
        
        # Full model detection for complex cases
        return await self._model_detection_async(transcription)
    
    def _quick_regex_detection(self, text: str) -> List[PIIDetection]:
        """Fast regex-based PII detection"""
        detections = []
        
        for pii_type, pattern in self.quick_patterns.items():
            for match in pattern.finditer(text):
                detection = PIIDetection(
                    pii_type=pii_type,
                    text=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.95,  # High confidence for regex matches
                    start_time=0.0,  # Will be calculated later
                    end_time=0.0,
                    word_indices=[]
                )
                detections.append(detection)
        
        return detections
    
    async def _model_detection_async(self, transcription: TranscriptionResult) -> List[PIIDetection]:
        """Full model-based detection for complex cases"""
        # This would use the full model - simplified for now
        return []


class UltraLowLatencyAudioRedactor:
    """Ultra low-latency audio redaction engine with streaming processing"""
    
    def __init__(
        self,
        whisper_model: str = "base",
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,  # 500ms chunks instead of 3s
        overlap_duration: float = 0.1,  # 100ms overlap
        max_parallel_chunks: int = 4
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.max_parallel_chunks = max_parallel_chunks
        
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.overlap_samples = int(sample_rate * overlap_duration)
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize optimized components
        self.vad = VoiceActivityDetector(sample_rate)
        self.whisper = OptimizedWhisperProcessor(whisper_model, device)
        self.pii_detector = LightweightPIIDetector(device=device)
        
        # Processing pipeline
        self.audio_buffer = deque(maxlen=self.sample_rate * 10)  # 10-second circular buffer
        self.chunk_counter = 0
        self.processing_queue = asyncio.Queue(maxsize=max_parallel_chunks * 2)
        self.result_queue = asyncio.Queue(maxsize=max_parallel_chunks * 2)
        
        # Pipeline workers
        self.workers_running = False
        self.pipeline_tasks = []
        
        # Performance metrics
        self.metrics = ProcessingMetrics()
        self.chunk_times = deque(maxlen=100)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Ultra low-latency redactor initialized: {chunk_duration:.1f}s chunks")
    
    async def start_pipeline(self):
        """Start the processing pipeline workers"""
        if self.workers_running:
            return
            
        self.workers_running = True
        
        # Start parallel processing workers
        for i in range(self.max_parallel_chunks):
            task = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self.pipeline_tasks.append(task)
        
        # Start result aggregator
        result_task = asyncio.create_task(self._result_aggregator())
        self.pipeline_tasks.append(result_task)
        
        self.logger.info(f"Started {len(self.pipeline_tasks)} pipeline workers")
    
    async def stop_pipeline(self):
        """Stop the processing pipeline"""
        self.workers_running = False
        
        # Cancel all tasks
        for task in self.pipeline_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self.pipeline_tasks, return_exceptions=True)
        self.pipeline_tasks.clear()
        
        self.logger.info("Pipeline stopped")
    
    async def add_audio_data_streaming(self, audio_data: bytes) -> None:
        """Add audio data for streaming processing"""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to circular buffer
        self.audio_buffer.extend(audio_array)
        
        # Process complete chunks
        while len(self.audio_buffer) >= self.chunk_samples:
            await self._extract_and_queue_chunk()
    
    async def _extract_and_queue_chunk(self):
        """Extract chunk from buffer and queue for processing"""
        if len(self.audio_buffer) < self.chunk_samples:
            return
        
        # Extract chunk
        chunk_data = np.array(list(self.audio_buffer)[:self.chunk_samples])
        
        # Voice activity detection
        is_speech = self.vad.is_speech(chunk_data)
        
        # Create chunk
        start_time = self.chunk_counter * (self.chunk_duration - self.overlap_duration)
        chunk = StreamingChunk(
            audio_data=chunk_data,
            chunk_id=f"chunk_{self.chunk_counter}_{int(time.time() * 1000)}",
            start_time=start_time,
            end_time=start_time + self.chunk_duration,
            is_speech=is_speech
        )
        
        # Queue for processing
        try:
            self.processing_queue.put_nowait((time.time(), chunk))
        except asyncio.QueueFull:
            self.logger.warning("Processing queue full, dropping chunk")
        
        # Advance buffer
        advance_samples = self.chunk_samples - self.overlap_samples
        for _ in range(advance_samples):
            if self.audio_buffer:
                self.audio_buffer.popleft()
        
        self.chunk_counter += 1
    
    async def _processing_worker(self, worker_id: str):
        """Worker that processes chunks in parallel"""
        self.logger.info(f"Processing worker {worker_id} started")
        
        while self.workers_running:
            try:
                # Get chunk from queue
                queue_time, chunk = await asyncio.wait_for(
                    self.processing_queue.get(), 
                    timeout=1.0
                )
                
                # Process chunk
                result = await self._process_chunk(chunk, queue_time)
                
                if result:
                    await self.result_queue.put(result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    async def _process_chunk(self, chunk: StreamingChunk, queue_time: float) -> Optional[AudioRedactionResult]:
        """Process a single chunk through the pipeline"""
        start_time = time.time()
        
        try:
            # Skip non-speech chunks
            if not chunk.is_speech:
                return None
            
            # Step 1: Transcribe
            whisper_start = time.time()
            transcription = await self.whisper.transcribe_chunk_async(chunk)
            whisper_time = time.time() - whisper_start
            
            if not transcription or not transcription.text.strip():
                return None
            
            # Step 2: PII Detection
            pii_start = time.time()
            detections = await self.pii_detector.detect_pii_async(transcription)
            pii_time = time.time() - pii_start
            
            # Step 3: Create redacted audio (simplified)
            redacted_audio = self._create_fast_redacted_audio(chunk.audio_data, detections)
            
            total_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(queue_time, whisper_time, pii_time, total_time)
            
            # Create result
            result = AudioRedactionResult(
                segment_id=chunk.chunk_id,
                original_audio=self._audio_to_bytes(chunk.audio_data),
                redacted_audio=self._audio_to_bytes(redacted_audio),
                transcription=transcription.text,
                redacted_transcription=self._create_redacted_text(transcription.text, detections),
                pii_detections=detections,
                timestamp=time.time(),
                processing_time=total_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chunk processing failed: {e}")
            return None
    
    def _create_fast_redacted_audio(self, audio: np.ndarray, detections: List[PIIDetection]) -> np.ndarray:
        """Fast audio redaction using noise replacement"""
        if not detections:
            return audio
        
        redacted = audio.copy()
        
        for detection in detections:
            # Estimate audio boundaries (simplified)
            start_sample = int(detection.start_time * self.sample_rate)
            end_sample = int(detection.end_time * self.sample_rate)
            
            # Clamp to audio bounds
            start_sample = max(0, min(start_sample, len(redacted)))
            end_sample = max(start_sample, min(end_sample, len(redacted)))
            
            if start_sample < end_sample:
                # Replace with low-level noise
                noise_level = np.std(audio) * 0.05
                noise = np.random.normal(0, noise_level, end_sample - start_sample)
                redacted[start_sample:end_sample] = noise.astype(np.float32)
        
        return redacted
    
    def _create_redacted_text(self, text: str, detections: List[PIIDetection]) -> str:
        """Create redacted text"""
        if not detections:
            return text
        
        # Sort by position (reverse order for replacement)
        sorted_detections = sorted(detections, key=lambda x: x.start_char, reverse=True)
        
        redacted = text
        for detection in sorted_detections:
            redacted = (
                redacted[:detection.start_char] + 
                f"[{detection.pii_type.value}]" + 
                redacted[detection.end_char:]
            )
        
        return redacted
    
    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio to bytes"""
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    def _update_metrics(self, queue_time: float, whisper_time: float, pii_time: float, total_time: float):
        """Update performance metrics"""
        current_time = time.time()
        
        self.chunk_times.append(total_time)
        self.metrics.chunk_latency = total_time
        self.metrics.whisper_latency = whisper_time
        self.metrics.pii_latency = pii_time
        self.metrics.total_latency = current_time - queue_time
        self.metrics.queue_time = current_time - queue_time - total_time
        
        if self.chunk_times:
            self.metrics.throughput = len(self.chunk_times) / sum(self.chunk_times)
    
    async def _result_aggregator(self):
        """Aggregate and deliver results"""
        while self.workers_running:
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                
                # Deliver result (placeholder - would integrate with existing system)
                self.logger.info(
                    f"Result: {result.segment_id}, "
                    f"PII: {len(result.pii_detections)}, "
                    f"Latency: {result.processing_time:.3f}s"
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Result aggregator error: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "chunk_latency_ms": self.metrics.chunk_latency * 1000,
            "whisper_latency_ms": self.metrics.whisper_latency * 1000,
            "pii_latency_ms": self.metrics.pii_latency * 1000,
            "total_latency_ms": self.metrics.total_latency * 1000,
            "queue_time_ms": self.metrics.queue_time * 1000,
            "throughput_chunks_per_sec": self.metrics.throughput,
            "chunk_duration_ms": self.chunk_duration * 1000,
            "processing_efficiency": (
                self.chunk_duration / self.metrics.chunk_latency 
                if self.metrics.chunk_latency > 0 else 0
            ),
            "buffer_size": len(self.audio_buffer),
            "queue_size": self.processing_queue.qsize()
        }


# Example usage and testing
async def main():
    """Test the ultra low-latency redactor"""
    logging.basicConfig(level=logging.INFO)
    
    # Create redactor with 500ms chunks
    redactor = UltraLowLatencyAudioRedactor(
        whisper_model="base",
        chunk_duration=0.5,
        overlap_duration=0.1,
        max_parallel_chunks=4
    )
    
    # Start pipeline
    await redactor.start_pipeline()
    
    try:
        # Simulate streaming audio
        for i in range(10):
            # Generate 100ms of test audio
            test_audio = np.random.normal(0, 0.1, 1600).astype(np.float32)  # 100ms at 16kHz
            audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
            
            await redactor.add_audio_data_streaming(audio_bytes)
            await asyncio.sleep(0.1)  # Simulate real-time streaming
        
        # Wait for processing to complete
        await asyncio.sleep(2.0)
        
        # Print performance metrics
        metrics = redactor.get_performance_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    finally:
        await redactor.stop_pipeline()


if __name__ == "__main__":
    asyncio.run(main())