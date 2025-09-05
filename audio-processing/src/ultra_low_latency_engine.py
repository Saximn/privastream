"""
Ultra Low-Latency Audio PII Redaction Engine - Integrated System
Combines all optimizations for sub-200ms latency processing
"""

import asyncio
import torch
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Import our optimization modules
from low_latency_redactor import (
    StreamingChunk, ProcessingMetrics, VoiceActivityDetector,
    OptimizedWhisperProcessor, LightweightPIIDetector
)
from model_optimizations import (
    ModelOptimizer, InferenceCache, BatchProcessor, 
    PredictiveProcessor, AdaptiveQualityManager, 
    MemoryOptimizer, OptimizedModelManager
)
from audio_optimizations import (
    AudioConfig, AudioBuffer, FastVAD, AudioPreprocessor,
    AudioFormatConverter, StreamingAudioProcessor
)
from pipeline_types import PIIDetection, PIIType
from realtime_audio_redactor import AudioRedactionResult


@dataclass
class UltraLatencyConfig:
    """Configuration for ultra low-latency processing"""
    # Audio processing
    sample_rate: int = 16000
    chunk_duration: float = 0.2  # 200ms chunks for ultra-low latency
    overlap_duration: float = 0.05  # 50ms overlap
    
    # Model configuration
    whisper_model: str = "tiny"  # Use tiny model for speed
    pii_model_name: str = "microsoft/deberta-v3-base"  # Base model for speed
    
    # Processing configuration
    max_parallel_chunks: int = 6
    processing_threads: int = 4
    enable_caching: bool = True
    enable_batching: bool = True
    enable_prediction: bool = True
    
    # Quality vs speed tradeoffs
    pii_confidence_threshold: float = 0.85  # Higher threshold for speed
    vad_threshold: float = 0.02
    enable_aggressive_optimizations: bool = True
    
    # Target performance
    target_latency_ms: float = 200.0
    max_queue_size: int = 20


class UltraLatencyMetrics:
    """Comprehensive metrics tracking"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.chunk_latencies = []
        self.whisper_latencies = []
        self.pii_latencies = []
        self.total_latencies = []
        self.queue_times = []
        self.throughput_samples = []
        
        self.cache_hits = 0
        self.cache_misses = 0
        self.chunks_processed = 0
        self.chunks_skipped_vad = 0
        self.chunks_skipped_prediction = 0
        self.pii_detections_total = 0
        
    def add_processing_time(self, chunk_time: float, whisper_time: float, 
                          pii_time: float, total_time: float, queue_time: float):
        """Add processing time measurements"""
        self._add_to_window(self.chunk_latencies, chunk_time)
        self._add_to_window(self.whisper_latencies, whisper_time)
        self._add_to_window(self.pii_latencies, pii_time)
        self._add_to_window(self.total_latencies, total_time)
        self._add_to_window(self.queue_times, queue_time)
        
        self.chunks_processed += 1
        
    def _add_to_window(self, window: List[float], value: float):
        """Add value to sliding window"""
        window.append(value)
        if len(window) > self.window_size:
            window.pop(0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        def avg(lst): return np.mean(lst) if lst else 0.0
        def p95(lst): return np.percentile(lst, 95) if lst else 0.0
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "latency_ms": {
                "chunk_avg": avg(self.chunk_latencies) * 1000,
                "whisper_avg": avg(self.whisper_latencies) * 1000,
                "pii_avg": avg(self.pii_latencies) * 1000,
                "total_avg": avg(self.total_latencies) * 1000,
                "queue_avg": avg(self.queue_times) * 1000,
                "total_p95": p95(self.total_latencies) * 1000,
            },
            "throughput": {
                "chunks_per_second": len(self.total_latencies) / sum(self.total_latencies) if self.total_latencies else 0,
                "real_time_factor": 0.2 / avg(self.total_latencies) if avg(self.total_latencies) > 0 else float('inf'),
            },
            "efficiency": {
                "cache_hit_rate": cache_hit_rate,
                "chunks_processed": self.chunks_processed,
                "chunks_skipped_vad": self.chunks_skipped_vad,
                "chunks_skipped_prediction": self.chunks_skipped_prediction,
                "skip_rate": (self.chunks_skipped_vad + self.chunks_skipped_prediction) / max(1, self.chunks_processed),
            },
            "detection": {
                "total_pii_detections": self.pii_detections_total,
                "pii_per_chunk": self.pii_detections_total / max(1, self.chunks_processed),
            }
        }


class UltraLowLatencyEngine:
    """Ultra low-latency audio PII redaction engine"""
    
    def __init__(self, config: UltraLatencyConfig = None):
        self.config = config or UltraLatencyConfig()
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Audio processing components
        self.audio_config = AudioConfig(
            sample_rate=self.config.sample_rate,
            enable_vad=True,
            enable_noise_suppression=True
        )
        
        self.audio_buffer = AudioBuffer(
            max_duration=30.0, 
            sample_rate=self.config.sample_rate
        )
        
        self.vad = FastVAD(self.config.sample_rate)
        self.preprocessor = AudioPreprocessor(self.audio_config)
        self.format_converter = AudioFormatConverter()
        
        # Model components (initialized lazily for performance)
        self._whisper_processor = None
        self._pii_detector = None
        self._model_manager = None
        
        # Processing pipeline
        self.chunk_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.workers = []
        self.is_running = False
        
        # Metrics and monitoring
        self.metrics = UltraLatencyMetrics()
        self.adaptive_manager = AdaptiveQualityManager()
        self.adaptive_manager.target_latency = self.config.target_latency_ms / 1000.0
        
        # Result callbacks
        self.result_callbacks = []
        
        # Threading
        self.chunk_counter = 0
        self.processing_lock = threading.Lock()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Ultra low-latency engine initialized: {self.config.chunk_duration*1000:.0f}ms chunks")
    
    @property
    def whisper_processor(self):
        """Lazy initialization of Whisper processor"""
        if self._whisper_processor is None:
            model_name = self.adaptive_manager.get_processing_config().get("whisper_model", self.config.whisper_model)
            self._whisper_processor = OptimizedWhisperProcessor(
                model_name=model_name,
                device=self.device,
                max_workers=2
            )
        return self._whisper_processor
    
    @property
    def pii_detector(self):
        """Lazy initialization of PII detector"""
        if self._pii_detector is None:
            self._pii_detector = LightweightPIIDetector(
                model_name=self.config.pii_model_name,
                device=self.device,
                confidence_threshold=self.config.pii_confidence_threshold
            )
        return self._pii_detector
    
    @property
    def model_manager(self):
        """Lazy initialization of model manager"""
        if self._model_manager is None:
            self._model_manager = OptimizedModelManager()
        return self._model_manager
    
    async def start(self):
        """Start the processing pipeline"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start processing workers
        for i in range(self.config.processing_threads):
            worker = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start result aggregator
        aggregator = asyncio.create_task(self._result_aggregator())
        self.workers.append(aggregator)
        
        self.logger.info(f"Started {len(self.workers)} workers")
    
    async def stop(self):
        """Stop the processing pipeline"""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        # Cleanup GPU memory
        MemoryOptimizer.cleanup_gpu_memory()
        
        self.logger.info("Engine stopped")
    
    def add_result_callback(self, callback: Callable[[AudioRedactionResult], None]):
        """Add callback for processing results"""
        self.result_callbacks.append(callback)
    
    async def process_audio_stream(self, audio_bytes: bytes) -> None:
        """Process streaming audio data"""
        # Convert to numpy array
        audio_array = self.format_converter.bytes_to_float32(audio_bytes)
        
        # Add to buffer
        samples_written = self.audio_buffer.write(audio_array)
        if samples_written < len(audio_array):
            self.logger.warning("Audio buffer overflow, dropping samples")
        
        # Extract and queue chunks
        await self._extract_chunks()
    
    async def _extract_chunks(self):
        """Extract chunks from buffer and queue for processing"""
        chunk_samples = int(self.config.chunk_duration * self.config.sample_rate)
        
        while self.audio_buffer.available_samples() >= chunk_samples:
            # Extract chunk
            chunk_data = self.audio_buffer.read(chunk_samples)
            if chunk_data is None:
                break
            
            # Preprocess audio
            processed_audio = self.preprocessor.process_chunk(chunk_data, in_place=False)
            
            # Voice activity detection (ultra-fast)
            is_speech, vad_confidence = self.vad.detect_speech(processed_audio)
            
            # Skip non-speech if enabled
            if not is_speech and vad_confidence < self.config.vad_threshold:
                self.metrics.chunks_skipped_vad += 1
                self.chunk_counter += 1
                continue
            
            # Create chunk
            start_time = self.chunk_counter * (self.config.chunk_duration - self.config.overlap_duration)
            chunk = StreamingChunk(
                audio_data=processed_audio,
                chunk_id=f"chunk_{self.chunk_counter}_{int(time.time() * 1000)}",
                start_time=start_time,
                end_time=start_time + self.config.chunk_duration,
                is_speech=is_speech
            )
            
            # Queue for processing
            try:
                await asyncio.wait_for(
                    self.chunk_queue.put((time.time(), chunk)), 
                    timeout=0.01
                )
            except asyncio.TimeoutError:
                self.logger.warning("Processing queue full, dropping chunk")
            
            self.chunk_counter += 1
    
    async def _processing_worker(self, worker_id: str):
        """Processing worker that handles chunks"""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get chunk from queue
                queue_time, chunk = await asyncio.wait_for(
                    self.chunk_queue.get(),
                    timeout=1.0
                )
                
                # Process chunk
                result = await self._process_chunk_optimized(chunk, queue_time)
                
                if result:
                    await self.result_queue.put(result)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    async def _process_chunk_optimized(self, chunk: StreamingChunk, queue_time: float) -> Optional[AudioRedactionResult]:
        """Process chunk with all optimizations"""
        start_time = time.time()
        
        try:
            # Step 1: Check predictive processor
            if self.config.enable_prediction and hasattr(self.model_manager, 'predictive_processor'):
                # Quick text estimation for prediction (placeholder)
                estimated_text = ""  # Would implement quick text estimation
                if self.model_manager.predictive_processor.should_skip_processing(estimated_text):
                    self.metrics.chunks_skipped_prediction += 1
                    return None
            
            # Step 2: Transcription with caching
            whisper_start = time.time()
            
            if self.config.enable_caching:
                cached_result = self.model_manager.cache.get(chunk.audio_data)
                if cached_result:
                    transcription_result = cached_result.get('transcription')
                    self.metrics.cache_hits += 1
                else:
                    self.metrics.cache_misses += 1
                    transcription_result = await self.whisper_processor.transcribe_chunk_async(chunk)
                    if transcription_result:
                        self.model_manager.cache.put(chunk.audio_data, {'transcription': transcription_result})
            else:
                transcription_result = await self.whisper_processor.transcribe_chunk_async(chunk)
            
            whisper_time = time.time() - whisper_start
            
            if not transcription_result or not transcription_result.text.strip():
                return None
            
            # Step 3: PII detection with optimizations
            pii_start = time.time()
            detections = await self.pii_detector.detect_pii_async(transcription_result)
            pii_time = time.time() - pii_start
            
            # Step 4: Create redacted audio (fast method)
            redacted_audio = self._create_optimized_redacted_audio(chunk.audio_data, detections)
            
            # Step 5: Create result
            total_time = time.time() - start_time
            queue_wait_time = start_time - queue_time
            
            # Update metrics
            self.metrics.add_processing_time(
                chunk_time=total_time,
                whisper_time=whisper_time,
                pii_time=pii_time,
                total_time=total_time + queue_wait_time,
                queue_time=queue_wait_time
            )
            self.metrics.pii_detections_total += len(detections)
            
            # Update adaptive quality manager
            self.adaptive_manager.update_metrics(total_time, self.chunk_queue.qsize())
            
            # Create result
            result = AudioRedactionResult(
                segment_id=chunk.chunk_id,
                original_audio=self.format_converter.float32_to_bytes(chunk.audio_data),
                redacted_audio=self.format_converter.float32_to_bytes(redacted_audio),
                transcription=transcription_result.text,
                redacted_transcription=self._create_redacted_text(transcription_result.text, detections),
                pii_detections=detections,
                timestamp=time.time(),
                processing_time=total_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chunk processing failed: {e}")
            return None
    
    def _create_optimized_redacted_audio(self, audio: np.ndarray, detections: List[PIIDetection]) -> np.ndarray:
        """Create redacted audio with minimal processing"""
        if not detections:
            return audio
        
        redacted = audio.copy()
        
        for detection in detections:
            # Rough time-to-sample mapping (simplified)
            start_sample = max(0, int(detection.start_time * self.config.sample_rate))
            end_sample = min(len(redacted), int(detection.end_time * self.config.sample_rate))
            
            if start_sample < end_sample:
                # Replace with shaped noise matching local characteristics
                segment_length = end_sample - start_sample
                if segment_length > 0:
                    # Use local noise characteristics
                    local_noise_level = np.std(audio[max(0, start_sample-1000):start_sample+1000]) * 0.1
                    replacement = np.random.normal(0, local_noise_level, segment_length).astype(np.float32)
                    
                    # Apply envelope to avoid clicks
                    envelope_length = min(100, segment_length // 4)
                    if envelope_length > 0:
                        envelope = np.ones(segment_length)
                        envelope[:envelope_length] = np.linspace(0, 1, envelope_length)
                        envelope[-envelope_length:] = np.linspace(1, 0, envelope_length)
                        replacement *= envelope
                    
                    redacted[start_sample:end_sample] = replacement
        
        return redacted
    
    def _create_redacted_text(self, text: str, detections: List[PIIDetection]) -> str:
        """Create redacted text efficiently"""
        if not detections:
            return text
        
        # Sort by position (reverse for safe replacement)
        sorted_detections = sorted(detections, key=lambda x: x.start_char, reverse=True)
        
        redacted = text
        for detection in sorted_detections:
            redacted = (
                redacted[:detection.start_char] + 
                f"[{detection.pii_type.value}]" + 
                redacted[detection.end_char:]
            )
        
        return redacted
    
    async def _result_aggregator(self):
        """Aggregate and deliver results"""
        while self.is_running:
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                
                # Execute callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        self.logger.error(f"Result callback error: {e}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Result aggregator error: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        base_metrics = self.metrics.get_summary()
        
        # Add additional engine-specific metrics
        base_metrics.update({
            "engine": {
                "chunk_duration_ms": self.config.chunk_duration * 1000,
                "target_latency_ms": self.config.target_latency_ms,
                "queue_size": self.chunk_queue.qsize(),
                "result_queue_size": self.result_queue.qsize(),
                "buffer_utilization": self.audio_buffer.available_samples() / self.audio_buffer.max_samples,
                "is_running": self.is_running,
                "worker_count": len(self.workers),
            },
            "adaptive": self.adaptive_manager.get_status(),
        })
        
        # Add model manager stats if available
        if self._model_manager:
            base_metrics["model_manager"] = self.model_manager.get_comprehensive_stats()
        
        return base_metrics
    
    def get_latency_summary(self) -> Dict[str, float]:
        """Get key latency metrics"""
        metrics = self.get_performance_metrics()
        latency = metrics.get("latency_ms", {})
        
        return {
            "total_avg_latency_ms": latency.get("total_avg", 0.0),
            "total_p95_latency_ms": latency.get("total_p95", 0.0),
            "chunk_processing_ms": latency.get("chunk_avg", 0.0),
            "queue_wait_ms": latency.get("queue_avg", 0.0),
            "real_time_factor": metrics.get("throughput", {}).get("real_time_factor", 0.0),
            "target_latency_ms": self.config.target_latency_ms,
            "latency_target_met": latency.get("total_avg", float('inf')) <= self.config.target_latency_ms
        }


# Example usage and integration
async def demo_ultra_low_latency():
    """Demonstrate the ultra low-latency engine"""
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration optimized for speed
    config = UltraLatencyConfig(
        chunk_duration=0.15,  # 150ms chunks
        overlap_duration=0.03,  # 30ms overlap
        whisper_model="tiny",
        target_latency_ms=150.0,
        enable_aggressive_optimizations=True
    )
    
    # Create engine
    engine = UltraLowLatencyEngine(config)
    
    # Add result callback
    results_received = []
    def handle_result(result):
        results_received.append(result)
        print(f"Result: {result.segment_id}, PII: {len(result.pii_detections)}, "
              f"Latency: {result.processing_time*1000:.1f}ms")
    
    engine.add_result_callback(handle_result)
    
    # Start engine
    await engine.start()
    
    try:
        # Simulate streaming audio for 5 seconds
        print("Simulating 5 seconds of streaming audio...")
        
        for i in range(50):  # 50 x 100ms = 5 seconds
            # Generate test audio with speech characteristics
            duration = 0.1  # 100ms chunks
            samples = int(duration * config.sample_rate)
            
            if i % 20 < 10:  # Alternate speech and silence
                # Speech-like signal
                t = np.linspace(0, duration, samples)
                freq = 200 + i * 5  # Varying fundamental frequency
                
                # Create voice-like signal with harmonics
                signal = (
                    0.4 * np.sin(2 * np.pi * freq * t) +
                    0.2 * np.sin(2 * np.pi * freq * 2 * t) +
                    0.1 * np.sin(2 * np.pi * freq * 3 * t)
                )
                
                # Add formant-like filtering (simplified)
                signal *= (1 + 0.5 * np.sin(2 * np.pi * 8 * t))  # 8 Hz modulation
                
                # Add some noise
                signal += np.random.normal(0, 0.02, len(signal))
            else:
                # Background noise
                signal = np.random.normal(0, 0.005, samples)
            
            # Convert to bytes
            audio_bytes = (signal * 32767).astype(np.int16).tobytes()
            
            # Process
            await engine.process_audio_stream(audio_bytes)
            
            # Simulate real-time streaming
            await asyncio.sleep(0.1)
        
        # Wait for processing to complete
        await asyncio.sleep(2.0)
        
        # Print comprehensive metrics
        print(f"\n{'='*60}")
        print("ULTRA LOW-LATENCY ENGINE PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        latency_summary = engine.get_latency_summary()
        for key, value in latency_summary.items():
            if isinstance(value, bool):
                print(f"{key}: {'✅ YES' if value else '❌ NO'}")
            elif isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        print(f"\nResults received: {len(results_received)}")
        
        # Detailed metrics
        full_metrics = engine.get_performance_metrics()
        print(f"\nDetailed Performance Metrics:")
        print(f"Cache hit rate: {full_metrics['efficiency']['cache_hit_rate']:.2%}")
        print(f"Chunks skipped (VAD): {full_metrics['efficiency']['chunks_skipped_vad']}")
        print(f"Real-time factor: {full_metrics['throughput']['real_time_factor']:.1f}x")
        
    finally:
        await engine.stop()
        print("\nEngine stopped.")


if __name__ == "__main__":
    asyncio.run(demo_ultra_low_latency())