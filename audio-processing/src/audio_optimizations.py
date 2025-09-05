"""
Audio Processing Optimizations for Ultra Low-Latency Pipeline
Reduces format conversions, implements efficient audio operations, and optimizes memory usage
"""

import numpy as np
import torch
import logging
import time
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass
from scipy import signal
import librosa
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    channels: int = 1
    dtype: np.dtype = np.float32
    chunk_size: int = 8192
    enable_vad: bool = True
    enable_noise_suppression: bool = True
    enable_echo_cancellation: bool = False


class AudioBuffer:
    """High-performance circular audio buffer with zero-copy operations"""
    
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.lock = threading.RLock()
        
    def write(self, data: np.ndarray) -> int:
        """Write data to buffer, returns number of samples written"""
        with self.lock:
            data = data.astype(np.float32)
            samples_to_write = min(len(data), self.max_samples - self.size)
            
            if samples_to_write == 0:
                return 0
            
            # Handle wrap-around
            if self.write_pos + samples_to_write <= self.max_samples:
                self.buffer[self.write_pos:self.write_pos + samples_to_write] = data[:samples_to_write]
            else:
                # Split write
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:samples_to_write - first_part] = data[first_part:samples_to_write]
            
            self.write_pos = (self.write_pos + samples_to_write) % self.max_samples
            self.size += samples_to_write
            
            return samples_to_write
    
    def read(self, num_samples: int, consume: bool = True) -> Optional[np.ndarray]:
        """Read data from buffer"""
        with self.lock:
            if self.size < num_samples:
                return None
            
            result = np.zeros(num_samples, dtype=np.float32)
            
            # Handle wrap-around
            if self.read_pos + num_samples <= self.max_samples:
                result[:] = self.buffer[self.read_pos:self.read_pos + num_samples]
            else:
                # Split read
                first_part = self.max_samples - self.read_pos
                result[:first_part] = self.buffer[self.read_pos:]
                result[first_part:] = self.buffer[:num_samples - first_part]
            
            if consume:
                self.read_pos = (self.read_pos + num_samples) % self.max_samples
                self.size -= num_samples
            
            return result
    
    def peek(self, num_samples: int) -> Optional[np.ndarray]:
        """Peek at data without consuming it"""
        return self.read(num_samples, consume=False)
    
    def available_samples(self) -> int:
        """Get number of available samples"""
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.read_pos = 0
            self.write_pos = 0
            self.size = 0


class FastVAD:
    """Ultra-fast Voice Activity Detection using energy and spectral features"""
    
    def __init__(self, sample_rate: int = 16000, frame_duration: float = 0.025):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_length = int(sample_rate * frame_duration)
        self.hop_length = self.frame_length // 2
        
        # Adaptive thresholds
        self.energy_threshold = 0.01
        self.spectral_threshold = 1200.0
        self.zero_crossing_threshold = 0.1
        
        # Background noise estimation
        self.background_energy = 0.001
        self.background_alpha = 0.95
        
        # Pre-compute window
        self.window = np.hanning(self.frame_length)
        
    def detect_speech(self, audio: np.ndarray) -> Tuple[bool, float]:
        """Detect speech in audio chunk, returns (is_speech, confidence)"""
        if len(audio) < self.frame_length:
            return False, 0.0
        
        # Energy-based detection
        energy = np.mean(audio ** 2)
        
        # Update background noise estimate
        if energy < self.background_energy * 3:  # Likely background noise
            self.background_energy = self.background_alpha * self.background_energy + (1 - self.background_alpha) * energy
        
        # Dynamic threshold based on background noise
        dynamic_threshold = max(self.energy_threshold, self.background_energy * 5)
        
        if energy < dynamic_threshold:
            return False, 0.0
        
        # Spectral features for voice detection
        confidence = energy / dynamic_threshold
        
        try:
            # Zero crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
            
            # Spectral centroid (rough approximation)
            if len(audio) >= 256:
                windowed = audio[:len(audio) - len(audio) % self.frame_length].reshape(-1, self.frame_length)
                fft_data = np.abs(np.fft.fft(windowed * self.window, axis=1))
                freqs = np.fft.fftfreq(self.frame_length, 1/self.sample_rate)
                
                # Compute spectral centroid for each frame
                centroids = []
                for frame_fft in fft_data:
                    spectrum = frame_fft[:len(frame_fft)//2]
                    freqs_pos = freqs[:len(freqs)//2]
                    if np.sum(spectrum) > 0:
                        centroid = np.sum(freqs_pos * spectrum) / np.sum(spectrum)
                        centroids.append(centroid)
                
                if centroids:
                    avg_centroid = np.mean(centroids)
                    # Voice typically has centroid between 500-3000 Hz
                    if 500 <= avg_centroid <= 3000:
                        confidence *= 1.5
                    
                    # Zero crossing rate should be moderate for voice
                    if 0.05 <= zero_crossings <= 0.3:
                        confidence *= 1.2
            
        except Exception:
            pass  # Use energy-based detection as fallback
        
        is_speech = confidence > 1.0
        return is_speech, min(confidence, 1.0)
    
    def process_streaming(self, audio_buffer: AudioBuffer, chunk_duration: float = 0.1) -> List[Tuple[int, bool, float]]:
        """Process streaming audio from buffer"""
        chunk_samples = int(chunk_duration * self.sample_rate)
        results = []
        
        start_sample = 0
        while audio_buffer.available_samples() >= chunk_samples:
            chunk = audio_buffer.read(chunk_samples)
            if chunk is None:
                break
            
            is_speech, confidence = self.detect_speech(chunk)
            results.append((start_sample, is_speech, confidence))
            start_sample += chunk_samples
        
        return results


class AudioPreprocessor:
    """Optimized audio preprocessing with minimal allocations"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pre-compute filters
        self._init_filters()
        
        # Processing state
        self.dc_offset_alpha = 0.995
        self.dc_estimate = 0.0
        self.previous_sample = 0.0
        
        # Noise gate
        self.noise_gate_threshold = 0.001
        self.noise_gate_ratio = 0.1
        
    def _init_filters(self):
        """Initialize audio filters"""
        # High-pass filter for DC removal (butterworth, 80 Hz cutoff)
        self.hp_b, self.hp_a = signal.butter(2, 80.0, btype='high', fs=self.config.sample_rate)
        self.hp_zi = signal.lfilter_zi(self.hp_b, self.hp_a)
        
        # Low-pass filter for anti-aliasing
        self.lp_b, self.lp_a = signal.butter(4, self.config.sample_rate * 0.4, btype='low', fs=self.config.sample_rate)
        self.lp_zi = signal.lfilter_zi(self.lp_b, self.lp_a)
    
    def process_chunk(self, audio: np.ndarray, in_place: bool = True) -> np.ndarray:
        """Process audio chunk with optimizations"""
        if not in_place:
            audio = audio.copy()
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # DC removal using IIR filter
        if len(audio) > 0:
            audio, self.hp_zi = signal.lfilter(self.hp_b, self.hp_a, audio, zi=self.hp_zi)
        
        # Noise gate
        if self.config.enable_noise_suppression:
            mask = np.abs(audio) > self.noise_gate_threshold
            audio[~mask] *= self.noise_gate_ratio
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio *= 0.95 / max_val
        
        return audio
    
    def resample_optimized(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Optimized resampling using librosa"""
        if orig_sr == target_sr:
            return audio
        
        try:
            # Use librosa for high-quality resampling
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_fast')
        except Exception as e:
            self.logger.warning(f"Resampling failed, using scipy: {e}")
            # Fallback to scipy
            from scipy.signal import resample
            new_length = int(len(audio) * target_sr / orig_sr)
            return resample(audio, new_length).astype(np.float32)


class AudioFormatConverter:
    """Efficient audio format converter with minimal copying"""
    
    @staticmethod
    def bytes_to_float32(audio_bytes: bytes, dtype: str = 'int16') -> np.ndarray:
        """Convert bytes to float32 numpy array"""
        if dtype == 'int16':
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_int.astype(np.float32) / 32768.0
        elif dtype == 'int32':
            audio_int = np.frombuffer(audio_bytes, dtype=np.int32)
            return audio_int.astype(np.float32) / 2147483648.0
        elif dtype == 'float32':
            return np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    
    @staticmethod
    def float32_to_bytes(audio: np.ndarray, dtype: str = 'int16') -> bytes:
        """Convert float32 numpy array to bytes"""
        if dtype == 'int16':
            audio_int = (audio * 32767).astype(np.int16)
            return audio_int.tobytes()
        elif dtype == 'int32':
            audio_int = (audio * 2147483647).astype(np.int32)
            return audio_int.tobytes()
        elif dtype == 'float32':
            return audio.astype(np.float32).tobytes()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    
    @staticmethod
    def numpy_to_torch(audio: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Convert numpy array to torch tensor efficiently"""
        tensor = torch.from_numpy(audio)
        if device != "cpu":
            tensor = tensor.to(device, non_blocking=True)
        return tensor
    
    @staticmethod
    def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array efficiently"""
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return tensor.detach().numpy()


class StreamingAudioProcessor:
    """High-performance streaming audio processor"""
    
    def __init__(
        self, 
        config: AudioConfig,
        buffer_duration: float = 10.0,
        processing_interval: float = 0.1
    ):
        self.config = config
        self.buffer_duration = buffer_duration
        self.processing_interval = processing_interval
        
        # Components
        self.buffer = AudioBuffer(buffer_duration, config.sample_rate)
        self.vad = FastVAD(config.sample_rate) if config.enable_vad else None
        self.preprocessor = AudioPreprocessor(config)
        self.converter = AudioFormatConverter()
        
        # Processing thread
        self.processing_thread = None
        self.is_running = False
        self.callbacks = []
        
        # Performance metrics
        self.total_samples_processed = 0
        self.processing_times = []
        self.max_processing_time_samples = 100
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_callback(self, callback: callable):
        """Add callback for processed audio chunks"""
        self.callbacks.append(callback)
    
    def start_processing(self):
        """Start the processing thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("Started streaming audio processor")
    
    def stop_processing(self):
        """Stop the processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        self.logger.info("Stopped streaming audio processor")
    
    def add_audio(self, audio_bytes: bytes) -> int:
        """Add audio data to the processing buffer"""
        audio_array = self.converter.bytes_to_float32(audio_bytes)
        return self.buffer.write(audio_array)
    
    def _processing_loop(self):
        """Main processing loop"""
        chunk_samples = int(self.processing_interval * self.config.sample_rate)
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Check if we have enough data
                if self.buffer.available_samples() < chunk_samples:
                    time.sleep(0.01)  # Short sleep to prevent busy waiting
                    continue
                
                # Read chunk
                chunk = self.buffer.read(chunk_samples)
                if chunk is None:
                    continue
                
                # Process chunk
                processed_chunk = self._process_chunk(chunk)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > self.max_processing_time_samples:
                    self.processing_times.pop(0)
                
                self.total_samples_processed += len(chunk)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(processed_chunk)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)
    
    def _process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single audio chunk"""
        # Preprocess audio
        processed_audio = self.preprocessor.process_chunk(chunk, in_place=False)
        
        # Voice activity detection
        is_speech = True
        vad_confidence = 1.0
        
        if self.vad:
            is_speech, vad_confidence = self.vad.detect_speech(processed_audio)
        
        # Convert to different formats if needed
        audio_bytes = self.converter.float32_to_bytes(processed_audio)
        
        return {
            'audio_array': processed_audio,
            'audio_bytes': audio_bytes,
            'is_speech': is_speech,
            'vad_confidence': vad_confidence,
            'sample_rate': self.config.sample_rate,
            'channels': self.config.channels,
            'timestamp': time.time()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.processing_times:
            return {}
        
        avg_processing_time = np.mean(self.processing_times)
        max_processing_time = np.max(self.processing_times)
        
        return {
            'total_samples_processed': self.total_samples_processed,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': max_processing_time * 1000,
            'real_time_factor': self.processing_interval / avg_processing_time if avg_processing_time > 0 else float('inf'),
            'buffer_utilization': self.buffer.available_samples() / self.buffer.max_samples,
            'is_real_time': avg_processing_time < self.processing_interval
        }


# Example usage and testing
def test_audio_optimizations():
    """Test the audio optimization components"""
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        enable_vad=True,
        enable_noise_suppression=True
    )
    
    # Create streaming processor
    processor = StreamingAudioProcessor(config, processing_interval=0.1)
    
    # Add callback to handle processed chunks
    def audio_callback(chunk_data):
        is_speech = chunk_data['is_speech']
        confidence = chunk_data['vad_confidence']
        print(f"Processed chunk: speech={is_speech}, confidence={confidence:.2f}")
    
    processor.add_callback(audio_callback)
    
    # Start processing
    processor.start_processing()
    
    try:
        # Simulate streaming audio
        for i in range(50):  # 5 seconds of audio
            # Generate test audio (mix of speech-like signal and silence)
            if i % 10 < 5:  # Alternate between signal and silence
                # Speech-like signal
                t = np.linspace(0, 0.1, 1600)  # 100ms at 16kHz
                frequency = 440 + i * 10  # Varying frequency
                audio_signal = 0.3 * np.sin(2 * np.pi * frequency * t)
                # Add some modulation
                audio_signal *= (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
            else:
                # Silence with small noise
                audio_signal = np.random.normal(0, 0.01, 1600)
            
            # Convert to bytes
            audio_bytes = (audio_signal * 32767).astype(np.int16).tobytes()
            
            # Add to processor
            processor.add_audio(audio_bytes)
            
            # Simulate real-time streaming
            time.sleep(0.1)
        
        # Wait for processing to complete
        time.sleep(1.0)
        
        # Print performance metrics
        metrics = processor.get_performance_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    finally:
        processor.stop_processing()


if __name__ == "__main__":
    test_audio_optimizations()