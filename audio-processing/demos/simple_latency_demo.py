"""
Simple Latency Reduction Demo
Demonstrates the key optimization techniques and their impact
"""

import asyncio
import time
import numpy as np
import logging
from typing import Dict, List
import sys
from pathlib import Path

# Add audio processing path
sys.path.insert(0, str(Path(__file__).parent / "audio-processing" / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_test_audio(duration: float = 0.2, sample_rate: int = 16000) -> np.ndarray:
    """Generate realistic speech-like test audio"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Create voice-like signal with harmonics
    fundamental = 200  # Hz
    voice_signal = (
        0.4 * np.sin(2 * np.pi * fundamental * t) +
        0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +
        0.2 * np.sin(2 * np.pi * fundamental * 3 * t)
    )
    
    # Add modulation for naturalness
    modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
    voice_signal *= modulation
    
    # Add background noise
    noise = np.random.normal(0, 0.05, samples)
    audio = voice_signal + noise
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


class OriginalProcessingSimulator:
    """Simulates the original processing pipeline with realistic delays"""
    
    def __init__(self):
        self.segment_duration = 3.0  # 3-second segments
        self.overlap_duration = 0.5
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        
    def add_audio_data(self, audio_data: np.ndarray) -> List[Dict]:
        """Simulate original processing with 3-second segments"""
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_data])
        
        results = []
        segment_samples = int(self.segment_duration * self.sample_rate)
        
        # Process complete segments
        while len(self.buffer) >= segment_samples:
            start_time = time.time()
            
            # Extract segment
            segment = self.buffer[:segment_samples]
            
            # Simulate processing delays
            whisper_delay = self._simulate_whisper_processing(segment)
            pii_delay = self._simulate_pii_detection()
            audio_redaction_delay = self._simulate_audio_redaction(segment)
            
            total_processing_time = whisper_delay + pii_delay + audio_redaction_delay
            
            results.append({
                'processing_time': total_processing_time,
                'whisper_time': whisper_delay,
                'pii_time': pii_delay,
                'audio_redaction_time': audio_redaction_delay,
                'segment_duration': self.segment_duration
            })
            
            # Advance buffer
            advance_samples = segment_samples - int(self.overlap_duration * self.sample_rate)
            self.buffer = self.buffer[advance_samples:]
        
        return results
    
    def _simulate_whisper_processing(self, audio_segment: np.ndarray) -> float:
        """Simulate Whisper large-v3 processing time"""
        # Simulate realistic Whisper processing time for 3-second audio
        base_time = 0.8  # Base processing time
        complexity_factor = len(audio_segment) / (3 * self.sample_rate)  # Adjust for length
        processing_time = base_time * complexity_factor + np.random.normal(0, 0.1)
        
        # Simulate actual processing by sleeping
        time.sleep(max(0.1, processing_time))  # Minimum 100ms
        return processing_time
    
    def _simulate_pii_detection(self) -> float:
        """Simulate DeBERTa v3 large PII detection"""
        # Simulate PII model processing time
        processing_time = 0.3 + np.random.normal(0, 0.05)  # ~300ms average
        time.sleep(max(0.05, processing_time))
        return processing_time
    
    def _simulate_audio_redaction(self, audio_segment: np.ndarray) -> float:
        """Simulate audio redaction processing"""
        # Simulate audio processing time
        processing_time = 0.05 + np.random.normal(0, 0.01)  # ~50ms average
        time.sleep(max(0.01, processing_time))
        return processing_time


class OptimizedProcessingSimulator:
    """Simulates the optimized ultra-low-latency pipeline"""
    
    def __init__(self):
        self.chunk_duration = 0.2  # 200ms chunks
        self.overlap_duration = 0.05  # 50ms overlap
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        self.cache = {}  # Simulation of inference cache
        self.vad_enabled = True
        
    async def add_audio_data_async(self, audio_data: np.ndarray) -> List[Dict]:
        """Process with optimizations"""
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_data])
        
        results = []
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        # Process complete chunks
        while len(self.buffer) >= chunk_samples:
            chunk = self.buffer[:chunk_samples]
            
            # Voice Activity Detection (ultra-fast)
            vad_time, is_speech = self._fast_vad(chunk)
            
            if not is_speech:
                # Skip processing for non-speech
                advance_samples = chunk_samples - int(self.overlap_duration * self.sample_rate)
                self.buffer = self.buffer[advance_samples:]
                continue
            
            start_time = time.time()
            
            # Optimized processing
            whisper_delay = await self._optimized_whisper_processing(chunk)
            pii_delay = await self._optimized_pii_detection(chunk)
            audio_redaction_delay = await self._fast_audio_redaction(chunk)
            
            total_processing_time = vad_time + whisper_delay + pii_delay + audio_redaction_delay
            
            results.append({
                'processing_time': total_processing_time,
                'whisper_time': whisper_delay,
                'pii_time': pii_delay,
                'audio_redaction_time': audio_redaction_delay,
                'vad_time': vad_time,
                'chunk_duration': self.chunk_duration,
                'cached': whisper_delay < 0.01  # Indicate if result was cached
            })
            
            # Advance buffer
            advance_samples = chunk_samples - int(self.overlap_duration * self.sample_rate)
            self.buffer = self.buffer[advance_samples:]
        
        return results
    
    def _fast_vad(self, audio_chunk: np.ndarray) -> tuple:
        """Ultra-fast Voice Activity Detection"""
        start_time = time.time()
        
        # Energy-based detection (very fast)
        energy = np.mean(audio_chunk ** 2)
        is_speech = energy > 0.01
        
        vad_time = time.time() - start_time
        return vad_time, is_speech
    
    async def _optimized_whisper_processing(self, audio_chunk: np.ndarray) -> float:
        """Optimized Whisper processing with caching"""
        # Check cache first
        chunk_hash = hash(audio_chunk.tobytes())
        if chunk_hash in self.cache:
            return 0.005  # Cache hit - very fast
        
        # Simulate Whisper Tiny model processing (much faster)
        base_time = 0.08  # Much faster than large model
        complexity_factor = len(audio_chunk) / (self.chunk_duration * self.sample_rate)
        processing_time = base_time * complexity_factor + np.random.normal(0, 0.01)
        
        # Simulate processing
        await asyncio.sleep(max(0.02, processing_time))
        
        # Cache result
        self.cache[chunk_hash] = "cached_result"
        
        return processing_time
    
    async def _optimized_pii_detection(self, audio_chunk: np.ndarray) -> float:
        """Optimized PII detection with fast patterns"""
        # Quick regex patterns first (very fast)
        quick_check_time = 0.002
        
        # Only do full model processing if needed (reduced model size)
        base_time = 0.05  # Much faster than large DeBERTa
        processing_time = quick_check_time + base_time + np.random.normal(0, 0.005)
        
        await asyncio.sleep(max(0.01, processing_time))
        return processing_time
    
    async def _fast_audio_redaction(self, audio_chunk: np.ndarray) -> float:
        """Fast audio redaction with minimal processing"""
        # Optimized audio redaction
        processing_time = 0.01 + np.random.normal(0, 0.002)
        await asyncio.sleep(max(0.005, processing_time))
        return processing_time


async def run_latency_comparison():
    """Run comprehensive latency comparison"""
    logger.info("Starting Latency Optimization Demonstration")
    logger.info("=" * 60)
    
    # Test parameters
    test_duration = 10.0  # 10 seconds of audio
    chunk_size = 0.1  # 100ms input chunks
    
    # Initialize processors
    original = OriginalProcessingSimulator()
    optimized = OptimizedProcessingSimulator()
    
    print(f"\nTest Test Configuration:")
    print(f"   • Test Duration: {test_duration}s")
    print(f"   • Input Chunk Size: {chunk_size * 1000}ms")
    print(f"   • Original: {original.segment_duration}s segments")
    print(f"   • Optimized: {optimized.chunk_duration * 1000}ms chunks")
    
    # Test Original System
    print(f"\n⏱️  Testing Original System...")
    original_results = []
    original_latencies = []
    original_start = time.time()
    
    total_chunks = int(test_duration / chunk_size)
    for i in range(total_chunks):
        # Generate test audio
        audio_chunk = generate_test_audio(chunk_size)
        
        # Process
        chunk_results = original.add_audio_data(audio_chunk)
        original_results.extend(chunk_results)
        
        for result in chunk_results:
            original_latencies.append(result['processing_time'])
        
        # Simulate real-time input
        await asyncio.sleep(chunk_size)
    
    original_total_time = time.time() - original_start
    
    # Test Optimized System
    print(f"\n⚡ Testing Optimized System...")
    optimized_results = []
    optimized_latencies = []
    optimized_start = time.time()
    
    for i in range(total_chunks):
        # Generate test audio
        audio_chunk = generate_test_audio(chunk_size)
        
        # Process asynchronously
        chunk_results = await optimized.add_audio_data_async(audio_chunk)
        optimized_results.extend(chunk_results)
        
        for result in chunk_results:
            optimized_latencies.append(result['processing_time'])
        
        # Simulate real-time input
        await asyncio.sleep(chunk_size)
    
    optimized_total_time = time.time() - optimized_start
    
    # Calculate metrics
    print(f"\n📈 Results Analysis:")
    print("=" * 60)
    
    # Original metrics
    if original_latencies:
        orig_avg = np.mean(original_latencies) * 1000  # Convert to ms
        orig_p95 = np.percentile(original_latencies, 95) * 1000
        orig_max = np.max(original_latencies) * 1000
        orig_buffering_latency = original.segment_duration * 1000  # Minimum latency from buffering
    else:
        orig_avg = orig_p95 = orig_max = orig_buffering_latency = 0
    
    # Optimized metrics
    if optimized_latencies:
        opt_avg = np.mean(optimized_latencies) * 1000
        opt_p95 = np.percentile(optimized_latencies, 95) * 1000
        opt_max = np.max(optimized_latencies) * 1000
        opt_buffering_latency = optimized.chunk_duration * 1000
        
        # Count cached results
        cached_count = sum(1 for r in optimized_results if r.get('cached', False))
        cache_hit_rate = cached_count / len(optimized_results) if optimized_results else 0
    else:
        opt_avg = opt_p95 = opt_max = opt_buffering_latency = 0
        cache_hit_rate = 0
    
    # Print comparison
    print(f"\n🎯 LATENCY COMPARISON")
    print(f"{'Metric':<25} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Buffering Latency':<25} {orig_buffering_latency:<14.0f}ms {opt_buffering_latency:<14.0f}ms {orig_buffering_latency/opt_buffering_latency if opt_buffering_latency > 0 else 0:.1f}x")
    print(f"{'Avg Processing':<25} {orig_avg:<14.0f}ms {opt_avg:<14.0f}ms {orig_avg/opt_avg if opt_avg > 0 else 0:.1f}x")
    print(f"{'P95 Processing':<25} {orig_p95:<14.0f}ms {opt_p95:<14.0f}ms {orig_p95/opt_p95 if opt_p95 > 0 else 0:.1f}x")
    print(f"{'Max Processing':<25} {orig_max:<14.0f}ms {opt_max:<14.0f}ms {orig_max/opt_max if opt_max > 0 else 0:.1f}x")
    
    # Total end-to-end latency (buffering + processing)
    orig_total_latency = orig_buffering_latency + orig_avg
    opt_total_latency = opt_buffering_latency + opt_avg
    
    print(f"{'TOTAL END-TO-END':<25} {orig_total_latency:<14.0f}ms {opt_total_latency:<14.0f}ms {orig_total_latency/opt_total_latency if opt_total_latency > 0 else 0:.1f}x")
    
    print(f"\n🏆 OPTIMIZATION BENEFITS")
    print(f"{'Segments Processed':<25} {len(original_results):<15} {len(optimized_results):<15}")
    print(f"{'Cache Hit Rate':<25} {'N/A':<15} {cache_hit_rate:<14.1%}")
    
    total_improvement = orig_total_latency / opt_total_latency if opt_total_latency > 0 else 0
    latency_reduction = ((orig_total_latency - opt_total_latency) / orig_total_latency * 100) if orig_total_latency > 0 else 0
    
    print(f"\n✨ KEY ACHIEVEMENTS:")
    print(f"   🎯 Total latency reduced by {latency_reduction:.1f}%")
    print(f"   ⚡ {total_improvement:.1f}x faster end-to-end processing")
    print(f"   📦 Chunk size reduced from {original.segment_duration}s to {optimized.chunk_duration}s")
    print(f"   🧠 Smart caching achieved {cache_hit_rate:.1%} hit rate")
    print(f"   🗣️  Voice activity detection skips non-speech segments")
    
    # Real-time capability assessment
    real_time_capable_orig = orig_total_latency <= (original.segment_duration * 1000)
    real_time_capable_opt = opt_total_latency <= (optimized.chunk_duration * 1000)
    
    print(f"\n🚦 REAL-TIME PERFORMANCE:")
    print(f"   Original System: {'✅ REAL-TIME CAPABLE' if real_time_capable_orig else '❌ TOO SLOW'}")
    print(f"   Optimized System: {'✅ REAL-TIME CAPABLE' if real_time_capable_opt else '❌ TOO SLOW'}")
    
    # Target achievement
    target_latency = 200  # 200ms target
    target_achieved = opt_total_latency <= target_latency
    
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    print(f"   Target Latency: {target_latency}ms")
    print(f"   Achieved Latency: {opt_total_latency:.0f}ms")
    print(f"   Target Met: {'✅ YES' if target_achieved else '❌ NO'}")
    
    print(f"\n{'='*60}")
    print("✅ Latency optimization demonstration completed!")
    print(f"The optimized system achieves {total_improvement:.1f}x better latency performance")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(run_latency_comparison())