"""
Model Optimization Utilities for Ultra Low-Latency Processing
Includes quantization, caching, batch processing, and model distillation techniques
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification
import whisper
from pathlib import Path
import pickle
import gc


class ModelOptimizer:
    """Model optimization utilities"""
    
    @staticmethod
    def quantize_model(model: nn.Module, device: str = "cpu") -> nn.Module:
        """Quantize model to INT8 for faster inference"""
        try:
            if device == "cpu":
                # Dynamic quantization for CPU
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {nn.Linear}, 
                    dtype=torch.qint8
                )
                return quantized_model
            else:
                # For CUDA, use FP16
                if hasattr(model, 'half'):
                    return model.half()
                return model
        except Exception as e:
            logging.warning(f"Quantization failed: {e}, using original model")
            return model
    
    @staticmethod
    def optimize_whisper_model(model, device: str = "cuda"):
        """Apply specific optimizations to Whisper model"""
        try:
            # Enable mixed precision
            if device == "cuda":
                model = model.half()
            
            # Enable torch.compile if available
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logging.info("Applied torch.compile optimization")
                except:
                    logging.warning("torch.compile failed, continuing without it")
            
            # Set to evaluation mode with optimizations
            model.eval()
            
            # Disable gradient computation globally for this model
            for param in model.parameters():
                param.requires_grad = False
            
            return model
            
        except Exception as e:
            logging.error(f"Whisper optimization failed: {e}")
            return model
    
    @staticmethod
    def create_onnx_model(pytorch_model, tokenizer, save_path: str, sample_text: str = "Hello world"):
        """Convert PyTorch model to ONNX for faster inference"""
        try:
            import torch.onnx
            
            # Prepare sample input
            inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
            
            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                (inputs['input_ids'], inputs['attention_mask']),
                save_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size', 1: 'sequence'}
                }
            )
            
            logging.info(f"Model exported to ONNX: {save_path}")
            return True
            
        except Exception as e:
            logging.error(f"ONNX export failed: {e}")
            return False


class InferenceCache:
    """Smart caching for model inference results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from input data"""
        if isinstance(data, np.ndarray):
            # For audio data, use hash of a subset for efficiency
            subset = data[::max(1, len(data) // 1000)]  # Sample every nth element
            return hashlib.md5(subset.tobytes()).hexdigest()
        elif isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, data: Any) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(data)
        current_time = time.time()
        
        if key in self.cache:
            # Check TTL
            if current_time - self.access_times[key] < self.ttl_seconds:
                self.access_times[key] = current_time
                self.hit_count += 1
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        
        self.miss_count += 1
        return None
    
    def put(self, data: Any, result: Any):
        """Cache result"""
        key = self._generate_key(data)
        current_time = time.time()
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }


class BatchProcessor:
    """Batch processing for improved throughput"""
    
    def __init__(self, batch_size: int = 4, timeout_ms: int = 50):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.pending_items = []
        self.last_batch_time = time.time()
        
    async def add_item(self, item: Any, callback: callable):
        """Add item to batch"""
        self.pending_items.append((item, callback))
        
        # Process if batch is full or timeout reached
        current_time = time.time()
        should_process = (
            len(self.pending_items) >= self.batch_size or
            (current_time - self.last_batch_time) * 1000 >= self.timeout_ms
        )
        
        if should_process:
            await self._process_batch()
    
    async def _process_batch(self):
        """Process accumulated batch"""
        if not self.pending_items:
            return
        
        items = [item for item, _ in self.pending_items]
        callbacks = [callback for _, callback in self.pending_items]
        
        # Process batch (placeholder - would call actual model)
        results = await self._batch_inference(items)
        
        # Execute callbacks
        for result, callback in zip(results, callbacks):
            callback(result)
        
        # Clear batch
        self.pending_items.clear()
        self.last_batch_time = time.time()
    
    async def _batch_inference(self, items: List[Any]) -> List[Any]:
        """Placeholder for batch inference"""
        # Would implement actual batch processing here
        return [f"result_for_{item}" for item in items]


class PredictiveProcessor:
    """Predictive processing to anticipate PII patterns"""
    
    def __init__(self):
        self.pii_patterns = {}
        self.context_window = []
        self.pattern_cache = InferenceCache(max_size=500)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def learn_patterns(self, text: str, detections: List[Any]):
        """Learn PII patterns from previous detections"""
        if not detections:
            return
        
        # Extract context around PII
        for detection in detections:
            context_start = max(0, detection.start_char - 20)
            context_end = min(len(text), detection.end_char + 20)
            context = text[context_start:context_end].lower()
            
            pii_type = detection.pii_type.value
            if pii_type not in self.pii_patterns:
                self.pii_patterns[pii_type] = []
            
            self.pii_patterns[pii_type].append(context)
            
            # Keep only recent patterns
            if len(self.pii_patterns[pii_type]) > 50:
                self.pii_patterns[pii_type].pop(0)
    
    def predict_pii_likelihood(self, text: str) -> float:
        """Predict likelihood of PII in text"""
        if not text.strip():
            return 0.0
        
        text_lower = text.lower()
        likelihood = 0.0
        
        # Check against learned patterns
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    likelihood = max(likelihood, 0.8)
                    break
        
        # Basic heuristics
        suspicious_words = ['phone', 'email', 'address', 'name', 'ssn', 'number']
        if any(word in text_lower for word in suspicious_words):
            likelihood = max(likelihood, 0.3)
        
        return likelihood
    
    def should_skip_processing(self, text: str) -> bool:
        """Determine if text can skip detailed processing"""
        # Skip if very unlikely to contain PII
        return self.predict_pii_likelihood(text) < 0.1


class AdaptiveQualityManager:
    """Manages processing quality based on system load and requirements"""
    
    def __init__(self):
        self.current_load = 0.0
        self.target_latency = 0.5  # 500ms target
        self.quality_level = "high"  # high, medium, low
        self.recent_latencies = []
        self.max_latency_samples = 20
        
    def update_metrics(self, processing_time: float, queue_size: int):
        """Update performance metrics"""
        self.recent_latencies.append(processing_time)
        if len(self.recent_latencies) > self.max_latency_samples:
            self.recent_latencies.pop(0)
        
        # Calculate current load based on latency and queue size
        avg_latency = np.mean(self.recent_latencies)
        self.current_load = min(1.0, (avg_latency / self.target_latency) * 0.7 + (queue_size / 10) * 0.3)
        
        # Adjust quality level
        self._adjust_quality_level()
    
    def _adjust_quality_level(self):
        """Adjust processing quality based on load"""
        if self.current_load > 0.8:
            self.quality_level = "low"
        elif self.current_load > 0.5:
            self.quality_level = "medium"
        else:
            self.quality_level = "high"
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get current processing configuration"""
        configs = {
            "high": {
                "whisper_model": "base",
                "pii_confidence_threshold": 0.7,
                "chunk_duration": 0.5,
                "enable_caching": True,
                "enable_batching": True
            },
            "medium": {
                "whisper_model": "tiny",
                "pii_confidence_threshold": 0.8,
                "chunk_duration": 0.3,
                "enable_caching": True,
                "enable_batching": True
            },
            "low": {
                "whisper_model": "tiny",
                "pii_confidence_threshold": 0.9,
                "chunk_duration": 0.2,
                "enable_caching": False,
                "enable_batching": False
            }
        }
        
        return configs[self.quality_level]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "current_load": self.current_load,
            "quality_level": self.quality_level,
            "target_latency": self.target_latency,
            "avg_recent_latency": np.mean(self.recent_latencies) if self.recent_latencies else 0.0,
            "samples_count": len(self.recent_latencies)
        }


class MemoryOptimizer:
    """Memory usage optimization utilities"""
    
    @staticmethod
    def cleanup_gpu_memory():
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def optimize_tensor_memory(tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory usage"""
        # Use contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Pin memory for faster CPU-GPU transfers
        if tensor.device.type == 'cpu' and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        
        return tensor
    
    @staticmethod
    def monitor_memory_usage() -> Dict[str, float]:
        """Monitor current memory usage"""
        import psutil
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        
        stats = {
            "cpu_memory_used_gb": cpu_memory.used / (1024**3),
            "cpu_memory_percent": cpu_memory.percent,
        }
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            allocated = gpu_memory.get("allocated_bytes.all.current", 0)
            reserved = gpu_memory.get("reserved_bytes.all.current", 0)
            
            stats.update({
                "gpu_memory_allocated_gb": allocated / (1024**3),
                "gpu_memory_reserved_gb": reserved / (1024**3),
            })
        
        return stats


# Integration example
class OptimizedModelManager:
    """Manages optimized models with caching and adaptive processing"""
    
    def __init__(self):
        self.cache = InferenceCache(max_size=1000)
        self.batch_processor = BatchProcessor(batch_size=4, timeout_ms=50)
        self.predictive_processor = PredictiveProcessor()
        self.quality_manager = AdaptiveQualityManager()
        self.memory_optimizer = MemoryOptimizer()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def process_with_optimizations(self, audio_data: np.ndarray, text: str = None) -> Dict[str, Any]:
        """Process data with all optimizations applied"""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get(audio_data)
        if cached_result:
            return cached_result
        
        # Predictive filtering
        if text and self.predictive_processor.should_skip_processing(text):
            result = {"transcription": text, "pii_detections": [], "skipped": True}
            self.cache.put(audio_data, result)
            return result
        
        # Get adaptive configuration
        config = self.quality_manager.get_processing_config()
        
        # Process with current quality settings
        result = await self._process_with_config(audio_data, config)
        
        # Update metrics
        processing_time = time.time() - start_time
        self.quality_manager.update_metrics(processing_time, 0)  # Queue size would be passed here
        
        # Cache result
        self.cache.put(audio_data, result)
        
        # Learn patterns for prediction
        if "pii_detections" in result and result["pii_detections"]:
            self.predictive_processor.learn_patterns(
                result.get("transcription", ""), 
                result["pii_detections"]
            )
        
        # Periodic memory cleanup
        if np.random.random() < 0.1:  # 10% chance
            self.memory_optimizer.cleanup_gpu_memory()
        
        return result
    
    async def _process_with_config(self, audio_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio with given configuration"""
        # Placeholder for actual processing
        # Would integrate with the optimized models based on config
        
        return {
            "transcription": "Sample transcription",
            "pii_detections": [],
            "processing_time": 0.1,
            "quality_level": config,
            "skipped": False
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "cache_stats": self.cache.get_stats(),
            "quality_manager_status": self.quality_manager.get_status(),
            "memory_usage": self.memory_optimizer.monitor_memory_usage(),
            "predictive_patterns": len(self.predictive_processor.pii_patterns)
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test model optimization
    manager = OptimizedModelManager()
    
    # Simulate processing
    test_audio = np.random.random(16000).astype(np.float32)  # 1 second of audio
    
    import asyncio
    result = asyncio.run(manager.process_with_optimizations(test_audio, "Hello world"))
    
    print("Processing result:", result)
    print("Performance stats:", manager.get_comprehensive_stats())