"""
Main pipeline orchestrator for livestream PII detection and redaction.
Combines Whisper transcription with DeBERTa-based PII detection.
"""

import asyncio
import logging
import time
import yaml
import json
import queue
import threading
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from .pipeline_types import (
    AudioSegment,
    TranscriptionResult,
    RedactionResult,
    VideoBlurInstruction,
    ProcessingResult,
    StreamConfig,
    PIIDetection
)
from .whisper_processor import WhisperProcessor  
from .pii_detector import PIIDetector


class LivestreamPIIPipeline:
    """
    Main pipeline for real-time PII detection and redaction in livestreams.
    Orchestrates audio transcription and PII detection processes.
    """
    
    def __init__(
        self,
        config_path: str = "configs/pipeline_config.yaml",
        stream_config: Optional[StreamConfig] = None
    ):
        """
        Initialize the livestream PII pipeline.
        
        Args:
            config_path: Path to pipeline configuration file
            stream_config: Stream-specific configuration
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.stream_config = stream_config or StreamConfig(stream_id="default")
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        # Initialize processors
        self.whisper_processor = self._init_whisper_processor()
        self.pii_detector = self._init_pii_detector()
        
        # Processing queues
        self.audio_queue = queue.Queue(maxsize=self.config['processing']['max_queue_size'])
        self.result_queue = queue.Queue(maxsize=self.config['processing']['max_queue_size'])
        
        # Processing control
        self.is_running = False
        self.processing_thread = None
        self.cleanup_thread = None
        
        # Statistics
        self.pipeline_stats = {
            'processed_segments': 0,
            'total_pipeline_time': 0.0,
            'average_pipeline_time': 0.0,
            'pii_detections': 0,
            'blur_instructions': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Result callbacks
        self.result_callbacks: List[Callable[[ProcessingResult], None]] = []
        
        self.logger.info("Livestream PII pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not available."""
        return {
            'whisper': {
                'model_name': 'large-v3',
                'device': 'cuda',
                'fp16': True,
                'language': 'en'
            },
            'deberta': {
                'model_path': './models/',
                'tokenizer_name': 'microsoft/deberta-v3-large',
                'device': 'cuda',
                'confidence_threshold': 0.7
            },
            'processing': {
                'batch_size': 8,
                'num_workers': 2,
                'max_queue_size': 100,
                'processing_timeout': 10.0
            },
            'output': {
                'save_transcripts': True,
                'save_redacted_text': True,
                'save_timestamps': True,
                'log_level': 'INFO'
            }
        }
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, self.config['output']['log_level'], logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _init_whisper_processor(self) -> WhisperProcessor:
        """Initialize Whisper processor."""
        whisper_config = self.config['whisper']
        return WhisperProcessor(
            model_name=whisper_config['model_name'],
            device=whisper_config['device'],
            compute_type="float16" if whisper_config.get('fp16', True) else "float32",
            language=whisper_config.get('language', 'en'),
            max_workers=self.config['processing']['num_workers'],
            enable_word_timestamps=True
        )
    
    def _init_pii_detector(self) -> PIIDetector:
        """Initialize PII detector."""
        deberta_config = self.config['deberta']
        return PIIDetector(
            model_path=deberta_config['model_path'],
            tokenizer_name=deberta_config['tokenizer_name'],
            device=deberta_config['device'],
            confidence_threshold=deberta_config['confidence_threshold'],
            max_length=deberta_config.get('max_length', 512),
            stride=deberta_config.get('stride', 128)
        )
    
    def add_result_callback(self, callback: Callable[[ProcessingResult], None]):
        """
        Add a callback function to handle processing results.
        
        Args:
            callback: Function that takes ProcessingResult as parameter
        """
        self.result_callbacks.append(callback)
    
    def process_audio_segment(self, audio_segment: AudioSegment) -> Optional[ProcessingResult]:
        """
        Process a single audio segment through the complete pipeline.
        
        Args:
            audio_segment: AudioSegment to process
            
        Returns:
            ProcessingResult or None if processing fails
        """
        start_time = time.time()
        
        try:
            # Step 1: Transcribe audio with Whisper
            self.logger.debug(f"Transcribing segment {audio_segment.segment_id}")
            transcription = self.whisper_processor.transcribe_segment(audio_segment)
            
            if not transcription.text.strip():
                self.logger.debug(f"Empty transcription for segment {audio_segment.segment_id}")
                return None
            
            # Step 2: Detect and redact PII
            self.logger.debug(f"Detecting PII in segment {audio_segment.segment_id}")
            redaction = self.pii_detector.process_transcription(transcription)
            
            # Step 3: Generate video blur instructions
            blur_instructions = self._generate_blur_instructions(
                redaction.detections, 
                audio_segment.start_time,
                audio_segment.end_time
            )
            
            # Create processing result
            result = ProcessingResult(
                segment_id=audio_segment.segment_id,
                audio_segment=audio_segment,
                transcription=transcription,
                redaction=redaction,
                blur_instructions=blur_instructions,
                timestamp=time.time()
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.pipeline_stats['processed_segments'] += 1
            self.pipeline_stats['total_pipeline_time'] += processing_time
            self.pipeline_stats['average_pipeline_time'] = (
                self.pipeline_stats['total_pipeline_time'] / 
                self.pipeline_stats['processed_segments']
            )
            self.pipeline_stats['pii_detections'] += len(redaction.detections)
            self.pipeline_stats['blur_instructions'] += len(blur_instructions)
            
            self.logger.info(
                f"Processed segment {audio_segment.segment_id} in {processing_time:.2f}s: "
                f"{len(redaction.detections)} PII detected, {len(blur_instructions)} blur instructions"
            )
            
            # Call result callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Error in result callback: {e}")
            
            return result
            
        except Exception as e:
            self.pipeline_stats['errors'] += 1
            self.logger.error(f"Error processing segment {audio_segment.segment_id}: {e}")
            return None
    
    def _generate_blur_instructions(
        self,
        detections: List[PIIDetection],
        segment_start_time: float,
        segment_end_time: float
    ) -> List[VideoBlurInstruction]:
        """
        Generate video blur instructions based on PII detections.
        
        Args:
            detections: List of PII detections
            segment_start_time: Start time of audio segment
            segment_end_time: End time of audio segment
            
        Returns:
            List of VideoBlurInstruction objects
        """
        blur_instructions = []
        
        for detection in detections:
            # Calculate frame numbers
            start_frame = self.stream_config.time_to_frames(detection.start_time)
            end_frame = self.stream_config.time_to_frames(detection.end_time)
            
            # Add buffer frames around detected PII
            buffer_frames = int(self.stream_config.fps * 0.5)  # 0.5 second buffer
            start_frame = max(0, start_frame - buffer_frames)
            end_frame = end_frame + buffer_frames
            
            blur_instruction = VideoBlurInstruction(
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=detection.start_time - 0.5,  # Buffer time
                end_time=detection.end_time + 0.5,
                blur_region=None,  # Full screen blur - can be customized
                blur_type="gaussian",
                blur_strength=10.0,
                reason=f"PII_DETECTED_{detection.pii_type.value}",
                pii_detections=[detection]
            )
            
            blur_instructions.append(blur_instruction)
        
        return blur_instructions
    
    async def process_audio_segment_async(
        self, 
        audio_segment: AudioSegment
    ) -> Optional[ProcessingResult]:
        """
        Asynchronously process an audio segment.
        
        Args:
            audio_segment: AudioSegment to process
            
        Returns:
            ProcessingResult or None if processing fails
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self.process_audio_segment,
                audio_segment
            )
        return result
    
    def start_pipeline(self):
        """Start the pipeline processing thread."""
        if self.is_running:
            self.logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        self.logger.info("Pipeline started")
    
    def stop_pipeline(self):
        """Stop the pipeline processing."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        self.logger.info("Pipeline stopped")
    
    def _processing_loop(self):
        """Main processing loop for handling queued audio segments."""
        while self.is_running:
            try:
                # Get audio segment from queue with timeout
                audio_segment = self.audio_queue.get(timeout=1.0)
                
                # Process the segment
                result = self.process_audio_segment(audio_segment)
                
                if result:
                    # Put result in result queue
                    self.result_queue.put(result)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
    
    def _cleanup_loop(self):
        """Cleanup loop for managing memory and old data."""
        while self.is_running:
            try:
                time.sleep(60)  # Run every minute
                
                # Clear old results from queue
                while not self.result_queue.empty():
                    try:
                        old_result = self.result_queue.get_nowait()
                        # Could save to disk or database here
                    except queue.Empty:
                        break
                
                # Clear GPU cache
                if hasattr(self.whisper_processor, 'device') and self.whisper_processor.device == 'cuda':
                    import torch
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    def submit_audio_segment(self, audio_segment: AudioSegment) -> bool:
        """
        Submit an audio segment for processing.
        
        Args:
            audio_segment: AudioSegment to process
            
        Returns:
            True if submitted successfully, False if queue is full
        """
        try:
            self.audio_queue.put_nowait(audio_segment)
            return True
        except queue.Full:
            self.logger.warning("Audio queue is full, dropping segment")
            return False
    
    def get_processing_result(self, timeout: float = 1.0) -> Optional[ProcessingResult]:
        """
        Get a processing result from the result queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            ProcessingResult or None if no result available
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline processing statistics."""
        stats = self.pipeline_stats.copy()
        stats['uptime'] = time.time() - stats['start_time']
        stats['queue_sizes'] = {
            'audio_queue': self.audio_queue.qsize(),
            'result_queue': self.result_queue.qsize()
        }
        stats['whisper_stats'] = self.whisper_processor.get_stats()
        stats['pii_detector_stats'] = self.pii_detector.get_stats()
        return stats
    
    def save_result_to_file(self, result: ProcessingResult, output_dir: str = "output"):
        """
        Save processing result to files.
        
        Args:
            result: ProcessingResult to save
            output_dir: Output directory
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        output_file = os.path.join(output_dir, f"result_{result.segment_id}_{int(result.timestamp)}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"Saved result to {output_file}")
    
    def cleanup(self):
        """Clean up pipeline resources."""
        self.stop_pipeline()
        
        if hasattr(self, 'whisper_processor'):
            self.whisper_processor.cleanup()
        
        if hasattr(self, 'pii_detector'):
            self.pii_detector.cleanup()
        
        self.logger.info("Pipeline cleanup completed")


# Example usage and integration helper
class LivestreamIntegration:
    """
    Helper class for integrating the pipeline with livestream systems.
    """
    
    def __init__(self, pipeline: LivestreamPIIPipeline):
        self.pipeline = pipeline
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def setup_audio_callback(self, audio_callback_func: Callable):
        """
        Set up audio input callback for real-time processing.
        This would be connected to your livestream audio source.
        """
        # This is a template - implement based on your audio source
        pass
    
    def setup_video_output_callback(self, video_callback_func: Callable):
        """
        Set up video output callback for blur instructions.
        This would be connected to your video processing system.
        """
        def blur_callback(result: ProcessingResult):
            if result.blur_instructions:
                video_callback_func(result.blur_instructions)
        
        self.pipeline.add_result_callback(blur_callback)
    
    def example_real_time_processing(self):
        """
        Example of how to integrate with a real-time system.
        """
        self.logger.info("Starting example real-time processing")
        
        # Start pipeline
        self.pipeline.start_pipeline()
        
        # Example: Create dummy audio segments (replace with real audio source)
        import uuid
        
        for i in range(10):  # Process 10 segments as example
            # Create dummy audio segment (replace with real audio data)
            segment = AudioSegment(
                audio_data=b"dummy_audio_data",  # Replace with real audio bytes
                start_time=i * 5.0,
                end_time=(i + 1) * 5.0,
                sample_rate=16000,
                channels=1,
                segment_id=str(uuid.uuid4())
            )
            
            # Submit for processing
            self.pipeline.submit_audio_segment(segment)
            
            # Get results
            result = self.pipeline.get_processing_result(timeout=10.0)
            if result:
                self.logger.info(f"Got result for segment {result.segment_id}")
                # Process blur instructions, save results, etc.
        
        # Clean up
        self.pipeline.cleanup()
