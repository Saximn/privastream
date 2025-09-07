"""
Live Stream Pipeline for Real-time Video Blur Processing
========================================================

This pipeline implements a complete live streaming solution with:
1. Video input from camera/stream
2. Frame scheduling at configurable FPS 
3. Unified video analysis (face, PII, plate detection)
4. Real-time blur application
5. Output stream at original FPS

Architecture:
- I/O LiveStream (30fps) -> VideoScheduler (4fps) -> UnifiedVideoAnalyzer -> BlurProcessor -> Output (30fps)
- Audio processing pipeline (placeholder for future implementation)
- Clean engineering with proper threading, buffering, and error handling
"""

import sys
import time
import threading
import queue
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import cv2

# Add model directories to path
sys.path.append(str(Path(__file__).parent))

from unified_detector import UnifiedBlurDetector
from blur_utils import apply_blur_regions


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame_id: int
    timestamp: float
    image: np.ndarray
    detection_results: Optional[Dict[str, Any]] = None
    blur_regions: Optional[Dict[str, Any]] = None
    processed: bool = False


@dataclass  
class PipelineConfig:
    """Configuration for the live stream pipeline."""
    # Video input settings
    input_source: int = 0  # 0 for webcam, or video file path
    input_fps: float = 30.0
    input_width: int = 1280
    input_height: int = 720
    
    # Processing settings
    processing_fps: float = 4.0  # FPS sent to unified analyzer
    output_fps: float = 30.0     # Output FPS (same as input)
    
    # Buffer sizes
    input_buffer_size: int = 60   # ~2 seconds at 30fps
    processing_buffer_size: int = 10  # ~2.5 seconds at 4fps
    output_buffer_size: int = 60  # ~2 seconds at 30fps
    
    # Blur settings
    blur_type: str = "gaussian"  # "gaussian", "pixelate", "fill"
    blur_kernel_size: int = 35
    blur_pixel_size: int = 16
    blur_hold_frames: int = 8    # Hold blur for N frames after detection
    
    # Detection settings
    enable_face: bool = True
    enable_pii: bool = True
    enable_plate: bool = True
    
    # Output settings
    save_output: bool = False
    output_path: str = "output_stream.mp4"
    show_preview: bool = True
    show_detections: bool = True
    
    # Performance monitoring
    enable_profiling: bool = True
    stats_interval: float = 5.0  # Print stats every N seconds


class FrameBuffer:
    """Thread-safe circular buffer for frames."""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
    
    def put(self, item: FrameData, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add item to buffer. Returns False if buffer is full and non-blocking."""
        try:
            self.queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[FrameData]:
        """Get item from buffer. Returns None if empty and non-blocking."""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.queue.qsize()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.queue.full()


class PerformanceMonitor:
    """Monitor pipeline performance metrics."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = time.time()
        self.frame_counts = {
            'input': 0,
            'processed': 0,
            'output': 0,
            'dropped': 0
        }
        self.processing_times = deque(maxlen=100)
        self.fps_measurements = {
            'input': deque(maxlen=30),
            'processing': deque(maxlen=10),
            'output': deque(maxlen=30)
        }
        self.last_stats_time = time.time()
    
    def record_frame(self, stage: str, processing_time: Optional[float] = None):
        """Record a frame processed at given stage."""
        self.frame_counts[stage] += 1
        current_time = time.time()
        
        if processing_time is not None:
            self.processing_times.append(processing_time)
        
        # Record FPS
        if stage in self.fps_measurements:
            self.fps_measurements[stage].append(current_time)
    
    def get_fps(self, stage: str) -> float:
        """Calculate current FPS for a stage."""
        if stage not in self.fps_measurements:
            return 0.0
        
        timestamps = list(self.fps_measurements[stage])
        if len(timestamps) < 2:
            return 0.0
        
        duration = timestamps[-1] - timestamps[0]
        return (len(timestamps) - 1) / max(duration, 0.001)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            'runtime_seconds': runtime,
            'frames': self.frame_counts.copy(),
            'fps': {stage: self.get_fps(stage) for stage in self.fps_measurements},
            'avg_processing_time_ms': avg_processing_time * 1000,
            'processing_efficiency': (self.frame_counts['processed'] / max(1, self.frame_counts['input'])) * 100
        }
    
    def should_print_stats(self) -> bool:
        """Check if it's time to print statistics."""
        current_time = time.time()
        if current_time - self.last_stats_time >= self.config.stats_interval:
            self.last_stats_time = current_time
            return True
        return False


class VideoScheduler:
    """Schedules frames from input stream to processing at reduced FPS."""
    
    def __init__(self, config: PipelineConfig, input_buffer: FrameBuffer, 
                 processing_buffer: FrameBuffer, monitor: PerformanceMonitor):
        self.config = config
        self.input_buffer = input_buffer
        self.processing_buffer = processing_buffer
        self.monitor = monitor
        self.running = False
        
        # Calculate frame skip ratio
        self.frame_skip_ratio = max(1, int(config.input_fps / config.processing_fps))
        self.frame_counter = 0
        
        logger.info(f"VideoScheduler: Processing every {self.frame_skip_ratio} frames "
                   f"({config.input_fps}fps -> {config.processing_fps}fps)")
    
    def start(self):
        """Start the video scheduler thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, name="VideoScheduler")
        self.thread.daemon = True
        self.thread.start()
        logger.info("VideoScheduler started")
    
    def stop(self):
        """Stop the video scheduler."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        logger.info("VideoScheduler stopped")
    
    def _run(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Get frame from input buffer
                frame_data = self.input_buffer.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                # Check if this frame should be processed
                if self.frame_counter % self.frame_skip_ratio == 0:
                    # Send to processing buffer
                    if not self.processing_buffer.put(frame_data, block=False):
                        # Processing buffer full, drop frame
                        self.monitor.record_frame('dropped')
                        logger.warning("Processing buffer full, dropping frame")
                else:
                    # Frame not selected for processing, but still count it
                    pass
                
                self.frame_counter += 1
                
            except Exception as e:
                logger.error(f"VideoScheduler error: {e}")
                time.sleep(0.01)


class UnifiedVideoAnalyzer:
    """Processes frames with unified detection models."""
    
    def __init__(self, config: PipelineConfig, processing_buffer: FrameBuffer, 
                 output_buffer: FrameBuffer, monitor: PerformanceMonitor):
        self.config = config
        self.processing_buffer = processing_buffer
        self.output_buffer = output_buffer
        self.monitor = monitor
        self.running = False
        
        # Initialize detection models
        detector_config = {
            "enable_face": config.enable_face,
            "enable_pii": config.enable_pii,
            "enable_plate": config.enable_plate,
            "face": {
                "embed_path": "face_blur/whitelist/creator_embedding.json",
                "threshold": 0.35,
                "dilate_px": 12
            },
            "pii": {
                "classifier_path": "pii_blur/pii_clf.joblib",
                "conf_thresh": 0.35
            },
            "plate": {
                "weights_path": "plate_blur/best.pt",
                "conf_thresh": 0.25
            }
        }
        
        try:
            self.detector = UnifiedBlurDetector(detector_config)
            logger.info("UnifiedVideoAnalyzer: Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector models: {e}")
            self.detector = None
        
        # Cache for holding detections across frames
        self.detection_cache = {}  # frame_id -> (detection_results, remaining_hold_frames)
    
    def start(self):
        """Start the video analyzer thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, name="VideoAnalyzer")
        self.thread.daemon = True
        self.thread.start()
        logger.info("UnifiedVideoAnalyzer started")
    
    def stop(self):
        """Stop the video analyzer."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5.0)
        logger.info("UnifiedVideoAnalyzer stopped")
    
    def _run(self):
        """Main analyzer loop."""
        while self.running:
            try:
                # Get frame for processing
                frame_data = self.processing_buffer.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                start_time = time.time()
                
                # Process frame if detector is available
                if self.detector is not None:
                    try:
                        detection_results = self.detector.process_frame(
                            frame_data.image, frame_data.frame_id
                        )
                        frame_data.detection_results = detection_results
                        
                        # Cache detection for blur holding
                        self.detection_cache[frame_data.frame_id] = (
                            detection_results, self.config.blur_hold_frames
                        )
                        
                    except Exception as e:
                        logger.error(f"Detection failed for frame {frame_data.frame_id}: {e}")
                        frame_data.detection_results = {"frame_id": frame_data.frame_id, "models": {}}
                
                processing_time = time.time() - start_time
                frame_data.processed = True
                
                # Send to output buffer
                if not self.output_buffer.put(frame_data, block=False):
                    logger.warning(f"Output buffer full, dropping processed frame {frame_data.frame_id}")
                
                self.monitor.record_frame('processed', processing_time)
                
                # Clean up old cache entries
                self._cleanup_cache()
                
            except Exception as e:
                logger.error(f"UnifiedVideoAnalyzer error: {e}")
                time.sleep(0.01)
    
    def _cleanup_cache(self):
        """Remove old entries from detection cache."""
        current_time = time.time()
        to_remove = []
        
        for frame_id, (results, remaining_frames) in self.detection_cache.items():
            # Remove if too old (more than 5 seconds)
            if current_time - results.get('timestamp', 0) > 5.0:
                to_remove.append(frame_id)
        
        for frame_id in to_remove:
            del self.detection_cache[frame_id]
    
    def get_blur_regions_for_frame(self, frame_id: int) -> Dict[str, Any]:
        """Get blur regions for a given frame, considering hold frames."""
        blur_regions = {"rectangles": [], "polygons": []}
        
        # Check all cached detections for active blur regions
        for cached_frame_id, (results, remaining_frames) in list(self.detection_cache.items()):
            if remaining_frames > 0 and cached_frame_id <= frame_id:
                # This detection is still active for blurring
                models = results.get("models", {})
                
                # Face rectangles
                if "face" in models and "rectangles" in models["face"]:
                    blur_regions["rectangles"].extend(models["face"]["rectangles"])
                
                # Plate rectangles
                if "plate" in models and "rectangles" in models["plate"]:
                    blur_regions["rectangles"].extend(models["plate"]["rectangles"])
                
                # PII polygons
                if "pii" in models and "polygons" in models["pii"]:
                    blur_regions["polygons"].extend(models["pii"]["polygons"])
                
                # Decrease remaining frames
                self.detection_cache[cached_frame_id] = (results, remaining_frames - 1)
        
        return blur_regions


class BlurProcessor:
    """Applies blur to frames and manages output."""
    
    def __init__(self, config: PipelineConfig, input_buffer: FrameBuffer, 
                 output_buffer: FrameBuffer, analyzer: UnifiedVideoAnalyzer, 
                 monitor: PerformanceMonitor):
        self.config = config
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.analyzer = analyzer
        self.monitor = monitor
        self.running = False
        
        # Output video writer
        self.video_writer = None
        if config.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                config.output_path, fourcc, config.output_fps,
                (config.input_width, config.input_height)
            )
            logger.info(f"Video output will be saved to: {config.output_path}")
    
    def start(self):
        """Start the blur processor thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, name="BlurProcessor")
        self.thread.daemon = True
        self.thread.start()
        logger.info("BlurProcessor started")
    
    def stop(self):
        """Stop the blur processor."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        
        if self.video_writer:
            self.video_writer.release()
            logger.info(f"Video saved to: {self.config.output_path}")
        
        logger.info("BlurProcessor stopped")
    
    def _run(self):
        """Main processor loop."""
        while self.running:
            try:
                # Get frame from input (original frames at full FPS)
                frame_data = self.input_buffer.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                # Get blur regions for this frame
                blur_regions = self.analyzer.get_blur_regions_for_frame(frame_data.frame_id)
                frame_data.blur_regions = blur_regions
                
                # Apply blur
                blurred_frame = apply_blur_regions(
                    frame_data.image.copy(),
                    rectangles=blur_regions.get("rectangles"),
                    polygons=blur_regions.get("polygons"),
                    blur_type=self.config.blur_type,
                    kernel_size=self.config.blur_kernel_size,
                    pixel_size=self.config.blur_pixel_size
                )
                
                # Show preview if enabled
                if self.config.show_preview:
                    self._show_preview(frame_data, blurred_frame, blur_regions)
                
                # Save to video file if enabled
                if self.video_writer:
                    self.video_writer.write(blurred_frame)
                
                self.monitor.record_frame('output')
                
            except Exception as e:
                logger.error(f"BlurProcessor error: {e}")
                time.sleep(0.01)
    
    def _show_preview(self, frame_data: FrameData, blurred_frame: np.ndarray, 
                     blur_regions: Dict[str, Any]):
        """Show preview window with original and blurred frames."""
        display_frame = blurred_frame.copy()
        
        if self.config.show_detections:
            # Draw detection regions
            for rect in blur_regions.get("rectangles", []):
                if len(rect) == 4:
                    x1, y1, x2, y2 = rect
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            for poly in blur_regions.get("polygons", []):
                if len(poly) > 0:
                    cv2.polylines(display_frame, [poly], True, (255, 0, 0), 2)
            
            # Add frame info
            info_text = f"Frame: {frame_data.frame_id}"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Live Stream Pipeline", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quit requested via 'q' key")
            self.running = False
        elif key == ord('s'):
            # Save current frame
            filename = f"frame_{frame_data.frame_id}_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_frame)
            logger.info(f"Frame saved to: {filename}")


class LiveStreamPipeline:
    """Main pipeline coordinator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.running = False
        
        # Performance monitoring
        self.monitor = PerformanceMonitor(config)
        
        # Create frame buffers
        self.input_buffer = FrameBuffer(config.input_buffer_size)
        self.processing_buffer = FrameBuffer(config.processing_buffer_size) 
        self.output_buffer = FrameBuffer(config.output_buffer_size)
        
        # Initialize pipeline components
        self.video_scheduler = VideoScheduler(config, self.input_buffer, self.processing_buffer, self.monitor)
        self.video_analyzer = UnifiedVideoAnalyzer(config, self.processing_buffer, self.output_buffer, self.monitor)
        self.blur_processor = BlurProcessor(config, self.input_buffer, self.output_buffer, self.video_analyzer, self.monitor)
        
        # Video capture
        self.cap = None
        
        logger.info("LiveStreamPipeline initialized")
    
    def start(self):
        """Start the complete pipeline."""
        logger.info("Starting Live Stream Pipeline...")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.config.input_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.input_source}")
        
        # Set capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.input_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.input_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.input_fps)
        
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video capture initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Start pipeline components
        self.video_scheduler.start()
        self.video_analyzer.start()
        self.blur_processor.start()
        
        self.running = True
        
        # Start main capture loop
        self._capture_loop()
    
    def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping Live Stream Pipeline...")
        
        self.running = False
        
        # Stop components
        self.blur_processor.stop()
        self.video_analyzer.stop()
        self.video_scheduler.stop()
        
        # Release video capture
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.monitor.get_stats()
        logger.info("=== Final Pipeline Statistics ===")
        logger.info(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        logger.info(f"Frames processed: {stats['frames']}")
        logger.info(f"FPS: {stats['fps']}")
        logger.info(f"Average processing time: {stats['avg_processing_time_ms']:.1f}ms")
        logger.info(f"Processing efficiency: {stats['processing_efficiency']:.1f}%")
        
        logger.info("Pipeline stopped")
    
    def _capture_loop(self):
        """Main video capture loop."""
        frame_id = 0
        target_frame_time = 1.0 / self.config.input_fps
        
        logger.info("Starting video capture loop...")
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Create frame data
                frame_data = FrameData(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    image=frame.copy()
                )
                
                # Add to input buffer
                if not self.input_buffer.put(frame_data, block=False):
                    # Input buffer full, drop frame
                    self.monitor.record_frame('dropped')
                    logger.warning("Input buffer full, dropping frame")
                else:
                    self.monitor.record_frame('input')
                
                # Print statistics periodically
                if self.config.enable_profiling and self.monitor.should_print_stats():
                    stats = self.monitor.get_stats()
                    logger.info(f"Pipeline Stats - Input:{stats['fps']['input']:.1f}fps "
                              f"Processing:{stats['fps']['processing']:.1f}fps "
                              f"Output:{stats['fps']['output']:.1f}fps "
                              f"Dropped:{stats['frames']['dropped']} "
                              f"AvgTime:{stats['avg_processing_time_ms']:.1f}ms")
                
                frame_id += 1
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Capture loop error: {e}")
        finally:
            self.stop()


# ============================================================================
# Audio Processing Pipeline (Placeholder)
# ============================================================================

class AudioPipeline:
    """
    Placeholder for audio processing pipeline.
    
    Future implementation will include:
    1. Audio capture from microphone/stream
    2. Whisper speech-to-text conversion
    3. Fine-tuned DeBerta PII detection in text
    4. Temporal alignment with video frames
    5. Mouth blur coordination with video pipeline
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info("AudioPipeline: Placeholder initialized (not implemented)")
    
    def start(self):
        logger.info("AudioPipeline: Would start audio processing here")
    
    def stop(self):
        logger.info("AudioPipeline: Would stop audio processing here")


# ============================================================================
# Main Application
# ============================================================================

def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration."""
    return PipelineConfig(
        # Input settings
        input_source=0,           # Webcam
        input_fps=30.0,          # 30fps input
        input_width=1280,
        input_height=720,
        
        # Processing settings  
        processing_fps=4.0,      # Process every 7-8 frames (30/4 ≈ 7.5)
        output_fps=30.0,         # Output at original FPS
        
        # Blur settings
        blur_type="gaussian",
        blur_kernel_size=35,
        blur_hold_frames=8,      # Hold blur for ~250ms at 30fps
        
        # Detection settings
        enable_face=True,
        enable_pii=True,
        enable_plate=True,
        
        # Output settings
        save_output=False,       # Set to True to save video file
        show_preview=True,
        show_detections=True,
        
        # Performance
        enable_profiling=True,
        stats_interval=5.0
    )


def main():
    """Main application entry point."""
    print("=== Live Stream Pipeline ===")
    print("Features:")
    print("- Real-time face, PII, and license plate detection")
    print("- Configurable processing FPS for performance optimization")
    print("- Multi-threaded pipeline with proper buffering")
    print("- Blur persistence across frames")
    print("- Performance monitoring and statistics")
    print()
    print("Controls:")
    print("- 'q': Quit application")
    print("- 's': Save current frame")
    print()
    
    # Create configuration
    config = create_default_config()
    
    # Create and start pipeline
    pipeline = LiveStreamPipeline(config)
    
    try:
        pipeline.start()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
