"""
Simple launcher for the Live Stream Pipeline.
Uses configuration from pipeline_config.py for easy customization.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from live_stream_pipeline import LiveStreamPipeline, PipelineConfig
import pipeline_config as config


def setup_logging():
    """Setup logging based on configuration."""
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # File handler if enabled
    if config.LOG_TO_FILE:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.LOG_FILE_PATH,
            maxBytes=config.LOG_MAX_SIZE_MB * 1024 * 1024,
            backupCount=config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def create_config_from_file() -> PipelineConfig:
    """Create pipeline configuration from config file."""
    return PipelineConfig(
        # Video input settings
        input_source=config.INPUT_SOURCE,
        input_fps=config.INPUT_FPS,
        input_width=config.INPUT_WIDTH,
        input_height=config.INPUT_HEIGHT,
        
        # Processing settings
        processing_fps=config.PROCESSING_FPS,
        output_fps=config.OUTPUT_FPS,
        
        # Buffer sizes
        input_buffer_size=config.INPUT_BUFFER_SIZE,
        processing_buffer_size=config.PROCESSING_BUFFER_SIZE,
        output_buffer_size=config.OUTPUT_BUFFER_SIZE,
        
        # Blur settings
        blur_type=config.BLUR_TYPE,
        blur_kernel_size=config.BLUR_KERNEL_SIZE,
        blur_pixel_size=config.BLUR_PIXEL_SIZE,
        blur_hold_frames=config.BLUR_HOLD_FRAMES,
        
        # Detection settings
        enable_face=config.ENABLE_FACE_DETECTION,
        enable_pii=config.ENABLE_PII_DETECTION,
        enable_plate=config.ENABLE_PLATE_DETECTION,
        
        # Output settings
        save_output=config.SAVE_OUTPUT_VIDEO,
        output_path=config.OUTPUT_VIDEO_PATH,
        show_preview=config.SHOW_PREVIEW,
        show_detections=config.SHOW_DETECTIONS,
        
        # Performance monitoring
        enable_profiling=config.ENABLE_PROFILING,
        stats_interval=config.STATS_INTERVAL
    )


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("🎥 LIVE STREAM PRIVACY PIPELINE 🎥")
    print("=" * 60)
    print()
    print("Real-time Video Privacy Protection System")
    print("- Face Detection & Blurring")
    print("- PII (Personal Information) Detection & Blurring") 
    print("- License Plate Detection & Blurring")
    print("- Optimized Multi-threaded Processing")
    print("- Configurable FPS and Quality Settings")
    print()
    print("Current Configuration:")
    print(f"  Input:      {config.INPUT_WIDTH}x{config.INPUT_HEIGHT} @ {config.INPUT_FPS}fps")
    print(f"  Processing: Every {int(config.INPUT_FPS/config.PROCESSING_FPS)} frames ({config.PROCESSING_FPS}fps)")
    print(f"  Output:     Same as input @ {config.OUTPUT_FPS}fps")
    print(f"  Models:     Face:{config.ENABLE_FACE_DETECTION} PII:{config.ENABLE_PII_DETECTION} Plate:{config.ENABLE_PLATE_DETECTION}")
    print(f"  Blur:       {config.BLUR_TYPE.title()} (hold {config.BLUR_HOLD_FRAMES} frames)")
    print()
    print("Controls:")
    print("  'q' - Quit application")
    print("  's' - Save current frame")
    print()
    print("Starting pipeline...")
    print("=" * 60)


def main():
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging()
        
        # Print banner
        print_banner()
        
        # Create configuration
        pipeline_config = create_config_from_file()
        
        # Create and start pipeline
        pipeline = LiveStreamPipeline(pipeline_config)
        pipeline.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user...")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        logging.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        print("👋 Thank you for using the Live Stream Privacy Pipeline!")


if __name__ == "__main__":
    main()
