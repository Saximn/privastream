#!/usr/bin/env python3
"""
Main entry point for Privastream CLI.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add privastream to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from privastream.core.config import Config, WebConfig, ProductionConfig
from privastream.core.logging import setup_logging
from privastream.models import UnifiedBlurDetector
from privastream.web.backend.app import create_app


def run_web_server(config: WebConfig = None):
    """Run the web server for streaming interface."""
    config = config or WebConfig()
    logger = setup_logging(config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else 'INFO')
    
    try:
        logger.info('Starting Privastream web server')
        app, socketio = create_app()
        socketio.run(
            app, 
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG
        )
    except Exception as e:
        logger.error(f'Failed to start web server: {e}')
        raise


def process_video(input_path: str, output_path: str, config: Config = None):
    """Process a video file for PII redaction."""
    import cv2
    from privastream.models.detection.blur_utils import apply_blur_regions
    
    config = config or Config()
    logger = setup_logging()
    
    logger.info(f'Processing video: {input_path} -> {output_path}')
    
    try:
        # Initialize detector
        detector = UnifiedBlurDetector(config)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        logger.info(f'Processing {total_frames} frames...')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame for detections
            results = detector.process_frame(frame, frame_id)
            
            # Apply blur to detected regions
            rectangles = detector.get_all_rectangles(results)
            polygons = detector.get_all_polygons(results)
            
            processed_frame = apply_blur_regions(
                frame, rectangles, polygons, config.GAUSSIAN_KERNEL_SIZE
            )
            
            # Write frame
            out.write(processed_frame)
            
            if frame_id % 100 == 0:
                logger.info(f'Processed {frame_id}/{total_frames} frames')
            
            frame_id += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        logger.info(f'Video processing completed: {output_path}')
        
    except Exception as e:
        logger.error(f'Video processing failed: {e}')
        raise


def process_audio(input_path: str, output_path: str, config: Config = None):
    """Process audio file for PII redaction."""
    config = config or Config()
    logger = setup_logging()
    
    logger.info(f'Processing audio: {input_path} -> {output_path}')
    
    try:
        # Import audio processing modules
        from privastream.models.audio import AudioPIIDetector
        
        # Initialize audio detector
        detector = AudioPIIDetector(config)
        
        # Process audio file
        detector.process_file(input_path, output_path)
        
        logger.info(f'Audio processing completed: {output_path}')
        
    except Exception as e:
        logger.error(f'Audio processing failed: {e}')
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Privastream - AI-Powered Privacy Streaming Platform'
    )
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Web server command
    web_parser = subparsers.add_parser('web', help='Start web server')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    web_parser.add_argument('--config', choices=['web', 'production'], default='web',
                           help='Configuration preset')
    
    # Video processing command
    video_parser = subparsers.add_parser('video', help='Process video file')
    video_parser.add_argument('input', help='Input video file path')
    video_parser.add_argument('output', help='Output video file path')
    video_parser.add_argument('--config', help='Configuration file path')
    
    # Audio processing command
    audio_parser = subparsers.add_parser('audio', help='Process audio file')
    audio_parser.add_argument('input', help='Input audio file path')
    audio_parser.add_argument('output', help='Output audio file path')
    audio_parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.command == 'web':
        # Create config
        if args.config == 'production':
            config = ProductionConfig()
        else:
            config = WebConfig()
        
        config.FLASK_HOST = args.host
        config.FLASK_PORT = args.port
        config.FLASK_DEBUG = args.debug
        
        run_web_server(config)
    
    elif args.command == 'video':
        config = Config()
        if args.config:
            # Load custom config from file
            pass
        
        process_video(args.input, args.output, config)
    
    elif args.command == 'audio':
        config = Config()
        if args.config:
            # Load custom config from file
            pass
        
        process_audio(args.input, args.output, config)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()