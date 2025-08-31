#!/usr/bin/env python3
"""
Real-time audio processor for livestream integration.
Wraps the existing pipeline for real-time sensitive word detection and beeping.
"""

import json
import sys
import time
import threading
import queue
import uuid
import argparse
from typing import Dict, List, Optional
import logging
import signal

from livestream_pipeline import LivestreamPIIPipeline
from pipeline_types import AudioSegment, StreamConfig, ProcessingResult

class RealtimeAudioProcessor:
    """
    Real-time processor that integrates with mediasoup server.
    Receives audio chunks and outputs beep instructions.
    """
    
    def __init__(self, room_id: str, segment_duration: float = 3.0):
        """
        Initialize real-time processor.
        
        Args:
            room_id: Room identifier for this stream
            segment_duration: Duration of audio segments to process (seconds)
        """
        self.room_id = room_id
        self.segment_duration = segment_duration
        self.is_running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'realtime_processor_{room_id}.log'),
                logging.StreamHandler(sys.stderr)
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize pipeline
        self.stream_config = StreamConfig(
            stream_id=room_id,
            fps=30.0,
            audio_sample_rate=16000
        )
        
        self.pipeline = LivestreamPIIPipeline(
            stream_config=self.stream_config
        )
        
        # Audio buffering
        self.audio_buffer = bytearray()
        self.buffer_start_time = time.time()
        self.segment_counter = 0
        
        # Processing queue for audio segments
        self.processing_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        
        # Setup result callback
        self.pipeline.add_result_callback(self._handle_processing_result)
        
        self.logger.info(f"Real-time processor initialized for room {room_id}")
    
    def start(self):
        """Start the real-time processor."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start pipeline
        self.pipeline.start_pipeline()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Real-time processor started")
        
        # Output ready signal to mediasoup
        self._output_message({
            "type": "PROCESSOR_READY",
            "room_id": self.room_id,
            "timestamp": time.time()
        })
    
    def stop(self):
        """Stop the real-time processor."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        if self.pipeline:
            self.pipeline.cleanup()
        
        self.logger.info("Real-time processor stopped")
    
    def process_audio_chunk(self, audio_data: bytes, timestamp: float):
        """
        Process incoming audio chunk.
        
        Args:
            audio_data: Raw audio bytes (16kHz, 16-bit, mono)
            timestamp: Timestamp of this audio chunk
        """
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Calculate expected buffer size for segment duration
        sample_rate = self.stream_config.audio_sample_rate
        bytes_per_second = sample_rate * 2  # 16-bit = 2 bytes per sample
        expected_buffer_size = int(self.segment_duration * bytes_per_second)
        
        # Check if we have enough audio for a segment
        if len(self.audio_buffer) >= expected_buffer_size:
            # Extract segment
            segment_data = bytes(self.audio_buffer[:expected_buffer_size])
            self.audio_buffer = self.audio_buffer[expected_buffer_size:]
            
            # Create audio segment
            segment_id = f"{self.room_id}_{self.segment_counter}"
            start_time = self.buffer_start_time
            end_time = start_time + self.segment_duration
            
            audio_segment = AudioSegment(
                audio_data=segment_data,
                start_time=start_time,
                end_time=end_time,
                sample_rate=sample_rate,
                channels=1,
                segment_id=segment_id
            )
            
            # Submit for processing
            try:
                self.processing_queue.put_nowait(audio_segment)
                self.segment_counter += 1
                self.buffer_start_time = end_time
            except queue.Full:
                self.logger.warning("Processing queue full, dropping audio segment")
    
    def _processing_loop(self):
        """Processing loop for handling queued audio segments."""
        while self.is_running:
            try:
                # Get audio segment from queue
                audio_segment = self.processing_queue.get(timeout=1.0)
                
                # Submit to pipeline
                success = self.pipeline.submit_audio_segment(audio_segment)
                if not success:
                    self.logger.warning(f"Failed to submit segment {audio_segment.segment_id}")
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
    
    def _handle_processing_result(self, result: ProcessingResult):
        """
        Handle processing result and generate beep instructions.
        
        Args:
            result: Processing result from pipeline
        """
        try:
            # Check if sensitive words were detected
            if not result.redaction.detections:
                return
            
            # Generate beep instructions for each detection
            beep_instructions = []
            
            for detection in result.redaction.detections:
                beep_instruction = {
                    "type": "BEEP_AUDIO",
                    "room_id": self.room_id,
                    "segment_id": result.segment_id,
                    "start_time": detection.start_time,
                    "end_time": detection.end_time,
                    "pii_type": detection.pii_type.value,
                    "confidence": detection.confidence,
                    "detected_text": detection.text,
                    "timestamp": time.time()
                }
                beep_instructions.append(beep_instruction)
            
            # Output beep instructions to mediasoup
            for instruction in beep_instructions:
                self._output_message(instruction)
                
            self.logger.info(
                f"Generated {len(beep_instructions)} beep instructions for segment {result.segment_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling processing result: {e}")
    
    def _output_message(self, message: Dict):
        """
        Output message to stdout for mediasoup server.
        
        Args:
            message: Message dictionary to output
        """
        try:
            json_message = json.dumps(message)
            print(json_message, flush=True)
        except Exception as e:
            self.logger.error(f"Error outputting message: {e}")
    
    def run_stdin_loop(self):
        """
        Main loop that reads audio data from stdin.
        Expected format: Each line contains base64 encoded audio data with timestamp.
        """
        import base64
        
        self.start()
        
        try:
            for line in sys.stdin:
                if not self.is_running:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse input: timestamp:base64_audio_data
                    parts = line.split(':', 1)
                    if len(parts) != 2:
                        continue
                    
                    timestamp = float(parts[0])
                    audio_data = base64.b64decode(parts[1])
                    
                    # Process audio chunk
                    self.process_audio_chunk(audio_data, timestamp)
                    
                except Exception as e:
                    self.logger.error(f"Error processing stdin input: {e}")
                    continue
        
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.stop()


def main():
    """Main entry point for real-time processor."""
    parser = argparse.ArgumentParser(description='Real-time audio processor for sensitive word detection')
    parser.add_argument('--room-id', required=True, help='Room ID for this stream')
    parser.add_argument('--segment-duration', type=float, default=3.0, 
                        help='Audio segment duration in seconds (default: 3.0)')
    parser.add_argument('--log-level', default='INFO', 
                        help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create processor
    processor = RealtimeAudioProcessor(
        room_id=args.room_id,
        segment_duration=args.segment_duration
    )
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down...")
        processor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run processor
    try:
        processor.run_stdin_loop()
    except Exception as e:
        logging.error(f"Fatal error in processor: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()