#!/usr/bin/env python3
"""
Test processor for audio processing integration without ML dependencies.
Simulates sensitive word detection for testing the integration.
"""

import json
import sys
import time
import argparse
import threading
import queue
import uuid
from typing import Dict, List, Optional

class TestProcessor:
    """
    Test processor that simulates sensitive word detection.
    """
    
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.is_running = False
        self.segment_counter = 0
        
        # Test words that trigger beeps
        self.sensitive_words = [
            "password", "secret", "confidential", 
            "phone", "email", "address",
            "social security", "credit card", "bank account"
        ]
        
    def start(self):
        """Start the test processor."""
        self.is_running = True
        
        # Output ready signal
        self._output_message({
            "type": "PROCESSOR_READY",
            "room_id": self.room_id,
            "timestamp": time.time()
        })
        
        print(f"Test processor started for room {self.room_id}", file=sys.stderr)
    
    def stop(self):
        """Stop the test processor."""
        self.is_running = False
        print(f"Test processor stopped for room {self.room_id}", file=sys.stderr)
    
    def process_audio_chunk(self, audio_data: bytes, timestamp: float):
        """
        Process audio chunk - simulate detection.
        For testing, we'll randomly trigger beeps.
        """
        self.segment_counter += 1
        
        # Simulate processing every few segments
        if self.segment_counter % 10 == 0:  # Every 10th segment (~30 seconds)
            # Simulate finding sensitive word
            detected_word = self.sensitive_words[self.segment_counter % len(self.sensitive_words)]
            
            beep_instruction = {
                "type": "BEEP_AUDIO",
                "room_id": self.room_id,
                "segment_id": f"{self.room_id}_{self.segment_counter}",
                "start_time": timestamp,
                "end_time": timestamp + 2.0,  # 2 second beep
                "pii_type": "TEST_SENSITIVE_WORD",
                "confidence": 0.95,
                "detected_text": detected_word,
                "timestamp": time.time()
            }
            
            self._output_message(beep_instruction)
            
            print(f"Simulated detection: {detected_word} at {timestamp}s", file=sys.stderr)
    
    def _output_message(self, message: Dict):
        """Output message to stdout for mediasoup server."""
        try:
            json_message = json.dumps(message)
            print(json_message, flush=True)
        except Exception as e:
            print(f"Error outputting message: {e}", file=sys.stderr)
    
    def run_stdin_loop(self):
        """Main loop that reads audio data from stdin."""
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
                    print(f"Error processing stdin input: {e}", file=sys.stderr)
                    continue
        
        except KeyboardInterrupt:
            print("Received interrupt signal", file=sys.stderr)
        finally:
            self.stop()


def main():
    """Main entry point for test processor."""
    parser = argparse.ArgumentParser(description='Test audio processor for sensitive word detection')
    parser.add_argument('--room-id', required=True, help='Room ID for this stream')
    parser.add_argument('--segment-duration', type=float, default=3.0, 
                        help='Audio segment duration in seconds (default: 3.0)')
    
    args = parser.parse_args()
    
    # Create test processor
    processor = TestProcessor(room_id=args.room_id)
    
    # Run processor
    try:
        processor.run_stdin_loop()
    except Exception as e:
        print(f"Fatal error in test processor: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()