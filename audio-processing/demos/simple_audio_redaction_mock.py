#!/usr/bin/env python3
"""
Simplified Audio Redaction Service Mock
Demonstrates the audio redaction integration without complex model dependencies
"""

import asyncio
import json
import logging
import numpy as np
import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading

# Mock data classes
@dataclass
class PIIDetection:
    text: str
    pii_type: str
    confidence: float
    start_time: float
    end_time: float

@dataclass
class AudioRedactionResult:
    segment_id: str
    original_audio: bytes
    redacted_audio: bytes
    transcription: str
    redacted_transcription: str
    pii_detections: List[PIIDetection]
    timestamp: float
    processing_time: float

class MockAudioRedactionEngine:
    """Mock engine that simulates PII detection and redaction"""
    
    def __init__(self):
        self.stats = {
            'processed_segments': 0,
            'total_pii_detections': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Mock Audio Redaction Engine initialized")
        
        # Mock PII patterns for demonstration
        self.pii_patterns = [
            {"text": "John Smith", "type": "PERSON", "confidence": 0.95},
            {"text": "555-123-4567", "type": "PHONE_NUMBER", "confidence": 0.92},
            {"text": "john@example.com", "type": "EMAIL", "confidence": 0.88},
            {"text": "123 Main Street", "type": "ADDRESS", "confidence": 0.85},
            {"text": "4532-1234-5678-9012", "type": "CREDIT_CARD", "confidence": 0.90}
        ]
    
    def add_audio_data(self, audio_data: bytes) -> List[AudioRedactionResult]:
        """Mock processing of audio data"""
        start_time = time.time()
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Mock transcription
        sample_texts = [
            "Hello, my name is John Smith",
            "You can reach me at 555-123-4567",
            "Send emails to john@example.com",
            "I live at 123 Main Street",
            "My card number is 4532-1234-5678-9012",
            "This is just regular conversation",
            "No sensitive information here"
        ]
        
        segment_id = f"mock_segment_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        transcription = np.random.choice(sample_texts)
        
        # Mock PII detection
        pii_detections = []
        redacted_text = transcription
        
        for pattern in self.pii_patterns:
            if pattern["text"].lower() in transcription.lower():
                detection = PIIDetection(
                    text=pattern["text"],
                    pii_type=pattern["type"],
                    confidence=pattern["confidence"],
                    start_time=0.0,
                    end_time=2.0
                )
                pii_detections.append(detection)
                
                # Redact the text
                redacted_text = redacted_text.replace(pattern["text"], "[REDACTED]")
        
        # Create mock redacted audio (just add some noise)
        original_audio = audio_data
        redacted_audio = self.create_mock_redacted_audio(audio_data, len(pii_detections) > 0)
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.stats['processed_segments'] += 1
        self.stats['total_pii_detections'] += len(pii_detections)
        self.stats['total_processing_time'] += processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['processed_segments']
        )
        
        result = AudioRedactionResult(
            segment_id=segment_id,
            original_audio=original_audio,
            redacted_audio=redacted_audio,
            transcription=transcription,
            redacted_transcription=redacted_text,
            pii_detections=pii_detections,
            timestamp=time.time(),
            processing_time=processing_time
        )
        
        self.logger.info(
            f"Processed {segment_id}: '{transcription}' -> {len(pii_detections)} PII detected"
        )
        
        return [result] if result else []
    
    def create_mock_redacted_audio(self, audio_data: bytes, has_pii: bool) -> bytes:
        """Create mock redacted audio"""
        if not has_pii:
            return audio_data
        
        # Convert to numpy array, add noise where PII would be, convert back
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add some noise to simulate redaction
            noise_level = 0.1
            noise = np.random.normal(0, noise_level, len(audio_array))
            redacted_array = audio_array * 0.3 + noise * 0.7  # Mix original with noise
            
            # Convert back to bytes
            redacted_int16 = (redacted_array * 32767).astype(np.int16)
            return redacted_int16.tobytes()
        except:
            # Fallback: just return modified original
            return audio_data
    
    def get_stats(self) -> Dict:
        return self.stats.copy()

class MockAudioRedactionService:
    """Mock service for handling real-time audio redaction"""
    
    def __init__(self, host="0.0.0.0", port=5002):
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'mock-audio-redaction-secret'
        CORS(self.app, origins="*")
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False
        )
        
        # Initialize mock engine
        self.redaction_engine = MockAudioRedactionEngine()
        
        # Active connections
        self.active_connections = {}
        self.room_redactors = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_routes()
        self.setup_socketio_handlers()
        
        self.logger.info("Mock Audio Redaction Service initialized")
    
    def setup_routes(self):
        @self.app.route('/health')
        def health():
            return {
                'status': 'healthy',
                'service': 'mock-audio-redaction',
                'stats': self.redaction_engine.get_stats()
            }
        
        @self.app.route('/stats')
        def stats():
            return {
                'engine_stats': self.redaction_engine.get_stats(),
                'active_connections': len(self.active_connections),
                'active_rooms': len(self.room_redactors)
            }
    
    def setup_socketio_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            session_id = str(uuid.uuid4())
            self.active_connections[session_id] = {
                'connected_at': time.time(),
                'room_id': None,
                'role': None
            }
            emit('connected', {'session_id': session_id})
            self.logger.info(f"Client connected: {session_id}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            # Clean up connection
            self.logger.info("Client disconnected")
        
        @self.socketio.on('join_redaction_room')
        def handle_join_redaction_room(data):
            room_id = data.get('room_id')
            role = data.get('role', 'viewer')
            
            if not room_id:
                emit('error', {'message': 'Room ID required'})
                return
            
            # Create room redactor if needed
            if room_id not in self.room_redactors:
                self.room_redactors[room_id] = MockAudioRedactionEngine()
            
            emit('joined_redaction_room', {
                'room_id': room_id,
                'role': role,
                'status': 'ready'
            })
            self.logger.info(f"Client joined redaction room {room_id} as {role}")
        
        @self.socketio.on('audio_data')
        def handle_audio_data(data):
            """Handle incoming audio data for processing"""
            try:
                # Extract audio data
                audio_bytes = data.get('audio')
                if not audio_bytes:
                    return
                
                # Process audio data with mock engine
                results = self.redaction_engine.add_audio_data(audio_bytes)
                
                # Send results
                for result in results:
                    result_data = {
                        'segment_id': result.segment_id,
                        'original_transcription': result.transcription,
                        'redacted_transcription': result.redacted_transcription,
                        'pii_count': len(result.pii_detections),
                        'processing_time': result.processing_time,
                        'timestamp': result.timestamp
                    }
                    
                    # Send redacted audio
                    self.socketio.emit('redacted_audio', {
                        **result_data,
                        'audio_data': result.redacted_audio
                    })
                    
                    # Send PII detection details
                    emit('pii_detections', {
                        **result_data,
                        'pii_detections': [
                            {
                                'text': det.text,
                                'pii_type': det.pii_type,
                                'confidence': det.confidence,
                                'start_time': det.start_time,
                                'end_time': det.end_time
                            }
                            for det in result.pii_detections
                        ]
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing audio data: {e}")
                emit('error', {'message': str(e)})
        
        @self.socketio.on('get_redaction_stats')
        def handle_get_stats():
            stats = self.redaction_engine.get_stats()
            emit('redaction_stats', stats)
    
    def run(self, debug=False):
        """Run the service"""
        self.logger.info(f"Starting Mock Audio Redaction Service on {self.host}:{self.port}")
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            use_reloader=False
        )

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the mock service
    service = MockAudioRedactionService()
    service.run(debug=False)