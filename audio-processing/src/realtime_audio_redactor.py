"""
Real-time Audio PII Redaction Service
Integrates Whisper v3 and DeBERTa v3 for live audio stream processing with redaction
"""

import asyncio
import json
import logging
import numpy as np
import soundfile as sf
import io
import time
import uuid
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import socketio
import requests
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import queue
import wave

from whisper_processor import WhisperProcessor
from pii_detector import PIIDetector
from pipeline_types import AudioSegment, TranscriptionResult, PIIDetection


@dataclass
class AudioRedactionResult:
    """Result of audio redaction processing"""
    segment_id: str
    original_audio: bytes
    redacted_audio: bytes
    transcription: str
    redacted_transcription: str
    pii_detections: List[PIIDetection]
    timestamp: float
    processing_time: float


class AudioRedactionEngine:
    """Core engine for real-time audio PII redaction"""
    
    def __init__(
        self,
        whisper_model: str = "large-v3",
        deberta_model_path: str = "./models/",
        sample_rate: int = 16000,
        segment_duration: float = 3.0,
        overlap_duration: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.segment_samples = int(sample_rate * segment_duration)
        self.overlap_samples = int(sample_rate * overlap_duration)
        
        # Detect available device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Initialize processors
        self.whisper_processor = WhisperProcessor(
            model_name=whisper_model,
            device=device,
            enable_word_timestamps=True
        )
        
        self.pii_detector = PIIDetector(
            model_path=deberta_model_path,
            device=device,
            confidence_threshold=0.7
        )
        
        # Audio processing buffers
        self.audio_buffer = np.array([], dtype=np.float32)
        self.segment_counter = 0
        
        # Statistics
        self.stats = {
            'processed_segments': 0,
            'total_pii_detections': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Audio redaction engine initialized")
    
    def add_audio_data(self, audio_data: bytes) -> List[AudioRedactionResult]:
        """
        Add new audio data and process complete segments
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Returns:
            List of processed redaction results
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
        
        results = []
        
        # Process complete segments
        while len(self.audio_buffer) >= self.segment_samples:
            # Extract segment with overlap
            segment_data = self.audio_buffer[:self.segment_samples]
            
            # Create audio segment - convert numpy array to bytes for AudioSegment
            segment_data_bytes = self.audio_to_bytes(segment_data)
            segment = AudioSegment(
                audio_data=segment_data_bytes,
                start_time=self.segment_counter * (self.segment_duration - self.overlap_duration),
                end_time=self.segment_counter * (self.segment_duration - self.overlap_duration) + self.segment_duration,
                sample_rate=self.sample_rate,
                channels=1,
                segment_id=f"segment_{self.segment_counter}_{int(time.time())}"
            )
            
            # Process segment
            result = self.process_audio_segment(segment)
            if result:
                results.append(result)
            
            # Move buffer forward (with overlap)
            advance_samples = self.segment_samples - self.overlap_samples
            self.audio_buffer = self.audio_buffer[advance_samples:]
            self.segment_counter += 1
        
        return results
    
    def process_audio_segment(self, segment: AudioSegment) -> Optional[AudioRedactionResult]:
        """Process a single audio segment for PII detection and redaction"""
        start_time = time.time()
        
        try:
            # Convert bytes back to numpy array for processing
            segment_audio_array = np.frombuffer(segment.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Create a new AudioSegment with numpy array for Whisper processor
            processing_segment = AudioSegment(
                audio_data=segment_audio_array,
                start_time=segment.start_time,
                end_time=segment.end_time,
                sample_rate=segment.sample_rate,
                channels=segment.channels,
                segment_id=segment.segment_id
            )
            
            # Step 1: Transcribe with Whisper
            transcription = self.whisper_processor.transcribe_segment(processing_segment)
            
            if not transcription.text.strip():
                return None
            
            # Step 2: Detect PII in transcription
            redaction_result = self.pii_detector.process_transcription(transcription)
            
            # Step 3: Create redacted audio using the numpy array
            redacted_audio = self.create_redacted_audio(
                segment_audio_array, 
                redaction_result.detections,
                transcription.word_timestamps
            )
            
            # Convert audio to bytes
            original_audio_bytes = segment.audio_data  # Already in bytes
            redacted_audio_bytes = self.audio_to_bytes(redacted_audio)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['processed_segments'] += 1
            self.stats['total_pii_detections'] += len(redaction_result.detections)
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['processed_segments']
            )
            
            result = AudioRedactionResult(
                segment_id=segment.segment_id,
                original_audio=original_audio_bytes,
                redacted_audio=redacted_audio_bytes,
                transcription=transcription.text,
                redacted_transcription=redaction_result.redacted_text,
                pii_detections=redaction_result.detections,
                timestamp=time.time(),
                processing_time=processing_time
            )
            
            self.logger.info(
                f"Processed {segment.segment_id}: {len(redaction_result.detections)} PII detected, "
                f"processing time: {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing segment {segment.segment_id}: {e}")
            return None
    
    def create_redacted_audio(
        self,
        audio_data: np.ndarray,
        pii_detections: List[PIIDetection],
        word_timestamps: List[Dict]
    ) -> np.ndarray:
        """
        Create redacted audio by replacing PII segments with noise/silence
        
        Args:
            audio_data: Original audio data
            pii_detections: List of detected PII
            word_timestamps: Word-level timestamps from Whisper
            
        Returns:
            Redacted audio data
        """
        redacted_audio = audio_data.copy()
        
        for detection in pii_detections:
            # Find corresponding audio timestamps for the detected PII
            start_sample, end_sample = self.find_audio_boundaries(
                detection, word_timestamps, len(audio_data)
            )
            
            if start_sample < end_sample:
                # Replace PII audio with noise or silence
                redaction_length = end_sample - start_sample
                
                # Option 1: Replace with silence
                # redacted_audio[start_sample:end_sample] = 0
                
                # Option 2: Replace with white noise (more natural sounding)
                noise_level = np.std(audio_data) * 0.1  # Low-level noise
                noise = np.random.normal(0, noise_level, redaction_length).astype(np.float32)
                redacted_audio[start_sample:end_sample] = noise
                
                # Option 3: Replace with beep tone
                # beep_freq = 1000  # 1kHz beep
                # t = np.linspace(0, redaction_length / self.sample_rate, redaction_length)
                # beep = 0.1 * np.sin(2 * np.pi * beep_freq * t).astype(np.float32)
                # redacted_audio[start_sample:end_sample] = beep
        
        return redacted_audio
    
    def find_audio_boundaries(
        self,
        detection: PIIDetection,
        word_timestamps: List[Dict],
        audio_length: int
    ) -> tuple:
        """Find audio sample boundaries for a PII detection"""
        # Find the start and end times from word timestamps
        start_time = detection.start_time
        end_time = detection.end_time
        
        # Convert to sample indices
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        # Ensure boundaries are within audio length
        start_sample = max(0, start_sample)
        end_sample = min(audio_length, end_sample)
        
        # Add small buffer around PII
        buffer_samples = int(0.1 * self.sample_rate)  # 100ms buffer
        start_sample = max(0, start_sample - buffer_samples)
        end_sample = min(audio_length, end_sample + buffer_samples)
        
        return start_sample, end_sample
    
    def audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert audio numpy array to bytes"""
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()


class RealtimeAudioRedactionService:
    """Service for handling real-time audio redaction via WebSocket/SocketIO"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5002,
        backend_url: str = "http://localhost:5000",
        mediasoup_url: str = "http://localhost:3001"
    ):
        self.host = host
        self.port = port
        self.backend_url = backend_url
        self.mediasoup_url = mediasoup_url
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'audio-redaction-secret'
        CORS(self.app, origins="*")
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=True,
            engineio_logger=False
        )
        
        # Initialize audio redaction engine
        self.redaction_engine = AudioRedactionEngine()
        
        # Active connections and rooms
        self.active_connections = {}  # session_id -> connection_info
        self.room_redactors = {}  # room_id -> redaction_engine
        
        # Result callbacks
        self.result_callbacks = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_routes()
        self.setup_socketio_handlers()
        
        self.logger.info("Realtime Audio Redaction Service initialized")
    
    def setup_routes(self):
        """Setup HTTP routes"""
        @self.app.route('/health')
        def health():
            return {
                'status': 'healthy',
                'service': 'audio-redaction',
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
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect(auth=None):
            session_id = request.sid
            self.active_connections[session_id] = {
                'connected_at': time.time(),
                'room_id': None,
                'role': None
            }
            emit('connected', {'session_id': session_id})
            self.logger.info(f"Client connected: {session_id}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            # Just clean up any stale connections
            self.logger.info("Client disconnected")
            
            # Remove any old connections (older than 1 hour)
            current_time = time.time()
            stale_sessions = []
            for session_id, conn in self.active_connections.items():
                if current_time - conn['connected_at'] > 3600:  # 1 hour
                    stale_sessions.append(session_id)
            
            for session_id in stale_sessions:
                if session_id in self.active_connections:
                    del self.active_connections[session_id]
            
            if stale_sessions:
                self.logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
        
        @self.socketio.on('join_redaction_room')
        def handle_join_redaction_room(data):
            room_id = data.get('room_id')
            role = data.get('role', 'viewer')  # 'host' or 'viewer'
            session_id = request.sid
            
            if not room_id:
                emit('error', {'message': 'Room ID required'})
                return
            
            # Update connection info
            if session_id in self.active_connections:
                self.active_connections[session_id].update({
                    'room_id': room_id,
                    'role': role
                })
            
            # Create room redactor if needed
            if room_id not in self.room_redactors:
                self.room_redactors[room_id] = AudioRedactionEngine()
            
            emit('joined_redaction_room', {
                'room_id': room_id,
                'role': role,
                'status': 'ready'
            })
            self.logger.info(f"Client {session_id} joined redaction room {room_id} as {role}")
        
        @self.socketio.on('audio_data')
        def handle_audio_data(data):
            """Handle incoming audio data for processing"""
            session_id = request.sid
            connection_info = self.active_connections.get(session_id, {})
            room_id = connection_info.get('room_id')
            
            if not room_id or room_id not in self.room_redactors:
                emit('error', {'message': 'Not joined to redaction room'})
                return
            
            # Extract audio data
            audio_bytes = data.get('audio')
            if not audio_bytes:
                return
            
            # Process audio data
            redactor = self.room_redactors[room_id]
            results = redactor.add_audio_data(audio_bytes)
            
            # Send results to all viewers in the room
            for result in results:
                result_data = {
                    'segment_id': result.segment_id,
                    'original_transcription': result.transcription,
                    'redacted_transcription': result.redacted_transcription,
                    'pii_count': len(result.pii_detections),
                    'processing_time': result.processing_time,
                    'timestamp': result.timestamp
                }
                
                # Send redacted audio to viewers
                self.socketio.emit('redacted_audio', {
                    **result_data,
                    'audio_data': result.redacted_audio
                }, room=f"redaction_{room_id}")
                
                # Send PII detection details to host only
                if connection_info.get('role') == 'host':
                    emit('pii_detections', {
                        **result_data,
                        'pii_detections': [
                            {
                                'text': det.text,
                                'pii_type': det.pii_type.value,
                                'confidence': det.confidence,
                                'start_time': det.start_time,
                                'end_time': det.end_time
                            }
                            for det in result.pii_detections
                        ]
                    })
        
        @self.socketio.on('get_redaction_stats')
        def handle_get_stats():
            session_id = request.sid
            connection_info = self.active_connections.get(session_id, {})
            room_id = connection_info.get('room_id')
            
            if room_id and room_id in self.room_redactors:
                stats = self.room_redactors[room_id].get_stats()
                emit('redaction_stats', stats)
            else:
                emit('redaction_stats', self.redaction_engine.get_stats())
    
    def add_result_callback(self, callback: Callable[[AudioRedactionResult], None]):
        """Add callback for processing results"""
        self.result_callbacks.append(callback)
    
    def run(self, debug: bool = False):
        """Run the service"""
        self.logger.info(f"Starting Realtime Audio Redaction Service on {self.host}:{self.port}")
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            use_reloader=False
        )


# Integration with existing backend
class AudioRedactionIntegration:
    """Integration layer with existing backend and mediasoup"""
    
    def __init__(
        self,
        redaction_service_url: str = "http://localhost:5002",
        backend_url: str = "http://localhost:5000",
        mediasoup_url: str = "http://localhost:3001"
    ):
        self.redaction_service_url = redaction_service_url
        self.backend_url = backend_url
        self.mediasoup_url = mediasoup_url
        
        # Connect to redaction service
        self.sio = socketio.Client()
        self.setup_client_handlers()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def setup_client_handlers(self):
        """Setup client event handlers for redaction service"""
        
        @self.sio.on('connect')
        def on_connect():
            self.logger.info("Connected to audio redaction service")
        
        @self.sio.on('disconnect')
        def on_disconnect():
            self.logger.info("Disconnected from audio redaction service")
        
        @self.sio.on('redacted_audio')
        def on_redacted_audio(data):
            # Forward redacted audio to mediasoup or clients
            self.logger.info(f"Received redacted audio: {data.get('segment_id')}")
            # TODO: Integrate with mediasoup audio pipeline
        
        @self.sio.on('pii_detections')
        def on_pii_detections(data):
            # Handle PII detection notifications
            self.logger.info(f"PII detected: {data.get('pii_count')} items")
    
    def connect(self):
        """Connect to the redaction service"""
        self.sio.connect(self.redaction_service_url)
    
    def join_room(self, room_id: str, role: str = 'viewer'):
        """Join a redaction room"""
        self.sio.emit('join_redaction_room', {
            'room_id': room_id,
            'role': role
        })
    
    def send_audio_data(self, audio_data: bytes):
        """Send audio data for processing"""
        self.sio.emit('audio_data', {'audio': audio_data})


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the redaction service
    service = RealtimeAudioRedactionService()
    service.run(debug=True)