#!/usr/bin/env python3
"""
Audio Redaction Server with Faster-Whisper + BERT
Fast and Accurate ASR with high-accuracy PII detection
Based on the working Vosk implementation structure
"""

import os
import logging
import json
import numpy as np
import librosa
import base64
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline

# Faster-Whisper import
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("⚠️ faster-whisper not available. Install with: pip install faster-whisper")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)

class FasterWhisperBertRedactionService:
    def __init__(self):
        logger.info("Initializing Faster-Whisper + BERT Audio Redaction Service...")
        
        # Set up device (CPU/GPU for Faster-Whisper, prefer GPU for BERT if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        
        # Initialize models
        self.load_models()
        
        # Sensitive keywords for additional detection
        self.sensitive_keywords = [
            "password", "secret", "pin", "ssn", "social", "credit", "card",
            "account", "login", "private", "confidential", "address", "phone"
        ]
        
        logger.info("Faster-Whisper + BERT Audio Redaction Service initialized")
    
    def load_models(self):
        """Load Faster-Whisper ASR and BERT NER models"""
        # Load Faster-Whisper model
        try:
            if not FASTER_WHISPER_AVAILABLE:
                raise Exception("faster-whisper package not available")
                
            logger.info("Loading Faster-Whisper ASR model...")
            
            # Model size options: tiny, base, small, medium, large-v1, large-v2, large-v3
            model_size = "base"  # Good balance of speed and accuracy
            
            # Configure compute type based on device
            if self.device == "cuda":
                compute_type = "float16"  # Faster on GPU
            else:
                compute_type = "int8"     # Faster on CPU
            
            # Initialize Faster-Whisper model
            self.whisper_model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type,
                cpu_threads=4 if self.device == "cpu" else 0,
                num_workers=1  # Reduce memory usage
            )
            logger.info("✅ Faster-Whisper ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Faster-Whisper model: {e}")
            self.whisper_model = None
        
        # Load BERT NER model (keep the GPU-accelerated NER for accuracy)
        try:
            logger.info("Loading BERT NER model...")
            pipeline_kwargs = {
                "task": "ner",
                "model": "dslim/bert-base-NER",
                "aggregation_strategy": "simple",
                "device": 0 if self.device == "cuda" else -1
            }
            
            # Only add model_kwargs for GPU to avoid deprecated warning
            if self.device == "cuda":
                pipeline_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            
            self.ner_pipeline = pipeline(**pipeline_kwargs)
            logger.info(f"✅ BERT NER model loaded successfully on {self.device.upper()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load BERT NER model: {e}")
            self.ner_pipeline = None
    
    def transcribe_with_faster_whisper(self, audio_data, sample_rate=16000):
        """Fast and accurate transcription using Faster-Whisper"""
        if self.whisper_model is None:
            logger.warning("Faster-Whisper model not available")
            return "", []
            
        try:
            # Ensure audio is in the right format for Whisper (16kHz, mono, float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
                
            # Resample if necessary
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Use Faster-Whisper for transcription with word timestamps
            segments, info = self.whisper_model.transcribe(
                audio_data,
                word_timestamps=True,
                language="en",  # Specify language for better performance
                condition_on_previous_text=False,  # Disable context for shorter audio
                vad_filter=True,  # Voice Activity Detection to skip silent parts
                vad_parameters=dict(min_silence_duration_ms=500)  # Skip silence > 500ms
            )
            
            # Process segments and extract word-level information
            transcript_parts = []
            word_times = []
            
            for segment in segments:
                segment_text = segment.text.strip()
                if not segment_text:
                    continue
                    
                # Add segment text to transcript
                transcript_parts.append(segment_text)
                
                # Process word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_text = word.word.strip()
                        if word_text:
                            word_times.append({
                                'word': word_text,
                                'start': word.start,
                                'end': word.end,
                                'confidence': getattr(word, 'probability', 1.0)
                            })
                else:
                    # Fallback: estimate word timings from segment timing
                    words = segment_text.split()
                    word_duration = (segment.end - segment.start) / len(words) if words else 0
                    
                    for i, word in enumerate(words):
                        start_time = segment.start + (i * word_duration)
                        end_time = start_time + word_duration
                        
                        word_times.append({
                            'word': word,
                            'start': start_time,
                            'end': end_time,
                            'confidence': 1.0
                        })
            
            transcript = " ".join(transcript_parts)
            
            logger.info(f"📝 Faster-Whisper transcript: '{transcript}'")
            logger.info(f"📊 Word count: {len(word_times)} words")
            logger.info(f"🎯 Detected language: {info.language} (confidence: {info.language_probability:.3f})")
            
            return transcript, word_times
            
        except Exception as e:
            logger.error(f"Error in Faster-Whisper transcription: {e}")
            return "", []
    
    def detect_sensitive_spans_bert(self, transcript, word_times):
        """Detect PII using BERT NER + keywords"""
        redact_intervals = []
        logger.info("Detecting sensitive spans with BERT NER + keywords...")
        
        if not transcript or not word_times:
            return redact_intervals
        
        # Keyword-based detection (fast)
        keyword_detections = []
        for keyword in self.sensitive_keywords:
            if keyword.lower() in transcript.lower():
                keyword_detections.append(keyword)
                logger.info(f"🔑 Keyword detected: '{keyword}'")
                
                # Find word positions containing this keyword
                for i, word_info in enumerate(word_times):
                    if keyword.lower() in word_info['word'].lower():
                        start_sec = word_info['start']
                        # Include surrounding words for context
                        end_idx = min(i + 2, len(word_times) - 1)
                        end_sec = word_times[end_idx]['end']
                        redact_intervals.append((start_sec, end_sec))
        
        # BERT NER detection (high accuracy)
        ner_detections = []
        if self.ner_pipeline and transcript:
            logger.info("🔍 Running BERT NER analysis...")
            
            try:
                # Clear GPU cache before NER
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Run BERT NER
                with torch.no_grad():
                    ner_results = self.ner_pipeline(transcript)
                    
                # Process NER results
                logger.info(f"🔍 BERT NER found {len(ner_results)} entities")
                
                for entity in ner_results:
                    entity_text = transcript[entity["start"]:entity["end"]]
                    logger.info(f"   Entity: '{entity_text}' -> {entity['entity_group']} (confidence: {entity.get('score', 0):.3f})")
                    
                    # Consider PER, LOC, MISC as potentially sensitive
                    if entity['entity_group'] in ['PER', 'LOC', 'MISC']:
                        # Map character positions to time intervals
                        start_char = entity["start"]
                        end_char = entity["end"]
                        
                        # Find corresponding words
                        char_pos = 0
                        for i, word_info in enumerate(word_times):
                            word_start = char_pos
                            word_end = char_pos + len(word_info['word'])
                            
                            if word_start <= start_char < word_end or start_char <= word_start < end_char:
                                start_sec = word_info['start']
                                end_sec = word_times[min(i + 1, len(word_times) - 1)]['end']
                                redact_intervals.append((start_sec, end_sec))
                                ner_detections.append(entity_text)
                                logger.info(f"🚨 BERT NER DETECTION: '{entity_text}' ({entity['entity_group']}) at {start_sec:.2f}s-{end_sec:.2f}s")
                                break
                            
                            char_pos = word_end + 1  # +1 for space
                
                # Cleanup after NER
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
            except Exception as e:
                logger.error(f"Error in BERT NER: {e}")
        
        # Merge overlapping intervals
        if redact_intervals:
            logger.info(f"🔗 Merging {len(redact_intervals)} intervals...")
            redact_intervals = sorted(redact_intervals, key=lambda x: x[0])
            merged = [redact_intervals[0]]
            
            for start, end in redact_intervals[1:]:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))
            
            redact_intervals = merged
            logger.info(f"🔗 Merged to {len(redact_intervals)} final intervals")
        
        logger.info(f"🎯 PII DETECTION SUMMARY:")
        logger.info(f"   Keywords found: {len(keyword_detections)} - {keyword_detections}")
        logger.info(f"   BERT NER found: {len(ner_detections)} - {ner_detections}")
        logger.info(f"   Total intervals: {len(redact_intervals)}")
        
        return redact_intervals
    
    def mute_intervals(self, audio_data, sample_rate, intervals):
        """Mute specified time intervals in audio"""
        if not intervals:
            return audio_data
            
        try:
            logger.info(f"🔇 Muting {len(intervals)} intervals in audio")
            audio_copy = audio_data.copy()
            
            for start_sec, end_sec in intervals:
                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)
                
                # Ensure bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_copy), end_sample)
                
                # Mute by setting to zero
                audio_copy[start_sample:end_sample] = 0
                logger.info(f"   Muted {start_sec:.2f}s - {end_sec:.2f}s (samples {start_sample}-{end_sample})")
            
            return audio_copy
            
        except Exception as e:
            logger.error(f"Error in audio muting: {e}")
            return audio_data
    
    def process_audio_chunk(self, audio_data, sample_rate=16000):
        """Main processing: Faster-Whisper ASR -> BERT NER -> Muting"""
        try:
            total_start = time.time()
            logger.info(f"Processing {len(audio_data)} samples at {sample_rate}Hz with Faster-Whisper + BERT")
            
            # Step 1: Fast and accurate transcription with Faster-Whisper
            whisper_start = time.time()
            transcript, word_times = self.transcribe_with_faster_whisper(audio_data, sample_rate)
            whisper_time = time.time() - whisper_start
            
            # Step 2: PII detection with BERT NER + keywords
            bert_start = time.time()
            intervals = self.detect_sensitive_spans_bert(transcript, word_times)
            bert_time = time.time() - bert_start
            
            # Step 3: Mute sensitive intervals
            mute_start = time.time()
            redacted_audio = self.mute_intervals(audio_data, sample_rate, intervals)
            mute_time = time.time() - mute_start
            
            total_time = time.time() - total_start
            audio_duration = len(audio_data) / sample_rate
            speed_ratio = total_time / audio_duration
            
            # Detailed timing breakdown
            logger.info(f"⏱️  TIMING BREAKDOWN:")
            logger.info(f"   🎤 Faster-Whisper: {whisper_time:.3f}s ({whisper_time/audio_duration:.2f}x real-time)")
            logger.info(f"   🧠 BERT NER:       {bert_time:.3f}s ({bert_time/audio_duration:.2f}x real-time)")
            logger.info(f"   🔇 Audio Muting:   {mute_time:.3f}s ({mute_time/audio_duration:.2f}x real-time)")
            logger.info(f"   📊 TOTAL:          {total_time:.3f}s ({speed_ratio:.2f}x real-time)")
            logger.info(f"✅ Processing complete: {len(intervals)} redactions")
            
            return redacted_audio, {
                "success": True,
                "transcript": transcript,
                "redacted_intervals": intervals,
                "pii_count": len(intervals),
                "processing_time": total_time,
                "speed_ratio": speed_ratio,
                "timing_breakdown": {
                    "whisper_time": whisper_time,
                    "bert_time": bert_time,
                    "mute_time": mute_time,
                    "whisper_ratio": whisper_time / audio_duration,
                    "bert_ratio": bert_time / audio_duration,
                    "mute_ratio": mute_time / audio_duration
                },
                "sample_rate": sample_rate,
                "device_used": f"Faster-Whisper+BERT({self.device})"
            }
            
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            return audio_data, {
                "success": False,
                "error": str(e),
                "transcript": "",
                "redacted_intervals": [],
                "pii_count": 0,
                "sample_rate": sample_rate,
                "device_used": f"Faster-Whisper+BERT({self.device})"
            }

# Initialize service
logger.info("Starting Faster-Whisper + BERT Audio Redaction Service...")
redaction_service = FasterWhisperBertRedactionService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if redaction_service.device == "cuda":
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        }
    
    return jsonify({
        "status": "healthy",
        "service": "Faster-Whisper + BERT Audio Redaction API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "asr": f"Faster-Whisper ({redaction_service.device.upper()})",
            "ner": f"BERT ({redaction_service.device.upper()})"
        },
        "models_loaded": {
            "faster_whisper": redaction_service.whisper_model is not None,
            "bert": redaction_service.ner_pipeline is not None
        },
        "faster_whisper_available": FASTER_WHISPER_AVAILABLE,
        "gpu_info": gpu_info if redaction_service.device == "cuda" else None
    })

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio chunk for PII redaction"""
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        audio_base64 = data.get('audio_data')
        sample_rate = data.get('sample_rate', 16000)
        
        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        logger.info(f"📥 Received audio: {len(audio_data)} samples at {sample_rate}Hz")
        
        # Process audio
        redacted_audio, metadata = redaction_service.process_audio_chunk(audio_data, sample_rate)
        
        # Encode result
        redacted_audio_int16 = (redacted_audio * 32767).astype(np.int16)
        redacted_base64 = base64.b64encode(redacted_audio_int16.tobytes()).decode('utf-8')
        
        # Return response
        response = {
            "success": metadata["success"],
            "redacted_audio_data": redacted_base64,
            "transcript": metadata["transcript"],
            "redacted_intervals": metadata["redacted_intervals"],
            "pii_count": metadata["pii_count"],
            "processing_time": metadata["processing_time"],
            "speed_ratio": metadata["speed_ratio"],
            "sample_rate": metadata["sample_rate"],
            "device_used": metadata["device_used"],
            "timestamp": datetime.now().isoformat()
        }
        
        if not metadata["success"]:
            response["error"] = metadata.get("error", "Unknown error")
        
        logger.info(f"📤 Returning: {len(redacted_base64)} bytes, {metadata['pii_count']} redactions")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Unexpected error in process_audio: {e}")
        return jsonify({
            "success": False,
            "error": f"Unexpected server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Faster-Whisper + BERT Audio Redaction Server on port 5002...")
    if FASTER_WHISPER_AVAILABLE:
        logger.info("✅ Faster-Whisper is available and ready!")
    else:
        logger.error("❌ Faster-Whisper is not available. Install with: pip install faster-whisper")
    app.run(host='0.0.0.0', port=5002, debug=True)