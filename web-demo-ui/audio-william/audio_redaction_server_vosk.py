#!/usr/bin/env python3
"""
Audio Redaction Server with Vosk + BERT
Fast ASR with high-accuracy PII detection
"""

import os
import logging
import json
import numpy as np
import librosa
import base64
import time
import wave
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import vosk
import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)

class VoskBertRedactionService:
    def __init__(self):
        logger.info("Initializing Vosk + BERT Audio Redaction Service...")
        
        # Set up device (prefer CPU for Vosk, GPU for BERT if available)
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
        
        logger.info("Vosk + BERT Audio Redaction Service initialized")
    
    def save_audio_debug(self, audio_data, sample_rate, prefix="debug"):
        """Save audio data as WAV file for debugging"""
        try:
            # Create debug directory if it doesn't exist
            debug_dir = "debug_audio"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{debug_dir}/{prefix}_{timestamp}.wav"
            
            # Convert float audio to int16
            if audio_data.dtype == np.float32:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"🔊 DEBUG: Saved audio to {filename} ({len(audio_data)} samples)")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save debug audio: {e}")
            return None
    
    def load_models(self):
        """Load Vosk ASR and BERT NER models"""
        # Load Vosk model
        try:
            model_path = "vosk-models/vosk-model-small-en-us-0.15"
            if not os.path.exists(model_path):
                raise Exception(f"Vosk model not found at {model_path}")
                
            logger.info("Loading Vosk ASR model...")
            vosk.SetLogLevel(-1)  # Suppress Vosk logs
            self.vosk_model = vosk.Model(model_path)
            logger.info("✅ Vosk ASR model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Vosk model: {e}")
            self.vosk_model = None
        
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
    
    def transcribe_with_vosk(self, audio_data, sample_rate=16000):
        """Fast transcription using Vosk"""
        if self.vosk_model is None:
            logger.warning("Vosk model not available")
            return "", []
            
        try:
            # Ensure audio is in the right format for Vosk (16kHz, mono, int16)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
                
            # Resample if necessary
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Convert to int16 format expected by Vosk
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create Vosk recognizer
            rec = vosk.KaldiRecognizer(self.vosk_model, sample_rate)
            rec.SetWords(True)  # Enable word-level timestamps
            
            # Process audio
            audio_bytes = audio_data.tobytes()
            
            # Feed audio to recognizer
            if rec.AcceptWaveform(audio_bytes):
                result = json.loads(rec.Result())
            else:
                result = json.loads(rec.FinalResult())
            
            # Extract transcript and word timings
            transcript = result.get('text', '')
            words = result.get('result', [])
            
            # Convert Vosk word format to our expected format
            word_times = []
            for word_info in words:
                word_times.append({
                    'word': word_info.get('word', ''),
                    'start': word_info.get('start', 0),
                    'end': word_info.get('end', 0)
                })
            
            logger.info(f"📝 Vosk transcript: '{transcript}'")
            logger.info(f"📊 Word count: {len(word_times)} words")
            
            return transcript, word_times
            
        except Exception as e:
            logger.error(f"Error in Vosk transcription: {e}")
            return "", []
    
    def detect_sensitive_spans_bert(self, transcript, word_times):
        """Detect PII using BERT NER + keywords"""
        redact_intervals = []
        
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
        """Main processing: Vosk ASR -> BERT NER -> Muting"""
        try:
            total_start = time.time()
            logger.info(f"Processing {len(audio_data)} samples at {sample_rate}Hz with Vosk + BERT")
            
            # DEBUG: Save incoming audio as WAV file
            self.save_audio_debug(audio_data, sample_rate, "incoming")
            
            # Step 1: Fast transcription with Vosk
            vosk_start = time.time()
            transcript, word_times = self.transcribe_with_vosk(audio_data, sample_rate)
            vosk_time = time.time() - vosk_start
            
            # Step 2: PII detection with BERT NER + keywords
            bert_start = time.time()
            intervals = self.detect_sensitive_spans_bert(transcript, word_times)
            bert_time = time.time() - bert_start
            
            # Step 3: Mute sensitive intervals
            mute_start = time.time()
            redacted_audio = self.mute_intervals(audio_data, sample_rate, intervals)
            mute_time = time.time() - mute_start
            
            # DEBUG: Save redacted audio as WAV file
            if len(intervals) > 0:
                self.save_audio_debug(redacted_audio, sample_rate, "redacted")
            else:
                self.save_audio_debug(redacted_audio, sample_rate, "clean")
            
            total_time = time.time() - total_start
            audio_duration = len(audio_data) / sample_rate
            speed_ratio = total_time / audio_duration
            
            # Detailed timing breakdown
            logger.info(f"⏱️  TIMING BREAKDOWN:")
            logger.info(f"   🎤 Vosk ASR:      {vosk_time:.3f}s ({vosk_time/audio_duration:.2f}x real-time)")
            logger.info(f"   🧠 BERT NER:      {bert_time:.3f}s ({bert_time/audio_duration:.2f}x real-time)")
            logger.info(f"   🔇 Audio Muting:  {mute_time:.3f}s ({mute_time/audio_duration:.2f}x real-time)")
            logger.info(f"   📊 TOTAL:         {total_time:.3f}s ({speed_ratio:.2f}x real-time)")
            logger.info(f"✅ Processing complete: {len(intervals)} redactions")
            
            return redacted_audio, {
                "success": True,
                "transcript": transcript,
                "redacted_intervals": intervals,
                "pii_count": len(intervals),
                "processing_time": total_time,
                "speed_ratio": speed_ratio,
                "timing_breakdown": {
                    "vosk_time": vosk_time,
                    "bert_time": bert_time,
                    "mute_time": mute_time,
                    "vosk_ratio": vosk_time / audio_duration,
                    "bert_ratio": bert_time / audio_duration,
                    "mute_ratio": mute_time / audio_duration
                },
                "sample_rate": sample_rate,
                "device_used": f"Vosk+BERT({self.device})"
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
                "device_used": f"Vosk+BERT({self.device})"
            }

# Initialize service
logger.info("Starting Vosk + BERT Audio Redaction Service...")
redaction_service = VoskBertRedactionService()

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
        "service": "Vosk + BERT Audio Redaction API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "asr": "Vosk (CPU)",
            "ner": f"BERT ({redaction_service.device.upper()})"
        },
        "models_loaded": {
            "vosk": redaction_service.vosk_model is not None,
            "bert": redaction_service.ner_pipeline is not None
        },
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
    logger.info("🚀 Starting Vosk + BERT Audio Redaction Server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)