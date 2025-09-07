#!/usr/bin/env python3
"""
Audio Redaction Flask API Server
Receives audio chunks from Mediasoup, processes them for PII redaction, 
and returns muted/redacted audio chunks back to the client.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import soundfile as sf
import numpy as np
import librosa
from transformers import pipeline
import io
import tempfile
import os
import logging
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for now

class AudioRedactionService:
    def __init__(self):
        """Initialize the audio redaction service with ML models"""
        logger.info("Initializing Audio Redaction Service...")
        
        # Check for CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Using device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"🎮 GPU Details: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("⚠️ CUDA not available, falling back to CPU")
        
        # Load NER pipeline for PII detection with GPU support
        try:
            pipeline_kwargs = {
                "task": "ner",
                "model": "dslim/bert-base-NER", 
                "aggregation_strategy": "simple",
                "device": 0 if self.device == "cuda" else -1  # 0 for GPU, -1 for CPU
            }
            
            # Only add dtype for GPU to avoid the deprecated warning
            if self.device == "cuda":
                pipeline_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            
            self.ner_pipeline = pipeline(**pipeline_kwargs)
            logger.info(f"✅ NER pipeline loaded successfully on {self.device.upper()}")
        except Exception as e:
            logger.error(f"❌ Failed to load NER pipeline: {e}")
            self.ner_pipeline = None
        
        # Load ASR pipeline for speech-to-text with GPU support
        try:
            pipeline_kwargs = {
                "task": "automatic-speech-recognition",
                "model": "openai/whisper-small",
                "device": 0 if self.device == "cuda" else -1  # 0 for GPU, -1 for CPU
            }
            
            # Only add dtype for GPU to avoid the deprecated warning
            if self.device == "cuda":
                pipeline_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            
            self.asr_pipeline = pipeline(**pipeline_kwargs)
            logger.info(f"✅ ASR pipeline loaded successfully on {self.device.upper()}")
        except Exception as e:
            logger.error(f"❌ Failed to load ASR pipeline: {e}")
            self.asr_pipeline = None
            
        # Sensitive keywords fallback
        self.sensitive_keywords = ["password", "secret", "pin", "ssn", "social", "credit", "card"]
        
        logger.info("🎤 Audio Redaction Service initialized")
    
    def cleanup_gpu_memory(self):
        """Aggressive GPU memory cleanup"""
        if self.device == "cuda":
            # Clear all cached memory
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
            # Clear model internal caches if available
            try:
                if hasattr(self.asr_pipeline, 'model'):
                    if hasattr(self.asr_pipeline.model, 'clear_cache'):
                        self.asr_pipeline.model.clear_cache()
                    # Clear any cached past_key_values or attention caches
                    if hasattr(self.asr_pipeline.model, 'past_key_values'):
                        self.asr_pipeline.model.past_key_values = None
                        
                if hasattr(self.ner_pipeline, 'model'):
                    if hasattr(self.ner_pipeline.model, 'clear_cache'):
                        self.ner_pipeline.model.clear_cache()
            except:
                pass  # Ignore if models don't have these attributes
            
            # Force garbage collection multiple times
            import gc
            gc.collect()
            gc.collect()
            
            # Final cache clear
            torch.cuda.empty_cache()
    
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e6  # MB
            reserved = torch.cuda.memory_reserved() / 1e6    # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1e6  # MB
            
            # Fix calculation - use allocated memory for actual usage
            # Reserved can exceed total due to PyTorch caching
            actual_used = min(allocated, total)  # Can't use more than total
            free = max(0, total - actual_used)   # Can't have negative free
            usage_percent = min(100, (actual_used / total) * 100)  # Cap at 100%
            
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved, 
                "free_mb": free,
                "total_mb": total,
                "usage_percent": usage_percent
            }
        return None

    def transcribe_audio_chunk(self, audio_data, sample_rate=16000):
        """
        Transcribe audio chunk and return transcript with word-level timestamps
        """
        if self.asr_pipeline is None:
            logger.warning("ASR pipeline not available")
            return "", [], audio_data, sample_rate
            
        try:
            # Ensure audio is in the right format for Whisper (16kHz)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
                
            # Resample if necessary
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Clear GPU cache before ASR
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Run ASR with aggressive memory management
            with torch.no_grad():  # Disable gradient computation
                # Clear cache before inference
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Use generate method for better memory efficiency with longer audio
                if len(audio_data) > 480000:  # > 30 seconds at 16kHz
                    # For longer audio, use the model's native chunking
                    result = self.asr_pipeline(
                        audio_data, 
                        return_timestamps="word",
                        ignore_warning=True,  # Suppress the experimental warning
                        batch_size=1  # Force single batch to reduce memory
                    )
                else:
                    # For shorter audio, use chunk_length_s
                    result = self.asr_pipeline(
                        audio_data, 
                        chunk_length_s=30, 
                        return_timestamps="word",
                        ignore_warning=True  # Suppress the experimental warning
                    )
            
            transcript = result["text"]
            words = result.get("chunks", [])
            
            # Clear result from memory immediately
            del result
            
            word_times = []
            current_pos = 0
            
            for w in words:
                text = w.get("text", "").strip()
                timestamp = w.get("timestamp", [0, 0])
                start_time = timestamp[0] if len(timestamp) > 0 else 0
                end_time = timestamp[1] if len(timestamp) > 1 else start_time + 0.1
                
                for token in text.split():
                    char_start = transcript.find(token, current_pos)
                    char_end = char_start + len(token)
                    current_pos = char_end
                    
                    word_times.append({
                        "word": token,
                        "start": start_time,
                        "end": end_time,
                        "char_start": char_start,
                        "char_end": char_end
                    })
            
            logger.info(f"📝 TRANSCRIPT: '{transcript}'")
            logger.info(f"📊 Word count: {len(word_times)} words")
            
            # Clear intermediate variables
            del words
            
            # Aggressive cleanup after ASR
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                
            # Clear ASR result references to prevent memory retention
            if 'result' in locals():
                del result
            import gc
            gc.collect()
            
            # Debug: Print word timing details
            if word_times and len(word_times) <= 20:  # Only for short transcripts
                logger.info("🕐 Word timings:")
                for i, word_info in enumerate(word_times):
                    logger.info(f"   {i+1}. '{word_info['word']}' ({word_info['start']:.2f}s - {word_info['end']:.2f}s)")
            
            return transcript, word_times, audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            # Clear GPU cache on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return "", [], audio_data, sample_rate

    def detect_sensitive_spans(self, transcript, word_times):
        """
        Detect sensitive information spans using NER and keyword matching
        """
        redact_intervals = []
        
        try:
            logger.info("🔍 Starting PII detection...")
            
            # Keyword matching
            keyword_detections = []
            for idx, w in enumerate(word_times):
                if w["word"].lower() in self.sensitive_keywords:
                    start = w["start"]
                    # Redact 5 words ahead for context
                    end_idx = min(idx + 5, len(word_times) - 1)
                    end = word_times[end_idx]["end"]
                    redact_intervals.append((start, end))
                    keyword_detections.append(w['word'])
                    logger.info(f"🚨 KEYWORD DETECTED: '{w['word']}' at {start:.2f}s")
            
            if keyword_detections:
                logger.info(f"📝 Total keyword detections: {keyword_detections}")
            else:
                logger.info("✅ No sensitive keywords found")
            
            # NER-based detection
            ner_detections = []
            if self.ner_pipeline and transcript:
                logger.info("🔍 Running NER analysis...")
                
                # Clear GPU cache before NER
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Run NER with memory management
                with torch.no_grad():  # Disable gradient computation
                    ner_results = self.ner_pipeline(transcript)
                    
                # Aggressive cleanup after NER
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                logger.info(f"🔍 NER found {len(ner_results)} entities")
                
                for entity in ner_results:
                    entity_text = transcript[entity["start"]:entity["end"]]
                    logger.info(f"   Entity: '{entity_text}' -> {entity['entity_group']} (confidence: {entity.get('score', 0):.3f})")
                    
                    # Consider PER (person), LOC (location), MISC as potentially sensitive
                    if entity["entity_group"] in ["PER", "LOC", "MISC"]:
                        start_char = entity["start"]
                        end_char = entity["end"]
                        
                        # Find overlapping words
                        overlapping_words = [
                            w for w in word_times
                            if not (w["char_end"] <= start_char or w["char_start"] >= end_char)
                        ]
                        
                        if overlapping_words:
                            start_sec = min(w["start"] for w in overlapping_words)
                            end_sec = max(w["end"] for w in overlapping_words)
                            redact_intervals.append((start_sec, end_sec))
                            ner_detections.append(entity_text)
                            logger.info(f"🚨 NER DETECTION: '{entity_text}' ({entity['entity_group']}) at {start_sec:.2f}s-{end_sec:.2f}s")
                
                # Clear NER results and GPU cache after NER
                del ner_results
                import gc
                gc.collect()
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                if ner_detections:
                    logger.info(f"📝 Total NER detections: {ner_detections}")
                else:
                    logger.info("✅ No sensitive NER entities found")
            else:
                logger.info("⏭️ Skipping NER analysis (pipeline not available)")
            
            # Merge overlapping intervals
            original_count = len(redact_intervals)
            if redact_intervals:
                logger.info(f"🔗 Merging {original_count} overlapping intervals...")
                redact_intervals = sorted(redact_intervals, key=lambda x: x[0])
                merged = [redact_intervals[0]]
                
                for start, end in redact_intervals[1:]:
                    last_start, last_end = merged[-1]
                    if start <= last_end:
                        merged[-1] = (last_start, max(last_end, end))
                        logger.info(f"   Merged interval: {last_start:.2f}s-{max(last_end, end):.2f}s")
                    else:
                        merged.append((start, end))
                        
                redact_intervals = merged
                logger.info(f"🔗 Merged to {len(redact_intervals)} final intervals")
            
            # Final summary
            logger.info(f"🎯 PII DETECTION SUMMARY:")
            logger.info(f"   Keywords found: {len(keyword_detections)} - {keyword_detections}")
            logger.info(f"   NER entities found: {len(ner_detections)} - {ner_detections}")
            logger.info(f"   Total intervals to redact: {len(redact_intervals)}")
            
            for i, (start, end) in enumerate(redact_intervals):
                duration = end - start
                logger.info(f"   🔇 Interval {i+1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
            
            return redact_intervals
            
        except Exception as e:
            logger.error(f"Error in PII detection: {e}")
            return []

    def mute_intervals(self, audio_data, sample_rate, intervals):
        """
        Mute audio at specified time intervals
        """
        try:
            if not intervals:
                logger.info("🔊 No intervals to mute - returning original audio")
                return audio_data.copy()
            
            logger.info(f"🔇 Muting {len(intervals)} intervals in audio...")
            audio_copy = audio_data.copy()
            total_muted_samples = 0
            
            for i, (start, end) in enumerate(intervals):
                start_idx = int(start * sample_rate)
                end_idx = int(end * sample_rate)
                
                # Ensure indices are within bounds
                original_start_idx = start_idx
                original_end_idx = end_idx
                start_idx = max(0, min(start_idx, len(audio_copy)))
                end_idx = max(start_idx, min(end_idx, len(audio_copy)))
                
                # Calculate samples to mute
                samples_to_mute = end_idx - start_idx
                total_muted_samples += samples_to_mute
                
                # Mute the audio (set to zero)
                audio_copy[start_idx:end_idx] = 0.0
                
                duration = end - start
                logger.info(f"   🔇 Muted interval {i+1}: {start:.2f}s-{end:.2f}s (samples {start_idx}-{end_idx}, {samples_to_mute} samples, {duration:.2f}s)")
                
                if original_start_idx != start_idx or original_end_idx != end_idx:
                    logger.warning(f"   ⚠️ Bounds adjusted from {original_start_idx}-{original_end_idx} to {start_idx}-{end_idx}")
            
            total_audio_samples = len(audio_copy)
            muted_percentage = (total_muted_samples / total_audio_samples) * 100
            muted_duration = total_muted_samples / sample_rate
            
            logger.info(f"🔇 MUTING SUMMARY:")
            logger.info(f"   Total audio samples: {total_audio_samples}")
            logger.info(f"   Muted samples: {total_muted_samples} ({muted_percentage:.1f}%)")
            logger.info(f"   Muted duration: {muted_duration:.2f}s")
            
            return audio_copy
            
        except Exception as e:
            logger.error(f"❌ Error in audio muting: {e}")
            return audio_data

    def process_audio_chunk(self, audio_data, sample_rate=16000):
        """
        Main processing function: transcribe -> detect PII -> mute -> return
        """
        try:
            logger.info(f"Processing audio chunk of {len(audio_data)} samples at {sample_rate}Hz on {self.device.upper()}")
            
            # Log warning for unexpectedly large chunks (should be handled by client now)
            if len(audio_data) > 48000:  # 3 seconds at 16kHz
                logger.warning(f"⚠️ Unexpectedly large audio chunk: {len(audio_data)} samples = {len(audio_data)/sample_rate:.1f}s")
            
            # Aggressive memory cleanup before processing
            self.cleanup_gpu_memory()
            
            # Monitor GPU memory
            if self.device == "cuda":
                memory_info = self.get_gpu_memory_info()
                logger.info(f"🎮 GPU Memory before: {memory_info['allocated_mb']:.1f}MB allocated, {memory_info['free_mb']:.1f}MB free ({memory_info['usage_percent']:.1f}% used)")
                
                # Safety check - refuse processing if memory is critically high
                if memory_info['usage_percent'] > 95:
                    logger.error(f"🚨 CRITICAL: GPU memory usage at {memory_info['usage_percent']:.1f}% - refusing request to prevent OOM")
                    self.cleanup_gpu_memory()
                    return audio_data, {
                        "success": False,
                        "error": "GPU memory critically high - request refused to prevent crash",
                        "memory_usage_percent": memory_info['usage_percent'],
                        "transcript": "",
                        "redacted_intervals": [],
                        "pii_count": 0,
                        "sample_rate": sample_rate,
                        "processing_complete": False,
                        "device_used": self.device
                    }
                
                # Warning if memory is getting high
                if memory_info['usage_percent'] > 80:
                    logger.warning(f"⚠️ GPU memory usage high: {memory_info['usage_percent']:.1f}%")
                    self.cleanup_gpu_memory()  # Emergency cleanup
            
            # Step 1: Transcribe audio
            transcript, word_times, processed_audio, sr = self.transcribe_audio_chunk(audio_data, sample_rate)
            
            # Step 2: Detect sensitive spans
            intervals = self.detect_sensitive_spans(transcript, word_times)
            
            # Step 3: Mute sensitive intervals
            redacted_audio = self.mute_intervals(processed_audio, sr, intervals)
            
            # Aggressive cleanup after processing
            self.cleanup_gpu_memory()
            
            # Monitor final memory state
            if self.device == "cuda":
                memory_info = self.get_gpu_memory_info()
                logger.info(f"🎮 GPU Memory after: {memory_info['allocated_mb']:.1f}MB allocated, {memory_info['free_mb']:.1f}MB free ({memory_info['usage_percent']:.1f}% used)")
                
                # Force additional cleanup if still high
                if memory_info['usage_percent'] > 70:
                    logger.warning("🧹 Forcing additional memory cleanup...")
                    self.cleanup_gpu_memory()
                    # Final check
                    final_info = self.get_gpu_memory_info()
                    logger.info(f"🎮 GPU Memory final: {final_info['allocated_mb']:.1f}MB allocated ({final_info['usage_percent']:.1f}% used)")
            
            result = {
                "success": True,
                "transcript": transcript,
                "redacted_intervals": intervals,
                "pii_count": len(intervals),
                "sample_rate": sr,
                "processing_complete": True,
                "device_used": self.device
            }
            
            logger.info(f"✅ Processing complete on {self.device.upper()}: {len(intervals)} intervals redacted")
            return redacted_audio, result
            
        except Exception as e:
            logger.error(f"❌ Error processing audio chunk on {self.device.upper()}: {e}")
            # Clear GPU cache on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return audio_data, {
                "success": False,
                "error": str(e),
                "transcript": "",
                "redacted_intervals": [],
                "pii_count": 0,
                "sample_rate": sample_rate,
                "processing_complete": False,
                "device_used": self.device
            }

# Initialize the service
redaction_service = AudioRedactionService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if redaction_service.device == "cuda":
        memory_info = redaction_service.get_gpu_memory_info()
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": f"{memory_info['total_mb'] / 1000:.1f} GB",
            "memory_allocated": memory_info['allocated_mb'],
            "memory_reserved": memory_info['reserved_mb']
        }
    
    return jsonify({
        "status": "healthy",
        "service": "Audio Redaction API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "device": redaction_service.device.upper(),
        "gpu_info": gpu_info if redaction_service.device == "cuda" else None,
        "models_loaded": {
            "ner": redaction_service.ner_pipeline is not None,
            "asr": redaction_service.asr_pipeline is not None
        }
    })

@app.route('/cleanup', methods=['POST'])
def cleanup_gpu():
    """Manual GPU memory cleanup endpoint"""
    try:
        if redaction_service.device == "cuda":
            memory_before = redaction_service.get_gpu_memory_info()
            redaction_service.cleanup_gpu_memory()
            memory_after = redaction_service.get_gpu_memory_info()
            
            return jsonify({
                "status": "cleaned",
                "device": "CUDA",
                "memory_before": memory_before,
                "memory_after": memory_after,
                "freed_mb": memory_before['reserved_mb'] - memory_after['reserved_mb'],
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "skipped",
                "device": "CPU", 
                "message": "GPU cleanup not needed on CPU",
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process audio chunk for PII redaction
    Expects: JSON with audio data (base64 encoded) and metadata
    Returns: Redacted audio data (base64 encoded) and processing info
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract audio data
        if 'audio_data' not in data:
            return jsonify({"error": "No audio_data field provided"}), 400
        
        # Decode base64 audio data
        try:
            audio_b64 = data['audio_data']
            audio_bytes = base64.b64decode(audio_b64)
            
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
        except Exception as e:
            return jsonify({"error": f"Failed to decode audio data: {str(e)}"}), 400
        
        # Get sample rate (default 16000)
        sample_rate = data.get('sample_rate', 16000)
        
        logger.info(f"📥 Received audio chunk: {len(audio_array)} samples at {sample_rate}Hz")
        
        # Process the audio
        redacted_audio, processing_result = redaction_service.process_audio_chunk(
            audio_array, sample_rate
        )
        
        # Convert redacted audio back to bytes and base64 encode
        try:
            # Convert back to 16-bit PCM
            redacted_int16 = (redacted_audio * 32767.0).astype(np.int16)
            redacted_bytes = redacted_int16.tobytes()
            redacted_b64 = base64.b64encode(redacted_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to encode redacted audio: {e}")
            return jsonify({"error": f"Failed to encode redacted audio: {str(e)}"}), 500
        
        # Prepare response
        response = {
            "success": processing_result["success"],
            "redacted_audio_data": redacted_b64,
            "sample_rate": processing_result["sample_rate"],
            "transcript": processing_result["transcript"],
            "pii_count": processing_result["pii_count"],
            "redacted_intervals": processing_result["redacted_intervals"],
            "processing_time": 0,  # TODO: Add timing
            "timestamp": datetime.now().isoformat()
        }
        
        if not processing_result["success"]:
            response["error"] = processing_result.get("error", "Unknown error")
        
        logger.info(f"📤 Returning redacted audio: {len(redacted_bytes)} bytes, {processing_result['pii_count']} redactions")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in process_audio: {e}")
        return jsonify({
            "success": False,
            "error": f"Unexpected server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/process_audio_raw', methods=['POST'])
def process_audio_raw():
    """
    Process raw audio data (binary)
    Expects: Raw audio data in request body
    Headers: Content-Type: application/octet-stream, X-Sample-Rate: 16000
    Returns: Raw redacted audio data
    """
    try:
        # Get raw audio data
        audio_bytes = request.get_data()
        if not audio_bytes:
            return "No audio data provided", 400
        
        # Get sample rate from headers
        sample_rate = int(request.headers.get('X-Sample-Rate', 16000))
        
        # Convert bytes to numpy array
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        except Exception as e:
            return f"Failed to decode audio: {str(e)}", 400
        
        logger.info(f"📥 Received raw audio: {len(audio_array)} samples at {sample_rate}Hz")
        
        # Process the audio
        redacted_audio, processing_result = redaction_service.process_audio_chunk(
            audio_array, sample_rate
        )
        
        # Convert back to bytes
        redacted_int16 = (redacted_audio * 32767.0).astype(np.int16)
        redacted_bytes = redacted_int16.tobytes()
        
        # Add processing info to response headers
        response = app.response_class(
            response=redacted_bytes,
            mimetype='application/octet-stream'
        )
        response.headers['X-PII-Count'] = str(processing_result['pii_count'])
        response.headers['X-Processing-Success'] = str(processing_result['success'])
        response.headers['X-Sample-Rate'] = str(processing_result['sample_rate'])
        
        logger.info(f"📤 Returning raw redacted audio: {len(redacted_bytes)} bytes")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in process_audio_raw: {e}")
        return f"Server error: {str(e)}", 500

if __name__ == '__main__':
    logger.info("🚀 Starting Audio Redaction API Server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)