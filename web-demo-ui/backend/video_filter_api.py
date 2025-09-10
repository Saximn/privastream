"""
Fast HTTP API for real-time video filtering using UnifiedBlurDetector.
Processes single frames and returns blurred frame for WebRTC integration.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import json
import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path

# Add video_models to path
sys.path.append(str(Path(__file__).parent.parent / "video_models"))
from video_models.unified_detector import UnifiedBlurDetector

# InsightFace imports for face enrollment
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Face enrollment disabled.")

app = Flask(__name__)
CORS(app, origins="*")

# PERFORMANCE CONFIGURATION - Easy to adjust
DETECTION_FPS = 30.0  # FPS for face/privacy detection (lower = less latency, higher = more CPU load)
# Conversion: 30fps input -> stride calculation
DETECTION_STRIDE = max(1, int(30 / DETECTION_FPS))  # Process every Nth frame

EXPECTED_DELAY_SEC = DETECTION_STRIDE / 30.0  # Delay in seconds based on 30fps input
print(f"[CONFIG] Detection FPS: {DETECTION_FPS}, Stride: {DETECTION_STRIDE} (process every {DETECTION_STRIDE} frames)")
print(f"[CONFIG] Expected detection delay: {EXPECTED_DELAY_SEC:.2f} seconds")

# Detector configuration
DETECTOR_CONFIG = {
    "enable_face": True,
    "enable_pii": False,
    "enable_plate": True,
    "pii": {
        "classifier_path": "video_models/pii_blur/pii_clf.joblib",
        "conf_thresh": 0.35
    }
}

# Debug configuration
DEBUG_CONFIG = {
    "enabled": False,  # Set to False to disable debug output
    "output_dir": "debug_images",
    "save_input": True,
    "save_output": True,
    "max_images": 100  # Limit to prevent disk space issues
}

# Request queue protection configuration
QUEUE_CONFIG = {
    "max_request_age_ms": 1000,  # Drop requests older than 1 second
    "max_concurrent_requests": 15,  # Limit concurrent processing to prevent GPU overload
    "enable_request_dropping": True,  # Enable/disable request age checking
    "queue_monitoring": True  # Enable queue monitoring logs
}

detector = None
face_app = None  # Global face detection instance for enrollment
room_embeddings = {}  # Store face embeddings per room: roomId -> {'embedding': np.array, 'metadata': {...}}
active_requests = 0  # Track concurrent processing requests
request_lock = threading.Lock()  # Thread lock for request counting

def is_request_stale(request_timestamp_ms):
    """Check if request is too old to process."""
    if not QUEUE_CONFIG["enable_request_dropping"]:
        return False
    
    current_time_ms = int(time.time() * 1000)
    request_age_ms = current_time_ms - request_timestamp_ms
    
    if QUEUE_CONFIG["queue_monitoring"]:
        print(f"[QUEUE] Request age: {request_age_ms}ms (max: {QUEUE_CONFIG['max_request_age_ms']}ms)")
    
    return request_age_ms > QUEUE_CONFIG["max_request_age_ms"]

def can_process_request():
    """Check if we can process another request (not at concurrent limit)."""
    with request_lock:
        can_process = active_requests < QUEUE_CONFIG["max_concurrent_requests"]
        if QUEUE_CONFIG["queue_monitoring"]:
            print(f"[QUEUE] Active requests: {active_requests}/{QUEUE_CONFIG['max_concurrent_requests']}, Can process: {can_process}")
        return can_process

def start_request_processing():
    """Mark start of request processing."""
    global active_requests
    with request_lock:
        active_requests += 1
        if QUEUE_CONFIG["queue_monitoring"]:
            print(f"[QUEUE] Started processing request, active: {active_requests}")

def finish_request_processing():
    """Mark end of request processing."""
    global active_requests
    with request_lock:
        active_requests = max(0, active_requests - 1)  # Prevent negative counts
        if QUEUE_CONFIG["queue_monitoring"]:
            print(f"[QUEUE] Finished processing request, active: {active_requests}")

def setup_debug_directories():
    """Create debug output directories if they don't exist."""
    if not DEBUG_CONFIG["enabled"]:
        return
    
    try:
        debug_dir = Path(DEBUG_CONFIG["output_dir"])
        debug_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for input and output images
        (debug_dir / "input").mkdir(exist_ok=True)
        (debug_dir / "output").mkdir(exist_ok=True)
        (debug_dir / "comparison").mkdir(exist_ok=True)
        
        print(f"[DEBUG] Debug directories created at: {debug_dir.absolute()}")
    except Exception as e:
        print(f"[DEBUG] Failed to create debug directories: {e}")

def save_debug_image(image, image_type, frame_id, rectangles=None):
    """Save debug images for input/output comparison."""
    if not DEBUG_CONFIG["enabled"]:
        return
    
    try:
        debug_dir = Path(DEBUG_CONFIG["output_dir"])
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Generate filename
        filename = f"frame_{frame_id:06d}_{timestamp}_{image_type}.jpg"
        
        if image_type == "input" and DEBUG_CONFIG["save_input"]:
            filepath = debug_dir / "input" / filename
        elif image_type == "output" and DEBUG_CONFIG["save_output"]:
            filepath = debug_dir / "output" / filename
        else:
            return
        
        # Save the image
        cv2.imwrite(str(filepath), image)
        
        # Create comparison image if we have rectangles
        if rectangles and len(rectangles) > 0 and image_type == "input":
            comparison_image = image.copy()
            
            # Draw bounding boxes on the comparison image
            for rect in rectangles:
                if len(rect) == 4:
                    x1, y1, x2, y2 = map(int, rect)
                    cv2.rectangle(comparison_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(comparison_image, "PII", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            comparison_filepath = debug_dir / "comparison" / filename.replace("_input", "_boxes")
            cv2.imwrite(str(comparison_filepath), comparison_image)
        
        # Clean up old images if we exceed max_images
        cleanup_old_debug_images(debug_dir / image_type)
        
        print(f"[DEBUG] Saved {image_type} image: {filepath.name}")
        
    except Exception as e:
        print(f"[DEBUG] Failed to save {image_type} image: {e}")

def cleanup_old_debug_images(directory):
    """Remove old debug images if we exceed the maximum count."""
    try:
        if not directory.exists():
            return
            
        image_files = list(directory.glob("*.jpg"))
        if len(image_files) > DEBUG_CONFIG["max_images"]:
            # Sort by modification time and remove oldest
            image_files.sort(key=lambda x: x.stat().st_mtime)
            files_to_remove = image_files[:-DEBUG_CONFIG["max_images"]]
            
            for file_path in files_to_remove:
                file_path.unlink()
                
            print(f"[DEBUG] Cleaned up {len(files_to_remove)} old images from {directory.name}")
            
    except Exception as e:
        print(f"[DEBUG] Failed to cleanup old images: {e}")

def init_detector():
    """Initialize the detector once."""
    global detector, face_app
    if detector is None:
        try:
            detector = UnifiedBlurDetector(DETECTOR_CONFIG)
            print("[API] Video filter detector initialized")
            
            # Set up debug directories
            setup_debug_directories()
            
        except Exception as e:
            print(f"[API] Failed to initialize detector: {e}")
            detector = "failed"
    
    # Initialize face detection for enrollment
    if face_app is None and INSIGHTFACE_AVAILABLE:
        try:
            print("[API] Initializing InsightFace Buffalo_S for enrollment...")
            face_app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("[API] ✅ InsightFace Buffalo_S initialized for enrollment")
        except Exception as e:
            print(f"[API] ❌ Failed to initialize InsightFace: {e}")
            face_app = "failed"

def filter_frame(frame, frame_id=0, blur_only=False, provided_rectangles=None, room_id=None):
    """
    Process a single frame with two modes:
    1. Full detection + blur (blur_only=False)
    2. CPU blur only using provided rectangles (blur_only=True)
    """
    global detector, room_embeddings
    rectangles = []
    
    # Save debug input image (copy before any modifications)
    input_frame_copy = frame.copy()
    
    if blur_only and provided_rectangles:
        # CPU-only blur mode: use provided rectangles, no GPU detection
        print(f"[API] CPU blur mode: applying blur to {len(provided_rectangles)} regions")
        rectangles = provided_rectangles
        
        # Save input image with provided rectangles for comparison
        save_debug_image(input_frame_copy, "input", frame_id, rectangles)
    else:
        # Full detection mode: run GPU detection
        if detector is None:
            init_detector()
        if detector == "failed":
            raise RuntimeError("Detector initialization failed")
        
        # Update face detector embedding if room has enrolled face
        print(f"[API] DEBUG: Checking room_id='{room_id}', available rooms: {list(room_embeddings.keys())}")
        if room_id and room_id in room_embeddings:
            print(f"[API] ✅ Updating face embedding for room {room_id}")
            embedding = room_embeddings[room_id]['embedding']  # Extract just the numpy array
            detector.update_face_embedding(embedding)
        else:
            print(f"[API] ⚠️  No embedding found for room {room_id} (available: {list(room_embeddings.keys())})")
        
        print(f"[API] Full detection mode: processing frame {frame_id}")
        results = detector.process_frame(frame, frame_id, stride=DETECTION_STRIDE)
        
        # Extract rectangles from detection results
        for model_name in ["face", "plate", "pii"]:
            model_data = results.get("models", {}).get(model_name, {})
            if "rectangles" in model_data:
                rectangles.extend(model_data["rectangles"])
        
        print(f"[API] Detected {len(rectangles)} regions")
        
        # Save input image with detection boxes for comparison
        save_debug_image(input_frame_copy, "input", frame_id, rectangles)
    
    # Apply Gaussian blur to all rectangles (CPU processing)
    blur_applied = 0
    for rect in rectangles:
        try:
            if len(rect) == 4:
                x1, y1, x2, y2 = rect
                # Convert to x, y, w, h format if needed
                if x2 < x1: x1, x2 = x2, x1
                if y2 < y1: y1, y2 = y2, y1
                
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                
                # Bounds check
                if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0] and w > 0 and h > 0:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (0, 0), sigmaX=75, sigmaY=75)
                        blur_applied += 1
        except Exception as e:
            print(f"[API] Error blurring rectangle {rect}: {e}")
    
    print(f"[API] Applied blur to {blur_applied}/{len(rectangles)} regions")
    
    # Save debug output image (after blur processing)
    save_debug_image(frame, "output", frame_id, rectangles)
    
    # Encode frame as base64 JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    frame_b64 = f"data:image/jpeg;base64,{frame_b64}"
    
    return frame_b64, rectangles

@app.route('/health')
def health():
    return {'status': 'healthy', 'detector_ready': detector is not None and detector != "failed"}

@app.route('/process-frame', methods=['POST'])
def process_frame_route():
    """Flask route to process a single frame with detection or blur-only mode."""
    processing_started = False
    
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "Missing frame data"}), 400

        # Get request timestamp (from client or use current time)
        request_timestamp = data.get('timestamp', int(time.time() * 1000))
        frame_id = data.get('frame_id', 0)
        room_id = data.get('room_id', None)  # Get room_id for whitelist lookup
        
        if room_id:
            print(f"[API] Processing frame {frame_id} for room {room_id}")
        
        # 1. Check if request is too old
        if is_request_stale(request_timestamp):
            print(f"[QUEUE] 🗑️ DROPPING STALE REQUEST - Frame {frame_id} is too old to process")
            return jsonify({
                "success": False,
                "error": "Request too old",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "stale_request"
            }), 429  # Too Many Requests
        
        # 2. Check concurrent request limit
        if not can_process_request():
            print(f"[QUEUE] 🚫 DROPPING REQUEST - Too many concurrent requests, Frame {frame_id}")
            return jsonify({
                "success": False,
                "error": "Server overloaded",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "overloaded"
            }), 503  # Service Unavailable
        
        # 3. Mark request as started
        start_request_processing()
        processing_started = True
        
        try:
            frame_data = data['frame']
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]

            img_bytes = base64.b64decode(frame_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Invalid image data"}), 400

            blur_only = data.get('blur_only', False)
            provided_rectangles = data.get('rectangles', [])
            detect_only = data.get('detect_only', False)
            
            # Log processing mode
            if blur_only:
                print(f"[API] Processing frame {frame_id} in BLUR_ONLY mode with {len(provided_rectangles)} rectangles")
            elif detect_only:
                print(f"[API] Processing frame {frame_id} in DETECT_ONLY mode (full detection)")
            else:
                print(f"[API] Processing frame {frame_id} in FULL mode")
            
            blurred_frame_b64, rectangles = filter_frame(
                frame, 
                frame_id, 
                blur_only=blur_only, 
                provided_rectangles=provided_rectangles,
                room_id=room_id
            )

            return jsonify({
                "success": True,
                "frame_id": frame_id,
                "frame": blurred_frame_b64,
                "rectangles": rectangles,
                "processing_mode": "blur_only" if blur_only else ("detect_only" if detect_only else "full"),
                "regions_processed": len(rectangles)
            })
        
        finally:
            # Always mark request as finished if we started processing
            if processing_started:
                finish_request_processing()

    except Exception as e:
        # Make sure to finish request processing if we started it
        if processing_started:
            try:
                finish_request_processing()
            except:
                pass  # Ignore errors in cleanup
        
        print(f"[API] Frame processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detector-info')
def detector_info():
    global detector
    if detector is None:
        init_detector()
    if detector == "failed":
        return jsonify({"error": "Detector not available"}), 500
    try:
        return jsonify(detector.get_model_info())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug-status')
def debug_status():
    """Get debug configuration status."""
    try:
        debug_dir = Path(DEBUG_CONFIG["output_dir"])
        status = {
            "debug_enabled": DEBUG_CONFIG["enabled"],
            "output_directory": str(debug_dir.absolute()),
            "save_input": DEBUG_CONFIG["save_input"],
            "save_output": DEBUG_CONFIG["save_output"],
            "max_images": DEBUG_CONFIG["max_images"],
            "directories_exist": debug_dir.exists(),
            "image_counts": {}
        }
        
        # Count images in each subdirectory
        if debug_dir.exists():
            for subdir in ["input", "output", "comparison"]:
                subdir_path = debug_dir / subdir
                if subdir_path.exists():
                    status["image_counts"][subdir] = len(list(subdir_path.glob("*.jpg")))
                else:
                    status["image_counts"][subdir] = 0
        
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug-config', methods=['POST'])
def update_debug_config():
    """Update debug configuration."""
    global DEBUG_CONFIG
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400
        
        # Update configuration
        if "enabled" in data:
            DEBUG_CONFIG["enabled"] = bool(data["enabled"])
        if "save_input" in data:
            DEBUG_CONFIG["save_input"] = bool(data["save_input"])
        if "save_output" in data:
            DEBUG_CONFIG["save_output"] = bool(data["save_output"])
        if "max_images" in data:
            DEBUG_CONFIG["max_images"] = int(data["max_images"])
        
        # Recreate debug directories if enabled
        if DEBUG_CONFIG["enabled"]:
            setup_debug_directories()
        
        return jsonify({
            "success": True, 
            "message": "Debug configuration updated",
            "config": DEBUG_CONFIG
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug-cleanup', methods=['POST'])
def cleanup_debug_images_endpoint():
    """Clean up all debug images."""
    try:
        if not DEBUG_CONFIG["enabled"]:
            return jsonify({"error": "Debug mode not enabled"}), 400
        
        debug_dir = Path(DEBUG_CONFIG["output_dir"])
        if not debug_dir.exists():
            return jsonify({"message": "Debug directory doesn't exist"})
        
        total_removed = 0
        for subdir in ["input", "output", "comparison"]:
            subdir_path = debug_dir / subdir
            if subdir_path.exists():
                image_files = list(subdir_path.glob("*.jpg"))
                for file_path in image_files:
                    file_path.unlink()
                total_removed += len(image_files)
        
        return jsonify({
            "success": True,
            "message": f"Removed {total_removed} debug images"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/queue-status')
def queue_status():
    """Get current queue and processing status."""
    global active_requests
    with request_lock:
        return jsonify({
            "active_requests": active_requests,
            "max_concurrent": QUEUE_CONFIG["max_concurrent_requests"],
            "can_accept_new": active_requests < QUEUE_CONFIG["max_concurrent_requests"],
            "max_request_age_ms": QUEUE_CONFIG["max_request_age_ms"],
            "request_dropping_enabled": QUEUE_CONFIG["enable_request_dropping"],
            "queue_monitoring": QUEUE_CONFIG["queue_monitoring"],
            "current_time_ms": int(time.time() * 1000)
        })

@app.route('/queue-config', methods=['POST'])
def update_queue_config():
    """Update queue configuration."""
    global QUEUE_CONFIG
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400
        
        # Update configuration
        if "max_request_age_ms" in data:
            QUEUE_CONFIG["max_request_age_ms"] = int(data["max_request_age_ms"])
        if "max_concurrent_requests" in data:
            QUEUE_CONFIG["max_concurrent_requests"] = int(data["max_concurrent_requests"])
        if "enable_request_dropping" in data:
            QUEUE_CONFIG["enable_request_dropping"] = bool(data["enable_request_dropping"])
        if "queue_monitoring" in data:
            QUEUE_CONFIG["queue_monitoring"] = bool(data["queue_monitoring"])
        
        return jsonify({
            "success": True, 
            "message": "Queue configuration updated",
            "config": QUEUE_CONFIG
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Face Enrollment Endpoints
@app.route('/face-detection', methods=['POST'])
def face_detection():
    """Live face detection endpoint for enrollment"""
    global face_app
    
    try:
        # Initialize face app if not already done
        if face_app is None:
            init_detector()
        
        if face_app == "failed" or not INSIGHTFACE_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Face detection not available',
                'faces_detected': []
            }), 503
        
        data = request.get_json()
        if not data or 'frame_data' not in data:
            return jsonify({
                'success': False,
                'error': 'No frame data provided',
                'faces_detected': []
            }), 400
        
        frame_data = data['frame_data']
        room_id = data.get('room_id', 'unknown')
        
        print(f"[ENROLLMENT] DEBUG - Received request with room_id: '{room_id}'")
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to decode image',
                    'faces_detected': []
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Image decode error: {str(e)}',
                'faces_detected': []
            }), 400
        
        # Detect faces using InsightFace
        start_time = time.time()
        faces = face_app.get(image)
        detection_time = time.time() - start_time
        
        detected_faces = []
        if faces:
            # Select ONLY the largest face (the one to be enrolled)
            max_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            
            bbox = max_face.bbox.astype(int)  # [x1, y1, x2, y2]
            confidence = float(max_face.det_score)
            
            detected_faces.append({
                'bbox': bbox.tolist(),
                'confidence': confidence
            })
            
            print(f"[ENROLLMENT] Detected {len(faces)} faces, showing largest one in {detection_time:.3f}s for room {room_id}")
        else:
            print(f"[ENROLLMENT] No faces detected in {detection_time:.3f}s for room {room_id}")
        
        return jsonify({
            'success': True,
            'faces_detected': detected_faces,
            'detection_time': detection_time,
            'room_id': room_id
        })
        
    except Exception as e:
        print(f"[ENROLLMENT] Face detection error: {e}")
        return jsonify({
            'success': False,
            'error': f'Detection failed: {str(e)}',
            'faces_detected': []
        }), 500

@app.route('/face-enrollment', methods=['POST'])
def face_enrollment():
    """Face enrollment endpoint"""
    global face_app, room_embeddings
    
    try:
        # Initialize face app if not already done
        if face_app is None:
            init_detector()
        
        if face_app == "failed" or not INSIGHTFACE_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Face enrollment not available',
                'enrollment_complete': False
            }), 503
        
        data = request.get_json()
        if not data or 'frames' not in data or 'room_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing frames or room_id',
                'enrollment_complete': False
            }), 400
        
        frames = data['frames']
        room_id = data['room_id']
        
        if not isinstance(frames, list) or len(frames) == 0:
            return jsonify({
                'success': False,
                'error': 'No frames provided for enrollment',
                'enrollment_complete': False
            }), 400
        
        print(f"[ENROLLMENT] Starting face enrollment for room: {room_id}")
        print(f"[ENROLLMENT] Processing {len(frames)} frames")
        
        all_embeddings = []
        valid_frames = 0
        
        for i, frame_data in enumerate(frames):
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    print(f"[ENROLLMENT] Failed to decode frame {i}")
                    continue
                
                # Extract embeddings from this frame - SELECT MAX SIZE FACE ONLY
                faces = face_app.get(image)
                
                if faces:
                    # Find the largest face (max bounding box area)
                    max_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    
                    if hasattr(max_face, 'normed_embedding') and max_face.normed_embedding is not None:
                        embedding = max_face.normed_embedding.astype(float)
                        all_embeddings.append(embedding)
                        valid_frames += 1
                        
                        # Calculate face area for logging
                        face_area = (max_face.bbox[2]-max_face.bbox[0])*(max_face.bbox[3]-max_face.bbox[1])
                        print(f"[ENROLLMENT] Frame {i}: extracted largest face embedding (area: {face_area:.0f}px, {len(faces)} total faces)")
                    else:
                        print(f"[ENROLLMENT] Frame {i}: no valid embedding from largest face")
                else:
                    print(f"[ENROLLMENT] Frame {i}: no faces detected")
                    
            except Exception as e:
                print(f"[ENROLLMENT] Error processing frame {i}: {e}")
                continue
        
        if not all_embeddings:
            return jsonify({
                'success': False,
                'message': f'No face embeddings extracted from {len(frames)} frames',
                'enrollment_complete': False
            })
        
        # Compute average embedding
        if len(all_embeddings) > 1:
            embeddings_array = np.stack(all_embeddings)
            average_embedding = np.mean(embeddings_array, axis=0)
            # Normalize the average embedding
            average_embedding = average_embedding / np.linalg.norm(average_embedding)
        else:
            average_embedding = all_embeddings[0]
        
        # Store in room embeddings
        room_embeddings[room_id] = {
            'embedding': average_embedding,
            'metadata': {
                'enrollment_time': datetime.now().isoformat(),
                'frames_processed': len(frames),
                'valid_frames': valid_frames,
                'embeddings_count': len(all_embeddings)
            }
        }
        
        print(f"[ENROLLMENT] ✅ Face enrollment complete for room {room_id}")
        print(f"[ENROLLMENT]    Processed: {valid_frames}/{len(frames)} frames")
        print(f"[ENROLLMENT]    Embeddings: {len(all_embeddings)} -> 1 average")
        
        return jsonify({
            'success': True,
            'message': f'Face enrolled successfully from {valid_frames} frames',
            'enrollment_complete': True,
            'metadata': room_embeddings[room_id]['metadata']
        })
        
    except Exception as e:
        print(f"[ENROLLMENT] Face enrollment error: {e}")
        return jsonify({
            'success': False,
            'message': f'Enrollment failed: {str(e)}',
            'enrollment_complete': False
        }), 500

@app.route('/room-status/<room_id>', methods=['GET'])
def get_room_status(room_id: str):
    """Get enrollment status for a room"""
    try:
        if room_id in room_embeddings:
            metadata = room_embeddings[room_id]['metadata']
            return jsonify({
                'enrolled': True,
                'room_id': room_id,
                'metadata': metadata
            })
        else:
            return jsonify({
                'enrolled': False,
                'room_id': room_id
            })
            
    except Exception as e:
        return jsonify({
            'error': f'Status check failed: {str(e)}'
        }), 500

@app.route('/cleanup-room/<room_id>', methods=['DELETE'])
def cleanup_room(room_id: str):
    """Clean up enrollment data for a room"""
    try:
        if room_id in room_embeddings:
            del room_embeddings[room_id]
            print(f"[ENROLLMENT] Cleaned up enrollment data for room: {room_id}")
        
        return jsonify({
            'success': True,
            'message': f'Room {room_id} cleaned up'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Cleanup failed: {str(e)}'
        }), 500

@app.route('/detect-faces-mouths', methods=['POST'])
def detect_faces_and_mouths():
    """Fast face + mouth landmark detection for immediate caching"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "Missing frame data"}), 400

        frame_data = data['frame']
        frame_id = data.get('frame_id', 0)
        room_id = data.get('room_id', None)
        
        # Decode frame
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Initialize detector if needed
        if detector is None:
            init_detector()
        if detector == "failed":
            return jsonify({"error": "Detector not available"}), 500
        
        # Update face detector embedding if room has enrolled face
        print(f"[API] DEBUG: Checking room_id='{room_id}', available rooms: {list(room_embeddings.keys())}")
        if room_id and room_id in room_embeddings:
            print(f"[API] ✅ Updating face embedding for room {room_id}")
            embedding = room_embeddings[room_id]['embedding']  # Extract just the numpy array
            detector.update_face_embedding(embedding)
        else:
            print(f"[API] ⚠️ No embedding found for room {room_id} (available: {list(room_embeddings.keys())})")
            
        # Get face blur regions and mouth landmarks
        start_time = time.time()
        frame_id_result, face_blur_regions, mouth_regions = detector.process_frame_with_mouth_landmarks(
            frame, frame_id, stride=1
        )
        detection_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "frame_id": frame_id,
            "face_blur_regions": face_blur_regions,
            "mouth_regions": mouth_regions,
            "detection_time": detection_time,
            "total_faces": len(mouth_regions),
            "faces_to_blur": len(face_blur_regions)
        })
        
    except Exception as e:
        print(f"[API] Fast detection error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/apply-conditional-blur', methods=['POST'])  
def apply_conditional_blur():
    """Apply face blur + conditional mouth blur based on PII events"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "Missing frame data"}), 400

        frame_data = data['frame']
        face_blur_regions = data.get('face_blur_regions', [])
        mouth_regions = data.get('mouth_regions', [])
        blur_mouths = data.get('blur_mouths', False)
        blur_mode = data.get('blur_mode', 'faces_only')
        pii_reason = data.get('pii_reason', None)
        
        # Decode frame
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Apply face blurring (always - privacy protection)
        faces_blurred = 0
        for region in face_blur_regions:
            apply_gaussian_blur_region(frame, region)
            faces_blurred += 1
        
        # Apply mouth blurring (conditional - PII protection)
        mouths_blurred = 0
        if blur_mouths and mouth_regions:
            print(f"[API] 👄 Applying mouth blur to {len(mouth_regions)} mouths due to PII: {pii_reason}")
            for mouth_data in mouth_regions:
                try:
                    mouth_bbox = mouth_data['bbox']
                    if mouth_data.get('landmarks'):
                        # Use precise landmarks for better blur
                        apply_landmark_mouth_blur(frame, mouth_data['landmarks'])
                    else:
                        # Use bbox fallback
                        apply_strong_mouth_blur(frame, mouth_bbox)
                    mouths_blurred += 1
                except Exception as e:
                    print(f"[API] Error processing mouth {mouths_blurred}: {e}")
                    print(f"[API] Mouth data: {mouth_data}")
                
        # Encode result
        try:
            encode_result = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if len(encode_result) != 2:
                print(f"[API] Warning: cv2.imencode returned {len(encode_result)} values instead of 2")
                return jsonify({"error": "Image encoding failed"}), 500
            
            success, buffer = encode_result
            if not success:
                return jsonify({"error": "Failed to encode image"}), 500
                
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[API] Image encoding error: {e}")
            return jsonify({"error": "Image encoding failed"}), 500
        frame_b64 = f"data:image/jpeg;base64,{frame_b64}"
        
        return jsonify({
            "success": True,
            "processed_frame": frame_b64,
            "faces_blurred": faces_blurred,
            "mouths_blurred": mouths_blurred,
            "blur_mode": blur_mode,
            "pii_triggered": blur_mouths
        })
        
    except Exception as e:
        print(f"[API] Conditional blur error: {e}")
        return jsonify({"error": str(e)}), 500

def apply_gaussian_blur_region(frame, region):
    """Apply Gaussian blur to a specific region"""
    x1, y1, x2, y2 = map(int, region)
    h, w = frame.shape[:2]
    
    # Bounds check
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), sigmaX=75, sigmaY=75)

def apply_landmark_mouth_blur(frame, mouth_landmarks):
    """Apply rectangular blur using mouth landmarks to determine bounds"""
    try:
        if not mouth_landmarks or len(mouth_landmarks) == 0:
            print("[API] No mouth landmarks provided")
            return
            
        # Convert landmarks to numpy array and ensure proper shape
        points = np.array(mouth_landmarks, dtype=np.float32)
        
        # Handle different landmark formats
        if points.ndim == 1:
            # If flattened, reshape to 2D
            points = points.reshape(-1, 2)
        elif points.ndim == 2 and points.shape[1] == 3:
            # If 3D landmarks, take only x,y coordinates
            points = points[:, :2]
            
        # Get bounding rectangle from landmarks
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        # Apply rectangular blur to mouth region
        apply_strong_mouth_blur(frame, [x_min, y_min, x_max, y_max])
        
        print(f"[API] ✅ Applied landmark-based mouth blur: bbox=[{x_min}, {y_min}, {x_max}, {y_max}]")
        
    except Exception as e:
        print(f"[API] Error in landmark mouth blur: {e}")
        print(f"[API] Landmark data type: {type(mouth_landmarks)}, length: {len(mouth_landmarks) if mouth_landmarks else 0}")
        # No additional fallback needed - error will be logged

def apply_strong_mouth_blur(frame, mouth_bbox):
    """Strong blur for mouth region using bbox"""
    x1, y1, x2, y2 = map(int, mouth_bbox)
    h, w = frame.shape[:2]
    
    # Bounds check
    x1, y1 = max(0, x1), max(0, y1) 
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            # Very strong blur for mouth (higher than face blur)
            blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=150, sigmaY=150)
            frame[y1:y2, x1:x2] = blurred

@app.route('/transfer-embedding', methods=['POST'])
def transfer_embedding():
    """Transfer face embedding from enrollment room to streaming room."""
    global room_embeddings
    
    try:
        data = request.get_json()
        if not data or 'from_room_id' not in data or 'to_room_id' not in data:
            return jsonify({"error": "Missing room IDs"}), 400
        
        from_room_id = data['from_room_id']
        to_room_id = data['to_room_id']
        
        if from_room_id not in room_embeddings:
            return jsonify({"error": f"Source room {from_room_id} has no embedding"}), 404
        
        # Copy embedding to new room
        room_embeddings[to_room_id] = room_embeddings[from_room_id].copy()
        print(f"[API] 🔄 Transferred embedding from room {from_room_id} to room {to_room_id}")
        print(f"[API] 🔄 Available rooms after transfer: {list(room_embeddings.keys())}")
        
        return jsonify({
            "success": True,
            "message": f"Embedding transferred from {from_room_id} to {to_room_id}"
        })
        
    except Exception as e:
        print(f"[API] ❌ Transfer embedding error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Video Filter API...")
    print(f"Detector config:\n{json.dumps(DETECTOR_CONFIG, indent=2)}")
    print(f"Debug config:\n{json.dumps(DEBUG_CONFIG, indent=2)}")
    print(f"Queue config:\n{json.dumps(QUEUE_CONFIG, indent=2)}")
    print("\n🛡️ QUEUE PROTECTION FEATURES:")
    print(f"   • Drop requests older than {QUEUE_CONFIG['max_request_age_ms']}ms")
    print(f"   • Max concurrent requests: {QUEUE_CONFIG['max_concurrent_requests']}")
    print(f"   • Request dropping: {'ENABLED' if QUEUE_CONFIG['enable_request_dropping'] else 'DISABLED'}")
    print(f"   • Queue monitoring: {'ENABLED' if QUEUE_CONFIG['queue_monitoring'] else 'DISABLED'}")
    
    # Initialize debug directories on startup
    setup_debug_directories()
    init_detector()
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
