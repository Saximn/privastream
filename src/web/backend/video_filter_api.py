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
import os
import time
import threading
from datetime import datetime
from pathlib import Path
import logging
from flask import g

# Structured logging (levels, timestamps) replaces ad-hoc print() calls.
# Tune verbosity with the LOG_LEVEL env var (default INFO).
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("video_filter_api")


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name, default, minimum=None):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if minimum is not None:
            parsed = max(minimum, parsed)
        return parsed
    except ValueError:
        logger.warning(f"[CONFIG] Invalid integer for {name}={value!r}; using {default}")
        return default

def _latency_ms(start_time):
    return round((time.perf_counter() - start_time) * 1000, 2)

from src.models.detection.unified_detector import UnifiedBlurDetector
from src.web.backend.video_api_utils import (
    RequestTracker,
    cleanup_expired_embeddings,
    decode_base64_image,
    decode_image_bytes,
    encode_jpeg,
    encode_jpeg_data_url,
    extract_model_regions,
    is_stale_request,
    json_header,
    room_ids_from_request,
    set_embedding_expiry,
    strip_data_url,
    apply_blur_region,
)

# InsightFace imports for face enrollment
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("Warning: InsightFace not available. Face enrollment disabled.")

def allowed_origins():
    """CORS allowlist from CORS_ALLOWED_ORIGINS (comma-separated).

    Defaults to the localhost frontend for development. Never wildcard — these
    endpoints drive GPU detection/blur and must not be callable from any site.
    """
    raw = os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000')
    return [o.strip() for o in raw.split(',') if o.strip()]


app = Flask(__name__)
CORS(app, origins=allowed_origins())

from functools import wraps
from src.web.backend import auth


def require_room_token(f):
    """Reject requests without a valid room token when REQUIRE_AUTH is on.

    No-op when REQUIRE_AUTH is unset (local dev). The token is read from the
    Authorization: Bearer header and must match the request's room_id (from the
    JSON body or the path parameter).
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not auth.auth_required():
            return f(*args, **kwargs)
        header = request.headers.get('Authorization', '')
        token = header[7:] if header.startswith('Bearer ') else header
        body = request.get_json(silent=True) or {}
        header_room_id = request.headers.get("X-Room-Id") or request.args.get("room_id")
        if header_room_id:
            body = {**body, "room_id": header_room_id}
        payload = auth.verify_room_token(token, None)
        room_ids = room_ids_from_request(kwargs.get("room_id"), body)
        if payload is None or (room_ids and payload.get("room_id") not in room_ids):
            return jsonify({"error": "Unauthorized"}), 401
        g.room_token_payload = payload
        return f(*args, **kwargs)
    return wrapper

# PERFORMANCE CONFIGURATION - Easy to adjust
DETECTION_FPS = 30.0  # FPS for face/privacy detection (lower = less latency, higher = more CPU load)
# Conversion: 30fps input -> stride calculation
DETECTION_STRIDE = max(1, int(30 / DETECTION_FPS))  # Process every Nth frame

EXPECTED_DELAY_SEC = DETECTION_STRIDE / 30.0  # Delay in seconds based on 30fps input
logger.info(f"[CONFIG] Detection FPS: {DETECTION_FPS}, Stride: {DETECTION_STRIDE} (process every {DETECTION_STRIDE} frames)")
logger.info(f"[CONFIG] Expected detection delay: {EXPECTED_DELAY_SEC:.2f} seconds")

# Detector configuration
DETECTOR_CONFIG = {
    "enable_face": True,
    "enable_pii": True,
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
    "max_request_age_ms": _env_int("VIDEO_FILTER_MAX_REQUEST_AGE_MS", 1000, minimum=0),
    "max_concurrent_requests": _env_int("VIDEO_FILTER_MAX_CONCURRENT_REQUESTS", 10, minimum=1),
    "enable_request_dropping": _env_bool("VIDEO_FILTER_ENABLE_REQUEST_DROPPING", True),
    "queue_monitoring": _env_bool("VIDEO_FILTER_QUEUE_MONITORING", True),
    "overload_status_code": _env_int("VIDEO_FILTER_OVERLOAD_STATUS_CODE", 503, minimum=400),
    "stale_status_code": _env_int("VIDEO_FILTER_STALE_STATUS_CODE", 429, minimum=400),
}
EMBEDDING_TTL_SECONDS = _env_int("ROOM_EMBEDDING_TTL_SECONDS", 6 * 3600, minimum=60)

detector = None
face_app = None  # Global face detection instance for enrollment
room_embeddings = {}  # Store face embeddings per room: roomId -> {'embedding': np.array, 'metadata': {...}}
active_requests = 0  # Track concurrent processing requests
request_lock = threading.Lock()  # Thread lock for request counting
request_tracker = RequestTracker(QUEUE_CONFIG["max_concurrent_requests"])

def is_request_stale(request_timestamp_ms):
    """Check if request is too old to process."""
    stale = is_stale_request(
        request_timestamp_ms,
        QUEUE_CONFIG["max_request_age_ms"],
        enabled=QUEUE_CONFIG["enable_request_dropping"],
    )
    
    if QUEUE_CONFIG["queue_monitoring"]:
        logger.debug(
            "[QUEUE] Request timestamp=%s max_age=%sms stale=%s",
            request_timestamp_ms,
            QUEUE_CONFIG["max_request_age_ms"],
            stale,
        )
    
    return stale

def can_process_request():
    """Check if we can process another request (not at concurrent limit)."""
    request_tracker.configure(QUEUE_CONFIG["max_concurrent_requests"])
    can_process = request_tracker.can_start()
    if QUEUE_CONFIG["queue_monitoring"]:
        logger.debug(
            "[QUEUE] Active requests: %s/%s, Can process: %s",
            request_tracker.active,
            request_tracker.max_concurrent,
            can_process,
        )
    return can_process

def start_request_processing():
    """Mark start of request processing."""
    global active_requests
    active_requests = request_tracker.start()
    if QUEUE_CONFIG["queue_monitoring"]:
        logger.debug(f"[QUEUE] Started processing request, active: {active_requests}")

def finish_request_processing():
    """Mark end of request processing."""
    global active_requests
    active_requests = request_tracker.finish()
    if QUEUE_CONFIG["queue_monitoring"]:
        logger.debug(f"[QUEUE] Finished processing request, active: {active_requests}")

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
        
        logger.debug(f"[DEBUG] Debug directories created at: {debug_dir.absolute()}")
    except Exception as e:
        logger.error(f"[DEBUG] Failed to create debug directories: {e}")

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
        
        logger.debug(f"[DEBUG] Saved {image_type} image: {filepath.name}")
        
    except Exception as e:
        logger.error(f"[DEBUG] Failed to save {image_type} image: {e}")

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
                
            logger.debug(f"[DEBUG] Cleaned up {len(files_to_remove)} old images from {directory.name}")
            
    except Exception as e:
        logger.error(f"[DEBUG] Failed to cleanup old images: {e}")

def init_detector():
    """Initialize the detector once."""
    global detector, face_app
    if detector is None:
        try:
            detector = UnifiedBlurDetector()
            logger.info("[API] Video filter detector initialized")
            
            # Set up debug directories
            setup_debug_directories()
            
        except Exception as e:
            logger.error(f"[API] Failed to initialize detector: {e}")
            detector = "failed"
    
    # Initialize face detection for enrollment
    if face_app is None and INSIGHTFACE_AVAILABLE:
        try:
            logger.info("[API] Initializing InsightFace Buffalo_S for enrollment...")
            face_app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("[API] ✅ InsightFace Buffalo_S initialized for enrollment")
        except Exception as e:
            logger.error(f"[API] ❌ Failed to initialize InsightFace: {e}")
            face_app = "failed"

def filter_frame(frame, frame_id=0, blur_only=False, provided_rectangles=None, room_id=None, detect_only=False):
    """
    Process a single frame with three modes:
    1. Full detection + blur (blur_only=False, detect_only=False)
    2. CPU blur only using provided rectangles (blur_only=True)
    3. Detection only with compact boxes (detect_only=True)
    """
    timings = {"decode_ms": 0.0, "detect_ms": 0.0, "blur_ms": 0.0, "encode_ms": 0.0, "total_ms": 0.0}
    total_start = time.perf_counter()
    global detector, room_embeddings
    rectangles = []
    
    # Save debug input image (copy before any modifications)
    input_frame_copy = frame.copy()
    
    if blur_only and provided_rectangles:
        # CPU-only blur mode: use provided rectangles, no GPU detection
        logger.info(f"[API] CPU blur mode: applying blur to {len(provided_rectangles)} regions")
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
        logger.info(f"[API] DEBUG: Checking room_id='{room_id}', available rooms: {list(room_embeddings.keys())}")
        if room_id and room_id in room_embeddings:
            logger.info(f"[API] ✅ Updating face embedding for room {room_id}")
            embedding = room_embeddings[room_id]['embedding']  # Extract just the numpy array
            detector.update_face_embedding(embedding)
        else:
            logger.warning(f"[API] ⚠️  No embedding found for room {room_id} (available: {list(room_embeddings.keys())})")
        
        logger.info(f"[API] Full detection mode: processing frame {frame_id}")
        detect_start = time.perf_counter()
        results = detector.process_frame(frame, frame_id, stride=DETECTION_STRIDE, room_id=room_id)
        timings["detect_ms"] = _latency_ms(detect_start)
        
        regions = extract_model_regions(results)
        rectangles.extend(regions["face"])
        rectangles.extend(regions["plate"])
        rectangles.extend(regions["pii"])
        
        logger.info(f"[API] Detected {len(rectangles)} regions")
        
        # Save input image with detection boxes for comparison
        save_debug_image(input_frame_copy, "input", frame_id, rectangles)
    
    if detect_only:
        timings["total_ms"] = _latency_ms(total_start)
        compact_rectangles = [[int(v) for v in rect[:4]] for rect in rectangles if len(rect) >= 4]
        logger.info(f"[API] Detect-only mode: returning {len(compact_rectangles)} compact boxes without blur/encode")
        return None, compact_rectangles, timings

    # Apply Gaussian blur to all rectangles (CPU processing)
    blur_start = time.perf_counter()
    blur_applied = 0
    for rect in rectangles:
        try:
            if apply_blur_region(frame, rect, kernel_size=75):
                blur_applied += 1
        except Exception as e:
            logger.error(f"[API] Error blurring rectangle {rect}: {e}")
    
    timings["blur_ms"] = _latency_ms(blur_start)
    logger.info(f"[API] Applied blur to {blur_applied}/{len(rectangles)} regions")
    
    # Save debug output image (after blur processing)
    save_debug_image(frame, "output", frame_id, rectangles)
    
    # Encode frame as base64 JPEG
    encode_start = time.perf_counter()
    frame_b64 = encode_jpeg_data_url(frame, quality=85)
    timings["encode_ms"] = _latency_ms(encode_start)
    timings["total_ms"] = _latency_ms(total_start)
    
    return frame_b64, rectangles, timings

@app.route('/health')
def health():
    expired = cleanup_expired_embeddings(room_embeddings, EMBEDDING_TTL_SECONDS)
    return {
        'status': 'healthy',
        'detector_ready': detector is not None and detector != "failed",
        'room_embeddings': len(room_embeddings),
        'expired_embeddings_cleaned': expired,
    }

@app.route('/process-frame', methods=['POST'])
@require_room_token
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
            logger.info(f"[API] Processing frame {frame_id} for room {room_id}")
        
        # 1. Check if request is too old
        if is_request_stale(request_timestamp):
            logger.debug(f"[QUEUE] 🗑️ DROPPING STALE REQUEST - Frame {frame_id} is too old to process")
            return jsonify({
                "success": False,
                "error": "Request too old",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "stale_request"
            }), QUEUE_CONFIG["stale_status_code"]
        
        # 2. Check concurrent request limit
        if not can_process_request():
            logger.debug(f"[QUEUE] 🚫 DROPPING REQUEST - Too many concurrent requests, Frame {frame_id}")
            return jsonify({
                "success": False,
                "error": "Server overloaded",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "overloaded"
            }), QUEUE_CONFIG["overload_status_code"]
        
        # 3. Mark request as started
        start_request_processing()
        processing_started = True
        
        try:
            decode_start = time.perf_counter()
            frame_data = data['frame']
            try:
                frame = decode_base64_image(frame_data)
            except ValueError:
                return jsonify({"error": "Invalid image data"}), 400
            decode_ms = _latency_ms(decode_start)

            blur_only = data.get('blur_only', False)
            provided_rectangles = data.get('rectangles', [])
            detect_only = data.get('detect_only', False)
            
            # Log processing mode
            if blur_only:
                logger.info(f"[API] Processing frame {frame_id} in BLUR_ONLY mode with {len(provided_rectangles)} rectangles")
            elif detect_only:
                logger.info(f"[API] Processing frame {frame_id} in DETECT_ONLY mode (full detection)")
            else:
                logger.info(f"[API] Processing frame {frame_id} in FULL mode")
            
            processed_frame_b64, rectangles, timings = filter_frame(
                frame, 
                frame_id, 
                blur_only=blur_only, 
                provided_rectangles=provided_rectangles,
                room_id=room_id,
                detect_only=detect_only
            )
            timings["decode_ms"] = decode_ms
            timings["total_ms"] = round(timings.get("total_ms", 0.0) + decode_ms, 2)

            response_payload = {
                "success": True,
                "frame_id": frame_id,
                "frame": processed_frame_b64,
                "rectangles": rectangles,
                "processing_mode": "blur_only" if blur_only else ("detect_only" if detect_only else "full"),
                "regions_processed": len(rectangles),
                "timings": timings,
            }

            return jsonify(response_payload)
        
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
        
        logger.error(f"[API] Frame processing error: {e}")
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/process-frame-binary', methods=['POST'])
@require_room_token
def process_frame_binary_route():
    """Binary frame variant of /process-frame.

    Request body is encoded image bytes. Metadata is carried in headers:
    X-Room-Id, X-Frame-Id, X-Timestamp-Ms, X-Blur-Only, and X-Rectangles.
    The response body is image/jpeg unless X-Detect-Only is true, in which
    case JSON metadata is returned.
    """
    processing_started = False

    try:
        room_id = request.headers.get("X-Room-Id") or request.args.get("room_id")
        frame_id = int(request.headers.get("X-Frame-Id", "0"))
        request_timestamp = request.headers.get("X-Timestamp-Ms", str(int(time.time() * 1000)))

        if is_request_stale(request_timestamp):
            return jsonify({
                "success": False,
                "error": "Request too old",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "stale_request",
            }), QUEUE_CONFIG["stale_status_code"]

        if not can_process_request():
            return jsonify({
                "success": False,
                "error": "Server overloaded",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "overloaded",
            }), QUEUE_CONFIG["overload_status_code"]

        start_request_processing()
        processing_started = True

        try:
            decode_start = time.perf_counter()
            try:
                frame = decode_image_bytes(request.get_data())
            except ValueError:
                return jsonify({"error": "Invalid image data"}), 400
            decode_ms = _latency_ms(decode_start)

            rectangles_header = request.headers.get("X-Rectangles", "[]")
            try:
                provided_rectangles = json.loads(rectangles_header)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid X-Rectangles header"}), 400

            blur_only = request.headers.get("X-Blur-Only", "false").lower() in {"1", "true", "yes"}
            detect_only = request.headers.get("X-Detect-Only", "false").lower() in {"1", "true", "yes"}
            processed_frame_b64, rectangles, timings = filter_frame(
                frame,
                frame_id,
                blur_only=blur_only,
                provided_rectangles=provided_rectangles,
                room_id=room_id,
                detect_only=detect_only,
            )
            timings["decode_ms"] = decode_ms
            timings["total_ms"] = round(timings.get("total_ms", 0.0) + decode_ms, 2)

            if detect_only:
                return jsonify({
                    "success": True,
                    "frame_id": frame_id,
                    "rectangles": rectangles,
                    "processing_mode": "detect_only",
                    "regions_processed": len(rectangles),
                    "timings": timings,
                })

            jpeg_bytes = base64.b64decode(strip_data_url(processed_frame_b64))
            response = app.response_class(jpeg_bytes, mimetype="image/jpeg")
            response.headers["X-Frame-Id"] = str(frame_id)
            response.headers["X-Regions-Processed"] = str(len(rectangles))
            response.headers["X-Timings"] = json_header(timings)
            return response

        finally:
            if processing_started:
                finish_request_processing()

    except Exception:
        if processing_started:
            try:
                finish_request_processing()
            except Exception:
                pass
        logger.exception("Unhandled error during binary frame processing")
        return jsonify({"error": "Internal server error"}), 500

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
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

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
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

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
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

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
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/queue-status')
def queue_status():
    """Get current queue and processing status."""
    return jsonify({
        "active_requests": request_tracker.active,
        "max_concurrent": QUEUE_CONFIG["max_concurrent_requests"],
        "can_accept_new": request_tracker.can_start(),
        "max_request_age_ms": QUEUE_CONFIG["max_request_age_ms"],
        "request_dropping_enabled": QUEUE_CONFIG["enable_request_dropping"],
        "queue_monitoring": QUEUE_CONFIG["queue_monitoring"],
        "overload_status_code": QUEUE_CONFIG["overload_status_code"],
        "stale_status_code": QUEUE_CONFIG["stale_status_code"],
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

        def _validate_and_clamp_status_code(value):
            if isinstance(value, bool):
                app.logger.warning("Ignoring boolean status code value for /queue-config: %s", value)
                return None
            try:
                return max(400, min(599, int(value)))
            except (TypeError, ValueError):
                app.logger.warning("Ignoring invalid status code value for /queue-config: %s", value)
                return None
        
        # Update configuration
        if "max_request_age_ms" in data:
            QUEUE_CONFIG["max_request_age_ms"] = int(data["max_request_age_ms"])
        if "max_concurrent_requests" in data:
            QUEUE_CONFIG["max_concurrent_requests"] = int(data["max_concurrent_requests"])
            request_tracker.configure(QUEUE_CONFIG["max_concurrent_requests"])
        if "enable_request_dropping" in data:
            QUEUE_CONFIG["enable_request_dropping"] = bool(data["enable_request_dropping"])
        if "queue_monitoring" in data:
            QUEUE_CONFIG["queue_monitoring"] = bool(data["queue_monitoring"])
        if "overload_status_code" in data:
            overload_code = _validate_and_clamp_status_code(data["overload_status_code"])
            if overload_code is not None:
                QUEUE_CONFIG["overload_status_code"] = overload_code
        if "stale_status_code" in data:
            stale_code = _validate_and_clamp_status_code(data["stale_status_code"])
            if stale_code is not None:
                QUEUE_CONFIG["stale_status_code"] = stale_code
        
        return jsonify({
            "success": True, 
            "message": "Queue configuration updated",
            "config": QUEUE_CONFIG
        })
    except Exception as e:
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

# Face Enrollment Endpoints
@app.route('/face-detection', methods=['POST'])
@require_room_token
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
        
        logger.info(f"[ENROLLMENT] DEBUG - Received request with room_id: '{room_id}'")
        
        # Decode base64 image
        try:
            image = decode_base64_image(frame_data)
        except ValueError:
            logger.exception("Image decode error")
            return jsonify({
                'success': False,
                'error': 'Failed to decode image',
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
            
            logger.info(f"[ENROLLMENT] Detected {len(faces)} faces, showing largest one in {detection_time:.3f}s for room {room_id}")
        else:
            logger.info(f"[ENROLLMENT] No faces detected in {detection_time:.3f}s for room {room_id}")
        
        return jsonify({
            'success': True,
            'faces_detected': detected_faces,
            'detection_time': detection_time,
            'room_id': room_id
        })
        
    except Exception as e:
        logger.exception("[ENROLLMENT] Face detection error")
        return jsonify({
            'success': False,
            'error': 'Face detection failed',
            'faces_detected': []
        }), 500

@app.route('/face-enrollment', methods=['POST'])
@require_room_token
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
        
        logger.info(f"[ENROLLMENT] Starting face enrollment for room: {room_id}")
        logger.info(f"[ENROLLMENT] Processing {len(frames)} frames")
        
        all_embeddings = []
        valid_frames = 0
        
        for i, frame_data in enumerate(frames):
            try:
                # Decode base64 image
                image = decode_base64_image(frame_data)
                
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
                        logger.info(f"[ENROLLMENT] Frame {i}: extracted largest face embedding (area: {face_area:.0f}px, {len(faces)} total faces)")
                    else:
                        logger.info(f"[ENROLLMENT] Frame {i}: no valid embedding from largest face")
                else:
                    logger.info(f"[ENROLLMENT] Frame {i}: no faces detected")
                    
            except Exception as e:
                logger.error(f"[ENROLLMENT] Error processing frame {i}: {e}")
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
        room_embeddings[room_id] = set_embedding_expiry({
            'embedding': average_embedding,
            'metadata': {
                'enrollment_time': datetime.now().isoformat(),
                'frames_processed': len(frames),
                'valid_frames': valid_frames,
                'embeddings_count': len(all_embeddings)
            }
        }, EMBEDDING_TTL_SECONDS)
        
        logger.info(f"[ENROLLMENT] ✅ Face enrollment complete for room {room_id}")
        logger.info(f"[ENROLLMENT]    Processed: {valid_frames}/{len(frames)} frames")
        logger.info(f"[ENROLLMENT]    Embeddings: {len(all_embeddings)} -> 1 average")
        
        return jsonify({
            'success': True,
            'message': f'Face enrolled successfully from {valid_frames} frames',
            'enrollment_complete': True,
            'metadata': room_embeddings[room_id]['metadata']
        })
        
    except Exception as e:
        logger.exception("[ENROLLMENT] Face enrollment error")
        return jsonify({
            'success': False,
            'message': 'Enrollment failed',
            'enrollment_complete': False
        }), 500

@app.route('/room-status/<room_id>', methods=['GET'])
@require_room_token
def get_room_status(room_id: str):
    """Get enrollment status for a room"""
    try:
        cleanup_expired_embeddings(room_embeddings, EMBEDDING_TTL_SECONDS)
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
        logger.exception("Room status check failed")
        return jsonify({
            'error': 'Status check failed'
        }), 500

@app.route('/cleanup-room/<room_id>', methods=['DELETE'])
@require_room_token
def cleanup_room(room_id: str):
    """Clean up enrollment data for a room"""
    try:
        if room_id in room_embeddings:
            del room_embeddings[room_id]
            logger.info(f"[ENROLLMENT] Cleaned up enrollment data for room: {room_id}")
        
        return jsonify({
            'success': True,
            'message': f'Room {room_id} cleaned up'
        })
        
    except Exception as e:
        logger.exception("Room cleanup failed")
        return jsonify({
            'success': False,
            'error': 'Cleanup failed'
        }), 500

@app.route('/detect-faces-mouths', methods=['POST'])
@require_room_token
def detect_faces_and_mouths():
    """Fast face + mouth landmark detection AND PII/plate detection for immediate caching"""
    processing_started = False
    
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "Missing frame data"}), 400

        frame_data = data['frame']
        frame_id = data.get('frame_id', 0)
        room_id = data.get('room_id', None)
        
        # Get request timestamp (from client or use current time)
        request_timestamp = data.get('timestamp', int(time.time() * 1000))
        
        if room_id:
            logger.info(f"[API] Processing frame {frame_id} for room {room_id}")
        
        # 1. Check if request is too old
        if is_request_stale(request_timestamp):
            logger.debug(f"[QUEUE] 🗑️ DROPPING STALE REQUEST - Frame {frame_id} is too old to process")
            return jsonify({
                "success": False,
                "error": "Request too old",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "stale_request"
            }), QUEUE_CONFIG["stale_status_code"]
        
        # 2. Check concurrent request limit
        if not can_process_request():
            logger.debug(f"[QUEUE] 🚫 DROPPING REQUEST - Too many concurrent requests, Frame {frame_id}")
            return jsonify({
                "success": False,
                "error": "Server overloaded",
                "frame_id": frame_id,
                "dropped": True,
                "reason": "overloaded"
            }), QUEUE_CONFIG["overload_status_code"]
        
        # 3. Mark request as started
        start_request_processing()
        processing_started = True
        
        try:
            timings = {"decode_ms": 0.0, "detect_ms": 0.0, "blur_ms": 0.0, "encode_ms": 0.0, "total_ms": 0.0}
            total_start = time.perf_counter()
            # Decode frame
            decode_start = time.perf_counter()
            try:
                frame = decode_base64_image(frame_data)
            except ValueError:
                return jsonify({"error": "Invalid image data"}), 400
            timings["decode_ms"] = _latency_ms(decode_start)
            
            # Initialize detector if needed
            if detector is None:
                init_detector()
            if detector == "failed":
                return jsonify({"error": "Detector not available"}), 500
            
            # Update face detector embedding if room has enrolled face
            logger.info(f"[API] DEBUG: Checking room_id='{room_id}', available rooms: {list(room_embeddings.keys())}")
            if room_id and room_id in room_embeddings:
                logger.info(f"[API] ✅ Updating face embedding for room {room_id}")
                embedding = room_embeddings[room_id]['embedding']  # Extract just the numpy array
                detector.update_face_embedding(embedding)
            else:
                logger.warning(f"[API] ⚠️ No embedding found for room {room_id} (available: {list(room_embeddings.keys())})")
                
            # Log processing mode
            logger.info(f"[API] Processing frame {frame_id} in FAST_DETECTION mode (face+mouth+PII+plate)")
                
            start_time = time.time()
            full_results = detector.process_frame(frame, frame_id, stride=1, room_id=room_id)
            regions = extract_model_regions(full_results)
            face_blur_regions = regions["face"]
            mouth_regions = regions["mouth"]
            pii_regions = regions["pii"]
            plate_regions = regions["plate"]
            logger.info(
                "[API] Detection found face=%s mouth=%s pii=%s plate=%s",
                len(face_blur_regions),
                len(mouth_regions),
                len(pii_regions),
                len(plate_regions),
            )
                
            detection_time = time.time() - start_time
            timings["detect_ms"] = round(detection_time * 1000, 2)
            timings["total_ms"] = _latency_ms(total_start)
            
            return jsonify({
                "success": True,
                "frame_id": frame_id,
                "face_blur_regions": face_blur_regions,
                "mouth_regions": mouth_regions,
                "pii_regions": pii_regions,
                "plate_regions": plate_regions,
                "detection_time": detection_time,
                "timings": timings,
                "total_faces": len(mouth_regions),
                "faces_to_blur": len(face_blur_regions),
                "pii_count": len(pii_regions),
                "plate_count": len(plate_regions),
                "processing_mode": "fast_detection"
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
        
        logger.error(f"[API] Fast detection error: {e}")
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/apply-conditional-blur', methods=['POST'])
@require_room_token
def apply_conditional_blur():
    """Apply face blur + conditional mouth blur + PII blur + plate blur"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "Missing frame data"}), 400

        timings = {"decode_ms": 0.0, "detect_ms": 0.0, "blur_ms": 0.0, "encode_ms": 0.0, "total_ms": 0.0}
        total_start = time.perf_counter()
        frame_data = data['frame']
        face_blur_regions = data.get('face_blur_regions', [])
        mouth_regions = data.get('mouth_regions', [])
        pii_regions = data.get('pii_regions', [])
        plate_regions = data.get('plate_regions', [])
        blur_mouths = data.get('blur_mouths', False)
        blur_mode = data.get('blur_mode', 'faces_only')
        pii_reason = data.get('pii_reason', None)
        
        # Decode frame
        decode_start = time.perf_counter()
        try:
            frame = decode_base64_image(frame_data)
        except ValueError:
            return jsonify({"error": "Invalid image data"}), 400
        timings["decode_ms"] = _latency_ms(decode_start)
        
        blur_start = time.perf_counter()
        # Apply face blurring (always - privacy protection)
        faces_blurred = 0
        for region in face_blur_regions:
            if apply_gaussian_blur_region(frame, region):
                faces_blurred += 1
        
        # Apply PII blurring (always - privacy protection)
        pii_blurred = 0
        for region in pii_regions:
            if apply_gaussian_blur_region(frame, region):
                pii_blurred += 1
        logger.info(f"[API] 🔒 Applied PII blur to {pii_blurred} regions")
        
        # Apply plate blurring (always - privacy protection)
        plates_blurred = 0
        for region in plate_regions:
            if apply_gaussian_blur_region(frame, region):
                plates_blurred += 1
        logger.info(f"[API] 🚗 Applied plate blur to {plates_blurred} regions")
        
        # Apply mouth blurring (conditional - PII protection)
        mouths_blurred = 0
        if blur_mouths and mouth_regions:
            logger.info(f"[API] 👄 Applying mouth blur to {len(mouth_regions)} mouths due to PII: {pii_reason}")
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
                    logger.error(f"[API] Error processing mouth {mouths_blurred}: {e}")
                    logger.info(f"[API] Mouth data: {mouth_data}")
                
        timings["blur_ms"] = _latency_ms(blur_start)
        # Encode result
        try:
            encode_start = time.perf_counter()
            frame_b64 = base64.b64encode(encode_jpeg(frame, quality=85)).decode('utf-8')
            timings["encode_ms"] = _latency_ms(encode_start)
        except Exception as e:
            logger.error(f"[API] Image encoding error: {e}")
            return jsonify({"error": "Image encoding failed"}), 500
        frame_b64 = f"data:image/jpeg;base64,{frame_b64}"
        timings["total_ms"] = _latency_ms(total_start)
        
        return jsonify({
            "success": True,
            "processed_frame": frame_b64,
            "faces_blurred": faces_blurred,
            "mouths_blurred": mouths_blurred,
            "pii_blurred": pii_blurred,
            "plates_blurred": plates_blurred,
            "blur_mode": blur_mode,
            "pii_triggered": blur_mouths,
            "timings": timings
        })
        
    except Exception as e:
        logger.error(f"[API] Conditional blur error: {e}")
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

def apply_gaussian_blur_region(frame, region):
    """Apply Gaussian blur to a specific region"""
    return apply_blur_region(frame, region, kernel_size=75)

def apply_landmark_mouth_blur(frame, mouth_landmarks):
    """Apply rectangular blur using mouth landmarks to determine bounds"""
    try:
        if not mouth_landmarks or len(mouth_landmarks) == 0:
            logger.info("[API] No mouth landmarks provided")
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
        
        logger.info(f"[API] ✅ Applied landmark-based mouth blur: bbox=[{x_min}, {y_min}, {x_max}, {y_max}]")
        
    except Exception as e:
        logger.error(f"[API] Error in landmark mouth blur: {e}")
        logger.info(f"[API] Landmark data type: {type(mouth_landmarks)}, length: {len(mouth_landmarks) if mouth_landmarks else 0}")
        # No additional fallback needed - error will be logged

def apply_strong_mouth_blur(frame, mouth_bbox):
    """Strong blur for mouth region using bbox"""
    return apply_blur_region(frame, mouth_bbox, kernel_size=150)

@app.route('/cleanup-room/<room_id>', methods=['POST'])
@require_room_token
def cleanup_room_endpoint(room_id: str):
    """Clean up all room-specific data."""
    global detector, room_embeddings
    
    try:
        # Clean up face detector room data
        if detector:
            detector.cleanup_room(room_id)
        
        # Clean up embeddings
        if room_id in room_embeddings:
            del room_embeddings[room_id]
            logger.info(f"[API] Cleaned up embedding for room: {room_id}")
            
        return jsonify({
            "success": True,
            "message": f"Cleaned up all data for room {room_id}"
        })
        
    except Exception as e:
        logger.error(f"[API] ❌ Room cleanup error: {e}")
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/transfer-embedding', methods=['POST'])
@require_room_token
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
        room_embeddings[to_room_id] = set_embedding_expiry(
            room_embeddings[from_room_id].copy(),
            EMBEDDING_TTL_SECONDS,
        )
        logger.info(f"[API] 🔄 Transferred embedding from room {from_room_id} to room {to_room_id}")
        logger.info(f"[API] 🔄 Available rooms after transfer: {list(room_embeddings.keys())}")
        
        return jsonify({
            "success": True,
            "message": f"Embedding transferred from {from_room_id} to {to_room_id}"
        })
        
    except Exception as e:
        logger.error(f"[API] ❌ Transfer embedding error: {e}")
        logger.exception("Unhandled error during request")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting Video Filter API...")
    logger.info(f"Detector config:\n{json.dumps(DETECTOR_CONFIG, indent=2)}")
    logger.info(f"Debug config:\n{json.dumps(DEBUG_CONFIG, indent=2)}")
    logger.info(f"Queue config:\n{json.dumps(QUEUE_CONFIG, indent=2)}")
    logger.info("\n🛡️ QUEUE PROTECTION FEATURES:")
    logger.info(f"   • Drop requests older than {QUEUE_CONFIG['max_request_age_ms']}ms")
    logger.info(f"   • Max concurrent requests: {QUEUE_CONFIG['max_concurrent_requests']}")
    logger.info(f"   • Request dropping: {'ENABLED' if QUEUE_CONFIG['enable_request_dropping'] else 'DISABLED'}")
    logger.info(f"   • Queue monitoring: {'ENABLED' if QUEUE_CONFIG['queue_monitoring'] else 'DISABLED'}")
    
    # Initialize debug directories on startup
    setup_debug_directories()
    init_detector()

    # NOTE: app.run() is Flask's development server and should not be used in
    # production. Run under a real WSGI server instead, e.g.:
    #   gunicorn --workers 1 --threads 8 --timeout 120 \
    #            --bind 0.0.0.0:5001 video_filter_api:app
    # A single worker keeps one GPU model resident; threads handle concurrency.
    # Routes initialise the detector lazily, so gunicorn needs no extra hook.
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
