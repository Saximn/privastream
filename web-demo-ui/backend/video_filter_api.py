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
from pathlib import Path

# Add video_models to path
sys.path.append(str(Path(__file__).parent.parent / "video_models"))
from video_models.unified_detector import UnifiedBlurDetector

app = Flask(__name__)
CORS(app, origins="*")

# Detector configuration
DETECTOR_CONFIG = {
    "enable_face": False,
    "enable_pii": False,
    "enable_plate": False,
    "pii": {
        "classifier_path": "video_models/pii_blur/pii_clf.joblib",
        "conf_thresh": 0.35
    }
}

detector = None

def init_detector():
    """Initialize the detector once."""
    global detector
    if detector is None:
        try:
            detector = UnifiedBlurDetector(DETECTOR_CONFIG)
            print("[API] Video filter detector initialized")
        except Exception as e:
            print(f"[API] Failed to initialize detector: {e}")
            detector = "failed"

def filter_frame(frame, frame_id=0, blur_only=False, provided_rectangles=None):
    """
    Process a single frame with two modes:
    1. Full detection + blur (blur_only=False)
    2. CPU blur only using provided rectangles (blur_only=True)
    """
    global detector
    rectangles = []
    
    if blur_only and provided_rectangles:
        # CPU-only blur mode: use provided rectangles, no GPU detection
        print(f"[API] CPU blur mode: applying blur to {len(provided_rectangles)} regions")
        rectangles = provided_rectangles
    else:
        # Full detection mode: run GPU detection
        if detector is None:
            init_detector()
        if detector == "failed":
            raise RuntimeError("Detector initialization failed")
        
        print(f"[API] Full detection mode: processing frame {frame_id}")
        results = detector.process_frame(frame, frame_id)
        
        # Extract rectangles from detection results
        for model_name in ["face", "plate", "pii"]:
            model_data = results.get("models", {}).get(model_name, {})
            if "rectangles" in model_data:
                rectangles.extend(model_data["rectangles"])
        
        print(f"[API] Detected {len(rectangles)} regions")
    
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
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "Missing frame data"}), 400

        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]

        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        frame_id = data.get('frame_id', 0)
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
            provided_rectangles=provided_rectangles
        )

        return jsonify({
            "success": True,
            "frame_id": frame_id,
            "frame": blurred_frame_b64,
            "rectangles": rectangles,
            "processing_mode": "blur_only" if blur_only else ("detect_only" if detect_only else "full"),
            "regions_processed": len(rectangles)
        })

    except Exception as e:
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

if __name__ == '__main__':
    print("Starting Video Filter API...")
    print(f"Detector config:\n{json.dumps(DETECTOR_CONFIG, indent=2)}")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
