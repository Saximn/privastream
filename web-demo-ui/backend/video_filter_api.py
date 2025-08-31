"""
Fast HTTP API for real-time video filtering using unified detector.
Processes single frames and returns blur regions for WebRTC integration.
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

# Initialize detector once at startup (start with minimal config)
DETECTOR_CONFIG = {
    "enable_face": False,
    "enable_pii": True,
    "enable_plate": True,
    "pii": {
        "classifier_path": "pii_blur/pii_clf.joblib", 
        "conf_thresh": 0.35
    }
}

# Global detector instance
detector = None

def init_detector():
    """Initialize the detector on first request."""
    global detector
    if detector is None:
        try:
            detector = UnifiedBlurDetector(DETECTOR_CONFIG)
            print("[API] Video filter detector initialized")
        except Exception as e:
            print(f"[API] Failed to initialize detector: {e}")
            detector = "failed"

@app.route('/health')
def health():
    return {'status': 'healthy', 'detector': detector is not None and detector != "failed"}

@app.route('/filter-frame', methods=['POST'])
def filter_frame():
    """Process a single frame and return blur regions."""
    global detector
    
    # Initialize detector if needed
    if detector is None:
        init_detector()
    
    if detector == "failed":
        return jsonify({"error": "Detector initialization failed"}), 500
    
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "Missing frame data"}), 400
        
        # Decode base64 image
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
        
        # Convert to OpenCV format
        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Process frame
        frame_id = data.get('frame_id', 0)
        results = detector.process_frame(frame, frame_id)
        
        # Extract regions for blurring
        rectangles = []
        polygons = []
        
        # Face rectangles
        face_data = results.get("models", {}).get("face", {})
        if "rectangles" in face_data:
            rectangles.extend(face_data["rectangles"])
        
        # Plate rectangles  
        plate_data = results.get("models", {}).get("plate", {})
        if "rectangles" in plate_data:
            rectangles.extend(plate_data["rectangles"])
        
        # PII polygons
        pii_data = results.get("models", {}).get("pii", {})
        if "polygons" in pii_data:
            # Convert numpy arrays to lists for JSON serialization
            for poly in pii_data["polygons"]:
                polygons.append(poly.tolist())
        
        return jsonify({
            "success": True,
            "frame_id": frame_id,
            "rectangles": rectangles,
            "polygons": polygons,
            "detection_counts": {
                "face": face_data.get("count", 0),
                "pii": pii_data.get("count", 0), 
                "plate": plate_data.get("count", 0)
            }
        })
        
    except Exception as e:
        print(f"[API] Frame processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detector-info')
def detector_info():
    """Get detector model information."""
    global detector
    
    if detector is None:
        init_detector()
    
    if detector == "failed":
        return jsonify({"error": "Detector not available"}), 500
    
    try:
        info = detector.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Video Filter API...")
    print(f"Detector config: {json.dumps(DETECTOR_CONFIG, indent=2)}")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)