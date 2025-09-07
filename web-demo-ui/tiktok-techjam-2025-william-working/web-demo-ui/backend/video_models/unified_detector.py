"""
Unified interface for all blur detection models.
Demonstrates how to use the refactored face, PII, and plate detection models.
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2

# Add model directories to path
sys.path.append(str(Path(__file__).parent / "face_blur"))
sys.path.append(str(Path(__file__).parent / "pii_blur"))
sys.path.append(str(Path(__file__).parent / "plate_blur"))

try:
    from face_detector import FaceDetector
except ImportError as e:
    print(f"[WARN] FaceDetector not available: {e}")
    FaceDetector = None

try:
    from pii_detector import PIIDetector
except ImportError as e:
    print(f"[WARN] PIIDetector not available: {e}")
    PIIDetector = None

try:
    from plate_detector import PlateDetector
except ImportError as e:
    print(f"[WARN] PlateDetector not available: {e}")
    PlateDetector = None


class UnifiedBlurDetector:
    """
    Unified interface for all blur detection models.
    Processes frames and returns regions to be blurred from multiple models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified detector.
        
        Args:
            config: Configuration dictionary with model-specific settings
        """
        self.config = config or {}
        self.models = {}
        
        # Initialize enabled models
        self._init_models()
    
    def _init_models(self):
        """Initialize the detection models based on configuration."""
        # Face detector
        if self.config.get("enable_face", True) and FaceDetector is not None:
            try:
                face_config = self.config.get("face", {})
                self.models["face"] = FaceDetector(
                    embed_path=face_config.get("embed_path", "video_models/face_blur/whitelist/creator_embedding.json"),
                    gpu_id=face_config.get("gpu_id", 0),
                    det_size=face_config.get("det_size", 960),
                    threshold=face_config.get("threshold", 0.35),
                    dilate_px=face_config.get("dilate_px", 12),
                    smooth_ms=face_config.get("smooth_ms", 300)
                )
                print("[UnifiedDetector] Face detector initialized")
            except Exception as e:
                print(f"[UnifiedDetector][WARN] Face detector initialization failed: {e}")
        
        # PII detector
        if self.config.get("enable_pii", True) and PIIDetector is not None:
            try:
                pii_config = self.config.get("pii", {})
                self.models["pii"] = PIIDetector(
                    classifier_path=pii_config.get("classifier_path", os.path.join(os.path.dirname(__file__), "pii_blur/pii_clf.joblib")),
                    conf_thresh=pii_config.get("conf_thresh", 0.35),
                    min_area=pii_config.get("min_area", 80),
                    K_confirm=pii_config.get("K_confirm", 2),
                    K_hold=pii_config.get("K_hold", 8)
                )
                print("[UnifiedDetector] PII detector initialized")
            except Exception as e:
                print(f"[UnifiedDetector][WARN] PII detector initialization failed: {e}")
        
        # Plate detector
        if self.config.get("enable_plate", True) and PlateDetector is not None:
            try:
                plate_config = self.config.get("plate", {})
                self.models["plate"] = PlateDetector(
                    weights_path=plate_config.get("weights_path", "models/plate_blur/best.pt"),
                    imgsz=plate_config.get("imgsz", 960),
                    conf_thresh=plate_config.get("conf_thresh", 0.25),
                    iou_thresh=plate_config.get("iou_thresh", 0.5),
                    pad=plate_config.get("pad", 4)
                )
                print("[UnifiedDetector] Plate detector initialized")
            except Exception as e:
                print(f"[UnifiedDetector][WARN] Plate detector initialization failed: {e}")
        
        print(f"[UnifiedDetector] Initialized with {len(self.models)} models: {list(self.models.keys())}")
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """
        Process a frame with all enabled models.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            Dictionary containing results from all models
        """
        results = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "models": {}
        }
        
        # Process with face detector
        if "face" in self.models:
            try:
                face_frame_id, face_rectangles = self.models["face"].process_frame(frame, frame_id)
                results["models"]["face"] = {
                    "frame_id": face_frame_id,
                    "rectangles": face_rectangles,
                    "count": len(face_rectangles)
                }
            except Exception as e:
                print(f"[UnifiedDetector][ERROR] Face detection failed: {e}")
                results["models"]["face"] = {"error": str(e)}
        
        # Process with PII detector
        if "pii" in self.models:
            try:
                pii_frame_id, pii_polygons = self.models["pii"].process_frame(frame, frame_id)
                results["models"]["pii"] = {
                    "frame_id": pii_frame_id,
                    "polygons": pii_polygons,
                    "count": len(pii_polygons)
                }
                print(f"[UnifiedTester] PII detections: {len(pii_polygons)}")  # Debug print

            except Exception as e:
                print(f"[UnifiedDetector][ERROR] PII detection failed: {e}")
                results["models"]["pii"] = {"error": str(e)}

        
        # Process with plate detector
        if "plate" in self.models:
            try:
                plate_frame_id, plate_rectangles = self.models["plate"].process_frame(frame, frame_id)
                results["models"]["plate"] = {
                    "frame_id": plate_frame_id,
                    "rectangles": plate_rectangles,
                    "count": len(plate_rectangles)
                }
            except Exception as e:
                print(f"[UnifiedDetector][ERROR] Plate detection failed: {e}")
                results["models"]["plate"] = {"error": str(e)}
        
        return results
    
    def get_all_rectangles(self, results: Dict[str, Any]) -> List[List[int]]:
        """
        Extract all rectangles from detection results.
        
        Args:
            results: Results from process_frame
            
        Returns:
            Combined list of all rectangles [x1, y1, x2, y2]
        """
        rectangles = []
        
        # Face rectangles
        face_data = results.get("models", {}).get("face", {})
        if "rectangles" in face_data:
            rectangles.extend(face_data["rectangles"])
        
        # Plate rectangles
        plate_data = results.get("models", {}).get("plate", {})
        if "rectangles" in plate_data:
            rectangles.extend(plate_data["rectangles"])
        
        return rectangles
    
    def get_all_polygons(self, results: Dict[str, Any]) -> List[np.ndarray]:
        """
        Extract all polygons from detection results.
        
        Args:
            results: Results from process_frame
            
        Returns:
            Combined list of all polygons
        """
        polygons = []
        
        # PII polygons
        pii_data = results.get("models", {}).get("pii", {})
        if "polygons" in pii_data:
            polygons.extend(pii_data["polygons"])
        
        return polygons
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        info = {
            "unified_detector": {
                "enabled_models": list(self.models.keys()),
                "model_count": len(self.models)
            }
        }
        
        for name, model in self.models.items():
            if hasattr(model, "get_model_info"):
                info[name] = model.get_model_info()
        
        return info


def demo_unified_detector():
    """Demonstration of the unified detector."""
    # Configuration for all models
    config = {
        "enable_face": True,
        "enable_pii": True,
        "enable_plate": True,
        "face": {
            "embed_path": "video_models/face_blur/whitelist/creator_embedding.json",
            "threshold": 0.35,
            "dilate_px": 12
        },
        "pii": {
            "classifier_path": "video_models/pii_blur/pii_clf.joblib",
            "conf_thresh": 0.35
        },
        "plate": {
            "weights_path": "video_models/plate_blur/best.pt",
            "conf_thresh": 0.25
        }
    }
    
    # Initialize detector
    detector = UnifiedBlurDetector(config)
    
    # Print model information
    model_info = detector.get_model_info()
    print("=== Model Information ===")
    for key, value in model_info.items():
        print(f"{key}: {value}")
    
    # Demo with webcam
    print("\n=== Starting webcam demo ===")
    print("Press 'q' to quit, 's' to save current frame results")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    
    frame_id = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = detector.process_frame(frame, frame_id)
            
            # Visualize results
            vis_frame = frame.copy()
            
            # Draw face rectangles (red)
            face_data = results.get("models", {}).get("face", {})
            if "rectangles" in face_data:
                for rect in face_data["rectangles"]:
                    x1, y1, x2, y2 = rect
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(vis_frame, "FACE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw PII polygons (green)
            pii_data = results.get("models", {}).get("pii", {})
            if "polygons" in pii_data:
                for poly in pii_data["polygons"]:
                    cv2.polylines(vis_frame, [poly], True, (0, 255, 0), 2)
                    if len(poly) > 0:
                        cv2.putText(vis_frame, "PII", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw plate rectangles (blue)
            plate_data = results.get("models", {}).get("plate", {})
            if "rectangles" in plate_data:
                for rect in plate_data["rectangles"]:
                    x1, y1, x2, y2 = rect
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis_frame, "PLATE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add info text
            total_detections = sum([
                face_data.get("count", 0),
                pii_data.get("count", 0),
                plate_data.get("count", 0)
            ])
            
            info_text = f"Frame: {frame_id}, Detections: {total_detections}"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Unified Blur Detector Demo", vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current results
                filename = f"detection_results_frame_{frame_id}.txt"
                with open(filename, 'w') as f:
                    f.write(f"Frame {frame_id} Detection Results:\n")
                    f.write(f"Face rectangles: {face_data.get('rectangles', [])}\n")
                    f.write(f"PII polygons: {len(pii_data.get('polygons', []))} polygons\n")
                    f.write(f"Plate rectangles: {plate_data.get('rectangles', [])}\n")
                print(f"Saved results to {filename}")
            
            frame_id += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_unified_detector()
