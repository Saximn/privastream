"""
Refactored unified interface for all blur detection models.
Uses centralized configuration and proper error handling.
"""
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2

from src.core.config import ModelConfig, default_config
from src.core.logging import (
    safe_import, handle_model_errors, ModelInitializationError, 
    ModelInferenceError, logger
)

# Import model classes safely
FaceDetector = safe_import("src.models.detection.face_blur.face_detector", "FaceDetector", logger)
PIIDetector = safe_import("src.models.detection.pii_blur.pii_detector", "PIIDetector", logger)
PlateDetector = safe_import("src.models.detection.plate_blur.plate_detector", "PlateDetector", logger)


class ModelFactory:
    """Factory class for creating detection models with proper configuration"""
    
    @staticmethod
    def create_face_detector(config: ModelConfig) -> Optional[Any]:
        """Create face detector with configuration"""
        if FaceDetector is None:
            logger.warning("FaceDetector not available")
            return None
        
        try:
            return FaceDetector(
                embed_path=str(config.face_embed_full_path),
                gpu_id=config.GPU_ID if config.USE_GPU else -1,
                det_size=config.DETECTION_SIZE,
                threshold=config.FACE_THRESHOLD,
                dilate_px=config.DILATION_PIXELS,
                smooth_ms=config.SMOOTH_DURATION_MS
            )
        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {e}")
            raise ModelInitializationError(f"Face detector initialization failed: {e}")
    
    @staticmethod
    def create_pii_detector(config: ModelConfig) -> Optional[Any]:
        """Create PII detector with configuration"""
        if PIIDetector is None:
            logger.warning("PIIDetector not available")
            return None
        
        try:
            return PIIDetector(
                classifier_path=str(config.pii_classifier_full_path),
                conf_thresh=config.PII_CONFIDENCE_THRESHOLD,
                K_confirm=config.CONFIRMATION_FRAMES,
                K_hold=config.HOLD_FRAMES,
                min_area=80
            )
        except Exception as e:
            logger.error(f"Failed to initialize PIIDetector: {e}")
            raise ModelInitializationError(f"PII detector initialization failed: {e}")
    
    @staticmethod 
    def create_plate_detector(config: ModelConfig) -> Optional[Any]:
        """Create plate detector with configuration"""
        if PlateDetector is None:
            logger.warning("PlateDetector not available")
            return None
        
        try:
            return PlateDetector(
                weights_path=str(config.plate_model_full_path),
                conf_thresh=config.PLATE_CONFIDENCE_THRESHOLD,
                device="cuda" if config.USE_GPU else "cpu",
                K_confirm=config.CONFIRMATION_FRAMES,
                K_hold=config.HOLD_FRAMES
            )
        except Exception as e:
            logger.error(f"Failed to initialize PlateDetector: {e}")
            raise ModelInitializationError(f"Plate detector initialization failed: {e}")


class UnifiedBlurDetector:
    """
    Unified interface for all blur detection models.
    Processes frames and returns regions to be blurred from multiple models.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the unified detector.
        
        Args:
            config: ModelConfig instance with detection settings
        """
        self.config = config if isinstance(config, ModelConfig) else default_config
        self.models = {}
        self.enabled_models = ["face", "pii", "plate"]
        
        logger.info("Initializing UnifiedBlurDetector")
        self._init_models()
    
    def _init_models(self):
        """Initialize all available detection models"""
        factory = ModelFactory()
        
        # Initialize face detector
        if "face" in self.enabled_models:
            try:
                self.models["face"] = factory.create_face_detector(self.config)
                if self.models["face"]:
                    logger.info("Face detector initialized successfully")
            except ModelInitializationError as e:
                logger.warning(f"Face detector unavailable: {e}")
        
        # Initialize PII detector  
        if "pii" in self.enabled_models:
            try:
                self.models["pii"] = factory.create_pii_detector(self.config)
                if self.models["pii"]:
                    logger.info("PII detector initialized successfully")
            except ModelInitializationError as e:
                logger.warning(f"PII detector unavailable: {e}")
        
        # Initialize plate detector
        if "plate" in self.enabled_models:
            try:
                self.models["plate"] = factory.create_plate_detector(self.config)
                if self.models["plate"]:
                    logger.info("Plate detector initialized successfully")
            except ModelInitializationError as e:
                logger.warning(f"Plate detector unavailable: {e}")
        
        available_models = [name for name, model in self.models.items() if model is not None]
        logger.info(f"Available models: {available_models}")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        stride: int = 1,
        room_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
                face_frame_id, face_rectangles = self.models["face"].process_frame(
                    frame, frame_id, stride=stride
                )
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
                    "rectangles": self._polygons_to_rectangles(pii_polygons),
                    "count": len(pii_polygons)
                }
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

    def update_face_embedding(self, embedding: np.ndarray):
        """Update the active face whitelist embedding when supported."""
        face_model = self.models.get("face")
        if face_model is not None and hasattr(face_model, "creator_embedding"):
            face_model.creator_embedding = embedding

    def cleanup_room(self, room_id: str):
        """Compatibility hook for room-scoped detectors."""
        for model in self.models.values():
            cleanup = getattr(model, "cleanup_room", None)
            if callable(cleanup):
                cleanup(room_id)

    def process_frame_with_mouth_landmarks(
        self,
        frame: np.ndarray,
        frame_id: int,
        stride: int = 1,
        room_id: Optional[str] = None,
    ):
        """Compatibility API returning face boxes and mouth landmarks if present."""
        results = self.process_frame(frame, frame_id, stride=stride, room_id=room_id)
        face_regions = results.get("models", {}).get("face", {}).get("rectangles", [])
        mouth_regions = results.get("models", {}).get("mouth", {}).get("rectangles", [])
        return frame_id, face_regions, mouth_regions

    def _polygons_to_rectangles(self, polygons: List[np.ndarray]) -> List[List[int]]:
        rectangles: List[List[int]] = []
        for poly in polygons:
            try:
                arr = np.asarray(poly)
                if arr.size == 0:
                    continue
                xs = arr[:, 0]
                ys = arr[:, 1]
                rectangles.append([
                    int(np.min(xs)),
                    int(np.min(ys)),
                    int(np.max(xs)),
                    int(np.max(ys)),
                ])
            except Exception:
                continue
        return rectangles
    
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
            "embed_path": "models/face_blur/whitelist/creator_embedding.json",
            "threshold": 0.35,
            "dilate_px": 12
        },
        "pii": {
            "classifier_path": "models/pii_blur/pii_clf.joblib",
            "conf_thresh": 0.35
        },
        "plate": {
            "weights_path": "models/plate_blur/best.pt",
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
