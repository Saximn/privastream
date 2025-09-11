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

import asyncio
import concurrent.futures
from threading import Lock

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
        self.model_locks = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
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
                self.model_locks["face"] = Lock()
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
                self.model_locks["pii"] = Lock()
                print("[UnifiedDetector] PII detector initialized")
            except Exception as e:
                print(f"[UnifiedDetector][WARN] PII detector initialization failed: {e}")
        
        # Plate detector
        if self.config.get("enable_plate", True) and PlateDetector is not None:
            try:
                plate_config = self.config.get("plate", {})
                self.models["plate"] = PlateDetector(
                    weights_path=plate_config.get("weights_path", os.path.join(os.path.dirname(__file__), "plate_blur/best.pt")),
                    imgsz=plate_config.get("imgsz", 960),
                    conf_thresh=plate_config.get("conf_thresh", 0.35),
                    iou_thresh=plate_config.get("iou_thresh", 0.5),
                    pad=plate_config.get("pad", 4)
                )
                self.model_locks["plate"] = Lock()
                print("[UnifiedDetector] Plate detector initialized")
            except Exception as e:
                print(f"[UnifiedDetector][WARN] Plate detector initialization failed: {e}")
        
        print(f"[UnifiedDetector] Initialized with {len(self.models)} models: {list(self.models.keys())}")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            self.process_frame(dummy_frame, frame_id=-1)
        print("[UnifiedDetector] Engine warmed up with dummy frame")
              # STAGE 2: Real-content warm-up (NEW - this is what you need)
        self._advanced_warmup()

    def _advanced_warmup(self):
        """Advanced warm-up with realistic content and varied sizes."""
        print("[UnifiedDetector] 🔥 Starting advanced warm-up with realistic content...")

        # Create varied frames with actual content
        warmup_frames = []

        # Different sizes (common webcam/phone resolutions)
        sizes = [(480, 640), (720, 1280), (1080, 1920), (480, 480)]

        for i, (h, w) in enumerate(sizes):
            # Create frame with realistic content
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

            # Add some geometric shapes (simulates faces/objects)
            cv2.rectangle(frame, (w//4, h//4), (w//2, h//2), (255, 255, 255), -1)
            cv2.circle(frame, (3*w//4, h//4), 50, (128, 128, 128), -1)

            # Add some text-like patterns (simulates PII)
            cv2.putText(frame, "ABC123", (w//2, 3*h//4), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),3)
            warmup_frames.append(frame)

        # Process varied frames multiple times
        for round_num in range(3):  # 3 rounds
            for i, frame in enumerate(warmup_frames):
                start_time = time.time()
                results = self.process_frame(frame, frame_id=f"warmup_{round_num}_{i}")
                warmup_time = time.time() - start_time
                print(f"[UnifiedDetector] 🔥 Warmup round {round_num+1}/3, frame {i+1}/4:{warmup_time:.3f}s")
        print("[UnifiedDetector] ✅ Advanced warm-up complete - models should be fully optimized")
    
    async def process_frame_async(self, frame: np.ndarray, frame_id: int, stride: int = 1, tta_every: int = 0, room_id: str = None) -> Dict[str, Any]:
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
        
        tasks = []

        if "face" in self.models:
            tasks.append(self._process_face_model_async("face", frame, frame_id, stride, tta_every, room_id=room_id))

        if "pii" in self.models:
            tasks.append(self._process_pii_model_async("pii", frame, frame_id, stride, tta_every, room_id=room_id))

        if "plate" in self.models:
            tasks.append(self._process_plate_model_async("plate", frame, frame_id, stride, tta_every))

        model_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in model_results:
            if isinstance(result, Exception):
                print(f"[UnifiedDetector][ERROR] Model processing failed: {result}")
                continue
            if result:
                model_type, model_data = result
                results["models"][model_type] = model_data

        print(f"[UnifiedDetector] Frame {frame_id} processed with results: {results}")
        return results

    async def _process_face_model_async(self, model_name: str, frame: np.ndarray, frame_id: int, stride: int, tta_every: int, room_id: str = None) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_event_loop()

        def face_task():
            with self.model_locks["face"]:
                start_time = time.time()
                face_frame_id, face_rectangles = self.models["face"].process_frame(frame, frame_id, stride, tta_every, room_id=room_id)
                end_time = time.time()
                print(f"[UnifiedDetector] Face detection time: {end_time - start_time:.5f}s for frame {frame_id}")
                return "face", {
                    "frame_id": face_frame_id,
                    "rectangles": face_rectangles,
                    "count": len(face_rectangles)
                }
            
        return await loop.run_in_executor(self.executor, face_task)
    
    async def _process_pii_model_async(self, model_name: str, frame: np.ndarray, frame_id: int, stride: int, tta_every: int, room_id: str = None) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_event_loop()

        def pii_task():
            with self.model_locks["pii"]:
                start_time = time.time()
                pii_frame_id, pii_rectangles = self.models["pii"].process_frame(frame, frame_id, room_id=room_id)
                end_time = time.time()
                print(f"[UnifiedDetector] PII detection time: {end_time - start_time:.5f}s for frame {frame_id}")
                return "pii", {
                    "frame_id": pii_frame_id,
                    "rectangles": pii_rectangles,
                    "count": len(pii_rectangles)
                }
            
        return await loop.run_in_executor(self.executor, pii_task)
    
    async def _process_plate_model_async(self, model_name: str, frame: np.ndarray, frame_id: int, stride: int, tta_every: int) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_event_loop()

        def plate_task():
            with self.model_locks["plate"]:
                start_time = time.time()
                plate_frame_id, plate_rectangles = self.models["plate"].process_frame(frame, frame_id)
                end_time = time.time()
                print(f"[UnifiedDetector] Plate detection time: {end_time - start_time:.5f}s for frame {frame_id}")
                return "plate", {
                    "frame_id": plate_frame_id,
                    "rectangles": plate_rectangles,
                    "count": len(plate_rectangles)
                }    
        
        return await loop.run_in_executor(self.executor, plate_task)
    
    def process_frame(self, frame: np.ndarray, frame_id: int, stride: int = 1, tta_every: int = 0, room_id: str = None) -> Dict[str, Any]:
        """
        Synchronous wrapper to process a frame with all enabled models.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            Dictionary containing results from all models
        """
        print("[UnifiedDetector] Processing frame", frame_id)
        return asyncio.run(self.process_frame_async(frame, frame_id, stride, tta_every))
    
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
        
        # PII rectangles
        pii_data = results.get("models", {}).get("pii", {})
        if "rectangles" in pii_data:
            rectangles.extend(pii_data["rectangles"])
        
        # Plate rectangles
        plate_data = results.get("models", {}).get("plate", {})
        if "rectangles" in plate_data:
            rectangles.extend(plate_data["rectangles"])
        
        return rectangles
    
    def get_all_polygons(self, results: Dict[str, Any]) -> List[np.ndarray]:
        """
        Extract all polygons from detection results.
        Note: PII detector now returns rectangles, not polygons.
        
        Args:
            results: Results from process_frame
            
        Returns:
            Combined list of all polygons (empty list since PII now uses rectangles)
        """
        polygons = []
        
        # PII now returns rectangles, not polygons
        # If polygons are needed, convert rectangles to polygons:
        # pii_data = results.get("models", {}).get("pii", {})
        # if "rectangles" in pii_data:
        #     for rect in pii_data["rectangles"]:
        #         x1, y1, x2, y2 = rect
        #         poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        #         polygons.append(poly)
        
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
    
    def update_face_embedding(self, embedding: np.ndarray) -> bool:
        """
        Update the face detector's dynamic embedding for whitelisting.
        
        Args:
            embedding: Face embedding to use for whitelisting
            
        Returns:
            True if successful, False otherwise
        """
        if "face" in self.models:
            try:
                return self.models["face"].set_dynamic_embedding(embedding)
            except Exception as e:
                print(f"[UnifiedDetector][ERROR] Failed to update face embedding: {e}")
                return False
        else:
            print("[UnifiedDetector][WARN] Face detector not available for embedding update")
            return False
    
    def cleanup_room(self, room_id: str) -> bool:
        """
        Clean up room-specific data from all models.
        
        Args:
            room_id: Room ID to clean up
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if "face" in self.models:
                self.models["face"].cleanup_room(room_id)
            
            if "pii" in self.models:
                self.models["pii"].cleanup_room(room_id)
            
            print(f"[UnifiedDetector] Cleaned up data for room: {room_id}")
            return True
        except Exception as e:
            print(f"[UnifiedDetector][ERROR] Failed to cleanup room {room_id}: {e}")
            return False

    def process_frame_with_mouth_landmarks(self, frame: np.ndarray, frame_id: int, 
                                         stride: int = 1, room_id: str = None) -> Tuple[int, List[List[int]], List[Dict]]:
        """
        Enhanced frame processing that returns both face blur regions and mouth landmarks.
        Routes to the face detector's mouth landmark extraction method.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            stride: Process every N frames for detection
            
        Returns:
            Tuple of (frame_id, face_blur_regions, mouth_regions)
        """
        if "face" in self.models:
            try:
                return self.models["face"].process_frame_with_mouth_landmarks(frame, frame_id, stride, room_id=room_id)
            except Exception as e:
                print(f"[UnifiedDetector][ERROR] Mouth landmark processing failed: {e}")
                # Fallback: return regular face processing with empty mouths
                regular_result = self.models["face"].process_frame(frame, frame_id, stride, room_id=room_id)
                return regular_result[0], regular_result[1], []  # frame_id, rectangles, empty mouths
        else:
            print("[UnifiedDetector][WARN] Face detector not available for mouth landmarks")
            return frame_id, [], []


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
            
            # Draw PII rectangles (green)
            pii_data = results.get("models", {}).get("pii", {})
            if "rectangles" in pii_data:
                for rect in pii_data["rectangles"]:
                    x1, y1, x2, y2 = rect
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, "PII", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
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
                    f.write(f"PII rectangles: {pii_data.get('rectangles', [])}\n")
                    f.write(f"Plate rectangles: {plate_data.get('rectangles', [])}\n")
                print(f"Saved results to {filename}")
            
            frame_id += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_unified_detector()
