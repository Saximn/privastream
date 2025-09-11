"""
License plate detection model for extracting blur regions.
Processes a single frame and returns rectangles to be blurred.
"""
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Use Ultralytics YOLO
try:
    import torch
    from ultralytics import YOLO
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    raise RuntimeError("This script requires 'ultralytics' and 'torch'. Install with: pip install ultralytics torch") from e


class PlateDetector:
    """
    License plate detection model that identifies plates to be blurred.
    Returns rectangles that should be blurred instead of performing blur directly.
    """
    
    def __init__(self,
                 weights_path: str = "best.engine",
                 imgsz: int = 640,
                 conf_thresh: float = 0.35,
                 iou_thresh: float = 0.5,
                 pad: int = 4):
        """
        Initialize the plate detector.
        
        Args:
            weights_path: Path to YOLO model weights
            imgsz: Model input image size
            conf_thresh: Confidence threshold for detections
            iou_thresh: IoU threshold for NMS
            pad: Padding around detected boxes in pixels
        """
        self.weights_path = weights_path
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.pad = pad
        
        # Determine device
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        # Load YOLO model
        self.model = YOLO(weights_path)
        
        print(f"[PlateDetector] Initialized with device={self.device}")
        
    
    def clamp(self, v: float, lo: int, hi: int) -> int:
        """Clamp value between bounds."""
        return max(lo, min(hi, int(v)))
    
    def yolo_predict(self, frame_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
        """
        Run YOLO inference on a frame.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            List of detections as (x1, y1, x2, y2, confidence, class_id)
        """
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )
        
        boxes = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
                
            b = r.boxes
            xyxy = b.xyxy.detach().cpu().numpy()  # (N, 4)
            confs = b.conf.detach().cpu().numpy() if b.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
            clss = b.cls.detach().cpu().numpy() if b.cls is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
            
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                boxes.append((float(x1), float(y1), float(x2), float(y2), float(c), int(k)))
        
        return boxes
    
    def pad_box(self, box: Tuple[float, float, float, float], frame_shape: Tuple[int, int]) -> List[int]:
        """
        Add padding to detection box and clamp to frame boundaries.
        
        Args:
            box: Detection box as (x1, y1, x2, y2)
            frame_shape: Frame shape as (height, width)
            
        Returns:
            Padded box as [x1, y1, x2, y2]
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        
        x1 = self.clamp(x1 - self.pad, 0, w - 1)
        y1 = self.clamp(y1 - self.pad, 0, h - 1)
        x2 = self.clamp(x2 + self.pad, 0, w - 1)
        y2 = self.clamp(y2 + self.pad, 0, h - 1)
        
        return [x1, y1, x2, y2]
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[int, List[List[int]]]:
        """
        Process a single frame and return rectangles to be blurred.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            Tuple of (frame_id, list of rectangles as [x1, y1, x2, y2])
        """
        # Run YOLO detection
        detections = self.yolo_predict(frame)
        
        print(detections)
        # Convert to padded rectangles
        rectangles = []
        for x1, y1, x2, y2, conf, cls in detections:
            padded_box = self.pad_box((x1, y1, x2, y2), frame.shape)
            rectangles.append(padded_box)
        
        return frame_id, rectangles
    
    def process_frame_with_metadata(self, frame: np.ndarray, frame_id: int) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Process a single frame and return rectangles with detection metadata.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            Tuple of (frame_id, list of detection dictionaries)
        """
        # Run YOLO detection
        detections = self.yolo_predict(frame)
        
        # Convert to rectangles with metadata
        detection_data = []
        for x1, y1, x2, y2, conf, cls in detections:
            padded_box = self.pad_box((x1, y1, x2, y2), frame.shape)
            detection_data.append({
                "rectangle": padded_box,
                "confidence": conf,
                "class_id": cls,
                "original_box": [x1, y1, x2, y2]
            })
        
        return frame_id, detection_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_type": "plate_detector",
            "weights_path": self.weights_path,
            "imgsz": self.imgsz,
            "conf_thresh": self.conf_thresh,
            "iou_thresh": self.iou_thresh,
            "pad": self.pad,
            "device": str(self.device),
            "torch_available": TORCH_OK,
            "cuda_available": torch.cuda.is_available() if TORCH_OK else False
        }
