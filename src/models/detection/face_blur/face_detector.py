"""
Face detection model for extracting blur regions.
Processes a single frame and returns polygons/rectangles to be blurred.
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import cv2
from insightface.app import FaceAnalysis


class FaceDetector:
    """
    Face detection model that identifies faces to be blurred while whitelisting enrolled faces.
    Returns rectangles/polygons that should be blurred instead of performing blur directly.
    """
    
    def __init__(self, 
                 embed_path: str = "whitelist/creator_embedding.json",
                 gpu_id: int = 0,
                 det_size: int = 960,
                 threshold: float = 0.35,
                 dilate_px: int = 12,
                 smooth_ms: int = 300,
                 lowlight_trigger: float = 60.0):
        """
        Initialize the face detector.
        
        Args:
            embed_path: Path to creator embedding JSON file
            gpu_id: GPU device ID (-1 for CPU)
            det_size: Detection model input size
            threshold: Cosine distance threshold for face matching
            dilate_px: Pixels to dilate detection boxes
            smooth_ms: Temporal smoothing duration in milliseconds
            lowlight_trigger: Mean pixel threshold to enable CLAHE enhancement
        """
        self.embed_path = embed_path
        self.threshold = threshold
        self.dilate_px = dilate_px
        self.smooth_ms = smooth_ms
        self.lowlight_trigger = lowlight_trigger
        
        # Load creator embedding
        self.creator_embedding = self._load_embedding(embed_path)
        
        # Initialize face analysis model
        self.ctx_id = self._pick_ctx_id(gpu_id)
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=self.ctx_id, det_size=(det_size, det_size))
        
        # Temporal tracking
        self.masks = []  # (expiry_time, box)
        self.vote_buf = deque(maxlen=3)  # temporal vote for whitelist decision
        self.panic_mode = False
        
        print(f"[FaceDetector] Initialized with ctx_id={self.ctx_id}")
    
    def _load_embedding(self, embed_path: str) -> Optional[np.ndarray]:
        """Load creator embedding from JSON file."""
        p = Path(embed_path)
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                emb = np.array(obj["embedding"], dtype=float)
                print(f"[FaceDetector] Loaded embedding: {p}")
                return emb
            except Exception as e:
                print(f"[FaceDetector][WARN] Failed to read embedding; will blur all faces. {e}")
        else:
            print(f"[FaceDetector][WARN] Embedding file not found: {embed_path}")
        return None
    
    def _pick_ctx_id(self, gpu_id: int) -> int:
        """Select appropriate context ID for face analysis."""
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return int(gpu_id)
            print("[FaceDetector][WARN] CUDAExecutionProvider not available; falling back to CPU.")
            return -1
        except Exception as e:
            print(f"[FaceDetector][WARN] onnxruntime not found or misconfigured ({e}); falling back to CPU.")
            return -1
    
    def reload_embedding(self) -> bool:
        """Reload creator embedding from disk."""
        try:
            obj = json.loads(Path(self.embed_path).read_text(encoding="utf-8"))
            self.creator_embedding = np.array(obj["embedding"], dtype=float)
            print("[FaceDetector] Reloaded embedding from disk.")
            return True
        except Exception as e:
            print(f"[FaceDetector][WARN] Reload failed: {e}")
            return False
    
    def set_panic_mode(self, panic: bool):
        """Toggle panic mode (blur entire frame)."""
        self.panic_mode = panic
        print(f"[FaceDetector] Panic mode: {'ON' if panic else 'OFF'}")
    
    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine distance between two embeddings."""
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return 1.0 - float(np.dot(a, b))
    
    def dilate_box(self, box: List[float], W: int, H: int) -> List[int]:
        """Dilate bounding box by specified pixels."""
        x1, y1, x2, y2 = box
        d = self.dilate_px
        return [
            max(0, int(x1 - d)),
            max(0, int(y1 - d)),
            min(W - 1, int(x2 + d)),
            min(H - 1, int(y2 + d))
        ]
    
    def enhance_lowlight(self, frame: np.ndarray) -> np.ndarray:
        """Enhance low-light frames using CLAHE."""
        if frame.mean() < self.lowlight_trigger:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)
        return frame
    
    def detect_faces_tta(self, frame_bgr: np.ndarray, big_size: int = 960, do_flip: bool = True) -> List[List[float]]:
        """Test-time augmentation for face detection."""
        H, W = frame_bgr.shape[:2]
        boxes = []
        
        # Regular detection
        for f in self.app.get(frame_bgr):
            boxes.append(list(map(float, f.bbox)))
        
        # Horizontal flip augmentation
        if do_flip:
            flipped = cv2.flip(frame_bgr, 1)
            for f in self.app.get(flipped):
                x1, y1, x2, y2 = map(float, f.bbox)
                boxes.append([W - x2, y1, W - x1, y2])
        
        # Scale augmentation
        if max(H, W) < big_size:
            scale = big_size / max(H, W)
            big = cv2.resize(frame_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
            for f in self.app.get(big):
                x1, y1, x2, y2 = map(float, f.bbox)
                boxes.append([x1 / scale, y1 / scale, x2 / scale, y2 / scale])
        
        return self._nms_union(boxes, thr=0.5)
    
    def _iou(self, a: List[float], b: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
        return inter / ua
    
    def _nms_union(self, boxes: List[List[float]], thr: float = 0.5) -> List[List[float]]:
        """Non-maximum suppression with union."""
        out = []
        for b in boxes:
            if not any(self._iou(b, o) > thr for o in out):
                out.append(b)
        return out
    
    def process_frame(self, frame: np.ndarray, frame_id: int, 
                     stride: int = 1, tta_every: int = 0) -> Tuple[int, List[List[int]]]:
        """
        Process a single frame and return rectangles to be blurred.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            stride: Process every N frames for detection
            tta_every: Apply TTA every N frames (0 to disable)
        
        Returns:
            Tuple of (frame_id, list of rectangles as [x1, y1, x2, y2])
        """
        H, W = frame.shape[:2]
        now = time.monotonic()
        
        # Enhance low-light frames
        frame_for_det = self.enhance_lowlight(frame)
        
        new_boxes = []
        
        if self.panic_mode:
            # Blur entire frame in panic mode
            new_boxes.append([0, 0, W - 1, H - 1])
        else:
            # Run detection on specified cadence
            if frame_id % max(1, stride) == 0:
                if tta_every > 0 and frame_id % tta_every == 0:
                    # Use TTA
                    face_boxes = self.detect_faces_tta(frame_for_det, big_size=960, do_flip=True)
                    faces = []  # TTA only returns boxes
                else:
                    # Regular detection
                    faces = self.app.get(frame_for_det)
                    face_boxes = [list(map(float, f.bbox)) for f in faces]
                
                # Decide per face: blur unless it's the creator
                for i, box in enumerate(face_boxes):
                    should_blur = True
                    
                    if (self.creator_embedding is not None and 
                        i < len(faces) and 
                        hasattr(faces[i], 'normed_embedding') and
                        faces[i].normed_embedding is not None):
                        
                        # Check if face matches creator
                        distance = self.cosine_distance(self.creator_embedding, faces[i].normed_embedding)
                        self.vote_buf.append(distance <= self.threshold)
                        
                        # Simple temporal voting: allow if majority of recent frames match
                        should_blur = sum(self.vote_buf) < 2
                    
                    if should_blur:
                        new_boxes.append(self.dilate_box(box, W, H))
        
        # Temporal smoothing: update mask list
        expiry = now + self.smooth_ms / 1000.0
        self.masks = [m for m in self.masks if m[0] > now] + [(expiry, b) for b in new_boxes]
        
        # Return all active masks as rectangles
        rectangles = [box for _, box in self.masks]
        
        return frame_id, rectangles
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_type": "face_detector",
            "embed_path": self.embed_path,
            "has_creator_embedding": self.creator_embedding is not None,
            "threshold": self.threshold,
            "dilate_px": self.dilate_px,
            "smooth_ms": self.smooth_ms,
            "panic_mode": self.panic_mode,
            "ctx_id": self.ctx_id
        }
