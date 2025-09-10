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
                 threshold: float = 0.45,  # More lenient threshold for better matching
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
        self.app = FaceAnalysis(name="buffalo_s")
        self.app.prepare(ctx_id=self.ctx_id, det_size=(det_size, det_size))
        
        # Temporal tracking
        self.masks = []  # (expiry_time, box)
        self.vote_buf = deque(maxlen=5)  # temporal vote for whitelist decision (longer buffer for stability)
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
    
    def set_dynamic_embedding(self, embedding: np.ndarray) -> bool:
        """Set creator embedding dynamically from memory (e.g., from enrollment)."""
        try:
            if embedding is not None and len(embedding) > 0:
                self.creator_embedding = np.array(embedding, dtype=float)
                print(f"[FaceDetector] ✅ Set dynamic embedding from memory (shape: {self.creator_embedding.shape})")
                # Reset vote buffer when embedding changes
                self.vote_buf.clear()
                return True
            else:
                print("[FaceDetector][WARN] Invalid embedding provided.")
                return False
        except Exception as e:
            print(f"[FaceDetector][WARN] Dynamic embedding failed: {e}")
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
            # Always run face detection, but only run expensive embedding comparison on stride intervals
            if tta_every > 0 and frame_id % tta_every == 0 and self.creator_embedding is None:
                # Use TTA only when no whitelist is needed and TTA is enabled
                face_boxes = self.detect_faces_tta(frame_for_det, big_size=960, do_flip=True)
                faces = []  # TTA only returns boxes
            else:
                # Regular detection (always when we have creator embedding)
                faces = self.app.get(frame_for_det)
                face_boxes = [list(map(float, f.bbox)) for f in faces]
            
            # Log detection results
            if self.creator_embedding is not None:
                print(f"[FaceDetector] Detected {len(face_boxes)} faces, {len(faces)} face objects, whitelist available: YES")
            else:
                print(f"[FaceDetector] Detected {len(face_boxes)} faces, no whitelist available - will blur all")
            
            # Find which faces (if any) match the creator by checking only the 3 largest faces
            # Always run embedding comparison when faces are detected (no caching/stride optimization)
            creator_matches = set()  # Indices of faces that match creator
            
            if self.creator_embedding is not None and len(faces) > 0:
                # Get the 3 largest faces for whitelist checking only
                face_areas = []
                for i, (face, box) in enumerate(zip(faces, face_boxes)):
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    face_areas.append((area, i, face, box))
                
                # Sort by area (largest first) and take top 3
                face_areas.sort(reverse=True)
                top_faces = face_areas[:3]
                
                print(f"[FaceDetector] Processing {len(top_faces)} largest faces for creator matching (out of {len(faces)} total)...")
                
                for area, i, face, box in top_faces:
                    try:
                        # Check if face has embedding
                        if not hasattr(face, 'normed_embedding'):
                            print(f"[FaceDetector] Face {i} has no normed_embedding attribute")
                            continue
                            
                        if face.normed_embedding is None:
                            print(f"[FaceDetector] Face {i} has None embedding")
                            continue
                            
                        # Validate embedding shape
                        if len(face.normed_embedding.shape) != 1 or face.normed_embedding.shape[0] == 0:
                            print(f"[FaceDetector] Face {i} has invalid embedding shape: {face.normed_embedding.shape}")
                            continue
                        
                        # Calculate distance
                        distance = self.cosine_distance(self.creator_embedding, face.normed_embedding)
                        match = distance <= self.threshold
                        
                        if match:
                            creator_matches.add(i)
                            print(f"[FaceDetector] Face {i} MATCHES creator (distance: {distance:.3f}, area: {area:.0f})")
                        else:
                            print(f"[FaceDetector] Face {i} does NOT match (distance: {distance:.3f}, area: {area:.0f})")
                            
                    except Exception as e:
                        print(f"[FaceDetector] Error processing face {i}: {e}")
                        continue
                
                # Update temporal voting with whether ANY face matched
                any_match = len(creator_matches) > 0
                self.vote_buf.append(any_match)
                
                # More stable voting: allow creator if at least 2 out of last 5 frames had a match
                # Special case: if we have fewer than 3 frames, be more lenient (50% threshold)
                vote_count = sum(self.vote_buf)
                buffer_size = len(self.vote_buf)
                
                if buffer_size < 3:
                    # Be more lenient for initial frames: require at least 50% matches
                    creator_allowed = vote_count >= max(1, buffer_size // 2)
                else:
                    # Standard voting: require at least 2 out of 5 frames
                    creator_allowed = vote_count >= 2
                
                print(f"[FaceDetector] Temporal voting: {vote_count}/{buffer_size} frames matched, threshold: {'lenient' if buffer_size < 3 else 'standard'}")
                
                print(f"[FaceDetector] Creator matches: {creator_matches}, allowed: {creator_allowed}")
            else:
                creator_allowed = False
                print(f"[FaceDetector] No creator embedding or no face embeddings available")
                
            # Decide per face: blur unless it's the creator
            for i, box in enumerate(face_boxes):
                # Simple logic: if creator is allowed and this face matches creator, don't blur
                if creator_allowed and i in creator_matches:
                    should_blur = False
                    print(f"[FaceDetector] Face {i} whitelisted (creator match)")
                else:
                    should_blur = True
                    print(f"[FaceDetector] Face {i} will be blurred")
                
                if should_blur:
                    new_boxes.append(self.dilate_box(box, W, H))
        
        # Temporal smoothing: update mask list
        expiry = now + self.smooth_ms / 1000.0
        self.masks = [m for m in self.masks if m[0] > now] + [(expiry, b) for b in new_boxes]
        
        # Return all active masks as rectangles
        rectangles = [box for _, box in self.masks]
        
        # Debug logging
        print(f"[FaceDetector] Frame {frame_id}: new_boxes={len(new_boxes)}, active_masks={len(rectangles)}")
        
        return frame_id, rectangles
    
    def extract_mouth_landmarks_buffalo(self, faces: List) -> List[Dict]:
        """
        Extract precise mouth coordinates using Buffalo's 68-point landmarks.
        Points 49-68 define outer and inner lips.
        """
        mouth_regions = []
        
        for i, face in enumerate(faces):
            try:
                # Check if Buffalo provides landmarks
                if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                    landmarks = face.landmark_3d_68
                    
                    # Mouth points: 49-68 (20 points total)
                    # Outer lip: points 49-60 (12 points)
                    # Inner lip: points 61-68 (8 points)
                    mouth_points = landmarks[48:68]  # 0-indexed: 48-67 = points 49-68
                    
                    # Get bounding box of mouth region
                    mouth_x = mouth_points[:, 0]
                    mouth_y = mouth_points[:, 1]
                    
                    x_min, x_max = np.min(mouth_x), np.max(mouth_x)
                    y_min, y_max = np.min(mouth_y), np.max(mouth_y)
                    
                    # Add padding for better coverage - increased for larger blur area
                    padding = 20  # pixels (increased from 8 to 20)
                    mouth_bbox = [
                        max(0, int(x_min - padding)),
                        max(0, int(y_min - padding)),
                        int(x_max + padding),
                        int(y_max + padding)
                    ]
                    
                    mouth_regions.append({
                        'face_index': i,
                        'bbox': mouth_bbox,
                        'landmarks': mouth_points.tolist(),  # Full landmark points
                        'confidence': float(face.det_score) if hasattr(face, 'det_score') else 1.0
                    })
                    
                    print(f"[FaceDetector] Extracted Buffalo mouth landmarks for face {i}: bbox={mouth_bbox}")
                    
                else:
                    # Fallback: estimate mouth region from face bbox
                    face_bbox = face.bbox
                    mouth_bbox = self._estimate_mouth_from_face(face_bbox)
                    
                    mouth_regions.append({
                        'face_index': i,
                        'bbox': mouth_bbox,
                        'landmarks': None,
                        'confidence': float(face.det_score) if hasattr(face, 'det_score') else 0.5
                    })
                    
                    print(f"[FaceDetector] Fallback mouth estimation for face {i}: bbox={mouth_bbox}")
                    
            except Exception as e:
                print(f"[FaceDetector] Error extracting mouth for face {i}: {e}")
                continue
        
        return mouth_regions

    def _estimate_mouth_from_face(self, face_bbox) -> List[int]:
        """Fallback mouth estimation when landmarks unavailable"""
        x1, y1, x2, y2 = map(int, face_bbox)
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Mouth is typically in bottom 1/3 of face, center 70% width (increased for larger area)
        mouth_x1 = x1 + int(face_width * 0.15)   # 15% from left (was 20%)
        mouth_x2 = x2 - int(face_width * 0.15)   # 15% from right (was 20%)
        mouth_y1 = y1 + int(face_height * 0.65)  # 65% from top (was 70%)
        mouth_y2 = y1 + int(face_height * 0.95)  # 95% from top (was 90%)
        
        return [mouth_x1, mouth_y1, mouth_x2, mouth_y2]

    def process_frame_with_mouth_landmarks(self, frame: np.ndarray, frame_id: int, 
                                         stride: int = 1) -> Tuple[int, List[List[int]], List[Dict]]:
        """
        Enhanced frame processing that returns both face blur regions and mouth landmarks.
        """
        H, W = frame.shape[:2]
        frame_for_det = self.enhance_lowlight(frame)
        
        if self.panic_mode:
            return frame_id, [[0, 0, W-1, H-1]], []
        
        # Get faces with full InsightFace data
        faces = self.app.get(frame_for_det)
        face_boxes = [list(map(float, f.bbox)) for f in faces]
        
        # Extract mouth landmarks for ALL detected faces
        mouth_regions = self.extract_mouth_landmarks_buffalo(faces)
        
        # Existing whitelist logic for face blurring
        now = time.monotonic()
        new_boxes = []
        creator_matches = set()
        
        if self.creator_embedding is not None and len(faces) > 0:
            # Get the 3 largest faces for whitelist checking only
            face_areas = []
            for i, (face, box) in enumerate(zip(faces, face_boxes)):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                face_areas.append((area, i, face, box))
            
            # Sort by area (largest first) and take top 3
            face_areas.sort(reverse=True)
            top_faces = face_areas[:3]
            
            print(f"[FaceDetector] Processing {len(top_faces)} largest faces for creator matching (out of {len(faces)} total)...")
            
            for area, i, face, box in top_faces:
                try:
                    # Check if face has embedding
                    if not hasattr(face, 'normed_embedding') or face.normed_embedding is None:
                        continue
                        
                    # Validate embedding shape
                    if len(face.normed_embedding.shape) != 1 or face.normed_embedding.shape[0] == 0:
                        continue
                    
                    # Calculate distance
                    distance = self.cosine_distance(self.creator_embedding, face.normed_embedding)
                    match = distance <= self.threshold
                    
                    if match:
                        creator_matches.add(i)
                        print(f"[FaceDetector] Face {i} MATCHES creator (distance: {distance:.3f}, area: {area:.0f})")
                    else:
                        print(f"[FaceDetector] Face {i} does NOT match (distance: {distance:.3f}, area: {area:.0f})")
                        
                except Exception as e:
                    print(f"[FaceDetector] Error processing face {i}: {e}")
                    continue
            
            # Update temporal voting with whether ANY face matched
            any_match = len(creator_matches) > 0
            self.vote_buf.append(any_match)
            
            # Voting logic
            vote_count = sum(self.vote_buf)
            buffer_size = len(self.vote_buf)
            
            if buffer_size < 3:
                creator_allowed = vote_count >= max(1, buffer_size // 2)
            else:
                creator_allowed = vote_count >= 2
            
            print(f"[FaceDetector] Temporal voting: {vote_count}/{buffer_size} frames matched")
        else:
            creator_allowed = False
            print(f"[FaceDetector] No creator embedding available")
        
        # Decide which faces to blur
        for i, box in enumerate(face_boxes):
            if creator_allowed and i in creator_matches:
                print(f"[FaceDetector] Face {i} whitelisted (creator match)")
            else:
                new_boxes.append(self.dilate_box(box, W, H))
                print(f"[FaceDetector] Face {i} will be blurred")
        
        # Temporal smoothing for face regions
        expiry = now + self.smooth_ms / 1000.0
        self.masks = [m for m in self.masks if m[0] > now] + [(expiry, b) for b in new_boxes]
        rectangles = [box for _, box in self.masks]
        
        print(f"[FaceDetector] Frame {frame_id}: faces={len(faces)}, mouths={len(mouth_regions)}, blur_regions={len(rectangles)}")
        
        return frame_id, rectangles, mouth_regions

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
