"""
PII (Personally Identifiable Information) detection model for extracting blur regions.
Processes a single frame and returns polygons/rectangles to be blurred.
"""
import os
import sys
import re
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

# Try GPU via PyTorch
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

DEVICE = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"


class OCRPipeline:
    """Unified OCR interface supporting docTR and EasyOCR."""
    
    def __init__(self, det_arch: str = "fast_small", reco_arch: str = "crnn_mobilenet_v3_small"):
        """
        Initialize OCR pipeline.
        
        Args:
            det_arch: Detection architecture for docTR
            reco_arch: Recognition architecture for docTR
        """
        self.kind = "doctr"
        try:
            from doctr.models import ocr_predictor
            self._ocr = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
            if TORCH_OK:
                self._ocr = self._ocr.to(DEVICE)
            self._ocr = self._ocr.eval()
        except Exception as e:
            print(f"[OCRPipeline][WARN] docTR not available -> falling back to EasyOCR: {e}", file=sys.stderr)
            try:
                import easyocr
                self.kind = "easyocr"
                self._reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))
            except Exception as e2:
                print(f"[OCRPipeline][ERROR] EasyOCR not available either: {e2}", file=sys.stderr)
                raise

    def infer(self, img_bgr: np.ndarray) -> dict:
        """Run OCR on image and return structured results."""
        if self.kind == "doctr":
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            doc = self._ocr([img_rgb])
            return doc.export()
        else:
            H, W = img_bgr.shape[:2]
            results = self._reader.readtext(img_bgr, detail=1, paragraph=False)
            words = []
            for box, text, conf in results:
                xs = [p[0] / W for p in box]
                ys = [p[1] / H for p in box]
                x0, x1 = float(min(xs)), float(max(xs))
                y0, y1 = float(min(ys)), float(max(ys))
                words.append({
                    "value": text,
                    "confidence": float(conf),
                    "geometry": ((x0, y0), (x1, y1))
                })
            return {
                "pages": [{
                    "blocks": [{
                        "lines": [{
                            "words": words,
                            "geometry": ((0.0, 0.0), (1.0, 1.0))
                        }]
                    }]
                }]
            }


class PIIDecider:
    """Hybrid PII decision engine using rules and optional ML classifier."""
    
    def __init__(self, classifier_path: Optional[str] = None, threshold: Optional[float] = None):
        """
        Initialize PII decision engine.
        
        Args:
            classifier_path: Path to joblib classifier bundle
            threshold: ML classifier threshold (overrides saved threshold)
        """
        # Address detection rules (extend per locale)
        street_tokens = r"(Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Terrace|Ter|Court|Ct|Crescent|Cres|Place|Pl|Highway|Hwy|Expressway|Expwy|Jalan|Jln|Lorong|Lor)"
        unit = r"#\s?\d{1,3}-\d{1,4}"
        postal_sg = r"\b(?:S\s*)?\d{6}\b"
        house_no = r"\b(?:Blk|Block)?\s?\d{1,5}[A-Z]?\b"
        composed = rf"{house_no}.*\b{street_tokens}\b"
        
        self._regexes = [re.compile(p, re.I) for p in [street_tokens, unit, postal_sg, composed]]
        
        # Optional ML classifier
        self._vec = None
        self._clf = None
        self._thr = None
        
        if classifier_path is None:
            classifier_path = "pii_clf.joblib"
        
        if os.path.exists(classifier_path):
            try:
                import joblib
                bundle = joblib.load(classifier_path)
                self._vec = bundle.get("vec", None)
                self._clf = bundle.get("clf", None)
                self._thr = float(bundle.get("thr", 0.5)) if threshold is None else float(threshold)
                print(f"[PIIDecider] Loaded classifier from {classifier_path} (thr={self._thr:.3f})")
            except Exception as e:
                print(f"[PIIDecider][WARN] Failed to load classifier at {classifier_path}: {e}", file=sys.stderr)
        else:
            print(f"[PIIDecider][INFO] No classifier found at {classifier_path}; using rules only.", file=sys.stderr)

    def _rule_is_pii(self, text: str) -> bool:
        """Check if text matches PII rules."""
        t = (text or "").strip()
        if not t:
            return False
        return any(rx.search(t) for rx in self._regexes)

    def _ml_prob(self, text: str) -> float:
        """Get ML classifier probability for PII."""
        if self._vec is None or self._clf is None or not text:
            return 0.0
        try:
            X = self._vec.transform([text])
            return float(self._clf.predict_proba(X)[0, 1])
        except Exception:
            return 0.0

    def decide(self, text: str, conf: float, conf_thresh: float = 0.35) -> bool:
        """
        Decide if text should be blurred.
        
        Args:
            text: Detected text
            conf: OCR confidence
            conf_thresh: Minimum confidence threshold
            
        Returns:
            True if text should be blurred
        """
        if not text or conf < conf_thresh:
            return False
        if self._rule_is_pii(text):
            return True
        if self._clf is not None:
            return self._ml_prob(text) >= self._thr
        return False


class Hysteresis:
    """Temporal stabilization for polygon tracking."""
    
    def __init__(self, iou_thresh: float = 0.3, K_confirm: int = 2, K_hold: int = 8):
        """
        Initialize hysteresis tracker.
        
        Args:
            iou_thresh: IoU threshold for polygon matching
            K_confirm: Frames to confirm before activating
            K_hold: Frames to hold after last detection
        """
        self.iou_thresh = iou_thresh
        self.K_confirm = K_confirm
        self.K_hold = K_hold
        self.tracks = {}
        self.next_id = 1
        self.frame = 0

    def iou(self, a: List[int], b: List[int]) -> float:
        """Calculate IoU between two bounding boxes."""
        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def aabb(self, poly: np.ndarray) -> List[int]:
        """Get axis-aligned bounding box from polygon."""
        xs, ys = poly[:, 0], poly[:, 1]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def update(self, polys: List[np.ndarray]) -> List[Tuple[np.ndarray, bool]]:
        """
        Update tracker with new polygons.
        
        Args:
            polys: List of detected polygons
            
        Returns:
            List of (polygon, is_active) tuples
        """
        self.frame += 1
        used = [False] * len(polys)
        
        # Match to existing tracks
        for tid, t in list(self.tracks.items()):
            ta = self.aabb(t["poly"])
            best, bj = 0.0, -1
            
            for j, p in enumerate(polys):
                if used[j]:
                    continue
                ov = self.iou(ta, self.aabb(p))
                if ov > best:
                    best, bj = ov, j
                    
            if best >= self.iou_thresh and bj >= 0:
                t["poly"] = polys[bj]
                t["hits"] += 1
                t["last"] = self.frame
                if not t["active"] and t["hits"] >= self.K_confirm:
                    t["active"] = True
                used[bj] = True
                
            if t["active"] and (self.frame - t["last"]) > self.K_hold:
                t["active"] = False

        # Create new tracks
        for j, p in enumerate(polys):
            if not used[j]:
                self.tracks[self.next_id] = {
                    "poly": p,
                    "hits": 1,
                    "active": (1 >= self.K_confirm),
                    "last": self.frame
                }
                self.next_id += 1

        # Garbage collect old tracks
        drop = [tid for tid, t in self.tracks.items() 
                if (self.frame - t["last"]) > (3 * self.K_hold)]
        for tid in drop:
            self.tracks.pop(tid, None)

        return [(t["poly"], t["active"]) for t in self.tracks.values()]


class PIIDetector:
    """
    PII detection model that identifies text regions containing personally identifiable information.
    Returns polygons that should be blurred instead of performing blur directly.
    """
    
    def __init__(self,
                 classifier_path: Optional[str] = None,
                 conf_thresh: float = 0.35,
                 min_area: int = 80,
                 K_confirm: int = 2,
                 K_hold: int = 8,
                 det_arch: str = "fast_small",
                 reco_arch: str = "crnn_mobilenet_v3_small"):
        """
        Initialize PII detector.
        
        Args:
            classifier_path: Path to ML classifier joblib file
            conf_thresh: OCR confidence threshold
            min_area: Minimum polygon area to consider
            K_confirm: Frames to confirm before blurring
            K_hold: Frames to hold blur after last detection
            det_arch: OCR detection architecture
            reco_arch: OCR recognition architecture
        """
        self.conf_thresh = conf_thresh
        self.min_area = min_area
        self.K_confirm = K_confirm
        self.K_hold = K_hold
        
        # Initialize OCR pipeline
        self.ocr = OCRPipeline(det_arch=det_arch, reco_arch=reco_arch)
        
        # Initialize PII decision engine
        self.decider = PIIDecider(classifier_path=classifier_path)
        
        # Per-room temporal stabilization (fixes cross-room contamination)
        self.room_stabilizers = {}  # roomId -> Hysteresis instance
        
        print("[PIIDetector] Initialized with per-room temporal isolation")
    
    def _get_room_stabilizer(self, room_id: str) -> 'Hysteresis':
        """Get or create temporal stabilizer for specific room."""
        if room_id not in self.room_stabilizers:
            self.room_stabilizers[room_id] = Hysteresis(
                iou_thresh=0.3, 
                K_confirm=self.K_confirm, 
                K_hold=self.K_hold
            )
        return self.room_stabilizers[room_id]
    
    def cleanup_room(self, room_id: str):
        """Clean up room-specific data when room closes."""
        if room_id in self.room_stabilizers:
            del self.room_stabilizers[room_id]
        print(f"[PIIDetector] Cleaned up temporal data for room: {room_id}")
    
    def rect_from_box_norm(self, box: Tuple[Tuple[float, float], Tuple[float, float]], 
                          W: int, H: int) -> List[int]:
        """Convert normalized box to rectangle coordinates [x1, y1, x2, y2]."""
        (x0, y0), (x1, y1) = box
        x0, x1 = int(x0 * W), int(x1 * W)
        y0, y1 = int(y0 * H), int(y1 * H)
        return [x0, y0, x1, y1]
    
    def collect_pii_rectangles(self, frame_bgr: np.ndarray, blur_all: bool = False) -> List[List[int]]:
        """
        Collect PII rectangles from a frame.
        
        Args:
            frame_bgr: Input frame
            blur_all: If True, blur all detected text regardless of PII classification
            
        Returns:
            List of rectangles to blur [x1, y1, x2, y2]
        """
        H, W = frame_bgr.shape[:2]
        data = self.ocr.infer(frame_bgr)
        pages = data.get("pages", [])
        rectangles = []
        
        if not pages:
            return rectangles
            
        for blk in pages[0].get("blocks", []):
            for line in blk.get("lines", []):
                for w in line.get("words", []):
                    text = w.get("value", "")
                    conf = float(w.get("confidence", 1.0))
                    geom = w.get("geometry")
                    
                    if not geom:
                        continue
                        
                    rect = self.rect_from_box_norm(geom, W, H)
                    
                    # Calculate area from rectangle
                    area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                    if area < self.min_area:
                        continue
                        
                    if blur_all or self.decider.decide(text, conf, self.conf_thresh):
                        rectangles.append(rect)
        
        return rectangles
    
    def process_frame(self, frame: np.ndarray, frame_id: int, 
                     blur_all: bool = False, room_id: str = None) -> Tuple[int, List[List[int]]]:
        """
        Process a single frame and return rectangles to be blurred.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            blur_all: If True, blur all detected text
            room_id: Room identifier for temporal isolation
            
        Returns:
            Tuple of (frame_id, list of rectangles as [x1, y1, x2, y2])
        """
        # Collect PII rectangles
        rectangles = self.collect_pii_rectangles(frame, blur_all=blur_all)
        
        # Convert rectangles to polygons for stabilization
        polys = []
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            polys.append(poly)
        
        # Get room-specific stabilizer (fixes cross-room contamination)
        if room_id:
            stabilizer = self._get_room_stabilizer(room_id)
        else:
            # Fallback for legacy calls without room_id
            if not hasattr(self, '_fallback_stabilizer'):
                self._fallback_stabilizer = Hysteresis(
                    iou_thresh=0.3, 
                    K_confirm=self.K_confirm, 
                    K_hold=self.K_hold
                )
            stabilizer = self._fallback_stabilizer
        
        # Apply ROOM-SPECIFIC temporal stabilization
        tracks = stabilizer.update(polys)
        
        # Convert active polygons back to rectangles
        active_rectangles = []
        for poly, is_active in tracks:
            if is_active:
                # Convert polygon back to rectangle
                xs, ys = poly[:, 0], poly[:, 1]
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                active_rectangles.append([x1, y1, x2, y2])
        
        # Debug logging
        print(f"[PIIDetector] Frame {frame_id} room {room_id}: new_rects={len(rectangles)}, active_rects={len(active_rectangles)}")
        
        return frame_id, active_rectangles
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "model_type": "pii_detector",
            "ocr_kind": self.ocr.kind,
            "conf_thresh": self.conf_thresh,
            "min_area": self.min_area,
            "K_confirm": self.K_confirm,
            "K_hold": self.K_hold,
            "has_ml_classifier": self.decider._clf is not None,
            "device": DEVICE
        }
