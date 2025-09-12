#!/usr/bin/env python3
"""
Live PII blur: DBNet + PARSeq (docTR) + hybrid Rules ∨ tiny ML classifier.
- Uses your pii_clf.joblib saved at repo root (override with --classifier).
- Works with webcam (source index) or a video file/RTSP URL.
"""
import argparse
import os
import sys
import time
import re
from typing import List, Tuple, Optional, Any

import numpy as np
import cv2

# Try GPU via PyTorch
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

DEVICE = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"


# ---------- OCR pipeline (docTR preferred; EasyOCR fallback) ----------
class OCRPipeline:
    """
    Unified OCR interface:
      infer(bgr) -> dict like docTR .export() with normalized word boxes.
    """
    def __init__(self, det_arch="db_resnet50", reco_arch="parseq"):
        self.kind = "doctr"
        try:
            from doctr.models import ocr_predictor
            self._ocr = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
            # Move to device if torch is present
            if TORCH_OK:
                self._ocr = self._ocr.to(DEVICE)
            self._ocr = self._ocr.eval()
        except Exception as e:
            print("[WARN] docTR not available -> falling back to EasyOCR:", e, file=sys.stderr)
            try:
                import easyocr
            except Exception as e2:
                print("[ERROR] EasyOCR not available either:", e2, file=sys.stderr)
                raise
            self.kind = "easyocr"
            self._reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"))

    def infer(self, img_bgr: np.ndarray) -> dict:
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
                words.append({"value": text, "confidence": float(conf), "geometry": ((x0, y0), (x1, y1))})
            return {"pages": [{"blocks": [{"lines": [{"words": words, "geometry": ((0.0, 0.0), (1.0, 1.0))}]}]}]}


# ---------- PII decision (Rules ∨ tiny classifier) ----------
class PIIHybridDecider:
    def __init__(self, classifier_path: Optional[str] = None, thr: Optional[float] = None):
        # Address rules (extend per locale)
        street_tokens = r"(Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Terrace|Ter|Court|Ct|Crescent|Cres|Place|Pl|Highway|Hwy|Expressway|Expwy|Jalan|Jln|Lorong|Lor)"
        unit          = r"#\s?\d{1,3}-\d{1,4}"
        postal_sg     = r"\b(?:S\s*)?\d{6}\b"
        house_no      = r"\b(?:Blk|Block)?\s?\d{1,5}[A-Z]?\b"
        composed      = rf"{house_no}.*\b{street_tokens}\b"
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
                self._thr = float(bundle.get("thr", 0.5)) if thr is None else float(thr)
                print(f"[OK] Loaded classifier from {classifier_path} (thr={self._thr:.3f})")
            except Exception as e:
                print(f"[WARN] Failed to load classifier at {classifier_path}: {e}", file=sys.stderr)
        else:
            print(f"[INFO] No classifier found at {classifier_path}; using rules only.", file=sys.stderr)

    def _rule_is_pii(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return any(rx.search(t) for rx in self._regexes)

    def _ml_prob(self, text: str) -> float:
        if self._vec is None or self._clf is None or not text:
            return 0.0
        try:
            import numpy as np
            X = self._vec.transform([text])
            return float(self._clf.predict_proba(X)[0, 1])
        except Exception:
            return 0.0

    def decide(self, text: str, conf: float, conf_thresh: float = 0.35) -> bool:
        """True -> blur"""
        if not text or conf < conf_thresh:
            return False
        if self._rule_is_pii(text):
            return True
        if self._clf is not None:
            return self._ml_prob(text) >= self._thr
        return False


# ---------- Geometry & blur ----------
def poly_from_box_norm(box, W, H):
    (x0, y0), (x1, y1) = box
    x0, x1 = int(x0 * W), int(x1 * W)
    y0, y1 = int(y0 * H), int(y1 * H)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)

def aabb(poly: np.ndarray):
    xs, ys = poly[:, 0], poly[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def iou(a, b) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def blur_polygon(img: np.ndarray, poly: np.ndarray, ksize: int = 41, pad: int = 3) -> np.ndarray:
    x0, y0, x1, y1 = aabb(poly)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(img.shape[1] - 1, x1 + pad), min(img.shape[0] - 1, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return img
    roi = img[y0:y1, x0:x1].copy()
    img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    return img


# ---------- Stabilization ----------
class Hysteresis:
    """Confirm blur after K_confirm hits; hold for K_hold frames after last match."""
    def __init__(self, iou_thresh=0.3, K_confirm=2, K_hold=8):
        self.iou_thresh = iou_thresh
        self.K_confirm = K_confirm
        self.K_hold = K_hold
        self.tracks = {}
        self.next_id = 1
        self.frame = 0

    def update(self, polys: List[np.ndarray]):
        self.frame += 1
        used = [False] * len(polys)
        # match to existing tracks
        for tid, t in list(self.tracks.items()):
            ta = aabb(t["poly"])
            best, bj = 0.0, -1
            for j, p in enumerate(polys):
                if used[j]:
                    continue
                ov = iou(ta, aabb(p))
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

        # new tracks
        for j, p in enumerate(polys):
            if not used[j]:
                self.tracks[self.next_id] = {
                    "poly": p, "hits": 1, "active": (1 >= self.K_confirm), "last": self.frame
                }
                self.next_id += 1

        # garbage collect
        drop = [tid for tid, t in self.tracks.items() if (self.frame - t["last"]) > (3 * self.K_hold)]
        for tid in drop:
            self.tracks.pop(tid, None)

        return [(t["poly"], t["active"]) for t in self.tracks.values()]


# ---------- Collect PII polygons from a frame ----------
def collect_pii_polys(ocr: OCRPipeline, decider: PIIHybridDecider,
                      frame_bgr: np.ndarray, conf_thresh=0.35, min_area=80, blur_all=False) -> List[np.ndarray]:
    H, W = frame_bgr.shape[:2]
    data = ocr.infer(frame_bgr)
    pages = data.get("pages", [])
    polys = []
    if not pages:
        return polys
    for blk in pages[0].get("blocks", []):
        for line in blk.get("lines", []):
            for w in line.get("words", []):
                text = w.get("value", "")
                conf = float(w.get("confidence", 1.0))
                geom = w.get("geometry")
                if not geom:
                    continue
                poly = poly_from_box_norm(geom, W, H)
                if cv2.contourArea(poly) < min_area:
                    continue
                if blur_all or decider.decide(text, conf, conf_thresh):
                    polys.append(poly)
    return polys


# ---------- Live & file processing ----------
def run_live(source: str,
             width: Optional[int], height: Optional[int],
             conf_thresh: float, K_confirm: int, K_hold: int,
             ksize: int, show_boxes: bool, blur_all: bool,
             classifier_path: Optional[str] = None):
    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    assert cap.isOpened(), f"Cannot open video source: {source}"

    ocr = OCRPipeline()
    decider = PIIHybridDecider(classifier_path=classifier_path)

    stab = Hysteresis(iou_thresh=0.3, K_confirm=K_confirm, K_hold=K_hold)
    print("[INFO] Live stream started. ESC to exit.")
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        polys = collect_pii_polys(ocr, decider, frame, conf_thresh=conf_thresh, blur_all=blur_all)
        tracks = stab.update(polys)

        out = frame.copy()
        active = 0
        for poly, is_active in tracks:
            if is_active:
                active += 1
                out = blur_polygon(out, poly, ksize=ksize, pad=3)
                if show_boxes:
                    cv2.polylines(out, [poly], True, (0, 255, 0), 2)
        fps = 1.0 / max(time.time() - t0, 1e-3)
        cv2.putText(out, f"PII Blur | FPS {fps:.1f} | Active {active}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, "Privacy Filter ON", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2)

        cv2.imshow("Live PII Blur (ESC to quit)", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Live stream ended.")


def run_video(src_path: str, out_path: str,
              conf_thresh: float, K_confirm: int, K_hold: int, ksize: int,
              blur_all: bool, classifier_path: Optional[str] = None):
    cap = cv2.VideoCapture(src_path)
    assert cap.isOpened(), f"Cannot open {src_path}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    ocr = OCRPipeline()
    decider = PIIHybridDecider(classifier_path=classifier_path)
    stab = Hysteresis(iou_thresh=0.3, K_confirm=K_confirm, K_hold=K_hold)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        polys = collect_pii_polys(ocr, decider, frame, conf_thresh=conf_thresh, blur_all=blur_all)
        tracks = stab.update(polys)

        out = frame.copy()
        for poly, is_active in tracks:
            if is_active:
                out = blur_polygon(out, poly, ksize=ksize, pad=3)
        writer.write(out)
        if idx % 30 == 0:
            print(f"[INFO] Processed frame {idx}")

    writer.release()
    cap.release()
    print(f"[OK] Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Live PII blur with OCR + hybrid classifier")
    p.add_argument("--mode", choices=["live", "video"], default="live")
    p.add_argument("--source", default="0", help="Camera index (e.g., 0) or RTSP/URL or file path")
    p.add_argument("--out", default="output_blurred.mp4", help="Output path (video mode)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--conf-thresh", type=float, default=0.35, help="OCR confidence gate for decisions")
    p.add_argument("--k-confirm", type=int, default=2, help="Frames to confirm blur")
    p.add_argument("--k-hold", type=int, default=8, help="Frames to hold blur after last hit")
    p.add_argument("--ksize", type=int, default=41, help="Gaussian blur kernel (odd)")
    p.add_argument("--show-boxes", action="store_true", help="Draw blurred polygons")
    p.add_argument("--blur-all", action="store_true", help="Blur all detected text (ignores classifier/rules)")
    p.add_argument("--classifier", default="pii_clf.joblib", help="Path to joblib bundle")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "live":
        run_live(
            source=args.source,
            width=args.width, height=args.height,
            conf_thresh=args.conf_thresh,
            K_confirm=args.k_confirm, K_hold=args.k_hold,
            ksize=args.ksize, show_boxes=args.show_boxes, blur_all=args.blur_all,
            classifier_path=args.classifier
        )
    else:
        run_video(
            src_path=args.source, out_path=args.out,
            conf_thresh=args.conf_thresh,
            K_confirm=args.k_confirm, K_hold=args.k_hold,
            ksize=args.ksize, blur_all=args.blur_all,
            classifier_path=args.classifier
        )


if __name__ == "__main__":
    main()
