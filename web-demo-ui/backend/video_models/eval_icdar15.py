#!/usr/bin/env python3
"""
Evaluate Video Text PII blurring on ICDAR 2015 (Incidental Scene Text).
- Uses OCR (docTR: DBNet + PARSeq) + rules ∨ tiny ML classifier to predict PII regions.
- Compares predicted PII polygons against ground-truth text polygons that match PII rules.
- Reports region-level Precision/Recall/F1, pixel-level coverage, residual OCR rate (optional).

Example:
  python eval_icdar15.py \
    --img-dir /path/to/ICDAR2015/ch4_test_images \
    --gt-dir  /path/to/ICDAR2015/ch4_test_localization_transcription_gt \
    --max-images 200 \
    --classifier models/pii_clf.joblib \
    --residual-ocr

Ground-truth format (per ICDAR15):
  Each GT file: gt_img_*.txt lines: x1,y1,x2,y2,x3,y3,x4,y4,transcription
  Lines with transcription "###" are 'don't care' and will be ignored.
"""
import argparse
import os
import re
import sys
import time
from typing import List, Tuple, Dict, Optional
import csv
import json

import numpy as np
import cv2

# Optional torch/doctr
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ---- OCR: docTR preferred, EasyOCR fallback ----
class OCRPipeline:
    def __init__(self, det_arch="db_resnet50", reco_arch="parseq", device=None):
        self.kind = "doctr"
        self.device = device if device is not None else ("cuda" if TORCH_OK and torch.cuda.is_available() else "cpu")
        try:
            from doctr.models import ocr_predictor
            self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
            if TORCH_OK:
                self.model = self.model.to(self.device)
            self.model = self.model.eval()
        except Exception as e:
            print("[WARN] docTR unavailable → falling back to EasyOCR:", e, file=sys.stderr)
            try:
                import easyocr
            except Exception as e2:
                print("[ERROR] EasyOCR unavailable too:", e2, file=sys.stderr)
                raise
            self.kind = "easyocr"
            self.reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"))

    def infer(self, img_bgr: np.ndarray) -> Dict:
        if self.kind == "doctr":
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            doc = self.model([img_rgb])
            return doc.export()
        else:
            H, W = img_bgr.shape[:2]
            res = self.reader.readtext(img_bgr, detail=1, paragraph=False)
            words = []
            for box, text, conf in res:
                xs = [p[0]/W for p in box]; ys = [p[1]/H for p in box]
                x0,x1 = float(min(xs)), float(max(xs)); y0,y1 = float(min(ys)), float(max(ys))
                words.append({"value": text, "confidence": float(conf), "geometry": ((x0,y0),(x1,y1))})
            return {"pages":[{"blocks":[{"lines":[{"words":words, "geometry": ((0,0),(1,1))}]}]}]}

# ---- PII decision: rules ∨ tiny classifier ----
class PIIHybrid:
    def __init__(self, classifier_path: Optional[str] = None, thr_bias: float = 0.0):
        # Address/PII-ish patterns (tune per locale)
        street_tokens = r"(Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Terrace|Ter|Court|Ct|Crescent|Cres|Place|Pl|Highway|Hwy|Expressway|Expwy|Jalan|Jln|Lorong|Lor)"
        unit          = r"#\s?\d{1,3}-\d{1,4}"
        postal_sg     = r"\b(?:S\s*)?\d{6}\b"
        house_no      = r"\b(?:Blk|Block)?\s?\d{1,5}[A-Z]?\b"
        composed      = rf"{house_no}.*\b{street_tokens}\b"
        phone         = r"\b(?:\+?\d[\d\- ]{7,}\d)\b"
        url           = r"(https?://|www\.)\S+"
        self._regexes = [re.compile(p, re.I) for p in [street_tokens, unit, postal_sg, composed, phone, url]]

        self.vec = None; self.clf = None; self.thr = 0.5
        if classifier_path and os.path.exists(classifier_path):
            try:
                import joblib
                bundle = joblib.load(classifier_path)
                self.vec = bundle.get("vec", None)
                self.clf = bundle.get("clf", None)
                self.thr = float(bundle.get("thr", 0.5)) + float(thr_bias)
                print(f"[OK] Loaded classifier: {classifier_path} (thr={self.thr:.3f})")
            except Exception as e:
                print(f"[WARN] Could not load classifier: {e}", file=sys.stderr)

    def rule(self, text: str) -> bool:
        t = (text or "").strip()
        if not t: return False
        return any(rx.search(t) for rx in self._regexes)

    def prob(self, text: str) -> float:
        if self.vec is None or self.clf is None or not text:
            return 0.0
        try:
            X = self.vec.transform([text])
            return float(self.clf.predict_proba(X)[0,1])
        except Exception:
            return 0.0

    def decide(self, text: str, conf: float, conf_gate: float = 0.35) -> bool:
        if not text or conf < conf_gate:
            return False
        if self.rule(text): return True
        return self.prob(text) >= self.thr

# ---- ICDAR15 loader ----
def load_icdar15(gt_dir: str, img_dir: str) -> List[Tuple[str, List[Tuple[np.ndarray, str]]]]:
    """
    Returns list of (img_path, [(poly4x2, transcription), ...])
    """
    items = []
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(".txt")]
    gt_files.sort()
    for gt in gt_files:
        img_id = gt
        for tok in ["gt_", ".txt"]:
            img_id = img_id.replace(tok, "")
        # img_id already contains "img_" prefix from the ground truth filename
        candidates = [f"{img_id}.jpg", f"{img_id}.jpeg", f"{img_id}.png"]
        img_path = None
        for c in candidates:
            p = os.path.join(img_dir, c)
            if os.path.exists(p):
                img_path = p; break
        if img_path is None:
            continue

        polys = []
        with open(os.path.join(gt_dir, gt), "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = [p.strip("\ufeff") for p in line.split(",")]
                if len(parts) < 9:  # 8 coords + text
                    continue
                try:
                    coords = list(map(int, parts[:8]))
                    txt = ",".join(parts[8:])  # transcription may contain commas
                except ValueError:
                    coords = list(map(float, parts[:8]))
                    txt = ",".join(parts[8:])
                if txt == "###":
                    continue
                poly = np.array([[coords[0], coords[1]],
                                 [coords[2], coords[3]],
                                 [coords[4], coords[5]],
                                 [coords[6], coords[7]]], dtype=np.int32)
                polys.append((poly, txt))
        items.append((img_path, polys))
    return items

# ---- Geometry & masks ----
def clip_poly(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    poly = poly.copy()
    poly[:,0] = np.clip(poly[:,0], 0, W-1)
    poly[:,1] = np.clip(poly[:,1], 0, H-1)
    return poly

def poly_from_norm_box(box, W, H) -> np.ndarray:
    (x0,y0),(x1,y1) = box
    x0, x1 = int(x0*W), int(x1*W)
    y0, y1 = int(y0*H), int(y1*H)
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.int32)

def mask_from_polys(polys: List[np.ndarray], H: int, W: int) -> np.ndarray:
    m = np.zeros((H,W), dtype=np.uint8)
    if polys:
        cv2.fillPoly(m, polys, 255)
    return m

def iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a>0, b>0).sum()
    if inter == 0: return 0.0
    union = np.logical_or(a>0, b>0).sum()
    return float(inter) / float(union + 1e-6)

# ---- Matching (greedy) ----
def match_greedy(gt_masks: List[np.ndarray], pr_masks: List[np.ndarray], iou_thr=0.5):
    M, N = len(gt_masks), len(pr_masks)
    used_pred = [False]*N
    TP = 0; matches = []
    for i in range(M):
        best_j, best_iou = -1, 0.0
        for j in range(N):
            if used_pred[j]: continue
            iou = iou_mask(gt_masks[i], pr_masks[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            used_pred[best_j] = True
            TP += 1
            matches.append((i, best_j, best_iou))
    FP = used_pred.count(True)
    FN = M - TP
    return TP, FP, FN, matches

# ---- Blur (for residual OCR) ----
def blur_polygon(img: np.ndarray, poly: np.ndarray, ksize: int = 41, pad: int = 3) -> None:
    x,y,w,h = cv2.boundingRect(poly)
    x = max(0, x - pad); y = max(0, y - pad)
    xe = min(img.shape[1]-1, x + w + 2*pad); ye = min(img.shape[0]-1, y + h + 2*pad)
    if xe <= x or ye <= y: return
    roi = img[y:ye, x:xe]
    if ksize % 2 == 0: ksize += 1
    img[y:ye, x:xe] = cv2.GaussianBlur(roi, (ksize, ksize), 0)

# ---- Evaluate on one image ----
def eval_image(img_path: str, gt_polys_txt: List[Tuple[np.ndarray,str]],
               ocr: OCRPipeline, pii: PIIHybrid,
               conf_gate=0.35, min_area=80, iou_thr=0.5,
               eval_residual=False) -> dict:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"skip": True, "img_path": img_path}
    H,W = img.shape[:2]

    # GT PII polygons via rule on transcription
    gt_pii_polys = []
    gt_all_text = []  # Store all text for analysis
    for poly, txt in gt_polys_txt:
        poly_c = clip_poly(poly, W, H)
        if poly_c.shape[0] != 4: continue
        if cv2.contourArea(poly_c) < min_area: continue
        gt_all_text.append(txt)
        if pii.rule(txt):
            gt_pii_polys.append(poly_c)
    
    if len(gt_pii_polys) == 0:
        return {"skip": True, "img_path": img_path, "reason": "no_pii_gt", 
                "total_text_regions": len(gt_all_text)}

    gt_masks = [mask_from_polys([p], H, W) for p in gt_pii_polys]
    gt_union = mask_from_polys(gt_pii_polys, H, W)

    # Predict PII polygons
    data = ocr.infer(img)
    pred_polys = []
    pages = data.get("pages", [])
    if pages:
        for blk in pages[0].get("blocks", []):
            for line in blk.get("lines", []):
                words = line.get("words", [])
                if not words: continue
                line_text = " ".join([w.get("value","") for w in words]).strip()
                line_conf = np.mean([float(w.get("confidence",1.0)) for w in words]) if words else 1.0
                word_polys = []
                for w in words:
                    geom = w.get("geometry", None)
                    if not geom: continue
                    poly = poly_from_norm_box(geom, W, H)
                    if cv2.contourArea(poly) < min_area: continue
                    word_polys.append((poly, w.get("value",""), float(w.get("confidence",1.0))))
                if pii.decide(line_text, line_conf, conf_gate):
                    pred_polys.extend([p for (p,_,_) in word_polys])
                else:
                    for (p, t, c) in word_polys:
                        if pii.decide(t, c, conf_gate):
                            pred_polys.append(p)
    pr_masks = [mask_from_polys([p], H, W) for p in pred_polys]
    pr_union = mask_from_polys(pred_polys, H, W)

    # Region-level
    TP, FP, FN, matches = match_greedy(gt_masks, pr_masks, iou_thr=iou_thr)
    prec = TP / max(1, TP+FP)
    rec  = TP / max(1, TP+FN)
    f1   = 2*prec*rec / max(1e-9, (prec+rec))

    # Pixel coverage
    inter = np.logical_and(gt_union>0, pr_union>0).sum()
    pix_cov = inter / max(1, (gt_union>0).sum())

    # Residual OCR
    residual_rate = None
    if eval_residual:
        blurred = img.copy()
        for p in pred_polys:
            blur_polygon(blurred, p, ksize=41, pad=3)
        data_blur = ocr.infer(blurred)
        words_blur = []
        if data_blur.get("pages"):
            for blk in data_blur["pages"][0].get("blocks", []):
                for line in blk.get("lines", []):
                    for w in line.get("words", []):
                        geom = w.get("geometry")
                        if not geom: continue
                        poly = poly_from_norm_box(geom, W, H)
                        if cv2.contourArea(poly) < min_area: continue
                        words_blur.append((poly, w.get("value",""), float(w.get("confidence",1.0))))
        gt_fail = 0
        for gp in gt_pii_polys:
            gm = mask_from_polys([gp], H, W)
            leaked = False
            for (pp, txt, conf) in words_blur:
                pm = mask_from_polys([pp], H, W)
                if iou_mask(gm, pm) > 0.2 and txt.strip():
                    leaked = True; break
            gt_fail += 1 if leaked else 0
        residual_rate = gt_fail / max(1, len(gt_pii_polys))

    return {"skip": False, "TP": TP, "FP": FP, "FN": FN,
            "precision": prec, "recall": rec, "f1": f1,
            "pixel_coverage": pix_cov,
            "residual_rate": residual_rate,
            "n_gt": len(gt_pii_polys), "n_pred": len(pred_polys),
            "img_path": img_path, "img_size": f"{W}x{H}",
            "total_text_regions": len(gt_all_text)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="ICDAR15 images directory (e.g., ch4_test_images)")
    ap.add_argument("--gt-dir",  required=True, help="ICDAR15 gt directory (e.g., ch4_test_localization_transcription_gt)")
    ap.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    ap.add_argument("--classifier", default="", help="Path to tiny PII classifier (joblib). If not set, rules-only.")
    ap.add_argument("--thr-bias", type=float, default=0.0, help="Additive bias to classifier threshold (privacy mode: negative)")
    ap.add_argument("--conf-gate", type=float, default=0.35, help="OCR confidence gate for decisions")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for region-level match")
    ap.add_argument("--min-area", type=int, default=80, help="Minimum polygon area (pixels) to consider")
    ap.add_argument("--residual-ocr", action="store_true", help="Re-run OCR after blur to estimate residual recognition")
    ap.add_argument("--output-csv", default="", help="Save detailed results to CSV file")
    ap.add_argument("--run-name", default="", help="Run name for identification in results")
    args = ap.parse_args()

    device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Torch:{torch.__version__ if TORCH_OK else 'N/A'} CUDA:{'yes' if device=='cuda' else 'no'}")

    ocr = OCRPipeline(device=device)
    pii = PIIHybrid(classifier_path=args.classifier or None, thr_bias=args.thr_bias)

    items = load_icdar15(args.gt_dir, args.img_dir)
    if args.max_images > 0:
        items = items[:args.max_images]
    print(f"[INFO] Found {len(items)} images with GT. Starting eval...")

    agg = {"TP":0,"FP":0,"FN":0,"n_eval":0,"pix_cov_sum":0.0,"residual_sum":0.0,"residual_count":0}
    detailed_results = []  # Store per-image results

    t0 = time.time()
    for i,(img_path, polys_txt) in enumerate(items, 1):
        res = eval_image(img_path, polys_txt, ocr, pii,
                         conf_gate=args.conf_gate, min_area=args.min_area,
                         iou_thr=args.iou_thr, eval_residual=args.residual_ocr)
        
        # Store detailed results
        if args.output_csv:
            result_row = {
                'run_name': args.run_name,
                'image_id': i,
                'img_path': os.path.basename(res.get('img_path', '')),
                'skip': res.get('skip', False),
                'skip_reason': res.get('reason', ''),
                'img_size': res.get('img_size', ''),
                'total_text_regions': res.get('total_text_regions', 0),
                'TP': res.get('TP', 0),
                'FP': res.get('FP', 0), 
                'FN': res.get('FN', 0),
                'n_gt_pii': res.get('n_gt', 0),
                'n_pred_pii': res.get('n_pred', 0),
                'precision': res.get('precision', 0.0),
                'recall': res.get('recall', 0.0),
                'f1': res.get('f1', 0.0),
                'pixel_coverage': res.get('pixel_coverage', 0.0),
                'residual_rate': res.get('residual_rate', None),
                'classifier_used': bool(args.classifier),
                'thr_bias': args.thr_bias,
                'conf_gate': args.conf_gate,
                'iou_thr': args.iou_thr
            }
            detailed_results.append(result_row)
        
        if res["skip"]:
            continue
        agg["n_eval"] += 1
        agg["TP"] += res["TP"]; agg["FP"] += res["FP"]; agg["FN"] += res["FN"]
        agg["pix_cov_sum"] += res["pixel_coverage"]
        if args.residual_ocr and res["residual_rate"] is not None:
            agg["residual_sum"] += res["residual_rate"]; agg["residual_count"] += 1
        if i % 25 == 0:
            P = agg["TP"] / max(1, agg["TP"]+agg["FP"]); R = agg["TP"] / max(1, agg["TP"]+agg["FN"])
            F1 = 2*P*R / max(1e-9, P+R)
            print(f"[{i}/{len(items)}] interim F1≈{F1:.3f}")

    P = agg["TP"] / max(1, agg["TP"]+agg["FP"])
    R = agg["TP"] / max(1, agg["TP"]+agg["FN"])
    F1 = 2*P*R / max(1e-9, P+R)
    avg_pix_cov = agg["pix_cov_sum"] / max(1, agg["n_eval"])
    avg_residual = (agg["residual_sum"] / max(1, agg["residual_count"])) if args.residual_ocr else None

    print("\n=== ICDAR15 PII Evaluation ===")
    print(f"Images evaluated:      {agg['n_eval']}")
    print(f"Region Precision:      {P:.3f}")
    print(f"Region Recall:         {R:.3f}")
    print(f"Region F1:             {F1:.3f}")
    print(f"Pixel Coverage (avg):  {avg_pix_cov:.3f}")
    if avg_residual is not None:
        print(f"Residual OCR rate:     {avg_residual:.3f} (lower is better)")
    print(f"Elapsed: {time.time()-t0:.1f}s")
    
    # Write CSV output if requested
    if args.output_csv and detailed_results:
        print(f"\nSaving detailed results to: {args.output_csv}")
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = detailed_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
        print(f"Saved {len(detailed_results)} rows to CSV")
        
        # Also save summary metrics
        summary_file = args.output_csv.replace('.csv', '_summary.json')
        summary_data = {
            'run_name': args.run_name,
            'timestamp': time.time(),
            'total_images': len(items),
            'images_evaluated': agg['n_eval'],
            'images_skipped': len(items) - agg['n_eval'],
            'region_precision': P,
            'region_recall': R,
            'region_f1': F1,
            'pixel_coverage_avg': avg_pix_cov,
            'residual_ocr_rate': avg_residual,
            'processing_time_sec': time.time() - t0,
            'parameters': {
                'classifier_used': bool(args.classifier),
                'thr_bias': args.thr_bias,
                'conf_gate': args.conf_gate,
                'iou_thr': args.iou_thr,
                'min_area': args.min_area,
                'residual_ocr_enabled': args.residual_ocr
            }
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary to: {summary_file}")

if __name__ == "__main__":
    main()
