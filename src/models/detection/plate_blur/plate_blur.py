#!/usr/bin/env python3
"""
License plate blurring (live webcam or video file) using Ultralytics YOLO weights (default: ./best.pt).
- Works with any YOLO model trained to detect plates (boxes). We blur ALL detections by default.
- Modes:
    --mode live  : webcam/RTSP (use --source 0 for default camera)
    --mode video : process a video file and save an output mp4
Example:
    python plate_blur.py --mode live --source 0 --weights best.pt --show-boxes
    python plate_blur.py --mode video --source input.mp4 --out output_blurred.mp4
"""
import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import cv2

# Use Ultralytics YOLO
try:
    import torch
    from ultralytics import YOLO
    TORCH_OK = True
except Exception as e:  # pragma: no cover
    TORCH_OK = False
    raise RuntimeError("This script requires 'ultralytics' and 'torch'. Install with: pip install ultralytics torch") from e


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def blur_box(img: np.ndarray, box_xyxy: Tuple[int, int, int, int], ksize: int = 35, pad: int = 4, mosaic: bool = False) -> None:
    """
    In-place blur of a rectangular region. ksize must be odd for Gaussian.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1 = clamp(int(x1) - pad, 0, w - 1)
    y1 = clamp(int(y1) - pad, 0, h - 1)
    x2 = clamp(int(x2) + pad, 0, w - 1)
    y2 = clamp(int(y2) + pad, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    if mosaic:
        # Pixelate: downscale then upscale
        mh = max(1, (y2 - y1) // 12)
        mw = max(1, (x2 - x1) // 12)
        small = cv2.resize(roi, (mw, mh), interpolation=cv2.INTER_LINEAR)
        pix   = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = pix
    else:
        if ksize % 2 == 0:
            ksize += 1
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)


def draw_box(img: np.ndarray, box_xyxy: Tuple[int, int, int, int], color=(0, 255, 0), thickness=2, label: Optional[str] = None):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(img, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def yolo_predict(model: "YOLO", frame_bgr: np.ndarray, imgsz: int, conf: float, iou: float, device: str):
    """Run YOLO on a single frame and return a list of (x1,y1,x2,y2, conf, cls)."""
    # Ultralytics supports numpy arrays directly
    results = model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
    boxes = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        b = r.boxes
        xyxy = b.xyxy.detach().cpu().numpy()  # (N,4)
        confs = b.conf.detach().cpu().numpy() if b.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
        clss = b.cls.detach().cpu().numpy() if b.cls is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(c), int(k)))
    return boxes


def run_live(weights: str, source: str, width: int, height: int, imgsz: int,
             conf: float, iou: float, ksize: int, pad: int, mosaic: bool, show_boxes: bool):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)

    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    assert cap.isOpened(), f"Cannot open video source: {source}"
    print("[INFO] Live started. Press ESC to quit.")

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        dets = yolo_predict(model, frame, imgsz=imgsz, conf=conf, iou=iou, device=device)
        out = frame.copy()

        for (x1, y1, x2, y2, c, k) in dets:
            blur_box(out, (x1, y1, x2, y2), ksize=ksize, pad=pad, mosaic=mosaic)
            if show_boxes:
                draw_box(out, (x1, y1, x2, y2), label=f"{c:.2f}")

        fps = 1.0 / max(1e-3, time.time() - t0)
        cv2.putText(out, f"Plate Blur | FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, "Privacy Filter ON", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2)

        cv2.imshow("License Plate Blur (ESC to quit)", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Live ended.")


def run_video(weights: str, src_path: str, out_path: str, imgsz: int, conf: float, iou: float,
              ksize: int, pad: int, mosaic: bool, show_boxes: bool):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)

    cap = cv2.VideoCapture(src_path)
    assert cap.isOpened(), f"Cannot open {src_path}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    print("[INFO] Processing video...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = yolo_predict(model, frame, imgsz=imgsz, conf=conf, iou=iou, device=device)
        out = frame.copy()
        for (x1, y1, x2, y2, c, k) in dets:
            blur_box(out, (x1, y1, x2, y2), ksize=ksize, pad=pad, mosaic=mosaic)
            if show_boxes:
                draw_box(out, (x1, y1, x2, y2), label=f"{c:.2f}")
        writer.write(out)

    writer.release()
    cap.release()
    print(f"[OK] Saved: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="License plate blurring with YOLO weights (best.pt)")
    ap.add_argument("--mode", choices=["live", "video"], default="live")
    ap.add_argument("--source", default="0", help="Camera index (0/1/...) or path/URL; for video mode this is the input path")
    ap.add_argument("--out", default="output_blurred.mp4", help="Output video path (video mode only)")
    ap.add_argument("--weights", default="best.pt", help="Path to YOLO weights (default: ./best.pt)")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    ap.add_argument("--width", type=int, default=1280, help="Capture width (live mode)")
    ap.add_argument("--height", type=int, default=720, help="Capture height (live mode)")
    ap.add_argument("--ksize", type=int, default=41, help="Gaussian blur kernel (odd)")
    ap.add_argument("--pad", type=int, default=4, help="Padding around box before blurring")
    ap.add_argument("--mosaic", action="store_true", help="Use pixelation instead of Gaussian blur")
    ap.add_argument("--show-boxes", action="store_true", help="Draw detection rectangles")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "live":
        run_live(args.weights, args.source, args.width, args.height, args.imgsz,
                 args.conf, args.iou, args.ksize, args.pad, args.mosaic, args.show_boxes)
    else:
        run_video(args.weights, args.source, args.out, args.imgsz,
                  args.conf, args.iou, args.ksize, args.pad, args.mosaic, args.show_boxes)


if __name__ == "__main__":
    main()
