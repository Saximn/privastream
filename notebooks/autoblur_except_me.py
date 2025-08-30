#!/usr/bin/env python
"""
Auto-blur everyone except the enrolled (whitelisted) creator.

Hotkeys:
  q  : quit
  p  : panic toggle (mask all faces regardless of whitelist)
  e  : re-enroll creator embedding live

Usage examples:
  # 1) First time: enroll + run on webcam 0 (CPU)
  python autoblur_except_me.py --source 0 --enroll

  # 2) Next time: just run (embedding file will be used)
  python autoblur_except_me.py --source 0

  # 3) If GUI issues on Windows/macOS, try specifying backend:
  python autoblur_except_me.py --source 0 --backend dshow     # Windows
  python autoblur_except_me.py --source 0 --backend avfoundation  # macOS
"""
import argparse, json, time, sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ------------------ Utils ------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return 1.0 - float(np.dot(a, b))

def blur_box(img, box, k=35):
    if k % 2 == 0: k += 1
    x1,y1,x2,y2 = map(int, box)
    roi = img[y1:y2, x1:x2]
    if roi.size:
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k,k), 0)
    return img

def pixelate_box(img, box, pix=16):
    x1,y1,x2,y2 = map(int, box)
    roi = img[y1:y2, x1:x2]
    if roi.size:
        h,w = roi.shape[:2]
        small = cv2.resize(roi, (max(1,w//pix), max(1,h//pix)), interpolation=cv2.INTER_LINEAR)
        img[y1:y2, x1:x2] = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
    return img

def fill_box(img, box, bgr=(0,0,0)):
    x1,y1,x2,y2 = map(int, box)
    img[y1:y2, x1:x2] = bgr
    return img

def dilate_box(box, d, W, H):
    x1,y1,x2,y2 = box
    return [max(0, int(x1-d)), max(0, int(y1-d)), min(W-1, int(x2+d)), min(H-1, int(y2+d))]

def open_capture(source, backend: str|None):
    """Try to open capture with optional backend hint."""
    if str(source).isdigit():
        idx = int(source)
        if backend == "dshow":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        elif backend == "msmf":
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        elif backend == "avfoundation":
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        elif backend == "v4l2":
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(idx)  # auto
    else:
        cap = cv2.VideoCapture(source)
    return cap

# ------------------ Enrollment ------------------
def enroll_creator(out_path="whitelist/creator_embedding.json", cam_source="0", device="cpu", shots=20, backend=None):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[Enroll] Preparing face analysis (first run may download models)...")
    app = FaceAnalysis(name="buffalo_l")
    ctx_id = -1 if device == "cpu" else 0
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("[Enroll] InsightFace ready.")

    cap = open_capture(cam_source, backend)
    if not cap.isOpened():
        print("[Enroll][ERROR] Unable to open camera/source:", cam_source)
        return False

    embs = []
    print("[Enroll] Look at the camera; press 'q' to stop early.")
    while len(embs) < shots:
        ok, frame = cap.read()
        if not ok:
            break
        faces = app.get(frame)
        if faces:
            # pick largest face
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            if getattr(f, "normed_embedding", None) is not None:
                embs.append(f.normed_embedding.astype(float))
            # feedback rectangle
            x1,y1,x2,y2 = map(int, f.bbox)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("Enroll (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    if not embs:
        print("[Enroll][ERROR] No face embeddings captured.")
        return False

    mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
    out.write_text(json.dumps({"embedding": mean_emb.tolist()}), encoding="utf-8")
    print(f"[Enroll] Saved embedding to: {out}")
    return True

# ------------------ Streaming ------------------
def run_stream(args):
    # Load embedding (fail-safe: mask everyone if missing)
    creator_emb = None
    p = Path(args.embed)
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            creator_emb = np.array(obj["embedding"], dtype=float)
            print(f"[Run] Loaded embedding: {p}")
        except Exception as e:
            print("[Run][WARN] Failed to read embedding; will mask all faces.", e)

    # Face model
    print("[Run] Preparing InsightFace (first run may download models)...")
    app = FaceAnalysis(name="buffalo_l")
    ctx_id = -1 if args.device == "cpu" else 0
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("[Run] InsightFace ready.")

    # Video source
    cap = open_capture(args.source, args.backend)
    print(f"[Run] Opening source={args.source} backend={args.backend!r}")
    if not cap.isOpened():
        print("[Run][ERROR] Unable to open source.")
        return

    masks = []  # list of (expiry_time, box)
    frame_idx = 0
    panic_mask_all = False

    print("[Run] Running. Hotkeys: q=quit, p=panic mask-all, e=re-enroll")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Run][INFO] End of stream or read failure.")
            break
        H, W = frame.shape[:2]
        now = time.monotonic()

        new_boxes = []
        # Panic mode: mask whole frame (or skip detection)
        if panic_mask_all:
            new_boxes.append([0,0,W-1,H-1])

        # Detection on stride
        if not panic_mask_all and frame_idx % max(1, args.stride) == 0:
            faces = app.get(frame)
            for f in faces:
                box = list(map(float, f.bbox))
                allow_show = False
                if (creator_emb is not None) and (getattr(f, "normed_embedding", None) is not None):
                    d = cosine_distance(creator_emb, f.normed_embedding)
                    if d <= args.threshold:
                        allow_show = True  # creator -> do not mask
                if not allow_show:
                    new_boxes.append(dilate_box(box, args.dilate_px, W, H))

        # Smooth: keep masks alive for a short time window
        expiry = now + args.smooth_ms / 1000.0
        masks = [m for m in masks if m[0] > now] + [(expiry, b) for b in new_boxes]

        # Apply masks
        for _, box in masks:
            if args.mode == "blur":
                frame = blur_box(frame, box, k=args.blur_kernel)
            elif args.mode == "pixelate":
                frame = pixelate_box(frame, box, pix=args.pixel_size)
            else:
                frame = fill_box(frame, box, bgr=tuple(args.fill_bgr))

        # Show
        if args.show:
            cv2.imshow("AutoBlur (creator visible)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                panic_mask_all = not panic_mask_all
                state = "ON" if panic_mask_all else "OFF"
                print(f"[Run] Panic mask-all: {state}")
            elif key == ord('e'):
                print("[Run] Re-enrolling...")
                cap.release()
                cv2.destroyAllWindows()
                ok = enroll_creator(out_path=args.embed, cam_source=args.source, device=args.device, shots=args.shots, backend=args.backend)
                if ok:
                    obj = json.loads(Path(args.embed).read_text(encoding="utf-8"))
                    creator_emb = np.array(obj["embedding"], dtype=float)
                    print("[Run] Re-enroll complete. Resuming stream...")
                cap = open_capture(args.source, args.backend)
                if not cap.isOpened():
                    print("[Run][ERROR] Failed to reopen source after re-enroll.")
                    return

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[Run] Stopped.")

# ------------------ Main ------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Auto-blur everyone except the enrolled creator (face whitelist).")
    ap.add_argument("--source", default="0", help="0/1... for webcam index, or file/RTSP/RTMP URL")
    ap.add_argument("--backend", default=None, choices=[None, "dshow", "msmf", "avfoundation", "v4l2"], help="Optional OpenCV backend hint")
    ap.add_argument("--device", choices=["cpu","gpu"], default="cpu", help="InsightFace runtime")
    ap.add_argument("--embed", default="whitelist/creator_embedding.json", help="Embedding file path")
    ap.add_argument("--enroll", action="store_true", help="Run enrollment first (captures your face embedding)")
    ap.add_argument("--shots", type=int, default=20, help="Frames to average during enrollment")

    # Redaction & thresholds
    ap.add_argument("--threshold", type=float, default=0.35, help="Cosine distance; lower=stricter")
    ap.add_argument("--mode", choices=["blur","pixelate","fill"], default="blur")
    ap.add_argument("--blur_kernel", type=int, default=35)
    ap.add_argument("--pixel_size", type=int, default=16)
    ap.add_argument("--fill_bgr", type=int, nargs=3, default=[0,0,0])
    ap.add_argument("--dilate_px", type=int, default=12)
    ap.add_argument("--smooth_ms", type=int, default=250, help="Keep masks alive this long")
    ap.add_argument("--stride", type=int, default=2, help="Run detector every N frames (1 = every frame)")
    ap.add_argument("--show", type=int, default=1, help="1=preview window, 0=headless")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.enroll or not Path(args.embed).exists():
        ok = enroll_creator(out_path=args.embed, cam_source=args.source, device=args.device, shots=args.shots, backend=args.backend)
        if not ok:
            sys.exit(1)
    run_stream(args)
