"""
Live redaction (GPU-first): keeps the enrolled creator visible; masks everyone else.
High-recall defaults: det_size=960, dilation, temporal smoothing, optional TTA.
Hotkeys: q=quit, p=panic mask-all, r=reload embedding from disk
"""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np
from collections import deque
from insightface.app import FaceAnalysis

# --------- helpers ----------
CREATOR_UNLOCK_MS = 800
creator_last_positive_ms = 0

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9
    return inter/ua

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a)+1e-9); b = b / (np.linalg.norm(b)+1e-9)
    return 1.0 - float(np.dot(a, b))

def blur_box(img, box, k=35):
    if k % 2 == 0: k += 1
    x1,y1,x2,y2 = map(int, box); roi = img[y1:y2, x1:x2]
    if roi.size: img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k,k), 0)
    return img

def pixelate_box(img, box, pix=16):
    x1,y1,x2,y2 = map(int, box); roi = img[y1:y2, x1:x2]
    if roi.size:
        h,w = roi.shape[:2]
        small = cv2.resize(roi, (max(1,w//pix), max(1,h//pix)), interpolation=cv2.INTER_LINEAR)
        img[y1:y2, x1:x2] = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
    return img

def fill_box(img, box, bgr=(0,0,0)):
    x1,y1,x2,y2 = map(int, box); img[y1:y2, x1:x2] = bgr; return img

def dilate_box(box, d, W, H):
    x1,y1,x2,y2 = box
    return [max(0,int(x1-d)), max(0,int(y1-d)), min(W-1,int(x2+d)), min(H-1,int(y2+d))]

def _pick_ctx_id(gpu_id: int) -> int:
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return int(gpu_id)
        print("[WARN] CUDAExecutionProvider not available; falling back to CPU.")
        return -1
    except Exception as e:
        print(f"[WARN] onnxruntime not found or misconfigured ({e}); falling back to CPU.")
        return -1

def _open_capture(source, backend):
    if str(source).isdigit():
        idx = int(source)
        if backend == "dshow": return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if backend == "msmf":  return cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if backend == "avf":   return cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if backend == "v4l2":  return cv2.VideoCapture(idx, cv2.CAP_V4L2)
        return cv2.VideoCapture(idx)
    return cv2.VideoCapture(source)

# ----- TTA (optional every N frames) -----
def _iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9
    return inter/ua

def _nms_union(boxes, thr=0.5):
    out=[]
    for b in boxes:
        if not any(_iou(b,o)>thr for o in out):
            out.append(b)
    return out

def detect_faces_tta(app, frame_bgr, big_size=960, do_flip=True):
    H, W = frame_bgr.shape[:2]
    boxes=[]

    for f in app.get(frame_bgr):
        boxes.append(list(map(float, f.bbox)))

    if do_flip:
        flipped = cv2.flip(frame_bgr, 1)
        for f in app.get(flipped):
            x1,y1,x2,y2 = map(float, f.bbox)
            boxes.append([W-x2, y1, W-x1, y2])

    if max(H, W) < big_size:
        scale = big_size / max(H, W)
        big = cv2.resize(frame_bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_LINEAR)
        for f in app.get(big):
            x1,y1,x2,y2 = map(float, f.bbox)
            boxes.append([x1/scale, y1/scale, x2/scale, y2/scale])

    return _nms_union(boxes, thr=0.5)

# --------- main loop ----------
def run(args):
    # Load embedding
    emb = None
    p = Path(args.embed)
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            emb = np.array(obj["embedding"], dtype=float)
            print(f"[Run] Loaded embedding: {p}")
        except Exception as e:
            print("[Run][WARN] Failed to read embedding; will mask all faces.", e)

    # Face model (GPU-first)
    ctx_id = _pick_ctx_id(args.gpu_id)
    print(f"[Run] Initializing InsightFace (ctx_id={ctx_id})...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))
    print("[Run] InsightFace ready.")

    # Video source
    cap = _open_capture(args.source, args.backend)
    if not cap.isOpened():
        print("[ERROR] Cannot open source:", args.source); return

    # Optional: ask for a higher camera base resolution (helps recall)
    if str(args.source).isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    masks = []               # (expiry_time, box)
    fidx = 0
    panic = False
    vote_buf = deque(maxlen=3)  # temporal vote for whitelist decision

    print("[Run] Running. Hotkeys: q=quit, p=panic mask-all, r=reload embedding")
    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        now = time.monotonic()

        # Optionally enhance low-light frames before detection
        frame_for_det = frame
        if frame.mean() < args.lowlight_trigger:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            y = clahe.apply(y)
            frame_for_det = cv2.cvtColor(cv2.merge((y,cr,cb)), cv2.COLOR_YCrCb2BGR)

        new_boxes = []
        if panic:
            new_boxes.append([0,0,W-1,H-1])
        else:
            # Detector cadence + TTA
            if fidx % max(1, args.stride) == 0:
                if args.tta_every > 0 and fidx % args.tta_every == 0:
                    face_boxes = detect_faces_tta(app, frame_for_det, big_size=args.det_size, do_flip=True)
                    faces = []  # we only got boxes from TTA path
                else:
                    faces = app.get(frame_for_det)
                    face_boxes = [list(map(float, f.bbox)) for f in faces]

                # Decide per face: mask unless it's the creator
                for i, box in enumerate(face_boxes):
                    allow_show = False
                    if emb is not None and i < len(faces) and getattr(faces[i], "normed_embedding", None) is not None:
                        d = cosine_distance(emb, faces[i].normed_embedding)
                        vote_buf.append(d <= args.threshold)
                        allow_show = (sum(vote_buf) >= 2)  # simple temporal vote
                    if not allow_show:
                        new_boxes.append(dilate_box(box, args.dilate_px, W, H))

        # Temporal smoothing
        expiry = now + args.smooth_ms/1000.0
        masks = [m for m in masks if m[0] > now] + [(expiry, b) for b in new_boxes]

        # Apply masks
        for _, box in masks:
            if args.mode == "blur":
                frame = blur_box(frame, box, k=args.blur_kernel)
            elif args.mode == "pixelate":
                frame = pixelate_box(frame, box, pix=args.pixel_size)
            else:
                frame = fill_box(frame, box, bgr=tuple(args.fill_bgr))

        if args.show:
            cv2.imshow("Live Redaction (creator visible)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('p'):
                panic = not panic
                print(f"[Run] Panic mask-all: {'ON' if panic else 'OFF'}")
            if k == ord('r'):
                try:
                    obj = json.loads(Path(args.embed).read_text(encoding="utf-8"))
                    emb = np.array(obj["embedding"], dtype=float)
                    print("[Run] Reloaded embedding from disk.")
                except Exception as e:
                    print("[Run][WARN] Reload failed:", e)

        fidx += 1

    cap.release(); cv2.destroyAllWindows()
    print("[Run] Stopped.")
    return

if __name__ == "__main__":
    ap = argparse.ArgumentParser("GPU-first Live Redaction (face whitelist)")
    ap.add_argument("--source", default="0", help="webcam index or RTSP/RTMP/file")
    ap.add_argument("--backend", choices=["dshow","msmf","avf","v4l2"], default=None)
    ap.add_argument("--embed", default="whitelist/creator_embedding.json")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--det_size", type=int, default=960, help="detector input (square)")
    ap.add_argument("--stride", type=int, default=1, help="run detector every N frames")
    ap.add_argument("--tta_every", type=int, default=5, help="every Nth frame run TTA (0=off)")
    ap.add_argument("--lowlight_trigger", type=float, default=60.0, help="mean pixel threshold to enable CLAHE")
    ap.add_argument("--threshold", type=float, default=0.35, help="cosine distance (lower=stricter)")
    ap.add_argument("--mode", choices=["blur","pixelate","fill"], default="blur")
    ap.add_argument("--blur_kernel", type=int, default=35)
    ap.add_argument("--pixel_size", type=int, default=16)
    ap.add_argument("--fill_bgr", type=int, nargs=3, default=[0,0,0])
    ap.add_argument("--dilate_px", type=int, default=12)
    ap.add_argument("--smooth_ms", type=int, default=300)
    ap.add_argument("--show", type=int, default=1)
    args = ap.parse_args()
    run(args)