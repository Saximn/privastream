"""
Enroll the creator's face (GPU-first): captures multiple frames, averages embedding,
and writes whitelist/creator_embedding.json
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np
from insightface.app import FaceAnalysis

def _pick_ctx_id(gpu_id: int) -> int:
    # Use GPU if CUDAExecutionProvider is present; otherwise fallback to CPU with a warning.
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

def enroll(out_path: str, cam_source: str, shots: int, gpu_id: int, det_size: int, backend: str|None):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ctx_id = _pick_ctx_id(gpu_id)
    print(f"[Enroll] Initializing InsightFace (ctx_id={ctx_id})...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    print("[Enroll] InsightFace ready.")

    cap = _open_capture(cam_source, backend)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera/source: {cam_source}")
        return 1

    embs = []
    print("[Enroll] Look at the camera; press 'q' to stop early.")
    while len(embs) < shots:
        ok, frame = cap.read()
        if not ok: break
        faces = app.get(frame)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            if getattr(f, "normed_embedding", None) is not None:
                embs.append(f.normed_embedding.astype(float))
            x1,y1,x2,y2 = map(int, f.bbox)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("Enroll (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    if not embs:
        print("[Enroll] No face embeddings captured."); return 2

    mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
    out.write_text(json.dumps({"embedding": mean_emb.tolist(),
                               "model":"buffalo_l","det_size":det_size}), encoding="utf-8")
    print(f"[Enroll] Saved: {out}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser("GPU-first face enrollment")
    ap.add_argument("--out", default="whitelist/creator_embedding.json")
    ap.add_argument("--cam", default="0", help="webcam index or URL")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--det_size", type=int, default=960)
    ap.add_argument("--backend", choices=["dshow","msmf","avf","v4l2"], default=None,
                    help="Optional OpenCV backend hint (Windows: dshow/msmf, macOS: avf)")
    args = ap.parse_args()
    raise SystemExit(enroll(args.out, args.cam, args.shots, args.gpu_id, args.det_size, args.backend))