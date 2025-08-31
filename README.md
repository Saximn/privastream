# Live Privacy Filter - PrivaStream: Audio & Video PII Blurring

A production-leaning, real-time **privacy filter** for livestreams and videos.  
It **detects and blurs** personally identifiable information (PII) across **video** (faces, license plates, street/address text) and **audio** (spoken PII), with **temporal stabilization** and clean, modular interfaces.
Built for TikTok Techjam 2025. 

---

## ✨ Key Capabilities

- **Video PII blur**
  - **Face blur** (detector model; optional mouth-only blur via landmarks/ROI)
  - **License-plate blur** (YOLO weights, e.g., `best.pt`)
  - **Street/address text blur** (OCR + PII classifier/rules)
  - **Unified Video Analyzer** merges all regions from the three models
  - **Temporal confirm/hold** stabilization to prevent blur flicker

- **Audio PII blur**
  - **Whisper** for speech-to-text
  - **Fine-tuned DeBERTa** to tag PII tokens (names, phone, address, etc.) every **5 seconds**
  - Marks **timestamps** for PII words and resolves them to **video frame IDs**
  - Triggers **mouth blur** in sync with the spoken PII segment

- **Scheduler & Throughput control**
  - Input stream may be **30 FPS**
  - **Video Scheduler** samples at **4 FPS** (configurable) to reduce compute while preserving privacy
  - All downstream modules use **frame IDs** to align and act

- **Engineering first**
  - Typed, simple interfaces: each model is a function `f(frame_id, frame) -> (frame_id, [boxes])`
  - Deterministic, testable components with clear contracts
  - CLI & YAML-config friendly

---

## 🧱 High-Level Architecture

```
┌───────────────┐   frames@30fps     ┌──────────────────┐     ┌────────────────────────┐
│ I/O Livestream ├───────────────────►│  Video Scheduler │─────► Unified Video Analyzer │
└───────────────┘                     │   (e.g., 4fps)   │     └──────────┬─────────────┘
                                      └──────────────────┘                │
                                                                          │ merges
                 ┌────────────────────┐    ┌────────────────────┐        ▼
                 │ Face Model         │    │ License Plate Model│   [Boxes to blur]
(frame_id,frame) │ → [(x1,y1,x2,y2)]  │    │ → [(x1,y1,x2,y2)]  │──────────────► Blur Engine
───────────────►  └────────────────────┘    └────────────────────┘
                 ┌────────────────────┐
                 │ PII Text Model     │  OCR + PII rules/ML
                 │ → [(x1,y1,x2,y2)]  │
                 └────────────────────┘
```

**Audio side:**

```
┌────────┐   audio stream   ┌───────────────┐   5s batch   ┌───────────────┐
│  I/O   ├──────────────────►│  Whisper STT  ├──────────────►│  DeBERTa PII  │
└────────┘                   └───────────────┘              └──────┬────────┘
                                                               PII tokens + timestamps
                                               map to frames  ┌─────▼───────────┐
                                               via scheduler  │   Mouth Blur     │
                                                              └──────────────────┘
```

> All computer-vision models share a **common I/O contract**:
> - **Input:** `(frame_id: int, frame: np.ndarray[BGR])`
> - **Output:** `(frame_id: int, boxes: List[Tuple[x1,y1,x2,y2]])`
> - Coordinates are **pixel-space integers** in the original frame size.


---

## ⚙️ Interfaces & Contracts

### Video model interface
```python
def infer(frame_id: int, frame_bgr: "np.ndarray") -> "tuple[int, list[tuple[int,int,int,int]]]":
    \"\"\"
    Returns (same frame_id, list of boxes) where boxes are (x1, y1, x2, y2) in pixels.
    \"\"\"
```

### Unified Video Analyzer
- Calls all enabled models (`face`, `license`, `pii_text`) **in parallel** (thread or process pool).
- Deduplicates overlaps (IoU threshold), merges boxes, returns a single list per frame.

### Video Scheduler
- Receives frames @ e.g., **30 FPS**, emits `(frame_id, frame)` at **4 FPS**.
- Maintains a **timebase** so audio timestamps map to frame IDs:  
  `frame_id = floor(t_seconds * target_fps)`

### Audio PII
- **Whisper** chunks audio, produces timestamped words.
- **DeBERTa (fine-tuned)** tags tokens with PII labels every **5 seconds** (configurable).
- For each PII word timestamp `t`, compute `frame_id = floor(t * target_fps)`; request **mouth blur** for a short window around `t` (e.g., ±250 ms).

### Blur Engine
- **Confirm/Hold** hysteresis to avoid flicker:  
  - `K_confirm`: frames required before a box becomes active  
  - `K_hold`: frames to keep blur after last positive
- Mask types: **Gaussian** (default) or **mosaic**; padding around boxes.

---

## 🛠️ Requirements

- Python 3.10+ (tested; 3.13-ready if deps support)
- CUDA-capable GPU recommended (PyTorch with CUDA)
- Core libraries (subset):
  - `ultralytics`, `torch`, `opencv-python[-headless]`, `numpy`
  - `python-doctr[torch]` (DBNet + PARSeq OCR)
  - `openai-whisper` or `faster-whisper` (for speed)
  - `transformers` (DeBERTa), `scikit-learn` (tiny PII classifier)
  - See `requirements.txt`

---

## 🔧 Configuration (example `config.yaml`)

```yaml
io:
  source: 0              # camera index or "path/to/video.mp4"
  out: outputs/blurred.mp4
  write_output: true

scheduler:
  input_fps: 30
  target_fps: 4

video:
  models:
    face:
      enabled: true
      weights: models/face_best.pt
      conf: 0.4
    license:
      enabled: true
      weights: models/best.pt
      conf: 0.25
    pii_text:
      enabled: true
      ocr: doctr             # or easyocr
      clf_path: models/pii_clf.joblib
      conf_gate: 0.35

  stabilization:
    k_confirm: 2
    k_hold: 8
    iou_thresh: 0.3

  blur:
    type: gaussian           # gaussian | mosaic
    ksize: 41
    pad: 4

audio:
  whisper_model: small       # tiny | base | small | medium | large-v3
  chunk_seconds: 5
  deberta_model: models/deberta-pii/   # local or HF path
  mouth_blur_window_ms: 500
```

---

## 🚀 Quick Start

**1) Install**  
```bash
pip install -r requirements.txt
```

**2) Put weights** in `models/`:
- `models/face_best.pt` (your face detector)
- `models/best.pt` (your license-plate detector)
- `models/pii_clf.joblib` (your text PII classifier)

**3) Live run (video only, unified analyzer)**  
```bash
python scripts/run_live.py --mode live --source 0 --show-boxes
```

**4) Process a video file**  
```bash
python scripts/run_live.py --mode video --source data/samples/demo.mp4 --out outputs/blurred.mp4 --show-boxes
```

**5) Standalone plate blur (YOLO)**  
```bash
python scripts/plate_blur.py --mode live --source 0 --weights models/best.pt --show-boxes
```

---

## 🧪 Evaluation

- **Video**: Measure PII-F1 on a small labeled set of frames:
  - Ground-truth boxes for faces/plates/street text
  - Compare **Rules-only** vs **Hybrid (Rules ∨ tiny ML)** for text PII decisions
- **Audio**: Per 5s window, precision/recall of PII-tagged tokens vs reference, plus AV alignment error (ms).
- **Latency**:
  - FPS (median, p95), and time breakdown (detector/recognizer/post)

---

## 🔒 Privacy & Safety

- **On-device inference** (no cloud calls at runtime)
- **Over-blur on uncertainty** (lower thresholds in “privacy” mode)
- No frames stored unless explicitly enabled
- Clear watermark **“Privacy Filter ON”** during blur

---

## 🤖 Implementation Notes

- **Mouth blur**: if facial landmarks are available, use inner-lip polygon; otherwise, approximate lower third of the face box.
- **OCR**: docTR `ocr_predictor(det="db_resnet50", reco="parseq")` by default; fallback to EasyOCR if unavailable.
- **Audio alignment**: `frame_id = floor(t * target_fps)`; expand to a small frame window for natural speech duration.
- **Throughput**: use batched OCR and async queues between Scheduler → Analyzer → Blur Engine for higher FPS.

---

## 🗺️ Roadmap

- Multi-language street/address lexicons
- ONNX export paths (YOLO, OCR, classifier) for mobile/embedded
- Virtual camera output for streaming platforms (OBS/VT Cam)
- Per-user ignore lists; UI for manual add/remove blur boxes

---

## 🐛 Troubleshooting

- Low FPS: reduce capture resolution or set `target_fps` to 2–3; use `faster-whisper` for audio.
- Missing CUDA: ensure the PyTorch wheel matches your NVIDIA driver.
- No detections: verify class names/IDs and confidence thresholds for each detector.

---

## 📝 License

BSD-3-Clause

---

## 🙌 Acknowledgements

- YOLO (Ultralytics) for object detection
- docTR (Mindee) for OCR
- OpenAI Whisper for speech-to-text
- DeBERTa for token-level PII tagging
- scikit-learn for lightweight text classifiers

## Datasets 📊

We relied on several community datasets and also built our own:

- [@nbroad's PII-DD mistral-generated dataset](https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated) — by far the most valuable external dataset. ⭐️
- [@mpware's Mixtral-generated essays](https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays).
- A custom dataset of ~2k samples (using @minhsienweng's notebook as a starting point).  
  This was released as external_data_v8.json and includes the nbroad dataset. 🔥
