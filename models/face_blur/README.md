# Face Blur Model

> **Note:** This model strictly requires a GPU for it to function.

## Setup
Make sure to install all required dependencies before running any scripts (good luck finding those).

---

## Commands

### Enroll Face
```bash
python enroll_face.py --cam 0 --gpu_id 0 --det_size 960
```

### Livestream
```bash
python run_stream.py --source 0 --gpu_id 0 --det_size 960 --stride 1 --tta_every 0
```

