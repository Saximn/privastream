# PrivaStream — Improvement & Best-Practices Report

> Repository-wide audit of architecture, performance, security, and code quality.
> Generated from a full read of the codebase. Each finding lists concrete
> `file:line` references, the problem, the impact, and a recommended fix.
>
> Findings are tagged by severity:
> **🔴 Critical** · **🟠 High** · **🟡 Medium** · **🟢 Low / polish**

---

## 0. Executive Summary

PrivaStream works, but the **real-time video path is architecturally inverted**:
the application already establishes an efficient WebRTC video track (VP8 over the
mediasoup SFU) and then **never uses it**. Instead it ships every frame as a
**base64-encoded JPEG over Socket.IO**, runs detection/blur over HTTP in a
Python service, and re-broadcasts the result as another base64 JPEG that the
viewer paints onto a `<canvas>` and re-captures with `canvas.captureStream()`.

This is the root cause of the symptoms the team has noticed ("`setTimeout` per
frame, resource-consuming, bandwidth-heavy"). The top three things to fix, in
order of impact:

1. **Stop encoding frames as JPEG-over-WebSocket.** Use the WebRTC media path
   that already exists (VP8/H.264, hardware-accelerated, inter-frame
   compression, congestion control). JPEG-per-frame has **no temporal
   compression** and base64 inflates every payload by **~33%**. (§1)
2. **Stop the host from over-sampling.** The capture loop runs at **20 FPS**
   despite a comment claiming 4 FPS, sending ~5× the frames the pipeline needs.
   (§1.2 — *fixed in this PR*)
3. **Fix the privacy-leak fallbacks.** Several error paths emit the **original,
   un-redacted** audio/frame when processing fails. (§4)

The remaining sections cover audio, backend, security, infra, testing, and
code-quality debt.

---

## 1. Real-Time Video Pipeline (the big one)

### Current data flow

```
Host camera (30fps)
  ├─► mediasoup producer  → VP8 WebRTC track ……… PRODUCED BUT NEVER CONSUMED ❌
  └─► <canvas>.toDataURL("image/jpeg")  (host/page.tsx:333)
        → socket.emit("video-frame", base64 JPEG)   (host/page.tsx:339)
            → mediasoup server  (server.js:610)
                → HTTP POST base64 → Python /detect-faces-mouths  (server.js:635)
                → HTTP POST base64 → Python /apply-conditional-blur (server.js:930)
                → setTimeout(deliveryDelay) per frame  (server.js:954) ❌
                    → socket.emit("processed-video-frame", base64 JPEG)
                        → viewer: new Image(); ctx.drawImage; canvas.captureStream(30)
                          (processed-video-viewer.tsx:213-247) ❌
```

### 1.1 🔴 The WebRTC video track is produced but never consumed

`mediasoup-client.ts:107` produces the camera's video track to the SFU. But the
server **explicitly refuses to let viewers consume it**:

- `server.js:771-773` — `getProducers` skips every `video` producer
  ("using processed frames via socket events").
- `processed-video-viewer.tsx` never calls `consume` for video at all.

**Impact:** the host pays full upload bandwidth and encoder cost for a stream
nobody watches, *in addition to* the JPEG stream that is actually used. Pure
waste.

**Fix:** pick one transport. The correct one is WebRTC (see §1.6). The JPEG path
should be deleted once the WebRTC blur path works.

### 1.2 🟠 Host captures at 20 FPS, not the intended 4 FPS *(fixed here)*

`host/page.tsx:322` — `setInterval(() => {...}, 50)` is **20 FPS**, but the
inline comment says "Process frames at 4 FPS (every 250ms)" and the architecture
doc (`docs/ARCHITECTURE.md:127`) specifies 4 FPS detection. The server is also
built around `processEveryNthFrame: 15` (`server.js:77`) i.e. ~2 detections/sec.

**Impact:** ~5× more JPEG frames are encoded, base64'd, and pushed over the
socket than the pipeline consumes — directly the "bandwidth-heavy" symptom.

**Fix applied in this change:** the capture loop now targets a configurable
**4 FPS**, is driven by `requestVideoFrameCallback` (falling back to a correctly
sized `setInterval`), and skips a tick if the previous frame's encode hasn't
finished — so frames can never pile up under load. See §1.3.

### 1.3 🟠 `canvas.toDataURL()` blocks the main thread every frame

`host/page.tsx:333` and the viewer's per-frame `new Image()` decode
(`processed-video-viewer.tsx:212`) run **synchronous, main-thread** image
codec work on every frame. `toDataURL` is fully synchronous and returns a
base64 string (33% larger than the raw bytes).

**Best practice:**
- Capture with `requestVideoFrameCallback` (fires once per *actually rendered*
  frame, integrates with the browser's frame scheduler) instead of a wall-clock
  `setInterval`.
- Encode with `canvas.toBlob()` / `OffscreenCanvas.convertToBlob()`
  (asynchronous, off the render-critical path) and send **binary** over
  Socket.IO (it supports `ArrayBuffer`/`Blob` natively) instead of base64.
- Better still, move capture+encode into an `OffscreenCanvas` in a Web Worker.

*(This change wires up rVFC + the FPS fix + an in-flight guard. Switching the
payload to binary and moving to a worker are follow-ups noted in §1.6.)*

### 1.4 🟠 Per-frame `setTimeout` scheduling on the server

`server.js:954` schedules **one `setTimeout` per frame** (plus another per audio
chunk at `:457` and `:470`) to implement a fixed 6 s viewer delay
(`TIMING_CONFIG.TOTAL_VIEWER_DELAY`). At 20 FPS that is 20 live timers/sec/room,
each closing over a frame payload kept alive on the heap until it fires.

**Impact:** GC pressure, unbounded timer fan-out under load, and no
back-pressure — if Python slows down, timers and buffered frames accumulate.

**Fix:** replace per-frame timers with a single **jitter buffer / ordered
delivery queue** per room drained by one interval (or, with the WebRTC path, let
mediasoup + the client jitter buffer handle timing natively and drop the manual
delay entirely).

### 1.5 🟡 `BufferedVideoPlayer` is dead/ineffective code

`buffered-video-player.ts` polls with recursive `setTimeout(checkBuffer, 100)`
(`:120`) and `setTimeout(..., 500)` on stall (`:182`), maintains a `frameBuffer`
array that is **never populated**, and references a non-standard
`setVideoBufferSize` API (`:74`) that does not exist. The processed-video viewer
doesn't use this class at all — it paints onto a canvas.

**Fix:** delete it, or replace with MSE/`managedMediaSource` if a real jitter
buffer is needed.

### 1.6 🟢 Recommended target architecture

Two viable designs, both removing JPEG-over-socket:

**Option A — Server-side processed track (best quality/scale).**
Host sends one WebRTC track to the SFU. A worker pipeline decodes via FFmpeg/
GStreamer (or mediasoup `PlainTransport` → RTP), runs detection/blur on raw
frames, and re-encodes a single processed track that all viewers `consume`.
Viewers get real adaptive video. Detection still runs at N FPS but blur is
applied to every frame server-side.

**Option B — Client-side blur with Insertable Streams (lowest infra).**
Use `MediaStreamTrackProcessor` + `WebCodecs` + `TransformStream`
(Insertable Streams) in the host to blur frames **before** they enter the
encoder, then produce the already-private track to the SFU. The Python service
only returns *coordinates*; pixel blur happens on the GPU in the browser. No
JPEG, no re-broadcast, viewers consume normally.

Either way: detection coordinates are small JSON; **pixels travel over the
codec, never as base64**.

---

## 2. Audio Pipeline

### 2.1 🔴 PCM shipped as a JSON number array

`host/page.tsx:419` — `sfuSocketRef.current.emit("audio-data", Array.from(pcmData))`
serializes every 16-bit sample as a **decimal string in a JSON array**. A
1366-sample chunk becomes thousands of comma-separated ASCII digits — easily
**5–10×** the size of the 2.7 KB of raw PCM. The viewer reverses this with
`new Int16Array(audioData)` over a JSON array (`processed-video-viewer.tsx:263`).

**Fix:** send the `Int16Array.buffer` (an `ArrayBuffer`) directly; Socket.IO
transmits it as a binary frame.

### 2.2 🟠 Deprecated `ScriptProcessorNode`

`host/page.tsx:382` and `audio-redaction-client.ts:247` use
`createScriptProcessor`, which is **deprecated** and runs on the main thread.

**Fix:** migrate to `AudioWorklet` (runs on the audio render thread, no main-
thread jank, supported in all modern browsers).

### 2.3 🟡 Naïve downsampling causes aliasing

`host/page.tsx:401-408` downsamples 48 kHz → 16 kHz by taking "every 3rd sample"
with **no anti-alias low-pass filter**, injecting aliasing artifacts that hurt
Whisper transcription accuracy (and therefore PII detection recall).

**Fix:** apply a low-pass filter before decimation, or resample with a proper
polyphase filter (e.g. in the AudioWorklet).

### 2.4 🔴 Audio fallback emits un-redacted audio
*(reported by sub-audit — verify before acting)*

The mediasoup audio processors reportedly return the **original** chunk on
processing failure (`audio-processor.js:~191`,
`audio-redaction-processor.js:~185`). That defeats redaction exactly when it
matters.

**Fix:** on failure, emit **silence** (or drop the chunk), never the raw audio.
Mirror the "drop frame on failure" policy the video processor already uses
(`webrtc-video-processor.js:99-120`).

### 2.5 🟢 Fake/test-audio fallbacks left in production code

`audio-redaction-plugin.js:412,607` spin up `setInterval` loops generating test
audio. Remove or gate behind an explicit debug flag.

---

## 3. Python Detection Service (`video_filter_api.py`)

- 🟠 **Re-encode tax.** Every endpoint decodes base64→JPEG→`cv2.imdecode`, then
  `cv2.imencode`→base64 on the way out (`:303-305`, `:1000-1015`). With the
  WebRTC path (§1.6) the service should accept/return **raw frames or just
  coordinates**, never JPEG.
- 🟠 **`debug=False, threaded=True` Flask dev server** (`:1168`) is not a
  production WSGI server. Use `gunicorn`/`uvicorn` workers; a single GIL-bound
  Flask process will bottleneck under concurrent rooms.
- 🟡 **Double detection per frame.** `/detect-faces-mouths` runs
  `process_frame_with_mouth_landmarks` *and* a full `process_frame`
  (`:881-886`) — two model passes where one would do.
- 🟡 **`print()` everywhere instead of `logging`** (17+ calls) — no levels, no
  timestamps, unbuffered noise in production.
- 🟡 **Module-global mutable state** (`room_embeddings`, `active_requests`)
  guarded by one lock; fine for a single process, breaks the moment you scale to
  multiple workers. Move to Redis (already in the architecture diagram, not yet
  used).
- 🟢 **Blur kernel is fixed `(75,75)` / `(150,150)`** regardless of region size
  (`:292`, `:1096`) — tiny regions get over-blurred, large ones may show
  structure. Scale the kernel to the region.

---

## 4. Privacy & Security

- 🔴 **`CORS(origin="*")`** on every service (`backend/app.py`,
  `video_filter_api.py:32`, `server.js:82,86`). Any site can drive these APIs.
  Restrict to known frontend origins.
- 🔴 **No authentication anywhere.** Anyone who can reach the SFU/Python ports
  can join rooms, push frames, or drain GPU. The frame endpoints have a
  concurrency limiter but **no authn/z** (`video_filter_api.py:342`). Add room
  tokens / signed join.
- 🟠 **`SECRET_KEY` may be `None`** if the env var is unset
  (`backend/app.py`) — fail loudly at startup instead.
- 🟠 **`str(e)` returned to clients** (`video_filter_api.py:411`,
  `backend/app.py:411`) leaks internal paths/stack detail. Log server-side,
  return a generic message.
- 🟠 **`announcedIp: '127.0.0.1'` hardcoded** (`server.js:96`,
  `docker-compose.yml` default) — WebRTC is unreachable from anything but
  localhost. Drive from `MEDIASOUP_ANNOUNCED_IP` with **no localhost default**
  in production.
- 🟡 **Unbounded room/user maps** with no expiry; orphaned rooms leak memory if
  cleanup misses.

---

## 5. Configuration & Secrets

- 🟡 Hardcoded `localhost` fallbacks duplicated across `frontend/lib/config.ts`,
  `mediasoup/config.js`, and `backend` defaults. Centralize and **fail loudly**
  when required URLs are missing in production.
- 🟡 **No `.env.example`.** Add one documenting every `NEXT_PUBLIC_*`,
  `MEDIASOUP_*`, `SECRET_KEY`, and service-URL variable.
- 🟢 `TIMING_CONFIG` lives in code (`server.js:25`); expose via env so the
  6 s delay can be tuned without a rebuild.

---

## 6. Infrastructure & Build

- 🟠 **Build artifacts and logs are committed to git:**
  `src/web/frontend/.next/**` (full Next.js build output incl. webpack cache
  `.pack.gz`), `src/web/backend/video.log`, `src/web/mediasoup/log_mediasoup.txt`.
  Add `.next/`, `*.log`, `build/`, `dist/` to `.gitignore` and `git rm --cached`
  them.
- 🟡 **`.gitignore` is thin** — doesn't cover `.next/`, `*.log`, `*.joblib`,
  Python `*.egg-info`, etc.
- 🟡 **No `HEALTHCHECK`** in `Dockerfile.frontend` / `Dockerfile.mediasoup`; the
  backend healthcheck depends on `requests` being present in the slim runtime —
  prefer `curl`.
- 🟡 **Loose dependency pinning** across `requirements.txt`, `package.json`
  (`^`), and `setup.py` (`>=`). Pin exact versions (pip-compile / lockfiles) for
  reproducible builds; `numpy==2.x` vs models expecting `<2.0` is a real risk.
- 🟢 **Python 3.10 base image** is nearing EOL; move to 3.11/3.12.
- 🟢 **No secret-scanning step** in CI.

---

## 7. Testing & CI

- 🟠 **Tests are empty stubs.** `src/tests/test_web.py` and `test_models.py`
  are all `pass`/`# TODO`. CI runs them and reports green — a false signal.
- 🟠 **No coverage** of the video/audio pipeline, the Flask routes, or any
  React component. Start with: Flask endpoint tests (decode/blur round-trip),
  a mediasoup signaling smoke test, and a frame-timing/back-pressure test.
- 🟡 CI is CPU-only; the GPU code path is never exercised.

---

## 8. Code Quality & Maintainability

- 🟡 **Logic duplicated** between `server.js` and `webrtc-video-processor.js`
  (`interpolateBoundingBoxes`, blur calls, processing state) — two copies of the
  same pipeline, one largely unused.
- 🟡 **Emoji-laden `console.log` on every frame** across the mediasoup server,
  processors, and frontend — these dominate the hot path. Gate behind a debug
  flag / use a leveled logger (`pino`, `debug`).
- 🟡 **`README` project structure is stale** — references a top-level
  `privastream/` package and `services/`, `api/`, `docs/` dirs that don't exist
  at those paths.
- 🟢 Two `## 🤝 Contributing` sections and a broken
  `your-username/tiktok-techjam-2025` link in the README.
- 🟢 `mediasoup/SLIDING_WINDOW_EXPLANATION.md` and committed `*.log` suggest docs
  and runtime output are mixed into source.

---

## 9. Prioritized Roadmap

| # | Action | Severity | Effort | Section |
|---|--------|----------|--------|---------|
| 1 | Move video to the existing WebRTC track; delete JPEG-over-socket | 🔴 | L | §1.1, §1.6 |
| 2 | Fix host capture FPS (20→4) + rVFC + in-flight guard | 🟠 | S | §1.2 *(done)* |
| 3 | Send audio/video as **binary**, not JSON arrays / base64 | 🔴 | S–M | §2.1, §1.3 |
| 4 | Fix un-redacted audio fallback (emit silence) | 🔴 | S | §2.4 |
| 5 | Replace per-frame `setTimeout` with one jitter buffer/room | 🟠 | M | §1.4 |
| 6 | Lock down CORS + add room auth tokens | 🔴 | M | §4 |
| 7 | `AudioWorklet` + anti-aliased resample | 🟠 | M | §2.2, §2.3 |
| 8 | Run Python under gunicorn; stop re-encoding JPEG | 🟠 | M | §3 |
| 9 | Remove committed `.next/`/logs; tighten `.gitignore` | 🟠 | S | §6 |
| 10 | Implement real tests; stop reporting empty green CI | 🟠 | L | §7 |

---

*Appendix — what was changed alongside this report:* the host frame-capture loop
in `app/host/page.tsx` was rewritten to address §1.2–§1.3 (correct, configurable
FPS via `requestVideoFrameCallback` with a back-pressure guard). All larger
architectural items above remain open and are scoped in §1.6 and §9.
