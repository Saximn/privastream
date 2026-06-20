"""Small testable helpers for the video filter API."""

from __future__ import annotations

import base64
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Optional


DATA_URL_PREFIX = "base64,"


def now_ms() -> int:
    return int(time.time() * 1000)


def is_stale_request(
    request_timestamp_ms: Any,
    max_age_ms: int,
    *,
    enabled: bool = True,
    current_time_ms: Optional[int] = None,
) -> bool:
    """Return True when a client timestamp is too old or malformed."""
    if not enabled:
        return False
    try:
        timestamp = int(request_timestamp_ms)
    except (TypeError, ValueError):
        return True
    current = current_time_ms if current_time_ms is not None else now_ms()
    return current - timestamp > max_age_ms


class RequestTracker:
    """Thread-safe concurrent request counter."""

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max(1, int(max_concurrent))
        self.active = 0
        self._lock = threading.Lock()

    def can_start(self) -> bool:
        with self._lock:
            return self.active < self.max_concurrent

    def start(self) -> int:
        with self._lock:
            self.active += 1
            return self.active

    def finish(self) -> int:
        with self._lock:
            self.active = max(0, self.active - 1)
            return self.active

    def configure(self, max_concurrent: int):
        with self._lock:
            self.max_concurrent = max(1, int(max_concurrent))

    @contextmanager
    def track(self) -> Iterator[None]:
        self.start()
        try:
            yield
        finally:
            self.finish()


def strip_data_url(frame_data: str) -> str:
    if frame_data.startswith("data:image") and DATA_URL_PREFIX in frame_data:
        return frame_data.split(DATA_URL_PREFIX, 1)[1]
    return frame_data


def decode_image_bytes(image_bytes: bytes):
    """Decode raw encoded image bytes into an OpenCV BGR frame."""
    import cv2
    import numpy as np

    if not image_bytes:
        raise ValueError("empty image")
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid image")
    return frame


def decode_base64_image(frame_data: str):
    """Decode a base64 or data URL encoded image into an OpenCV BGR frame."""
    try:
        image_bytes = base64.b64decode(strip_data_url(frame_data), validate=True)
    except Exception as exc:
        raise ValueError("invalid base64 image") from exc
    return decode_image_bytes(image_bytes)


def encode_jpeg(frame, quality: int = 85) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buffer.tobytes()


def encode_jpeg_data_url(frame, quality: int = 85) -> str:
    encoded = base64.b64encode(encode_jpeg(frame, quality=quality)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def clamp_region(frame_shape: tuple[int, ...], region: Iterable[Any]) -> Optional[tuple[int, int, int, int]]:
    """Normalize and clamp [x1, y1, x2, y2] to image bounds."""
    values = list(region)
    if len(values) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in values[:4]]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    h, w = frame_shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def apply_blur_region(frame, region: Iterable[Any], kernel_size: int = 75) -> bool:
    import cv2

    box = clamp_region(frame.shape, region)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    frame[y1:y2, x1:x2] = cv2.blur(roi, (kernel_size, kernel_size))
    return True


def polygons_to_rectangles(polygons: Iterable[Any]) -> list[list[int]]:
    """Convert polygon-like arrays/lists to bounding rectangles."""
    rectangles: list[list[int]] = []
    for poly in polygons or []:
        try:
            points = poly.tolist() if hasattr(poly, "tolist") else poly
            xs = [int(float(p[0])) for p in points]
            ys = [int(float(p[1])) for p in points]
            if xs and ys:
                rectangles.append([min(xs), min(ys), max(xs), max(ys)])
        except Exception:
            continue
    return rectangles


def extract_model_regions(results: Mapping[str, Any]) -> dict[str, list[Any]]:
    models = results.get("models", {}) if isinstance(results, Mapping) else {}
    face_regions = list(models.get("face", {}).get("rectangles", []) or [])
    plate_regions = list(models.get("plate", {}).get("rectangles", []) or [])
    pii_model = models.get("pii", {})
    pii_regions = list(pii_model.get("rectangles", []) or [])
    if not pii_regions:
        pii_regions = polygons_to_rectangles(pii_model.get("polygons", []) or [])
    return {
        "face": face_regions,
        "mouth": list(models.get("mouth", {}).get("rectangles", []) or []),
        "pii": pii_regions,
        "plate": plate_regions,
    }


def room_ids_from_request(path_room_id: Optional[str], body: Mapping[str, Any]) -> list[str]:
    candidates = [
        path_room_id,
        body.get("room_id"),
        body.get("roomId"),
        body.get("from_room_id"),
        body.get("to_room_id"),
    ]
    return [str(value) for value in candidates if value]


def cleanup_expired_embeddings(room_embeddings: dict[str, dict[str, Any]], ttl_seconds: int) -> int:
    now = time.time()
    expired = []
    for room_id, data in room_embeddings.items():
        metadata = data.get("metadata", {})
        expires_at = data.get("expires_at") or metadata.get("expires_at")
        if expires_at and float(expires_at) <= now:
            expired.append(room_id)
    for room_id in expired:
        room_embeddings.pop(room_id, None)
    return len(expired)


def set_embedding_expiry(entry: dict[str, Any], ttl_seconds: int) -> dict[str, Any]:
    expires_at = time.time() + ttl_seconds
    entry["expires_at"] = expires_at
    entry.setdefault("metadata", {})["expires_at"] = expires_at
    return entry


def json_header(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))

