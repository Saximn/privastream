import base64
import time

import pytest

from src.web.backend import auth
from src.web.backend import video_filter_api as api


def _jpeg_data_url():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((8, 8, 3), 255, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


@pytest.fixture(autouse=True)
def reset_video_api(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUTH", "false")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    api.room_embeddings.clear()
    api.detector = None
    api.request_tracker.active = 0
    api.QUEUE_CONFIG["enable_request_dropping"] = True
    api.QUEUE_CONFIG["max_request_age_ms"] = 1000
    api.QUEUE_CONFIG["max_concurrent_requests"] = 10
    api.request_tracker.configure(10)
    yield
    api.request_tracker.active = 0
    api.room_embeddings.clear()
    api.detector = None


def test_video_health_and_malformed_frame_rejection():
    client = api.app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "healthy"

    response = client.post(
        "/process-frame",
        json={"frame": "not-valid-base64", "timestamp": int(time.time() * 1000)},
    )
    assert response.status_code == 400


def test_process_frame_stale_and_overload_rejections():
    client = api.app.test_client()

    stale = client.post(
        "/process-frame",
        json={"frame": "not-valid-base64", "timestamp": int(time.time() * 1000) - 5000},
    )
    assert stale.status_code == api.QUEUE_CONFIG["stale_status_code"]
    assert stale.get_json()["reason"] == "stale_request"

    api.QUEUE_CONFIG["max_concurrent_requests"] = 1
    api.request_tracker.configure(1)
    api.request_tracker.start()
    try:
        overloaded = client.post(
            "/process-frame",
            json={"frame": "not-valid-base64", "timestamp": int(time.time() * 1000)},
        )
    finally:
        api.request_tracker.finish()

    assert overloaded.status_code == api.QUEUE_CONFIG["overload_status_code"]
    assert overloaded.get_json()["reason"] == "overloaded"


def test_process_frame_blur_only_success_cleans_request_counter():
    client = api.app.test_client()
    response = client.post(
        "/process-frame",
        json={
            "frame": _jpeg_data_url(),
            "timestamp": int(time.time() * 1000),
            "blur_only": True,
            "rectangles": [[-4, -4, 20, 20]],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["regions_processed"] == 1
    assert api.request_tracker.active == 0


def test_auth_required_room_status_cleanup_and_transfer(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    client = api.app.test_client()
    api.room_embeddings["source-room"] = {"embedding": [1, 2, 3], "metadata": {}}

    missing = client.get("/room-status/source-room")
    assert missing.status_code == 401

    wrong_token = auth.issue_room_token("other-room")
    wrong = client.get(
        "/room-status/source-room",
        headers={"Authorization": f"Bearer {wrong_token}"},
    )
    assert wrong.status_code == 401

    source_token = auth.issue_room_token("source-room")
    status = client.get(
        "/room-status/source-room",
        headers={"Authorization": f"Bearer {source_token}"},
    )
    assert status.status_code == 200
    assert status.get_json()["enrolled"] is True

    target_token = auth.issue_room_token("target-room")
    transfer = client.post(
        "/transfer-embedding",
        headers={"Authorization": f"Bearer {target_token}"},
        json={"from_room_id": "source-room", "to_room_id": "target-room"},
    )
    assert transfer.status_code == 200
    assert "target-room" in api.room_embeddings

    cleanup = client.delete(
        "/cleanup-room/target-room",
        headers={"Authorization": f"Bearer {target_token}"},
    )
    assert cleanup.status_code == 200
    assert "target-room" not in api.room_embeddings


def test_detect_faces_mouths_uses_single_detector_pass():
    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def process_frame(self, frame, frame_id, stride=1, room_id=None):
            self.calls += 1
            return {
                "models": {
                    "face": {"rectangles": [[1, 1, 4, 4]]},
                    "pii": {"polygons": [[[2, 2], [5, 2], [5, 5], [2, 5]]]},
                    "plate": {"rectangles": [[0, 0, 2, 2]]},
                }
            }

    api.detector = FakeDetector()
    client = api.app.test_client()
    response = client.post(
        "/detect-faces-mouths",
        json={
            "frame": _jpeg_data_url(),
            "timestamp": int(time.time() * 1000),
            "frame_id": 7,
            "room_id": "room-1",
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["faces_to_blur"] == 1
    assert body["pii_count"] == 1
    assert body["plate_count"] == 1
    assert api.detector.calls == 1

