import pytest

from src.web.backend.video_api_utils import (
    RequestTracker,
    apply_blur_region,
    cleanup_expired_embeddings,
    clamp_region,
    is_stale_request,
    room_ids_from_request,
)


def test_stale_request_detection_handles_valid_and_malformed_timestamps():
    assert is_stale_request(900, 50, current_time_ms=1000) is True
    assert is_stale_request(980, 50, current_time_ms=1000) is False
    assert is_stale_request("bad", 50, current_time_ms=1000) is True
    assert is_stale_request("bad", 50, enabled=False, current_time_ms=1000) is False


def test_request_tracker_cleans_up_after_exception():
    tracker = RequestTracker(max_concurrent=1)

    with pytest.raises(RuntimeError):
        with tracker.track():
            assert tracker.active == 1
            assert tracker.can_start() is False
            raise RuntimeError("boom")

    assert tracker.active == 0
    assert tracker.can_start() is True


def test_region_clamping_and_blur_bounds():
    np = pytest.importorskip("numpy")

    frame = np.full((10, 10, 3), 255, dtype=np.uint8)

    assert clamp_region(frame.shape, [-5, -5, 20, 20]) == (0, 0, 10, 10)
    assert clamp_region(frame.shape, [5, 5, 5, 9]) is None
    assert apply_blur_region(frame, [-5, -5, 20, 20]) is True
    assert apply_blur_region(frame, [5, 5, 5, 9]) is False


def test_room_id_extraction_and_embedding_cleanup():
    assert room_ids_from_request("path-room", {"room_id": "body-room"}) == [
        "path-room",
        "body-room",
    ]

    embeddings = {
        "old": {"metadata": {"expires_at": 1}},
        "fresh": {"metadata": {"expires_at": 9999999999}},
    }

    assert cleanup_expired_embeddings(embeddings, ttl_seconds=60) == 1
    assert "old" not in embeddings
    assert "fresh" in embeddings

