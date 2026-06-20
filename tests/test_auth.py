import time

import pytest

from src.web.backend import auth


def test_issue_and_verify_room_token(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret")

    token = auth.issue_room_token("room-1", role="host", ttl_seconds=60)
    payload = auth.verify_room_token(token, "room-1")

    assert payload["room_id"] == "room-1"
    assert payload["role"] == "host"


def test_verify_room_token_rejects_wrong_room_expired_and_malformed(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret")

    token = auth.issue_room_token("room-1", ttl_seconds=1)

    assert auth.verify_room_token(token, "room-2") is None
    assert auth.verify_room_token("not-a-token", "room-1") is None

    real_time = time.time
    monkeypatch.setattr(auth.time, "time", lambda: real_time() + 120)
    assert auth.verify_room_token(token, "room-1") is None


def test_issue_room_token_requires_secret(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError):
        auth.issue_room_token("room-1")
