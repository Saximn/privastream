"""Signed room-join tokens (HMAC-SHA256, stdlib only).

A compact `base64url(payload).base64url(sig)` token, signed with SECRET_KEY.
Deliberately dependency-free so the mediasoup (Node) server can verify the same
token with built-in crypto (see server.js verifyRoomToken) and so it is unit
testable without installing anything.

Enforcement across services is gated by the REQUIRE_AUTH env var; the token
mechanism itself is always available for issuing.
"""
import base64
import hashlib
import hmac
import json
import os
import time


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _secret() -> bytes:
    secret = os.getenv("SECRET_KEY")
    if not secret:
        raise RuntimeError("SECRET_KEY must be set to sign/verify room tokens")
    return secret.encode("utf-8")


def auth_required() -> bool:
    """Whether token enforcement is on (default off for local dev)."""
    return os.getenv("REQUIRE_AUTH", "false").lower() in ("1", "true", "yes")


def issue_room_token(room_id: str, role: str = "viewer", ttl_seconds: int = 6 * 3600) -> str:
    """Mint a signed token binding a client to a room and role."""
    payload = {
        "room_id": room_id,
        "role": role,
        "exp": int(time.time()) + ttl_seconds,
    }
    body = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64u_encode(hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_room_token(token: str, room_id: str = None):
    """Return the payload dict if valid (and matches room_id), else None."""
    try:
        body, sig = token.split(".", 1)
        expected = _b64u_encode(
            hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64u_decode(body))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        if room_id is not None and payload.get("room_id") != room_id:
            return None
        return payload
    except Exception:
        return None
