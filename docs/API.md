# PrivaStream API

PrivaStream runs three backend-facing services:

- Room backend: Flask + Flask-SocketIO on `http://localhost:5000`
- Video filter API: Flask on `http://localhost:5001`
- Mediasoup SFU: Node.js on `http://localhost:3001`

Frontend configuration uses `NEXT_PUBLIC_BACKEND_URL`, `NEXT_PUBLIC_VIDEO_API_URL`, `NEXT_PUBLIC_SFU_URL`, and `NEXT_PUBLIC_AUDIO_API_URL`. Server-side SFU configuration uses `BACKEND_URL`, `VIDEO_API_URL`, `SFU_URL`, and `AUDIO_API_URL`.

## Authentication

Room tokens are HMAC-SHA256 tokens issued by the room backend. Enforcement is controlled by `REQUIRE_AUTH`.

- `REQUIRE_AUTH=false`: local development compatibility mode.
- `REQUIRE_AUTH=true`: protected room, SFU, and video-filter routes require `Authorization: Bearer <token>`.

Tokens bind a caller to a `room_id` and expire. The host receives a token from the `create_room` Socket.IO event. Viewers receive a token from `join_room`.

## Room Backend

### `GET /health`

Returns service readiness and in-memory room counts.

```json
{
  "status": "healthy",
  "rooms": 1,
  "users": 2,
  "expired_rooms_cleaned": 0
}
```

### Socket.IO

Path: `/backend/socket.io`

Client events:

- `create_room`
- `join_room` with `{ "roomId": "..." }`
- `sfu_streaming_started`
- `sfu_streaming_stopped`
- `get_room_info`

Server events:

- `connected`
- `room_created` with `{ "roomId": "...", "mediasoupUrl": "...", "token": "..." }`
- `joined_room` with `{ "roomId": "...", "mediasoupUrl": "...", "token": "..." }`
- `viewer_joined`
- `viewer_left`
- `host_disconnected`
- `streaming_started`
- `streaming_stopped`
- `room_info`
- `error`

## Video Filter API

### `GET /health`

Returns API readiness, detector readiness, and room embedding counts.

### `POST /process-frame`

JSON-compatible frame processing route retained for current clients.

Request:

```json
{
  "frame": "data:image/jpeg;base64,...",
  "timestamp": 1760000000000,
  "frame_id": 123,
  "room_id": "room-id",
  "blur_only": false,
  "detect_only": false,
  "rectangles": [[0, 0, 120, 90]]
}
```

Response:

```json
{
  "success": true,
  "frame_id": 123,
  "frame": "data:image/jpeg;base64,...",
  "rectangles": [[0, 0, 120, 90]],
  "processing_mode": "full",
  "regions_processed": 1,
  "timings": {
    "decode_ms": 1.2,
    "detect_ms": 14.3,
    "blur_ms": 0.8,
    "encode_ms": 2.1,
    "total_ms": 18.4
  }
}
```

### `POST /process-frame-binary`

Binary-compatible variant. The request body is encoded image bytes. Metadata is provided through headers:

- `X-Room-Id`
- `X-Frame-Id`
- `X-Timestamp-Ms`
- `X-Blur-Only`
- `X-Detect-Only`
- `X-Rectangles` as JSON

The response is `image/jpeg` unless `X-Detect-Only=true`, in which case JSON detection metadata is returned. Response headers include `X-Frame-Id`, `X-Regions-Processed`, and `X-Timings`.

### `POST /detect-faces-mouths`

Runs one detector pass and returns coordinate metadata for the host-side blur pipeline.

Response fields include:

- `face_blur_regions`
- `mouth_regions`
- `pii_regions`
- `plate_regions`
- `timings`
- `faces_to_blur`
- `pii_count`
- `plate_count`

### `POST /apply-conditional-blur`

Applies blur using provided face, mouth, PII, and plate regions. This is kept for compatibility with the SFU frame path.

### Enrollment Routes

- `POST /face-detection`
- `POST /face-enrollment`
- `GET /room-status/<room_id>`
- `DELETE /cleanup-room/<room_id>`
- `POST /cleanup-room/<room_id>`
- `POST /transfer-embedding`

When `REQUIRE_AUTH=true`, these routes require a valid token for the path room, source room, or target room depending on the request.

### Queue Routes

- `GET /queue-status`
- `POST /queue-config`

Queue protection can reject stale frames or overload with configured status codes. Defaults are controlled by `VIDEO_FILTER_MAX_REQUEST_AGE_MS`, `VIDEO_FILTER_MAX_CONCURRENT_REQUESTS`, `VIDEO_FILTER_STALE_STATUS_CODE`, and `VIDEO_FILTER_OVERLOAD_STATUS_CODE`.

## Mediasoup SFU

### `GET /health`

Returns SFU readiness.

Socket.IO uses `/mediasoup/socket.io`. When `REQUIRE_AUTH=true`, the client must connect with `auth: { token }`, and room-specific events are checked against the token room.

## Error Handling

Client errors use generic JSON responses such as:

```json
{ "error": "Invalid image data" }
```

Internal exceptions are logged server-side and returned as:

```json
{ "error": "Internal server error" }
```
