# PrivaStream API Documentation

## Overview

PrivaStream provides both REST API endpoints and WebSocket connections for real-time privacy filtering functionality.

## Base URLs

- **Development**: `http://localhost:5000`
- **Production**: Configure based on your deployment

## Authentication

Currently, the API doesn't require authentication. For production deployments, implement proper authentication mechanisms.

## REST API Endpoints

### Health Check

**GET** `/health`

Check if the service is running.

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Process Video/Audio

**POST** `/api/v1/process`

Process video or audio files for PII detection and redaction.

**Request Body**
```json
{
  "input_path": "path/to/input/file.mp4",
  "output_path": "path/to/output/file.mp4",
  "config": {
    "video": {
      "face_detection": true,
      "plate_detection": true,
      "text_detection": true
    },
    "audio": {
      "enabled": true,
      "chunk_seconds": 5
    }
  }
}
```

**Response**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "estimated_time": 120
}
```

### Job Status

**GET** `/api/v1/jobs/{job_id}`

Get the status of a processing job.

**Response**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 100,
  "output_path": "path/to/output/file.mp4",
  "processing_time": 95.2
}
```

### Model Information

**GET** `/api/v1/models`

Get information about available models.

**Response**
```json
{
  "models": {
    "face_detection": {
      "name": "YOLO Face Detector",
      "version": "1.0.0",
      "accuracy": 98.38
    },
    "plate_detection": {
      "name": "License Plate Detector",
      "version": "1.0.0",
      "map50": 96.47
    },
    "audio_pii": {
      "name": "DeBERTa PII Classifier",
      "version": "1.0.0",
      "accuracy": 96.99
    }
  }
}
```

## WebSocket API

Connect to WebSocket at `/socket.io/`

### Events from Client

#### `create_room`
Create a new streaming room.
```javascript
socket.emit('create_room');
```

#### `join_room`
Join an existing room.
```javascript
socket.emit('join_room', { roomId: 'room-uuid' });
```

#### `sfu_streaming_started`
Notify that SFU streaming has started.
```javascript
socket.emit('sfu_streaming_started', { roomId: 'room-uuid' });
```

#### `start_processing`
Start privacy filtering on the stream.
```javascript
socket.emit('start_processing', {
  roomId: 'room-uuid',
  config: {
    face_blur: true,
    plate_blur: true,
    text_blur: true,
    audio_processing: true
  }
});
```

### Events from Server

#### `room_created`
Room successfully created.
```javascript
socket.on('room_created', (data) => {
  console.log('Room ID:', data.roomId);
  console.log('Mediasoup URL:', data.mediasoupUrl);
});
```

#### `streaming_started`
Streaming has started in the room.
```javascript
socket.on('streaming_started', (data) => {
  console.log('Streaming started in room:', data.roomId);
});
```

#### `processing_update`
Real-time processing updates.
```javascript
socket.on('processing_update', (data) => {
  console.log('FPS:', data.fps);
  console.log('Detections:', data.detections);
});
```

#### `error`
Error occurred during processing.
```javascript
socket.on('error', (data) => {
  console.error('Error:', data.message);
});
```

## WebRTC SFU Integration

PrivaStream uses Mediasoup for WebRTC SFU functionality on port 3001.

### Connection Flow

1. Client connects to main backend via WebSocket
2. Backend creates room and returns Mediasoup endpoint
3. Client connects to Mediasoup SFU for media transport
4. Privacy filtering is applied to media streams in real-time

### Mediasoup Endpoints

- **HTTP API**: `http://localhost:3001/rooms`
- **WebSocket**: `ws://localhost:3001/`

## Configuration

### Video Processing Config
```json
{
  "video": {
    "models": {
      "face": {
        "enabled": true,
        "confidence": 0.4
      },
      "license": {
        "enabled": true,
        "confidence": 0.25
      },
      "pii_text": {
        "enabled": true,
        "ocr": "doctr",
        "confidence_gate": 0.35
      }
    },
    "blur": {
      "type": "gaussian",
      "kernel_size": 41,
      "padding": 4
    }
  }
}
```

### Audio Processing Config
```json
{
  "audio": {
    "whisper_model": "small",
    "chunk_seconds": 5,
    "mouth_blur_window_ms": 500
  }
}
```

## Error Handling

All API endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found
- `500` - Internal Server Error

Error responses include details:
```json
{
  "error": "Invalid input format",
  "message": "Unsupported file type: .xyz",
  "code": "INVALID_FORMAT"
}
```

## Rate Limiting

Currently no rate limiting is implemented. Consider implementing rate limiting for production deployments.

## SDK Examples

### Python
```python
import requests

# Process video
response = requests.post('http://localhost:5000/api/v1/process', json={
    'input_path': 'video.mp4',
    'output_path': 'blurred.mp4',
    'config': {'video': {'face_detection': True}}
})

print(response.json())
```

### JavaScript
```javascript
// WebSocket connection
const socket = io('http://localhost:5000');

socket.emit('create_room');
socket.on('room_created', (data) => {
    console.log('Room created:', data.roomId);
});
```

### cURL
```bash
# Health check
curl http://localhost:5000/health

# Process video
curl -X POST http://localhost:5000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"input_path": "input.mp4", "output_path": "output.mp4"}'
```

## Performance Considerations

- Video processing is CPU/GPU intensive
- Real-time streaming requires sufficient bandwidth
- Consider using smaller models for faster processing
- GPU acceleration significantly improves performance

## Security

- Validate all file paths to prevent directory traversal
- Implement proper authentication for production
- Use HTTPS in production environments
- Sanitize user inputs