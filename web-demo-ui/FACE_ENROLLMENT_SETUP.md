# Face Enrollment Setup Guide

This guide explains how to set up the Creator Face Enrollment system using InsightFace Buffalo_S.

## Overview

The face enrollment system provides:
- **Live face detection** with bounding boxes overlay on webcam video
- **Face enrollment** using multiple frames to compute average embeddings
- **Temporary storage** of face embeddings per room (in-memory only)
- **Integration** with the existing Mediasoup server

## Architecture

```
Frontend (React/Next.js) → Next.js API Routes → Python Flask Server (InsightFace Buffalo_S)
```

## Files Created/Modified

### Backend (Python Flask)
- `face_enrollment_server.py` - Main Flask server with InsightFace Buffalo_S
- `face_enrollment_requirements.txt` - Python dependencies
- `install_face_enrollment.bat` - Installation script
- `test_face_enrollment_api.py` - Test script

### Frontend (React/Next.js)
- `frontend-vindy-enrollment/components/enrollment/cameracapture.tsx` - Modified with live face detection
- `frontend-vindy-enrollment/app/api/face-detection/route.ts` - Next.js API route
- `frontend-vindy-enrollment/app/api/face-enrollment/route.ts` - Next.js API route
- `frontend-vindy-enrollment/enrollment/page.tsx` - Updated enrollment page

## Installation Steps

### 1. Install Python Dependencies

```bash
# Run the installation script
./install_face_enrollment.bat

# Or manually install:
pip install -r face_enrollment_requirements.txt
```

### 2. Start the Face Enrollment Server

```bash
python face_enrollment_server.py
```

The server will start on `http://localhost:5003`

### 3. Configure Environment (Optional)

Create `.env.local` in your Next.js app:

```env
FACE_ENROLLMENT_API_URL=http://localhost:5003
```

### 4. Start the Frontend

```bash
cd frontend-vindy-enrollment
npm run dev
```

## Usage Flow

### 1. Enrollment Process

1. **User visits enrollment page** (`/enrollment`)
2. **Progress through steps**:
   - Intro: Welcome and explanation
   - **Capture: Live face detection with webcam**
   - Form: Enter personal details
   - Preview: Review enrollment
   - Complete: Success confirmation

3. **Face Capture Step Features**:
   - Live webcam video with mirror effect
   - Real-time face detection with green bounding boxes
   - Face detection status indicator
   - "Enroll Face" button (primary action)
   - Manual photo capture (backup option)

4. **Enrollment Process**:
   - Collects 10 frames over 3 seconds
   - Shows progress bar during enrollment
   - Computes average face embedding using Buffalo_S
   - Stores embedding temporarily per room

### 2. Integration with Mediasoup

1. **Room ID Generation**: Each enrollment session gets a unique room ID
2. **Temporary Storage**: Face embeddings stored in memory per room
3. **Session Transfer**: Room ID passed to host page via sessionStorage
4. **Cleanup**: Embeddings automatically cleaned up when room ends

## API Endpoints

### Face Detection
```http
POST /api/face-detection
{
  "frame_data": "base64_image_data",
  "room_id": "room_123",
  "detect_only": true
}
```

### Face Enrollment
```http
POST /api/face-enrollment
{
  "frames": ["base64_frame1", "base64_frame2", ...],
  "room_id": "room_123"
}
```

### Room Status
```http
GET /api/room-status/{room_id}
```

### Cleanup
```http
DELETE /api/cleanup-room/{room_id}
```

## Technical Details

### InsightFace Buffalo_S Integration

- **Model**: Buffalo_S (lightweight, accurate face detection/recognition)
- **Detection**: Real-time face bounding boxes with confidence scores
- **Embedding**: 512-dimensional normalized face embeddings
- **Performance**: GPU-accelerated when available, CPU fallback

### Frontend Features

- **Live Detection**: 500ms intervals for face detection
- **Canvas Overlay**: Bounding boxes drawn on HTML5 canvas
- **Mirror Effect**: Video horizontally flipped for better UX
- **Progress Tracking**: Visual progress during enrollment
- **Error Handling**: Graceful fallbacks and user feedback

### Memory Management

- **Temporary Storage**: Embeddings stored per room in Python dict
- **Auto Cleanup**: Embeddings removed when room ends
- **No Persistence**: No database or file storage required
- **Privacy**: Face data never permanently saved

## Testing

### Test the API
```bash
python test_face_enrollment_api.py
```

### Health Check
```bash
curl http://localhost:5003/health
```

## Troubleshooting

### Common Issues

1. **InsightFace Installation**:
   ```bash
   pip install insightface
   pip install onnxruntime-gpu  # For GPU acceleration
   ```

2. **Model Download**: Buffalo_S model downloads automatically on first run

3. **CORS Issues**: Flask-CORS is configured for frontend access

4. **Port Conflicts**: Change port in `face_enrollment_server.py` if needed

### GPU Acceleration

- **CUDA**: Automatically used if available
- **CPU Fallback**: Works on CPU-only systems
- **Performance**: GPU provides 2-5x speedup for face detection

## Security Considerations

- **No Permanent Storage**: Face embeddings never saved to disk
- **Memory Only**: All data cleared when server restarts
- **Local Processing**: Face analysis happens locally
- **Privacy First**: No external API calls for face data

## Performance

- **Face Detection**: ~50-200ms per frame (GPU/CPU dependent)
- **Enrollment**: ~3 seconds for 10 frames + embedding computation
- **Memory Usage**: ~1KB per enrolled face embedding
- **Concurrent Rooms**: Supports multiple enrollment sessions

## Next Steps

1. **Integration**: Connect enrolled room ID to Mediasoup server
2. **Face Recognition**: Use stored embeddings for live face recognition
3. **Scaling**: Consider Redis for distributed deployments
4. **Monitoring**: Add logging and metrics for production use

## Dependencies

### Python
- `flask` - Web server
- `flask-cors` - Cross-origin requests
- `insightface` - Face detection/recognition
- `opencv-python` - Image processing
- `numpy` - Numerical computations
- `onnxruntime` - Model inference

### Frontend
- React/Next.js components
- HTML5 Canvas API
- WebRTC getUserMedia API
- TypeScript interfaces

## Support

For issues or questions:
1. Check server logs for error messages
2. Verify all dependencies are installed
3. Test with the provided test script
4. Ensure camera permissions are granted in browser