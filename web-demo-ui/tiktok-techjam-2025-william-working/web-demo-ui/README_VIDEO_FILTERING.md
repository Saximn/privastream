# Video Filtering Integration

## Quick Start

1. **Start the Video Filter API:**
   ```bash
   cd backend
   pip install -r requirements_video_filter.txt
   python video_filter_api.py
   ```
   Or run: `run_video_filter.bat`

2. **Start your existing servers:**
   - Backend: `python backend/app.py` (port 5000)
   - MediaSoup: `node mediasoup-server/server.js` (port 3001)
   - Frontend: `npm run dev` (port 3000)

3. **Go to Host page and enable Video Filter before starting stream**

## Architecture

**Client-side filtering with WebRTC insertable streams:**

```
Camera → WebRTC Transform → Python API (4fps detection) → Blur Application → MediaSoup → Viewers
```

## Components Added

### 1. Backend API (`video_filter_api.py`)
- HTTP endpoint: `POST /filter-frame` 
- Uses your `unified_detector.py` and `video_models`
- Processes single frames, returns blur regions

### 2. Frontend Transform (`video-filter.ts`)
- WebRTC insertable streams transform
- Processes 30fps video, sends 4fps to API
- Applies blur client-side using canvas
- Supports gaussian blur, pixelation, solid fill

### 3. MediaSoup Integration (`mediasoup-client.ts`)
- Added `produce(stream, enableFiltering)` parameter  
- Creates filtered stream before sending to MediaSoup
- Provides filter stats and config updates

### 4. UI Controls (`host/page.tsx`)
- Video filter toggle (must enable before streaming)
- Real-time statistics display
- Filter configuration options

## Browser Compatibility

Requires browsers with insertable streams support:
- Chrome 86+
- Firefox 94+
- Safari 15.4+

## Performance

- **4fps detection** → **30fps output**
- Client-side blur application (no server processing load)
- Async detection pipeline (no frame drops)
- Cached blur regions with persistence

## Detection Models Used

Your existing `video_models/`:
- Face detection with whitelist
- PII text detection 
- License plate detection
- Unified blur utilities

## Testing

1. Enable video filter on host page
2. Start streaming
3. Show face/text/license plate to camera
4. Verify blur applied in real-time
5. Check filter stats for detection counts