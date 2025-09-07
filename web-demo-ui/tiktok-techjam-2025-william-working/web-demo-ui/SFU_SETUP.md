# SFU (Mediasoup) Setup Instructions

Your live streaming app has been successfully converted from P2P to SFU architecture using Mediasoup. This enables scalable streaming to hundreds of concurrent viewers.

## Architecture Overview

**Before (P2P):**
- Host creates individual connections to each viewer
- Limited to ~5-10 viewers due to bandwidth constraints
- High CPU/bandwidth usage on host

**After (SFU):**
- Host uploads once to Mediasoup server
- Server distributes to all viewers
- Scales to 100s-1000s of concurrent viewers
- Low latency, efficient bandwidth usage

## New Components

### 1. Mediasoup Server (`mediasoup-server/`)
- **Purpose**: Media routing and distribution
- **Technology**: Node.js + Mediasoup library
- **Port**: 3001
- **Features**: Room management, producer/consumer handling, WebRTC transport

### 2. Updated Backend (`backend/app.py`)
- **Changes**: Added SFU signaling events
- **New Events**: `sfu_streaming_started`, `sfu_streaming_stopped`
- **Purpose**: Coordinates between users and SFU server

### 3. Updated Frontend
- **New Library**: `mediasoup-client` for SFU communication
- **New File**: `lib/mediasoup-client.ts` - SFU client wrapper
- **Updated**: Host and viewer interfaces for SFU architecture

## Installation & Setup

### Step 1: Install Mediasoup Server Dependencies
```bash
cd mediasoup-server
npm install
```

### Step 2: Update Frontend Dependencies
```bash
cd frontend
npm install
```

### Step 3: Update Backend Dependencies
```bash
cd backend
# Activate your virtual environment first
pip install -r requirements.txt
```

## Running the SFU Application

You now need to run **3 servers** simultaneously:

### Terminal 1: Mediasoup Server
```bash
cd mediasoup-server
npm start
# Server runs on http://localhost:3001
```

### Terminal 2: Flask Backend (Signaling)
```bash
cd backend
python app.py
# Server runs on http://localhost:5000
```

### Terminal 3: Next.js Frontend
```bash
cd frontend
npm run dev
# Server runs on http://localhost:3000
```

## Testing the SFU Implementation

1. **Start all 3 servers** in the order above
2. **Host**: Navigate to `http://localhost:3000/host`
   - Click "Start SFU Streaming"
   - Copy the Room ID
3. **Viewers**: Open multiple browser tabs/windows to `http://localhost:3000/viewer/[ROOM_ID]`
4. **Verify**: All viewers should see the host's stream simultaneously

## Key Differences from P2P Version

### For Hosts:
- Button now says "Start SFU Streaming"
- Status shows "Live (SFU)"
- Single upload stream distributed by server

### For Viewers:
- Status shows "Watching Live Stream (SFU)"
- Lower latency due to optimized routing
- Better quality with server-side optimization

## Scalability Benefits

| Metric | P2P (Old) | SFU (New) |
|--------|-----------|-----------|
| Max Viewers | ~10 | 1000+ |
| Host Upload | N × stream | 1 × stream |
| Host CPU | High | Low |
| Latency | Variable | Consistent |
| Quality | Degrades | Stable |

## Troubleshooting

### Common Issues:

1. **"Connection failed"**
   - Ensure all 3 servers are running
   - Check console for specific errors

2. **No video/audio**
   - Verify camera/microphone permissions
   - Check browser console for WebRTC errors

3. **Port conflicts**
   - Mediasoup: 3001
   - Flask backend: 5000  
   - Next.js frontend: 3000

### Production Deployment Notes:

1. **Update IP addresses** in `mediasoup-server/server.js`:
   ```javascript
   announcedIp: 'YOUR_SERVER_PUBLIC_IP'
   ```

2. **Configure firewall** for UDP ports 10000-10100 (RTC)

3. **Use HTTPS** for production (required for camera/microphone access)

## Next Steps

Your SFU implementation is now ready for:
1. **Production deployment** with proper IP configuration
2. **Filter implementation** - Add media processing in Mediasoup server
3. **Recording features** - Save streams server-side
4. **Analytics** - Track viewer metrics and stream quality

The foundation is solid for scaling to large audiences! 🚀