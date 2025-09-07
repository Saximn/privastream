# 🔧 FFmpeg Audio Debug Guide

## ✅ **Current Status:**
- **Audio to viewers**: ✅ Working (you can hear audio)
- **FFmpeg integration**: 🔧 Added with comprehensive logging
- **Debug logging**: ✅ Added to trace the entire pipeline

## 🎯 **What to Look For When Testing:**

### **1. Restart MediaSoup Server:**
```bash
cd mediasoup-server
node server.js
```

### **2. Expected Logs When Starting Stream:**

**Audio Producer Registration:**
```
[SERVER] Setting up audio redaction for producer: [producer-id]
🎤 Registering audio producer for background redaction processing in room [room-id]: [producer-id]
🎬 Setting up FFmpeg audio consumption for room [room-id]
✅ Created PlainTransport for FFmpeg: [transport-id]
   RTP Port: [port-number]
✅ Created audio consumer: [consumer-id]
   Codec: audio/opus
   Payload Type: [payload-type]
🎬 PlainTransport details:
   Local RTP port: [port]
   Codec: audio/opus, Payload: [payload]
🎬 Starting FFmpeg process for room [room-id]...
🎬 FFmpeg command: ffmpeg -protocol_whitelist file,udp,rtp -f rtp -i rtp://127.0.0.1:[port] -f wav -acodec pcm_s16le -ar 16000 -ac 1 -loglevel info pipe:1
```

**FFmpeg Connection:**
```
🔗 Connecting PlainTransport to start RTP flow...
✅ PlainTransport connected - RTP data should now flow to FFmpeg
✅ FFmpeg audio consumption started for room [room-id]
```

**Audio Data Flow:**
```
📊 FFmpeg stdout chunk: [X] bytes, total: [Y] bytes
📊 FFmpeg stdout chunk: [X] bytes, total: [Y] bytes
...
🎵 Processing [96000] bytes of audio from FFmpeg for room [room-id]
🎵 processAccumulatedAudio called for room [room-id]
📊 Audio buffer size: [96000] bytes
🎵 Sending [96000] bytes to Flask API for processing (room: [room-id])
✅ Audio processed: [N] PII detections, processing complete
```

### **3. What Each Log Means:**

| Log Message | Meaning |
|-------------|---------|
| `🎬 Setting up FFmpeg audio consumption` | Audio redaction plugin activated |
| `✅ Created PlainTransport for FFmpeg` | MediaSoup transport created |
| `🎬 Starting FFmpeg process` | FFmpeg process launched |
| `📊 FFmpeg stdout chunk` | **CRITICAL**: FFmpeg receiving RTP data |
| `🎵 Processing X bytes` | Audio accumulation reached threshold |
| `🎵 processAccumulatedAudio called` | Starting to send to Flask API |
| `✅ Audio processed` | Flask API successfully processed audio |

### **4. Debug Files Created:**

**Location:** `audio-william/debug_audio/`

**Files:**
- `incoming_YYYYMMDD_HHMMSS.wav` - Original audio from microphone
- `redacted_YYYYMMDD_HHMMSS.wav` - Processed audio (muted if PII found)

### **5. Troubleshooting:**

**If you see FFmpeg library versions but no stdout chunks:**
- RTP data is not flowing from MediaSoup to FFmpeg
- Check PlainTransport connection timing

**If you see stdout chunks but no processing:**
- Audio accumulation threshold not reached (need 96KB)
- Check if chunks are small or infrequent

**If processing happens but no debug files:**
- Flask API issue
- Check `audio-william/debug_audio/` folder permissions

**If debug files created but empty:**
- FFmpeg PCM conversion issue
- Check codec compatibility

## 🧪 **Quick Test Command:**

After starting both servers, run:
```bash
node debug_audio_status.js
```

This will show you the current status of all components.

## 🎬 **Expected FFmpeg Behavior:**

1. **Startup**: FFmpeg shows library versions (what you're seeing now)
2. **Waiting**: FFmpeg waits for RTP data on assigned port
3. **Receiving**: When you speak, RTP packets arrive
4. **Processing**: FFmpeg converts RTP → PCM → stdout chunks
5. **Accumulation**: Node.js collects chunks until 3 seconds worth
6. **API Call**: Sends accumulated audio to Flask API
7. **Debug Files**: Flask API saves incoming/redacted audio files

The fact that you're seeing FFmpeg library version logs means the process is starting correctly. The next step is to see if the `📊 FFmpeg stdout chunk` logs appear when you speak into the microphone.