// server.js
const express = require('express');
const { Server } = require('socket.io');
const http = require('http');
const cors = require('cors');
const mediasoup = require('mediasoup');
const { Transform } = require('stream');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

require('dotenv').config();
const API_CONFIG = require('./config').default;

// Try to import node-fetch, fallback to http if needed
let fetch;
try {
  fetch = require('node-fetch');
} catch (err) {
  console.warn('[SERVER] node-fetch not available, using http fallback');
  fetch = null;
}

// TIMING CONFIGURATION - Easy to adjust
const TIMING_CONFIG = {
  AUDIO_CHUNK_DURATION: 3000,      // Audio processing takes 3s
  TOTAL_VIEWER_DELAY: 6000,        // Configurable total delay (can change to 5s later)
  PROCESSING_BUFFER: 100,          // Small buffer after audio completes (100ms)
  CACHE_CLEANUP_INTERVAL: 30000    // Cleanup every 30s
};

// Calculate when to deliver based on original capture time
function calculateDeliveryDelay(originalTimestamp) {
  // Handle invalid timestamps
  if (!originalTimestamp || isNaN(originalTimestamp)) {
    console.warn('[TIMING] Invalid timestamp provided, using current time as fallback');
    originalTimestamp = Date.now();
  }
  
  const targetDeliveryTime = originalTimestamp + TIMING_CONFIG.TOTAL_VIEWER_DELAY;
  const currentTime = Date.now();
  const delay = Math.max(0, targetDeliveryTime - currentTime);
  
  // Additional safety check for NaN results
  if (isNaN(delay)) {
    console.warn('[TIMING] Calculated delay is NaN, using minimum delay of 100ms');
    return 100;
  }
  
  return delay;
}

console.log(`[CONFIG] Audio processing: ${TIMING_CONFIG.AUDIO_CHUNK_DURATION}ms`);
console.log(`[CONFIG] Total viewer delay: ${TIMING_CONFIG.TOTAL_VIEWER_DELAY}ms`);
console.log(`[CONFIG] Video processing triggers: T+${TIMING_CONFIG.AUDIO_CHUNK_DURATION + TIMING_CONFIG.PROCESSING_BUFFER}ms`);

console.log(API_CONFIG);

// Import Audio Redaction Processor
const { AudioRedactionProcessor } = require('./audio-redaction-processor');
// Import WebRTC Video Processor
const { WebRTCVideoProcessor } = require('./webrtc-video-processor');

// Initialize Audio Redaction Processor  
const audioRedactionProcessor = new AudioRedactionProcessor({
  redactionServiceUrl: API_CONFIG.AUDIO_API_URL,
  sampleRate: 16000,  // Match Vosk requirements
  channels: 1,        // Mono for better transcription
  bufferDurationMs: 3000
});

// Initialize WebRTC Video Processor
const videoProcessor = new WebRTCVideoProcessor({
  videoServiceUrl: `${API_CONFIG.VIDEO_API_URL}`,
  frameRate: 30,
  processEveryNthFrame: 15,
  bufferDurationMs: 3000  // Match audio processing timing
});

const app = express();
const server = http.createServer(app);
app.use(cors());
app.use(express.json());

const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] }
});

// Mediasoup codec config
const mediaCodecs = [
  { kind: 'audio', mimeType: 'audio/opus', clockRate: 48000, channels: 2 },
  { kind: 'video', mimeType: 'video/VP8', clockRate: 90000, parameters: { 'x-google-start-bitrate': 1000 } }
];

const webRtcTransportOptions = {
  listenIps: [{ ip: '0.0.0.0', announcedIp: '127.0.0.1' }],
  enableUdp: true,
  enableTcp: true,
  preferUdp: true
};

// Global Mediasoup objects
let worker, router;
const rooms = new Map();          // roomId -> room object
const transports = new Map();     // transportId -> transport
const producers = new Map();      // producerId -> producer
const consumers = new Map();      // consumerId -> consumer
const processedProducers = new Map(); // originalProducerId -> processedProducerId
const frameProcessingState = new Map(); // roomId -> { frameCount, lastDetection, lastProcessedFrame }

// Enhanced caches for mouth blurring coordination
const detectionCache = new Map();     // frameId -> detection results
const piiEventsBuffer = new Map();    // roomId -> PII events array  
const frameAudioMapping = new Map();  // roomId -> {pendingFrames: [frameId], audioChunkCount}

// Frame buffer for video filters
const frameBuffer = new Map(); // roomId -> [{frame, timestamp}]

// Audio chunk timing tracking for sync
const audioChunkStartTimes = new Map(); // roomId -> [startTime1, startTime2, ...]

// Initialize Mediasoup
async function createWorker() {
  worker = await mediasoup.createWorker({ rtcMinPort: 10000, rtcMaxPort: 10100 });
  worker.on('died', () => setTimeout(() => process.exit(1), 2000));
}

async function createRouter() {
  router = await worker.createRouter({ mediaCodecs });
}

async function init() {
  await createWorker();
  await createRouter();
}

// Python frame processing
async function sendToPython(frameB64, roomId = null) {
  if (fetch) {
    // Use node-fetch
    const res = await fetch(`${API_CONFIG.VIDEO_API_URL}/process-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: frameB64, room_id: roomId })
    });
    const data = await res.json();
    return data.frame;
  } else {
    // Fallback to http module
    
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({ frame: frameB64, room_id: roomId });
      const apiUrl = new URL(`${API_CONFIG.VIDEO_API_URL}/process-frame`);
      const isHttps = apiUrl.protocol === 'https:';
      const options = {
        hostname: apiUrl.hostname,
        port: apiUrl.port || (isHttps ? 443 : 80),
        path: apiUrl.pathname + apiUrl.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        }
      };
      
      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            const result = JSON.parse(data);
            resolve(result.frame);
          } catch (err) {
            reject(err);
          }
        });
      });
      
      req.on('error', (err) => reject(err));
      req.write(postData);
      req.end();
    });
  }
}

// Send frame to Python for detection only (using existing process-frame endpoint)
async function sendToPythonForDetection(frameB64, roomId = null) {
  if (fetch) {
    const res = await fetch(`${API_CONFIG.VIDEO_API_URL}/process-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: frameB64, detect_only: true, room_id: roomId })
    });
    const data = await res.json();
    return {
      boundingBoxes: data.rectangles || [],
      polygons: data.polygons || [],
      detectionCounts: data.detection_counts || { face: 0, pii: 0, plate: 0 },
      processedFrame: data.frame // Still get the processed frame for fallback
    };
  } else {
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({ frame: frameB64, detect_only: true, room_id: roomId });
      const apiUrl = new URL(`${API_CONFIG.VIDEO_API_URL}/process-frame`);
      const isHttps = apiUrl.protocol === 'https:';
      const options = {
        hostname: apiUrl.hostname,
        port: apiUrl.port || (isHttps ? 443 : 80),
        path: apiUrl.pathname + apiUrl.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        }
      };
      
      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            const result = JSON.parse(data);
            resolve({
              boundingBoxes: result.rectangles || [],
              polygons: result.polygons || [],
              detectionCounts: result.detection_counts || { face: 0, pii: 0, plate: 0 },
              processedFrame: result.frame
            });
          } catch (err) {
            reject(err);
          }
        });
      });
      
      req.on('error', (err) => reject(err));
      req.write(postData);
      req.end();
    });
  }
}

// Interpolate bounding boxes for intermediate frames
function interpolateBoundingBoxes(originalBoxes, framesSinceDetection) {
  if (!originalBoxes || originalBoxes.length === 0) return [];
  
  // Simple interpolation: slightly expand boxes over time to account for movement
  const expansionFactor = 1 + (framesSinceDetection * 0.02); // 2% expansion per frame
  const maxExpansion = 1.3; // Max 30% expansion
  const actualExpansion = Math.min(expansionFactor, maxExpansion);
  
  return originalBoxes.map(box => {
    if (box.length !== 4) return box;
    
    const [x1, y1, x2, y2] = box;
    const width = x2 - x1;
    const height = y2 - y1;
    const centerX = x1 + width / 2;
    const centerY = y1 + height / 2;
    
    const newWidth = width * actualExpansion;
    const newHeight = height * actualExpansion;
    
    return [
      Math.max(0, centerX - newWidth / 2),
      Math.max(0, centerY - newHeight / 2),
      centerX + newWidth / 2,
      centerY + newHeight / 2
    ];
  });
}

// Apply blur to frame using OpenCV/Python without GPU processing
async function applyBlurToFrame(frameB64, boundingBoxes, roomId = null) {
  if (!boundingBoxes || boundingBoxes.length === 0) {
    return frameB64; // No blur needed
  }
  
  try {
    // Call Python script for CPU-only blur using cv2
    const blurredFrame = await callPythonBlur(frameB64, boundingBoxes, roomId);
    return blurredFrame;
  } catch (error) {
    console.error('[SERVER] CPU blur failed:', error);
    return frameB64; // Return original on error
  }
}

// Call Python script for CPU-only blur
async function callPythonBlur(frameB64, boundingBoxes, roomId = null) {
  if (fetch) {
    // Use existing endpoint with blur_only flag
    const res = await fetch(`${API_CONFIG.VIDEO_API_URL}/process-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        frame: frameB64, 
        blur_only: true,
        rectangles: boundingBoxes,
        polygons: [], // Only using rectangles for interpolated frames
        room_id: roomId
      })
    });
    const data = await res.json();
    return data.frame;
  } else {
    // HTTP fallback
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({ 
        frame: frameB64, 
        blur_only: true,
        rectangles: boundingBoxes,
        polygons: [],
        room_id: roomId
      });
      const apiUrl = new URL(`${API_CONFIG.VIDEO_API_URL}/process-frame`);
      const isHttps = apiUrl.protocol === 'https:';
      const options = {
        hostname: apiUrl.hostname,
        port: apiUrl.port || (isHttps ? 443 : 80),
        path: apiUrl.pathname + apiUrl.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        }
      };
      
      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            const result = JSON.parse(data);
            resolve(result.frame);
          } catch (err) {
            reject(err);
          }
        });
      });
      
      req.on('error', (err) => reject(err));
      req.write(postData);
      req.end();
    });
  }
}

// Handle video frames and buffering (legacy - kept for compatibility)
async function handleVideoFrame(roomId, frameB64) {
  const blurredFrame = await sendToPython(frameB64, roomId);
  if (!frameBuffer.has(roomId)) frameBuffer.set(roomId, []);
  frameBuffer.get(roomId).push({ frame: blurredFrame, timestamp: Date.now() });

  // Remove old frames (>3s)
  const cutoff = Date.now() - 3000;
  frameBuffer.set(roomId, frameBuffer.get(roomId).filter(f => f.timestamp >= cutoff));
}

// Create a processed video producer that will receive processed frames
async function createProcessedVideoProducer(originalProducer, roomId) {
  console.log('[SERVER] Setting up processed video pipeline for room:', roomId);
  
  // Store reference to original producer for the room
  const room = rooms.get(roomId);
  if (room) {
    room.originalVideoProducer = originalProducer;
    room.processedFrames = new Map(); // Store processed frames
  }
  
  // For now, return the original producer
  // The processing will happen when host sends video-frame events
  console.log('[SERVER] Video processing pipeline ready');
  return originalProducer;
}

// Socket.IO handlers
io.on('connection', socket => {
  console.log(`Client connected: ${socket.id}`);

  // Create room
  socket.on('create-room', (data, callback) => {
    const roomId = data.roomId || Math.random().toString(36).substr(2, 8);
    const room = { 
      id: roomId, 
      host: socket.id, 
      viewers: new Set(), 
      hostProducers: new Map(), 
      hostTransports: new Map(),
      viewerTransports: new Map(),
      isStreaming: false,
      streamStartTime: null
    };
    rooms.set(roomId, room);
    socket.join(roomId);
    
    // Initialize audio chunk timing tracking
    audioChunkStartTimes.set(roomId, []);
    
    // Set up audio processing for this room (since audio is handled via socket events, not WebRTC producers)
    console.log('[AUDIO-SERVER] 🎤 Setting up audio redaction for room:', roomId);
    
    socket.on('audio-data', async (audioData) => {
      try {
        // Convert array back to Int16Array, then to Buffer
        const int16Array = new Int16Array(audioData);
        const audioBuffer = Buffer.from(int16Array.buffer);
        
        console.log(`[AUDIO-SERVER] 🎤 Received audio data for room ${roomId}:`, audioBuffer.length, 'bytes');
        
        // Process audio through redaction service
        const result = await audioRedactionProcessor.addAudioChunk(roomId, audioBuffer);
        
        if (result && result.success) {
          const processingCompleteTime = Date.now();
          console.log('[AUDIO-SERVER] ✅ Audio chunk completed, triggering video processing');
          
          // Store PII events if any detected
          if (result.metadata.redacted_intervals?.length > 0) {
            // TIMING FIX: Audio chunks are processed with sliding window
            // The audio we just processed was captured roughly 3-6 seconds ago
            // Let's use a more aggressive timing adjustment to match video frames
            const estimatedCaptureTime = processingCompleteTime - (TIMING_CONFIG.AUDIO_CHUNK_DURATION * 1.5); // 4.5 seconds ago
            
            const piiEvents = result.metadata.redacted_intervals.map(interval => ({
              // Map PII intervals to estimated capture time
              startTime: estimatedCaptureTime + (interval[0] * 1000),
              endTime: estimatedCaptureTime + (interval[1] * 1000),
              confidence: interval.confidence || 1.0,
              words: result.metadata.detected_words || [],
              transcript: result.metadata.transcript || ""
            }));
            
            console.log(`[AUDIO-SERVER] 🕐 TIMING DEBUG:`);
            console.log(`[AUDIO-SERVER] Processing complete: ${processingCompleteTime}`);
            console.log(`[AUDIO-SERVER] Estimated capture start: ${estimatedCaptureTime}`);
            console.log(`[AUDIO-SERVER] PII events: ${piiEvents.map(e => `${e.startTime}-${e.endTime}`).join(', ')}`);
            console.log(`[AUDIO-SERVER] PII words: ${piiEvents.map(e => e.words?.join(' ')).join(', ')}`);
            
            // Store PII events with immediate cleanup of old events
            if (!piiEventsBuffer.has(roomId)) {
              piiEventsBuffer.set(roomId, []);
            }
            
            // Clean up old events before adding new ones (keep only last 10 seconds)
            const cutoffTime = Date.now() - 10000; // 10 seconds ago
            const existingEvents = piiEventsBuffer.get(roomId);
            const cleanedEvents = existingEvents.filter(event => event.endTime > cutoffTime);
            
            // Add new events
            cleanedEvents.push(...piiEvents);
            piiEventsBuffer.set(roomId, cleanedEvents);
            
            console.log(`[AUDIO-SERVER] 🧹 Cleaned up old PII events. Before: ${existingEvents.length}, After: ${cleanedEvents.length}`);
          }
          
          // TRIGGER VIDEO PROCESSING IMMEDIATELY (T+3s+100ms)
          setTimeout(() => {
            processFramesForCompletedAudio(roomId);
          }, TIMING_CONFIG.PROCESSING_BUFFER);
          
          // Send processed audio to viewers
          // SYNC FIX: Both audio and video should use the same timing reference
          // Audio chunk represents the MOST RECENT 3 seconds of captured audio
          // For sync purposes, use the END of the audio chunk (most recent timestamp)
          const audioChunkEndTime = Date.now() - (TIMING_CONFIG.AUDIO_CHUNK_DURATION / 2); // Middle of the chunk for better sync
          const deliveryDelay = calculateDeliveryDelay(audioChunkEndTime);
          
          console.log(`[AUDIO-SERVER] 🕐 Audio sync timing: chunk end ${audioChunkEndTime}, delivery delay ${deliveryDelay}ms`);
          
          setTimeout(() => {
            const room = rooms.get(roomId);
            if (room) {
              room.viewers.forEach(viewerId => {
                io.to(viewerId).emit('processed-audio', {
                  audioData: Array.from(new Int16Array(result.processedAudio.buffer)),
                  metadata: result.metadata,
                  timestamp: Date.now()
                });
              });
              console.log(`[AUDIO-SERVER] 📤 Delivered audio to ${room.viewers.size} viewers (delay: ${deliveryDelay}ms)`);
            }
          }, deliveryDelay);
          
        } else {
          console.log('[AUDIO-SERVER] ⚠️ Audio processing failed or not ready yet');
        }
        
      } catch (error) {
        console.error('[AUDIO-SERVER] ❌ Audio processing error:', error);
      }
    });
    
    console.log(`[AUDIO-SERVER] Room created: ${roomId} with audio processing enabled`);
    callback({ success: true, roomId, mediasoupUrl: API_CONFIG.SFU_URL });
  });

  // Join room
  socket.on('join-room', (data, callback) => {
    const room = rooms.get(data.roomId);
    if (!room) return callback({ success:false, error:'Room not found' });
    room.viewers.add(socket.id);
    socket.join(data.roomId);
    
    // If host is already streaming, notify new viewer
    if (room.isStreaming) {
      socket.emit('host-streaming-started', {
        roomId: room.id,
        timestamp: room.streamStartTime
      });
    }
    console.log("[SERVER] Viewer joined room:", data.roomId, "Total viewers:", room.viewers.size);

    // Notify viewer of existing producers
    const existingProducers = Array.from(room.hostProducers.entries()).map(([kind, producer]) => ({
      id: producer.id,
      kind: kind
    }));
    
    callback({ success:true, producers: existingProducers });
  });

  // Get router RTP capabilities
  socket.on('getRouterRtpCapabilities', (data, callback) => {
    if (!router) return callback({ error: 'Router not ready' });
    callback({ rtpCapabilities: router.rtpCapabilities });
  });

  // Create WebRTC transport
  socket.on('createProducerTransport', async (data, callback) => {
    try {
      const transport = await router.createWebRtcTransport(webRtcTransportOptions);
      transports.set(transport.id, transport);
      
      // Store transport with room for easier lookup
      const room = rooms.get(data.roomId);
      if (room && room.host === socket.id) {
        room.hostTransports.set(socket.id, transport);
      }
      
      callback({
        id: transport.id,
        iceParameters: transport.iceParameters,
        iceCandidates: transport.iceCandidates,
        dtlsParameters: transport.dtlsParameters
      });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // Connect transport
  socket.on('connectProducerTransport', async (data, callback) => {
    try {
      const room = rooms.get(data.roomId);
      if (!room || room.host !== socket.id) {
        return callback({ error: 'Room not found or unauthorized' });
      }
      
      const transport = room.hostTransports.get(socket.id);
      if (!transport) return callback({ error: 'Transport not found' });
      
      await transport.connect({ dtlsParameters: data.dtlsParameters });
      callback({ success: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // Produce audio/video
  socket.on('produce', async (data, callback) => {
    try {
      const { kind, rtpParameters, roomId } = data;
      const room = rooms.get(roomId);
      if (!room || room.host !== socket.id) return callback({ error: 'Unauthorized' });
      
      const transport = room.hostTransports.get(socket.id);
      if (!transport) return callback({ error: 'Transport not found' });

      let producer = await transport.produce({ kind, rtpParameters });
      producers.set(producer.id, producer);
      
      // Audio is handled via socket events (not WebRTC producers), so skip audio producer processing
      if (kind === 'audio') {
        console.log('[SERVER] ⚠️ Audio producer created but audio processing is handled via socket events');
        room.hostProducers.set(kind, producer);
      }
      // Video processing setup using WebRTC tracks
      else if (kind === 'video') {
        console.log('[VIDEO-SERVER] 🎬 Starting WebRTC video processing setup');
        console.log('[VIDEO-SERVER] Producer details:', {
          id: producer.id,
          kind: producer.kind,
          roomId: roomId,
          rtpParameters: {
            codecs: producer.rtpParameters.codecs.map(c => `${c.mimeType}@${c.clockRate}`),
            headerExtensions: producer.rtpParameters.headerExtensions.length
          }
        });
        
        try {
          console.log('[VIDEO-SERVER] 🔧 Setting up real video frame processing (client-side capture)...');
          
          // Store original video producer (viewers will consume this)
          room.hostProducers.set('video', producer);
          
          // Track if we've notified viewers about streaming start
          let hasNotifiedViewers = false;
          
          // Set up real video frame processing handler with mouth blur coordination
          socket.on('video-frame', async (data) => {
            try {
              const { roomId, frame, frameId, timestamp } = data;
              console.log('[VIDEO-SERVER] 🎯 Fast detection for frame', frameId, 'room:', roomId, 'timestamp:', timestamp);
              
              const room = rooms.get(roomId);
              if (room && !room.isStreaming) {
                room.isStreaming = true;
                room.streamStartTime = Date.now();

                // Notify ALL current viewers
                room.viewers.forEach(viewerId => {
                  io.to(viewerId).emit('host-streaming-started', {
                    roomId: roomId,
                    timestamp: room.streamStartTime
                  });
                });
              }
              
              if (!frame) {
                console.error('[VIDEO-SERVER] ❌ No frame data found');
                return;
              }
              
              // STAGE 1: Immediate detection (T+0ms)
              const detectionResult = await fetch(`${API_CONFIG.VIDEO_API_URL}/detect-faces-mouths`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                  frame: frame,
                  frame_id: frameId,
                  room_id: roomId
                }),
                timeout: 2000
              });
              
              if (detectionResult.ok) {
                const result = await detectionResult.json();
                
                // Cache detection results with original timestamp
                detectionCache.set(frameId, {
                  face_blur_regions: result.face_blur_regions || [],
                  mouth_regions: result.mouth_regions || [],
                  pii_regions: result.pii_regions || [],
                  plate_regions: result.plate_regions || [],
                  originalFrame: frame,
                  captureTimestamp: timestamp || Date.now(),  // Use current time as fallback if timestamp is undefined
                  roomId: roomId,
                  cacheTime: Date.now()
                });
                
                // Track frames waiting for PII analysis
                if (!frameAudioMapping.has(roomId)) {
                  frameAudioMapping.set(roomId, {
                    pendingFrames: [],
                    audioChunkCount: 0
                  });
                }
                frameAudioMapping.get(roomId).pendingFrames.push(frameId);
                
                console.log(`[VIDEO-SERVER] 📦 Cached frame ${frameId}: ${result.total_faces} faces, ${result.mouth_regions?.length || 0} mouths, ${result.pii_count || 0} PII, ${result.plate_count || 0} plates. Pending: ${frameAudioMapping.get(roomId).pendingFrames.length}`);
                
              } else {
                console.warn('[VIDEO-SERVER] ⚠️ Detection failed for frame', frameId, '- Status:', detectionResult.status);
              }
              
            } catch (error) {
              console.error('[VIDEO-SERVER] ❌ Detection error:', error);
            }
          });
          
          console.log('[VIDEO-SERVER] ✅ Video producer stored, ready for frame processing');
          console.log('[VIDEO-SERVER] 📊 Room now has producers:', Array.from(room.hostProducers.keys()));
          
        } catch (error) {
          console.error('[VIDEO-SERVER] ❌ WebRTC video processing setup failed:', error);
          console.error('[VIDEO-SERVER] Error details:', {
            message: error.message,
            stack: error.stack,
            roomId: roomId,
            producerId: producer.id
          });
          room.hostProducers.set(kind, producer); // Fallback to original
          console.log('[VIDEO-SERVER] 🔄 Falling back to original video producer');
        }
      }
      
      // Notify viewers about new producer
      socket.to(roomId).emit('new-producer', {
        producerId: producer.id,
        kind: kind
      });

      callback({ id: producer.id });
      
      console.log('[SERVER] Producer created and configured:', {
        producerId: producer.id,
        kind: kind,
        roomId: roomId,
        hasProcessing: kind === 'video'
      });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // Create consumer transport
  socket.on('createConsumerTransport', async (data, callback) => {
    try {
      const transport = await router.createWebRtcTransport(webRtcTransportOptions);
      transports.set(transport.id, transport);
      
      // Store transport with room for easier lookup
      const room = rooms.get(data.roomId);
      if (room && room.viewers.has(socket.id)) {
        if (!room.viewerTransports) room.viewerTransports = new Map();
        room.viewerTransports.set(socket.id, transport);
      }
      
      callback({
        id: transport.id,
        iceParameters: transport.iceParameters,
        iceCandidates: transport.iceCandidates,
        dtlsParameters: transport.dtlsParameters
      });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // Connect consumer transport
  socket.on('connectConsumerTransport', async (data, callback) => {
    try {
      const room = rooms.get(data.roomId);
      if (!room || !room.viewers.has(socket.id)) {
        return callback({ error: 'Room not found or unauthorized' });
      }
      
      const transport = room.viewerTransports?.get(socket.id);
      if (!transport) return callback({ error: 'Transport not found' });
      
      await transport.connect({ dtlsParameters: data.dtlsParameters });
      callback({ success: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // Get existing producers
  socket.on('getProducers', (data, callback) => {
    const room = rooms.get(data.roomId);
    if (!room) return callback({ error: 'Room not found' });
    
    console.log('[VIDEO-SERVER] 🔍 getProducers called for room:', data.roomId);
    console.log('[VIDEO-SERVER] Available producers in room:', Array.from(room.hostProducers.keys()));
    
    // For processed video architecture, viewers should NOT consume any video producers
    // Video is handled via processed-video-frame socket events
    // Audio is handled via processed-audio socket events
    const producers = Array.from(room.hostProducers.entries()).map(([kind, producer]) => {
      // Skip video producers - video handled via socket events
      if (kind === 'video') {
        console.log('[VIDEO-SERVER] ❌ Skipping video producer - using processed frames via socket events');
        return null;
      }
      
      // Skip audio producers - audio handled via socket events  
      if (kind === 'audio') {
        console.log('[VIDEO-SERVER] ❌ Skipping audio producer - using processed audio via socket events');
        return null;
      }
      
      // Skip processed producers
      if (kind.startsWith('processed-')) return null;
      
      return { id: producer.id, kind: kind };
    }).filter(Boolean);
    
    console.log('[VIDEO-SERVER] 📤 Returning producers to viewer:', producers);
    
    callback({ producers });
  });

  // Consume a producer
  socket.on('consume', async (data, callback) => {
    try {
      const { roomId, producerId, rtpCapabilities } = data;
      const room = rooms.get(roomId);
      if (!room || !room.viewers.has(socket.id)) {
        return callback({ error: 'Room not found or unauthorized' });
      }
      
      const transport = room.viewerTransports?.get(socket.id);
      if (!transport) return callback({ error: 'Transport not found' });
      
      const producer = producers.get(producerId);
      if (!producer) return callback({ error: 'Producer not found' });
      
      // Check if can consume
      if (!router.canConsume({ producerId, rtpCapabilities })) {
        return callback({ error: 'Cannot consume' });
      }
      
      const consumer = await transport.consume({
        producerId,
        rtpCapabilities,
        paused: true // Start paused
      });
      
      consumers.set(consumer.id, consumer);
      
      callback({
        id: consumer.id,
        producerId: consumer.producerId,
        kind: consumer.kind,
        rtpParameters: consumer.rtpParameters
      });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // Resume consumer
  socket.on('resumeConsumer', async (data, callback) => {
    try {
      const consumer = consumers.get(data.consumerId);
      if (!consumer) return callback({ error: 'Consumer not found' });
      
      await consumer.resume();
      callback({ success: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
    
    // Clean up room associations
    for (const [roomId, room] of rooms.entries()) {
      if (room.host === socket.id) {
        // Host disconnected - notify viewers and cleanup audio redaction
        room.isStreaming = false;
        room.streamStartTime = null;

        socket.to(roomId).emit('host-disconnected');
        
        // Clean up audio redaction for this room
        try {
          audioRedactionProcessor.cleanup(roomId);
        } catch (error) {
          console.error('[SERVER] Error cleaning up audio redaction:', error);
        }
        
        // Clean up audio timing tracking
        audioChunkStartTimes.delete(roomId);
        
        rooms.delete(roomId);
      } else if (room.viewers.has(socket.id)) {
        // Viewer disconnected
        room.viewers.delete(socket.id);
        if (room.viewerTransports) {
          const transport = room.viewerTransports.get(socket.id);
          if (transport) transport.close();
          room.viewerTransports.delete(socket.id);
        }
      }
    }
  });
});

// Event-driven video processing functions
async function processFramesForCompletedAudio(roomId) {
  const mapping = frameAudioMapping.get(roomId);
  if (!mapping || mapping.pendingFrames.length === 0) {
    console.log(`[VIDEO-SERVER] 📭 No pending frames for room ${roomId}`);
    return;
  }
  
  const framesToProcess = [...mapping.pendingFrames];
  mapping.pendingFrames = []; // Clear pending list
  mapping.audioChunkCount++;
  
  console.log(`[VIDEO-SERVER] 🎬 Processing ${framesToProcess.length} frames for audio chunk ${mapping.audioChunkCount}`);
  
  // Process all frames that were waiting for this audio analysis
  for (const frameId of framesToProcess) {
    await processVideoFrameWithPII(frameId);
  }
}

async function processVideoFrameWithPII(frameId) {
  const cached = detectionCache.get(frameId);
  if (!cached) {
    console.warn('[VIDEO-SERVER] ⚠️ No cached data for frame', frameId);
    return;
  }
  
  const { face_blur_regions, mouth_regions, pii_regions, plate_regions, originalFrame, captureTimestamp, roomId } = cached;
  
  // Check PII events for this frame's timestamp
  const shouldBlurMouths = checkPIIEventsForTimestamp(roomId, captureTimestamp);
  const piiReason = shouldBlurMouths ? getPIIReasonForTimestamp(roomId, captureTimestamp) : null;
  
  console.log(`[VIDEO-SERVER] 🎭 Processing frame ${frameId} (captured at ${captureTimestamp})`);
  console.log(`[VIDEO-SERVER] 👄 Mouth blur: ${shouldBlurMouths}${piiReason ? ` (${piiReason.words?.join(',')})` : ''}`);
  
  try {
    // Apply conditional blurring
    const blurResult = await fetch(`${API_CONFIG.VIDEO_API_URL}/apply-conditional-blur`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame: originalFrame,
        face_blur_regions: face_blur_regions,
        mouth_regions: mouth_regions,
        pii_regions: pii_regions,
        plate_regions: plate_regions,
        blur_mouths: shouldBlurMouths,
        blur_mode: shouldBlurMouths ? 'faces_and_mouths' : 'faces_only',
        pii_reason: piiReason
      }),
      timeout: 3000
    });
    
    if (blurResult.ok) {
      const result = await blurResult.json();
      
      // Calculate delivery delay based on ORIGINAL capture time
      const deliveryDelay = calculateDeliveryDelay(captureTimestamp);
      
      console.log(`[VIDEO-SERVER] ⏰ Frame ${frameId} will be delivered in ${deliveryDelay}ms (total delay: ${TIMING_CONFIG.TOTAL_VIEWER_DELAY}ms)`);
      
      setTimeout(() => {
        // Deliver to viewers
        const room = rooms.get(roomId);
        if (room && room.viewers.size > 0) {
          room.viewers.forEach(viewerId => {
            io.to(viewerId).emit('processed-video-frame', {
              frame: result.processed_frame,
              frameId: frameId,
              facesBlurred: result.faces_blurred,
              mouthsBlurred: result.mouths_blurred,
              piiTriggered: shouldBlurMouths,
              piiReason: piiReason?.words || null,
              captureTimestamp: captureTimestamp,
              deliveryTimestamp: Date.now() + deliveryDelay
            });
          });
          
          console.log(`[VIDEO-SERVER] 📤 Delivered frame ${frameId} to ${room.viewers.size} viewers (faces: ${result.faces_blurred}, mouths: ${result.mouths_blurred}, PII: ${result.pii_blurred || 0}, plates: ${result.plates_blurred || 0})`);
        }
      }, deliveryDelay);
      
    } else {
      console.error('[VIDEO-SERVER] ❌ Blur processing failed for frame', frameId, '- Status:', blurResult.status);
    }
    
  } catch (error) {
    console.error('[VIDEO-SERVER] ❌ PII processing error:', error);
  } finally {
    // Cleanup cache
    detectionCache.delete(frameId);
  }
}

function checkPIIEventsForTimestamp(roomId, videoTimestamp) {
  const piiEvents = piiEventsBuffer.get(roomId) || [];
  
  // Add timing tolerance to account for processing delays and frame timing variations
  const TIMING_TOLERANCE = 1500; // 1.5s tolerance (increased to handle timing variations)
  
  const matchFound = piiEvents.some(event => {
    // Check if video frame timestamp falls within PII event time range (with tolerance)
    const frameInRange = (videoTimestamp >= (event.startTime - TIMING_TOLERANCE)) && 
                        (videoTimestamp <= (event.endTime + TIMING_TOLERANCE));
    
    if (frameInRange) {
      console.log(`[SYNC] 🎯 Frame ${videoTimestamp} matches PII event ${event.startTime}-${event.endTime} (tolerance: ±${TIMING_TOLERANCE}ms)`);
    }
    
    return frameInRange;
  });
  
  if (!matchFound && piiEvents.length > 0) {
    console.log(`[SYNC] ❌ Frame ${videoTimestamp} no match. Available PII events: ${piiEvents.map(e => `${e.startTime}-${e.endTime}`).join(', ')}`);
  }
  
  return matchFound;
}

function getPIIReasonForTimestamp(roomId, videoTimestamp) {
  const piiEvents = piiEventsBuffer.get(roomId) || [];
  const TIMING_TOLERANCE = 1500; // Same tolerance as checkPIIEventsForTimestamp
  
  const matchingEvent = piiEvents.find(event => 
    (videoTimestamp >= (event.startTime - TIMING_TOLERANCE)) && 
    (videoTimestamp <= (event.endTime + TIMING_TOLERANCE))
  );
  
  if (matchingEvent) {
    return {
      words: matchingEvent.words,
      transcript: matchingEvent.transcript,
      confidence: matchingEvent.confidence,
      timeRange: `${matchingEvent.startTime}-${matchingEvent.endTime}`
    };
  }
  return null;
}

// Cleanup old cached data
setInterval(() => {
  const detectionCutoff = Date.now() - 60000; // 1 minute for detection cache
  const piiCutoff = Date.now() - 15000; // 15 seconds for PII events (more aggressive)
  
  // Cleanup detection cache
  for (const [frameId, data] of detectionCache.entries()) {
    if (data.cacheTime < detectionCutoff) {
      detectionCache.delete(frameId);
    }
  }
  
  // Cleanup PII events more aggressively
  let totalPiiBefore = 0;
  let totalPiiAfter = 0;
  
  for (const [roomId, events] of piiEventsBuffer.entries()) {
    totalPiiBefore += events.length;
    const cleanedEvents = events.filter(event => event.endTime > piiCutoff);
    totalPiiAfter += cleanedEvents.length;
    piiEventsBuffer.set(roomId, cleanedEvents);
    
    if (events.length !== cleanedEvents.length) {
      console.log(`[CLEANUP] Room ${roomId}: Removed ${events.length - cleanedEvents.length} old PII events`);
    }
  }
  
  console.log(`[CLEANUP] Cache sizes - Detection: ${detectionCache.size}, PII events: ${totalPiiAfter} (removed ${totalPiiBefore - totalPiiAfter})`);
}, 5000); // Run cleanup every 5 seconds instead of 30s

app.get('/health', (req, res) => res.json({ status:'healthy', mediasoup:'ready' }));

// Timing configuration endpoints
app.get('/timing-config', (req, res) => {
  res.json({
    audioChunkDuration: TIMING_CONFIG.AUDIO_CHUNK_DURATION,
    totalViewerDelay: TIMING_CONFIG.TOTAL_VIEWER_DELAY,
    processingBuffer: TIMING_CONFIG.PROCESSING_BUFFER,
    videoProcessingTrigger: TIMING_CONFIG.AUDIO_CHUNK_DURATION + TIMING_CONFIG.PROCESSING_BUFFER,
    currentTime: Date.now()
  });
});

app.post('/set-delay', (req, res) => {
  const { delay } = req.body;
  if (delay && delay >= 3000 && delay <= 15000) {
    TIMING_CONFIG.TOTAL_VIEWER_DELAY = delay;
    console.log(`[CONFIG] Updated total viewer delay to ${delay}ms`);
    res.json({ success: true, newDelay: delay });
  } else {
    res.json({ success: false, error: 'Invalid delay (must be 3-15 seconds)' });
  }
});

const PORT = process.env.PORT || 3001;
init().then(() => server.listen(PORT, () => console.log(`Server running on port ${PORT}`)));
