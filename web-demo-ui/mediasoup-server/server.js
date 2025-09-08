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

// Try to import node-fetch, fallback to http if needed
let fetch;
try {
  fetch = require('node-fetch');
} catch (err) {
  console.warn('[SERVER] node-fetch not available, using http fallback');
  fetch = null;
}

// Import Audio Redaction Processor
const { AudioRedactionProcessor } = require('./audio-redaction-processor');
// Import WebRTC Video Processor
const { WebRTCVideoProcessor } = require('./webrtc-video-processor');

// Initialize Audio Redaction Processor  
const audioRedactionProcessor = new AudioRedactionProcessor({
  redactionServiceUrl: 'http://localhost:5002',
  sampleRate: 16000,  // Match Vosk requirements
  channels: 1,        // Mono for better transcription
  bufferDurationMs: 3000
});

// Initialize WebRTC Video Processor
const videoProcessor = new WebRTCVideoProcessor({
  videoServiceUrl: 'http://localhost:5001',
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
async function sendToPython(frameB64) {
  if (fetch) {
    // Use node-fetch
    const res = await fetch('http://localhost:5001/process-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: frameB64 })
    });
    const data = await res.json();
    return data.frame;
  } else {
    // Fallback to http module
    return new Promise((resolve, reject) => {
      const postData = JSON.stringify({ frame: frameB64 });
      const options = {
        hostname: 'localhost',
        port: 5001,
        path: '/process-frame',
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
async function sendToPythonForDetection(frameB64) {
  if (fetch) {
    const res = await fetch('http://localhost:5001/process-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: frameB64, detect_only: true })
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
      const postData = JSON.stringify({ frame: frameB64, detect_only: true });
      const options = {
        hostname: 'localhost',
        port: 5001,
        path: '/process-frame',
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
async function applyBlurToFrame(frameB64, boundingBoxes) {
  if (!boundingBoxes || boundingBoxes.length === 0) {
    return frameB64; // No blur needed
  }
  
  try {
    // Call Python script for CPU-only blur using cv2
    const blurredFrame = await callPythonBlur(frameB64, boundingBoxes);
    return blurredFrame;
  } catch (error) {
    console.error('[SERVER] CPU blur failed:', error);
    return frameB64; // Return original on error
  }
}

// Call Python script for CPU-only blur
async function callPythonBlur(frameB64, boundingBoxes) {
  if (fetch) {
    // Use existing endpoint with blur_only flag
    const res = await fetch('http://localhost:5001/process-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        frame: frameB64, 
        blur_only: true,
        rectangles: boundingBoxes,
        polygons: [] // Only using rectangles for interpolated frames
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
        polygons: []
      });
      const options = {
        hostname: 'localhost',
        port: 5001,
        path: '/process-frame',
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
  const blurredFrame = await sendToPython(frameB64);
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
      viewerTransports: new Map()
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
        
        // Track audio chunk start time (when first data for new chunk arrives)
        const chunkStartTimes = audioChunkStartTimes.get(roomId) || [];
        const currentTime = Date.now();
        
        // Process audio through redaction service
        const result = await audioRedactionProcessor.addAudioChunk(roomId, audioBuffer);
        
        if (result && result.success) {
          // A 3-second chunk just completed - record when it started and calculate delay
          const chunkStartTime = currentTime - 3000; // Estimate: chunk started 3 seconds ago
          chunkStartTimes.push(chunkStartTime);
          audioChunkStartTimes.set(roomId, chunkStartTimes);
          
          console.log('[AUDIO-SERVER] ✅ Audio processed successfully, sending to viewers');
          console.log('[AUDIO-SERVER] 🎤 Processed audio metadata:', {
            piiCount: result.metadata?.pii_count || 0,
            processingTime: result.metadata?.processing_time || 0,
            transcript: result.metadata?.transcript ? result.metadata.transcript.substring(0, 50) + '...' : 'empty'
          });
          
          // SLIDING WINDOW FIX: Account for the fact that sliding window output represents
          // audio from 3 seconds earlier (we output the second half of [previous + current])
          // So we need to reduce the delay by 3 seconds to maintain sync with video
          const slidingWindowOffset = result.metadata?.hadPreviousChunk ? 3000 : 0;
          const effectiveChunkStartTime = chunkStartTime - slidingWindowOffset;
          
          // Calculate delay based on effective chunk start time to achieve 8 seconds total
          const targetOutputTime = effectiveChunkStartTime + 8000; // 8 seconds from effective start
          const delayNeeded = Math.max(0, targetOutputTime - Date.now());
          
          console.log('[AUDIO-SERVER] 🕐 Sliding window timing adjustment:', {
            originalChunkStart: chunkStartTime,
            slidingWindowOffset: slidingWindowOffset,
            effectiveChunkStart: effectiveChunkStartTime,
            delayNeeded: delayNeeded
          });
          
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
              console.log('[AUDIO-SERVER] 📤 Sent audio with', delayNeeded, 'ms delay (8s total from chunk start) to', room.viewers.size, 'viewers');
            }
          }, delayNeeded); // Dynamic delay to achieve 8 seconds total from chunk start
        } else {
          console.log('[AUDIO-SERVER] ⚠️ Audio processing failed or not ready yet');
        }
        
      } catch (error) {
        console.error('[AUDIO-SERVER] ❌ Audio processing error:', error);
      }
    });
    
    console.log(`[AUDIO-SERVER] Room created: ${roomId} with audio processing enabled`);
    callback({ success: true, roomId, mediasoupUrl: 'http://localhost:3001' });
  });

  // Join room
  socket.on('join-room', (data, callback) => {
    const room = rooms.get(data.roomId);
    if (!room) return callback({ success:false, error:'Room not found' });
    room.viewers.add(socket.id);
    socket.join(data.roomId);
    
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
          
          // Set up real video frame processing handler
          socket.on('video-frame', async (data) => {
            try {
              const { roomId, frame, frameId, timestamp } = data;
              console.log('[VIDEO-SERVER] 📸 Received real video frame for room:', roomId, 'frameId:', frameId, 'frame size:', frame?.length);
              
              // Notify viewers on first frame that host is streaming
              if (!hasNotifiedViewers) {
                const room = rooms.get(roomId);
                if (room && room.viewers.size > 0) {
                  room.viewers.forEach(viewerId => {
                    io.to(viewerId).emit('host-streaming-started', {
                      roomId: roomId,
                      timestamp: Date.now()
                    });
                  });
                  console.log('[VIDEO-SERVER] 📡 Notified', room.viewers.size, 'viewers that host started streaming');
                }
                hasNotifiedViewers = true;
              }
              
              // Get or initialize frame count
              if (!frameProcessingState.has(roomId)) {
                frameProcessingState.set(roomId, { frameCount: 0 });
              }
              const state = frameProcessingState.get(roomId);
              state.frameCount++;
              
              // Process frame through WebRTC video processor
              if (!frame) {
                console.error('[VIDEO-SERVER] ❌ No frame data found');
                return;
              }
              
              console.log(`[VIDEO-SERVER] 🎯 Processing frame ${state.frameCount} for room ${roomId} (every 15th: ${state.frameCount % 15 === 1})`);
              
              // Process frame through Python detection service (DIRECT CALL - like working version)
              console.log('[VIDEO-SERVER] 📡 Calling Python detection service directly...');
              const response = await fetch('http://localhost:5001/process-frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                  frame: frame, // Host already sends full data:image/jpeg;base64, URL
                  detect_only: true,
                  frame_id: state.frameCount,
                  timestamp: timestamp || Date.now()  // Add timestamp for request age validation
                }),
                timeout: 5000
              });
              
              if (response.ok) {
                const result = await response.json();
                
                // Check if request was dropped by Python API
                if (result.dropped) {
                  console.log(`[VIDEO-SERVER] 🗑️ Request dropped by Python API: ${result.reason} - Frame ${frameId}`);
                  return; // Don't send anything to viewers
                }
                
                console.log('[VIDEO-SERVER] ✅ Python service responded successfully:', {
                  rectangles: result.rectangles?.length || 0,
                  frame: result.frame ? 'received' : 'missing'
                });
                
                // Calculate delay based on capture time to achieve 8 seconds total from capture
                const captureTime = timestamp || Date.now();
                const targetOutputTime = captureTime + 8000; // 8 seconds from when media was captured
                const currentTime = Date.now();
                const delayNeeded = Math.max(0, targetOutputTime - currentTime);
                
                setTimeout(() => {
                  const room = rooms.get(roomId);
                  if (room && room.viewers.size > 0) {
                    room.viewers.forEach(viewerId => {
                      io.to(viewerId).emit('processed-video-frame', {
                        frame: result.frame,
                        frameId: frameId,
                        boundingBoxCount: result.rectangles?.length || 0,
                        wasDetectionFrame: true,
                        timestamp: timestamp
                      });
                    });
                    console.log('[VIDEO-SERVER] 📤 Sent video frame with', delayNeeded, 'ms delay (8s total from capture) to', room.viewers.size, 'viewers');
                  }
                }, delayNeeded); // Dynamic delay to achieve 8 seconds total from capture
              } else {
                // Handle dropped requests (429, 503) vs actual errors
                if (response.status === 429) {
                  console.log(`[VIDEO-SERVER] 🗑️ Request too old - Frame ${frameId} dropped by Python API`);
                } else if (response.status === 503) {
                  console.log(`[VIDEO-SERVER] 🚫 Server overloaded - Frame ${frameId} dropped by Python API`);
                } else {
                  console.log('[VIDEO-SERVER] ❌ Python service error:', response.status);
                }
                console.log('[VIDEO-SERVER] 🔒 PRIVACY PROTECTION: Dropping frame instead of sending unprocessed video');
                
                // PRIVACY FIX: Drop the frame entirely instead of sending unprocessed video
                // This ensures viewers NEVER see raw/unprocessed content
                // Better to have gaps in video than to leak private information
              }
              
            } catch (error) {
              console.error('[VIDEO-SERVER] ❌ Video frame processing error:', error);
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

app.get('/health', (req, res) => res.json({ status:'healthy', mediasoup:'ready' }));

const PORT = process.env.PORT || 3001;
init().then(() => server.listen(PORT, () => console.log(`Server running on port ${PORT}`)));
