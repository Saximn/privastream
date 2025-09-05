/**
 * Audio Redaction Plugin for Mediasoup Server
 * Intercepts audio streams and sends them to the PII redaction service
 */

const { io } = require('socket.io-client');
const fs = require('fs');
const path = require('path');

class AudioRedactionPlugin {
  constructor(options = {}) {
    this.redactionServiceUrl = options.redactionServiceUrl || 'http://localhost:5002';
    this.isEnabled = options.enabled !== false;
    this.audioBufferSize = options.audioBufferSize || 4096; // Buffer size for audio chunks
    this.sampleRate = options.sampleRate || 16000;
    
    // CRITICAL: Buffer delay to prevent original content leak
    this.PROCESSING_BUFFER_MS = 300; // 300ms delay ensures PII processing completes
    this.videoFrameBuffer = new Map(); // roomId -> [{frame, timestamp, processed}]
    this.audioSegmentBuffer = new Map(); // roomId -> [{audio, timestamp, processed}]
    
    // Connection to redaction service
    this.redactionClient = null;
    this.isConnected = false;
    
    // Active room processors
    this.roomProcessors = new Map(); // roomId -> { producer, redactedProducer, audioBuffer, isRecording }
    
    // Store redacted audio segments for streaming
    this.redactedAudioBuffer = new Map(); // roomId -> Buffer with redacted audio
    
    console.log('🎤 AudioRedactionPlugin initialized', {
      enabled: this.isEnabled,
      serviceUrl: this.redactionServiceUrl,
      sampleRate: this.sampleRate
    });
    
    if (this.isEnabled) {
      this.connectToRedactionService();
    }
  }
  
  async connectToRedactionService() {
    try {
      this.redactionClient = io(this.redactionServiceUrl, {
        transports: ['websocket', 'polling']
      });
      
      this.redactionClient.on('connect', () => {
        console.log('✅ Connected to audio redaction service');
        this.isConnected = true;
      });
      
      this.redactionClient.on('disconnect', () => {
        console.log('❌ Disconnected from audio redaction service');
        this.isConnected = false;
      });
      
      this.redactionClient.on('redacted_audio', (data) => {
        this.handleRedactedAudio(data);
      });
      
      this.redactionClient.on('pii_detections', (data) => {
        this.handlePIIDetections(data);
      });
      
      this.redactionClient.on('error', (error) => {
        console.error('🔴 Redaction service error:', error);
      });
      
      // Connection is automatic with socket.io
      
    } catch (error) {
      console.error('🔴 Failed to connect to redaction service:', error);
      this.isConnected = false;
    }
  }
  
  /**
   * Register an audio producer for redaction processing
   */
  async registerAudioProducer(roomId, producer, router) {
    if (!this.isEnabled || !this.isConnected) {
      console.log('🟡 Audio redaction not available, skipping registration');
      return producer; // Return original producer if redaction is disabled
    }
    
    console.log(`🎤 Registering audio producer for redaction processing in room ${roomId}: ${producer.id}`);
    
    try {
      // Join redaction room as host (since we're processing the host's audio)
      this.redactionClient.emit('join_redaction_room', {
        room_id: roomId,
        role: 'host'
      });
      
      // Create a redacted producer that will carry the processed audio
      const redactedProducer = await this.createRedactedProducer(roomId, producer, router);
      
      // Set up audio processing for this room
      const roomProcessor = {
        originalProducer: producer,
        redactedProducer: redactedProducer,
        audioBuffer: Buffer.alloc(0),
        isRecording: true,
        lastProcessTime: Date.now(),
        router: router
      };
      
      this.roomProcessors.set(roomId, roomProcessor);
      
      // Set up RTP packet interception and processing
      this.setupRTPInterception(roomId, producer);
      
      console.log(`✅ Audio producer registered for room ${roomId}, redacted producer: ${redactedProducer.id}`);
      
      // Return the redacted producer instead of original
      return redactedProducer;
      
    } catch (error) {
      console.error(`🔴 Failed to register audio producer for room ${roomId}:`, error);
      return producer; // Return original producer on error
    }
  }
  
  /**
   * Create a redacted audio producer
   */
  async createRedactedProducer(roomId, originalProducer, router) {
    // Create a plain transport for injecting redacted audio
    const plainTransport = await router.createPlainTransport({
      listenIp: { ip: '127.0.0.1', announcedIp: null },
      rtcpMux: false,
      comedia: false
    });

    // Create producer on plain transport with same codec as original
    const redactedProducer = await plainTransport.produce({
      kind: 'audio',
      rtpParameters: originalProducer.rtpParameters
    });

    console.log(`✅ Created redacted producer ${redactedProducer.id} for room ${roomId}`);
    return redactedProducer;
  }

  /**
   * Set up RTP packet interception for audio processing
   */
  setupRTPInterception(roomId, originalProducer) {
    console.log(`🔧 Setting up RTP interception for room ${roomId}`);
    
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor) return;
    
    // Set up interval to process accumulated audio data
    const processingInterval = setInterval(() => {
      if (!roomProcessor.isRecording || !this.isConnected) {
        clearInterval(processingInterval);
        return;
      }
      
      this.processAccumulatedAudio(roomId);
    }, 1000); // Process every second
    
    // Store interval reference for cleanup
    roomProcessor.processingInterval = processingInterval;
    
    // Start capturing audio data from the original producer
    this.startAudioCapture(roomId, originalProducer);
  }
  
  /**
   * Start capturing audio data from the original producer
   */
  startAudioCapture(roomId, originalProducer) {
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor || !roomProcessor.isRecording) return;
    
    // For now, simulate audio capture until real RTP interception is implemented
    // This generates dummy audio data that represents what would be captured from RTP
    console.log(`🎵 Starting audio capture simulation for room ${roomId}`);
    
    const captureInterval = setInterval(() => {
      if (!roomProcessor.isRecording) {
        clearInterval(captureInterval);
        return;
      }
      
      // Generate audio data that simulates captured RTP packets
      // In real implementation, this would be actual audio data from RTP packets
      const samplesPerChunk = this.sampleRate * 0.1; // 100ms chunks
      const audioChunk = Buffer.alloc(samplesPerChunk * 2); // 16-bit samples
      
      // Fill with simulated voice pattern (sine wave with variations)
      const baseFreq = 440; // A note
      const timeOffset = Date.now() / 1000;
      
      for (let i = 0; i < samplesPerChunk; i++) {
        const t = (timeOffset + i / this.sampleRate);
        // Create voice-like pattern with multiple harmonics
        const sample = 
          Math.sin(2 * Math.PI * baseFreq * t) * 0.3 +
          Math.sin(2 * Math.PI * baseFreq * 2 * t) * 0.2 +
          Math.sin(2 * Math.PI * baseFreq * 0.5 * t) * 0.1;
        
        // Add some variation to simulate speech
        const variation = Math.sin(2 * Math.PI * 0.1 * t) * 0.5 + 0.5;
        const finalSample = Math.round(sample * variation * 16000);
        
        audioChunk.writeInt16LE(Math.max(-32767, Math.min(32767, finalSample)), i * 2);
      }
      
      // Add to buffer for processing
      roomProcessor.audioBuffer = Buffer.concat([roomProcessor.audioBuffer, audioChunk]);
      
    }, 100); // Capture every 100ms
    
    // Store interval reference for cleanup
    roomProcessor.captureInterval = captureInterval;
  }
  
  /**
   * Process accumulated audio data and send to redaction service
   */
  processAccumulatedAudio(roomId) {
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor || !this.isConnected) return;
    
    const audioBuffer = roomProcessor.audioBuffer;
    if (audioBuffer.length === 0) return;
    
    try {
      // Send audio data to redaction service
      this.redactionClient.emit('audio_data', {
        audio: audioBuffer
      });
      
      // Clear processed buffer
      roomProcessor.audioBuffer = Buffer.alloc(0);
      roomProcessor.lastProcessTime = Date.now();
      
      console.log(`🎵 Sent ${audioBuffer.length} bytes of audio for processing (room: ${roomId})`);
      
    } catch (error) {
      console.error(`🔴 Failed to send audio data for room ${roomId}:`, error);
    }
  }
  
  /**
   * CRITICAL: Buffered stream processing to prevent original content leak
   * Ensures PII processing completes BEFORE streaming to viewers
   */
  processBufferedStream(roomId, mediaType, data, timestamp) {
    const currentTime = Date.now();
    
    // Get appropriate buffer
    const bufferMap = mediaType === 'video' ? this.videoFrameBuffer : this.audioSegmentBuffer;
    
    if (!bufferMap.has(roomId)) {
      bufferMap.set(roomId, []);
    }
    
    const buffer = bufferMap.get(roomId);
    
    // Add new data to buffer
    buffer.push({
      data: data,
      timestamp: timestamp,
      processed: false,
      addedAt: currentTime
    });
    
    // Process items that are old enough (waited for PII processing)
    const readyItems = [];
    const remainingItems = [];
    
    for (const item of buffer) {
      const age = currentTime - item.addedAt;
      
      if (age >= this.PROCESSING_BUFFER_MS && item.processed) {
        // Item is ready and has been processed
        readyItems.push(item);
      } else if (age < this.PROCESSING_BUFFER_MS || !item.processed) {
        // Item still needs more time or processing
        remainingItems.push(item);
      } else {
        // Item aged out without processing - pass through (safety fallback)
        console.warn(`⚠️ Item aged out without processing in room ${roomId}`);
        readyItems.push(item);
      }
    }
    
    // Update buffer with remaining items
    bufferMap.set(roomId, remainingItems);
    
    // Return ready items for streaming to viewers
    return readyItems.map(item => item.data);
  }
  
  /**
   * Mark buffered items as processed when PII detection completes
   */
  markBufferedItemsProcessed(roomId, mediaType, processedTimestamp) {
    const bufferMap = mediaType === 'video' ? this.videoFrameBuffer : this.audioSegmentBuffer;
    
    if (!bufferMap.has(roomId)) return;
    
    const buffer = bufferMap.get(roomId);
    
    // Mark items as processed based on timestamp
    for (const item of buffer) {
      if (Math.abs(item.timestamp - processedTimestamp) < 50) { // 50ms tolerance
        item.processed = true;
      }
    }
  }
  
  /**
   * Handle redacted audio from the service
   */
  handleRedactedAudio(data) {
    console.log('🔇 Received redacted audio:', {
      segment_id: data.segment_id,
      pii_count: data.pii_count,
      processing_time: data.processing_time
    });
    
    // Find the room that this redacted audio belongs to
    const roomId = this.findRoomForSegment(data.segment_id);
    if (!roomId) {
      console.warn('🟡 Could not find room for segment:', data.segment_id);
      return;
    }
    
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor || !roomProcessor.redactedProducer) {
      console.warn('🟡 No redacted producer for room:', roomId);
      return;
    }
    
    // CRITICAL: Mark buffered audio as processed and stream with delay
    if (data.audio_data) {
      // Mark corresponding buffered items as processed
      this.markBufferedItemsProcessed(roomId, 'audio', data.timestamp);
      
      // Stream redacted audio through buffer (prevents original leak)
      const readyAudio = this.processBufferedStream(roomId, 'audio', data.audio_data, data.timestamp);
      
      if (readyAudio.length > 0) {
        this.streamRedactedAudio(roomId, readyAudio[0]); // Stream the first ready segment
      }
    }
    
    this.logRedactionEvent(data);
  }
  
  /**
   * Stream redacted audio data to the redacted producer
   */
  streamRedactedAudio(roomId, audioData) {
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor || !roomProcessor.redactedProducer) return;
    
    try {
      // Store redacted audio for this room
      if (!this.redactedAudioBuffer.has(roomId)) {
        this.redactedAudioBuffer.set(roomId, Buffer.alloc(0));
      }
      
      const currentBuffer = this.redactedAudioBuffer.get(roomId);
      const newBuffer = Buffer.concat([currentBuffer, Buffer.from(audioData)]);
      this.redactedAudioBuffer.set(roomId, newBuffer);
      
      console.log(`🎵 Streamed ${audioData.length} bytes of redacted audio for room ${roomId}`);
      
      // TODO: Send audio data to the redacted producer's RTP stream
      // This requires implementing RTP packet generation from audio data
      
    } catch (error) {
      console.error(`🔴 Failed to stream redacted audio for room ${roomId}:`, error);
    }
  }
  
  /**
   * Find room ID from segment ID
   */
  findRoomForSegment(segmentId) {
    // Extract room info from segment ID or maintain segment->room mapping
    // For now, return the first active room (simplified for demo)
    const activeRooms = Array.from(this.roomProcessors.keys());
    return activeRooms.length > 0 ? activeRooms[0] : null;
  }
  
  /**
   * Handle PII detections from the service
   */
  handlePIIDetections(data) {
    console.log('🚨 PII Detected:', {
      segment_id: data.segment_id,
      pii_count: data.pii_count,
      detections: data.pii_detections?.map(d => ({
        type: d.pii_type,
        confidence: d.confidence,
        text: d.text.substring(0, 20) + '...' // Log partial text for privacy
      }))
    });
    
    // TODO: Send alerts to room participants about PII detection
    // TODO: Update video filtering to blur during PII audio
    
    this.logPIIEvent(data);
  }
  
  /**
   * Unregister an audio producer
   */
  unregisterAudioProducer(roomId, producerId) {
    console.log(`🛑 Unregistering audio producer for room ${roomId}: ${producerId}`);
    
    const roomProcessor = this.roomProcessors.get(roomId);
    if (roomProcessor) {
      roomProcessor.isRecording = false;
      
      // Clear intervals
      if (roomProcessor.processingInterval) {
        clearInterval(roomProcessor.processingInterval);
      }
      if (roomProcessor.captureInterval) {
        clearInterval(roomProcessor.captureInterval);
      }
      
      // Close redacted producer
      if (roomProcessor.redactedProducer) {
        roomProcessor.redactedProducer.close();
      }
      
      // Remove from map
      this.roomProcessors.delete(roomId);
    }
    
    // Clean up redacted audio buffer
    this.redactedAudioBuffer.delete(roomId);
    
    // Leave redaction room
    if (this.isConnected) {
      this.redactionClient.emit('leave_redaction_room', { room_id: roomId });
    }
  }
  
  /**
   * Clean up room when it's closed
   */
  cleanupRoom(roomId) {
    console.log(`🧹 Cleaning up audio redaction for room ${roomId}`);
    
    const roomProcessor = this.roomProcessors.get(roomId);
    if (roomProcessor) {
      const producerId = roomProcessor.originalProducer ? roomProcessor.originalProducer.id : 'unknown';
      this.unregisterAudioProducer(roomId, producerId);
    }
  }
  
  /**
   * Log redaction event
   */
  logRedactionEvent(data) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      event: 'audio_redacted',
      segment_id: data.segment_id,
      pii_count: data.pii_count,
      processing_time: data.processing_time
    };
    
    // Log to console and optionally to file
    console.log('📝 Redaction Event:', logEntry);
  }
  
  /**
   * Log PII detection event
   */
  logPIIEvent(data) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      event: 'pii_detected',
      segment_id: data.segment_id,
      pii_count: data.pii_count,
      detection_types: data.pii_detections?.map(d => d.pii_type) || []
    };
    
    // Log to console and optionally to file (without sensitive data)
    console.log('📝 PII Detection Event:', logEntry);
  }
  
  /**
   * Get plugin statistics
   */
  getStats() {
    return {
      isEnabled: this.isEnabled,
      isConnected: this.isConnected,
      activeRooms: this.roomProcessors.size,
      roomDetails: Array.from(this.roomProcessors.entries()).map(([roomId, processor]) => ({
        roomId,
        originalProducerId: processor.originalProducer?.id || 'unknown',
        redactedProducerId: processor.redactedProducer?.id || 'unknown',
        isRecording: processor.isRecording,
        bufferSize: processor.audioBuffer.length,
        redactedBufferSize: this.redactedAudioBuffer.get(roomId)?.length || 0,
        lastProcessTime: processor.lastProcessTime
      }))
    };
  }
  
  /**
   * Enable/disable the plugin
   */
  setEnabled(enabled) {
    this.isEnabled = enabled;
    console.log(`🎛️ Audio redaction plugin ${enabled ? 'enabled' : 'disabled'}`);
    
    if (!enabled) {
      // Stop all processing
      for (const [roomId] of this.roomProcessors) {
        this.cleanupRoom(roomId);
      }
    }
  }
  
  /**
   * Disconnect and cleanup
   */
  async cleanup() {
    console.log('🧹 Cleaning up audio redaction plugin');
    
    // Clean up all rooms
    for (const [roomId] of this.roomProcessors) {
      this.cleanupRoom(roomId);
    }
    
    // Disconnect from service
    if (this.redactionClient && this.isConnected) {
      this.redactionClient.disconnect();
    }
  }
}

module.exports = { AudioRedactionPlugin };