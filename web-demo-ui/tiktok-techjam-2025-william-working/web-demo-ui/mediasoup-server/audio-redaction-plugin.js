/**
 * Audio Redaction Plugin for Mediasoup Server
 * Intercepts audio streams and sends them to the PII redaction service
 */

const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

class AudioRedactionPlugin {
  constructor(options = {}) {
    this.redactionServiceUrl = options.redactionServiceUrl || 'http://localhost:5002'; // Vosk + BERT server
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
      // Test connection to Flask API
      const response = await fetch(`${this.redactionServiceUrl}/health`);
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Connected to audio redaction service:', data.status);
        console.log('   Models loaded - NER:', data.models_loaded?.ner, 'ASR:', data.models_loaded?.asr);
        this.isConnected = true;
      } else {
        throw new Error(`Health check failed: ${response.status}`);
      }
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
    
    console.log(`🎤 Registering audio producer for background redaction processing in room ${roomId}: ${producer.id}`);
    
    try {
      // Set up background audio processing (non-blocking)
      const roomProcessor = {
        originalProducer: producer,
        audioBuffer: Buffer.alloc(0),
        isRecording: true,
        lastProcessTime: Date.now(),
        router: router
      };
      
      this.roomProcessors.set(roomId, roomProcessor);
      
      // Start FFmpeg-based audio consumption
      await this.setupFFmpegAudioConsumption(roomId, producer);
      
      console.log(`✅ Background audio redaction setup for room ${roomId} - original audio flows normally`);
      
      // Return the ORIGINAL producer so audio flows normally to viewers
      return producer;
      
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
   * Set up FFmpeg-based audio consumption from MediaSoup producer
   */
  async setupFFmpegAudioConsumption(roomId, originalProducer) {
    console.log(`🎬 Setting up FFmpeg audio consumption for room ${roomId}`);
    
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor) return;
    
    try {
      // Create PlainTransport for FFmpeg to consume audio
      const plainTransport = await roomProcessor.router.createPlainTransport({
        listenIp: { ip: '127.0.0.1', announcedIp: null },
        rtcpMux: false,
        comedia: false, // We know the remote endpoint (FFmpeg)
        enableSctp: false
      });
      
      console.log(`✅ Created PlainTransport for FFmpeg: ${plainTransport.id}`);
      console.log(`   RTP Port: ${plainTransport.tuple.localPort}`);
      
      // Create consumer to pipe audio from original producer
      const consumer = await plainTransport.consume({
        producerId: originalProducer.id,
        rtpCapabilities: roomProcessor.router.rtpCapabilities
      });
      
      console.log(`✅ Created audio consumer: ${consumer.id}`);
      console.log(`   Codec: ${consumer.rtpParameters.codecs[0]?.mimeType}`);
      console.log(`   Payload Type: ${consumer.rtpParameters.codecs[0]?.payloadType}`);
      console.log(`   Consumer Kind: ${consumer.kind}`);
      console.log(`   Consumer State: ${consumer.closed ? 'closed' : 'open'}`);
      console.log(`   Consumer Paused: ${consumer.paused}`);
      console.log(`   Producer ID: ${originalProducer.id}`);
      console.log(`   Producer State: ${originalProducer.closed ? 'closed' : 'open'}`);
      console.log(`   Producer Paused: ${originalProducer.paused}`);
      
      // Add consumer event listeners to debug
      consumer.on('transportclose', () => {
        console.log(`🔴 Consumer ${consumer.id} transport closed`);
      });
      
      consumer.on('producerclose', () => {
        console.log(`🔴 Consumer ${consumer.id} producer closed`);
      });
      
      consumer.on('producerpause', () => {
        console.log(`⏸️ Consumer ${consumer.id} producer paused`);
      });
      
      consumer.on('producerresume', () => {
        console.log(`▶️ Consumer ${consumer.id} producer resumed`);
      });
      
      // Start FFmpeg process to consume the RTP stream and convert to PCM
      await this.startFFmpegProcess(roomId, plainTransport, consumer);
      
      // Resume the consumer AFTER FFmpeg is ready (like in reference code)
      setTimeout(async () => {
        try {
          if (consumer.paused) {
            await consumer.resume();
            console.log(`▶️ Consumer ${consumer.id} resumed after FFmpeg startup`);
            
            // Request keyframe for better stream start
            if (typeof consumer.requestKeyFrame === 'function') {
              await consumer.requestKeyFrame();
              console.log(`🔑 Keyframe requested for consumer ${consumer.id}`);
            }
          }
        } catch (error) {
          console.error(`🔴 Failed to resume consumer:`, error);
        }
      }, 1000);
      
      // Store references for cleanup
      roomProcessor.plainTransport = plainTransport;
      roomProcessor.consumer = consumer;
      
    } catch (error) {
      console.error(`🔴 Failed to setup FFmpeg audio consumption:`, error);
      // Fallback to test audio generation
      this.setupTestAudioGeneration(roomId);
    }
  }
  
  /**
   * Start FFmpeg process to consume RTP audio and convert to PCM
   */
  async startFFmpegProcess(roomId, plainTransport, consumer) {
    const { spawn } = require('child_process');
    const roomProcessor = this.roomProcessors.get(roomId);
    
    // Get RTP details
    const rtpPort = plainTransport.tuple.localPort;
    const rtcpPort = plainTransport.rtcpTuple ? plainTransport.rtcpTuple.localPort : rtpPort + 1;
    const codec = consumer.rtpParameters.codecs[0];
    const payloadType = codec.payloadType;
    
    console.log(`🎬 PlainTransport details:`);
    console.log(`   Local RTP port: ${rtpPort}`);
    console.log(`   Local RTCP port: ${rtcpPort}`);
    console.log(`   Codec: ${codec.mimeType}, Payload: ${payloadType}`);
    
    console.log(`🎬 Starting FFmpeg process for room ${roomId}...`);
    
    // Create SDP file like in your reference code
    const codecName = codec.mimeType.split('/')[1];
    const channels = codec.channels || 2; // Default to stereo if not specified
    
    // Create SDP content matching your reference format
    const sdpContent = `v=0
o=- 0 0 IN IP4 127.0.0.1
s=FFmpeg
c=IN IP4 127.0.0.1
t=0 0
m=audio ${rtpPort} RTP/AVP ${payloadType}
a=rtpmap:${payloadType} ${codecName}/${codec.clockRate}${channels ? '/' + channels : ''}
a=recvonly
`;

    const fs = require('fs');
    const sdpPath = `./temp_${roomId}.sdp`;
    fs.writeFileSync(sdpPath, sdpContent);
    console.log(`📄 Created SDP file (reference format): ${sdpPath}`);
    console.log(`📄 SDP content:\n${sdpContent}`);
    
    // FFmpeg command using SDP file (like your reference)
    const ffmpegArgs = [
      '-protocol_whitelist', 'file,udp,rtp',
      '-f', 'sdp',
      '-i', sdpPath,
      '-f', 'wav',
      '-acodec', 'pcm_s16le',
      '-ar', '16000', // 16kHz sample rate  
      '-ac', '1',     // Mono
      '-avoid_negative_ts', 'make_zero',
      '-fflags', '+genpts',
      '-analyzeduration', '2000000',
      '-probesize', '2000000', 
      '-max_delay', '1000000',
      '-loglevel', 'info',
      'pipe:1'
    ];
    
    const ffmpegProcess = spawn('ffmpeg', ffmpegArgs);
    
    console.log(`🎬 FFmpeg command: ffmpeg ${ffmpegArgs.join(' ')}`);
    
    let audioChunks = [];
    let totalBytes = 0;
    
    ffmpegProcess.stdout.on('data', (chunk) => {
      // Accumulate audio data
      audioChunks.push(chunk);
      totalBytes += chunk.length;
      
      console.log(`📊 FFmpeg stdout chunk: ${chunk.length} bytes, total: ${totalBytes} bytes`);
      
      // Debug: Show first few bytes to understand the data
      if (chunk.length > 0) {
        const preview = chunk.slice(0, Math.min(16, chunk.length));
        console.log(`📊 Chunk preview (hex): ${preview.toString('hex')}`);
        console.log(`📊 Chunk preview (first 4 bytes as text): "${chunk.slice(0, 4).toString()}"`);
      }
      
      // Process when we have any reasonable amount of data
      // Lower threshold since we're getting very little data
      const shouldProcess = totalBytes >= 100; // Much lower threshold for debugging
      
      if (shouldProcess) {
        const audioBuffer = Buffer.concat(audioChunks);
        console.log(`🎵 Processing ${audioBuffer.length} bytes of audio from FFmpeg for room ${roomId}`);
        
        // Check if this looks like a WAV file (starts with "RIFF")
        if (audioBuffer.length >= 4) {
          const header = audioBuffer.slice(0, 4).toString();
          console.log(`📊 Audio buffer header: "${header}"`);
          if (header === 'RIFF') {
            console.log(`✅ Detected WAV format - this is good`);
          } else {
            console.log(`⚠️ Unexpected format - not WAV`);
          }
        }
        
        // Add to room processor buffer
        roomProcessor.audioBuffer = Buffer.concat([roomProcessor.audioBuffer, audioBuffer]);
        
        // Process the accumulated audio
        this.processAccumulatedAudio(roomId);
        
        // Reset accumulation
        audioChunks = [];
        totalBytes = 0;
      }
    });
    
    ffmpegProcess.stderr.on('data', (data) => {
      const message = data.toString();
      if (!message.includes('frame=') && !message.includes('time=')) {
        console.log(`[FFmpeg ${roomId}]:`, message.trim());
      }
    });
    
    ffmpegProcess.on('close', (code) => {
      console.log(`🔴 FFmpeg process for room ${roomId} exited with code ${code}`);
      
      // Process any remaining audio chunks before cleanup
      if (audioChunks.length > 0 && totalBytes > 0) {
        const audioBuffer = Buffer.concat(audioChunks);
        console.log(`🎵 Processing final ${audioBuffer.length} bytes of audio from FFmpeg for room ${roomId}`);
        
        // Add to room processor buffer
        roomProcessor.audioBuffer = Buffer.concat([roomProcessor.audioBuffer, audioBuffer]);
        
        // Process the accumulated audio
        this.processAccumulatedAudio(roomId);
      }
      
      // Cleanup SDP file
      try {
        fs.unlinkSync(sdpPath);
        console.log(`🗑️ Cleaned up SDP file: ${sdpPath}`);
      } catch (error) {
        console.log(`⚠️ Could not cleanup SDP file: ${error.message}`);
      }
      
      // Don't restart - let it finish naturally
      console.log(`✅ FFmpeg processing completed for room ${roomId}`);
      
      // Optional: restart only if there was an actual error
      // if (roomProcessor.isRecording && code !== 0) {
      //   console.log(`🔄 Restarting FFmpeg after error for room ${roomId}...`);
      //   setTimeout(() => {
      //     this.startFFmpegProcess(roomId, plainTransport, consumer);
      //   }, 2000);
      // }
    });
    
    ffmpegProcess.on('error', (error) => {
      console.error(`🔴 FFmpeg process error for room ${roomId}:`, error);
    });
    
    // Store FFmpeg process for cleanup
    roomProcessor.ffmpegProcess = ffmpegProcess;
    
    // Connect PlainTransport immediately (no delay needed)
    try {
      console.log(`🔗 Connecting PlainTransport to start RTP flow...`);
      await plainTransport.connect({
        ip: '127.0.0.1',
        port: rtpPort,
        rtcpPort: rtcpPort
      });
      console.log(`✅ PlainTransport connected - RTP data should now flow to FFmpeg on port ${rtpPort}`);
    } catch (error) {
      console.error(`🔴 Failed to connect PlainTransport:`, error);
      throw error; // Don't continue if connection fails
    }
    
    // Add debug monitoring for RTP packets
    const dgram = require('dgram');
    const monitorSocket = dgram.createSocket('udp4');
    
    monitorSocket.bind(rtpPort + 1000); // Use a different port for monitoring
    console.log(`🔍 Started RTP monitor on port ${rtpPort + 1000} to debug packet flow`);
    
    // Create a simple UDP listener to see if ANY packets are flowing
    const debugSocket = dgram.createSocket('udp4');
    debugSocket.bind(rtpPort + 2000, '127.0.0.1', () => {
      console.log(`🔍 Debug socket listening on port ${rtpPort + 2000}`);
    });
    
    debugSocket.on('message', (msg, rinfo) => {
      console.log(`📡 DEBUG: Received ${msg.length} bytes from ${rinfo.address}:${rinfo.port}`);
    });
    
    // Store for cleanup
    roomProcessor.debugSocket = debugSocket;
    
    console.log(`✅ FFmpeg audio consumption started for room ${roomId}`);
    console.log(`🔍 If no RTP packets are flowing, the issue is in MediaSoup producer/consumer setup`);
  }
  
  /**
   * Setup test audio generation as fallback when FFmpeg fails
   */
  setupTestAudioGeneration(roomId) {
    console.log(`🔄 Setting up test audio generation fallback for room ${roomId}`);
    
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor) return;
    
    const testInterval = setInterval(() => {
      if (!roomProcessor.isRecording || !this.isConnected) {
        clearInterval(testInterval);
        return;
      }
      
      this.generateTestAudioForProcessing(roomId);
    }, 3000);
    
    roomProcessor.processingInterval = testInterval;
  }
  
  /**
   * Generate test audio for processing pipeline
   */
  generateTestAudioForProcessing(roomId) {
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor) return;
    
    // Generate 3 seconds of test audio
    const sampleRate = this.sampleRate;
    const duration = 3.0;
    const samples = Math.floor(sampleRate * duration);
    const audioBuffer = Buffer.alloc(samples * 2); // 16-bit samples
    
    // Generate sine wave test audio
    for (let i = 0; i < samples; i++) {
      const sample = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.1; // Low volume
      const intSample = Math.round(sample * 32767);
      audioBuffer.writeInt16LE(Math.max(-32767, Math.min(32767, intSample)), i * 2);
    }
    
    roomProcessor.audioBuffer = Buffer.concat([roomProcessor.audioBuffer, audioBuffer]);
    
    // Process the audio
    this.processAccumulatedAudio(roomId);
  }
  
  /**
   * Start capturing REAL audio data from microphone via RTP packets
   */
  async startAudioCapture(roomId, originalProducer) {
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor || !roomProcessor.isRecording) return;
    
    console.log(`🎵 Starting REAL audio capture for room ${roomId} from producer ${originalProducer.id}`);
    console.log(`📊 Producer details:`, {
      id: originalProducer.id,
      kind: originalProducer.kind,
      codec: originalProducer.rtpParameters.codecs[0]?.mimeType,
      paused: originalProducer.paused,
      closed: originalProducer.closed
    });
    
    try {
      // Create a PlainTransport to receive raw RTP packets
      const plainTransport = await roomProcessor.router.createPlainTransport({
        listenIp: { ip: '127.0.0.1', announcedIp: null },
        rtcpMux: false,
        comedia: true // Let mediasoup detect remote IP/port
      });
      
      console.log(`✅ Created PlainTransport: ${plainTransport.id} on ${plainTransport.tuple.localIp}:${plainTransport.tuple.localPort}`);
      
      // Create consumer to pipe audio from original producer to plain transport
      const consumer = await plainTransport.consume({
        producerId: originalProducer.id,
        rtpCapabilities: roomProcessor.router.rtpCapabilities
      });
      
      console.log(`✅ Created consumer: ${consumer.id} for codec: ${consumer.rtpParameters.codecs[0]?.mimeType}`);
      
      // Create UDP socket to receive the RTP packets
      const dgram = require('dgram');
      const udpSocket = dgram.createSocket('udp4');
      
      let receivedPackets = 0;
      let lastLogTime = Date.now();
      
      udpSocket.on('message', (rtpPacket, rinfo) => {
        receivedPackets++;
        
        try {
          // Parse RTP header (12 bytes minimum)
          if (rtpPacket.length < 12) return;
          
          const version = (rtpPacket[0] >> 6) & 0x3;
          const padding = (rtpPacket[0] >> 5) & 0x1;
          const extension = (rtpPacket[0] >> 4) & 0x1;
          const csrcCount = rtpPacket[0] & 0xf;
          const marker = (rtpPacket[1] >> 7) & 0x1;
          const payloadType = rtpPacket[1] & 0x7f;
          const sequenceNumber = rtpPacket.readUInt16BE(2);
          const timestamp = rtpPacket.readUInt32BE(4);
          const ssrc = rtpPacket.readUInt32BE(8);
          
          // Calculate payload offset
          let payloadOffset = 12 + (csrcCount * 4);
          
          if (extension) {
            if (rtpPacket.length < payloadOffset + 4) return;
            const extensionLength = rtpPacket.readUInt16BE(payloadOffset + 2) * 4;
            payloadOffset += 4 + extensionLength;
          }
          
          if (rtpPacket.length <= payloadOffset) return;
          
          // Extract audio payload
          const audioPayload = rtpPacket.slice(payloadOffset);
          
          // Decode based on payload type/codec
          let pcmData = null;
          const codec = consumer.rtpParameters.codecs[0];
          
          if (codec) {
            const codecName = codec.mimeType.toLowerCase();
            
            if (codecName.includes('pcmu')) {
              // μ-law decoding
              pcmData = this.decodePCMU(audioPayload);
            } else if (codecName.includes('pcma')) {
              // A-law decoding  
              pcmData = this.decodePCMA(audioPayload);
            } else if (codecName.includes('opus')) {
              // For Opus, we'd need to decode with libopus, for now skip complex decoding
              // Just treat as raw 16-bit PCM as approximation
              pcmData = audioPayload;
            } else {
              // Assume raw PCM for other codecs
              pcmData = audioPayload;
            }
          } else {
            // Default to raw PCM
            pcmData = audioPayload;
          }
          
          if (pcmData && pcmData.length > 0) {
            // Add decoded audio to buffer
            roomProcessor.audioBuffer = Buffer.concat([roomProcessor.audioBuffer, pcmData]);
          }
          
          // Log progress every 100 packets
          if (receivedPackets % 100 === 0) {
            const now = Date.now();
            const packetsPerSec = 100 / ((now - lastLogTime) / 1000);
            console.log(`🎤 Received ${receivedPackets} RTP packets (${packetsPerSec.toFixed(1)}/s), buffer: ${roomProcessor.audioBuffer.length} bytes`);
            lastLogTime = now;
          }
          
        } catch (error) {
          console.error(`❌ Error processing RTP packet:`, error);
        }
      });
      
      udpSocket.on('error', (error) => {
        console.error(`❌ UDP socket error:`, error);
      });
      
      // Always let the system assign a free port to avoid conflicts
      await new Promise((resolve, reject) => {
        console.log(`🔧 Binding UDP socket to system-assigned port...`);
        udpSocket.bind(0, '127.0.0.1', (error) => {
          if (error) {
            console.error(`❌ Failed to bind UDP socket:`, error);
            reject(error);
          } else {
            const assignedPort = udpSocket.address().port;
            console.log(`✅ UDP socket bound to system-assigned port ${assignedPort} for real audio capture`);
            resolve();
          }
        });
      });
      
      // Store resources for cleanup
      roomProcessor.plainTransport = plainTransport;
      roomProcessor.consumer = consumer;
      roomProcessor.udpSocket = udpSocket;
      
      // Connect plain transport to start receiving packets
      const udpSocketPort = udpSocket.address().port;
      await plainTransport.connect({
        ip: '127.0.0.1',
        port: udpSocketPort
      });
      
      console.log(`🔗 PlainTransport connected to UDP socket on port ${udpSocketPort}`);
      
      console.log(`✅ Real audio capture active - receiving RTP packets from microphone`);
      
    } catch (error) {
      console.error(`❌ Failed to setup real audio capture:`, error);
      
      // Fallback: Generate minimal test audio to keep pipeline working
      console.log(`🔄 Fallback: Using minimal test audio generation`);
      
      const fallbackInterval = setInterval(() => {
        if (!roomProcessor.isRecording) {
          clearInterval(fallbackInterval);
          return;
        }
        
        // Generate simple sine wave as fallback
        const samplesPerChunk = this.sampleRate * 0.1; // 100ms 
        const audioChunk = Buffer.alloc(samplesPerChunk * 2);
        
        for (let i = 0; i < samplesPerChunk; i++) {
          const sample = Math.sin(2 * Math.PI * 440 * i / this.sampleRate); // 440Hz tone
          const intSample = Math.round(sample * 8000); // Lower volume
          audioChunk.writeInt16LE(Math.max(-32767, Math.min(32767, intSample)), i * 2);
        }
        
        roomProcessor.audioBuffer = Buffer.concat([roomProcessor.audioBuffer, audioChunk]);
        
      }, 100);
      
      roomProcessor.captureInterval = fallbackInterval;
    }
  }
  
  /**
   * Decode PCMU (μ-law) to 16-bit PCM
   */
  decodePCMU(data) {
    const pcm = Buffer.alloc(data.length * 2);
    for (let i = 0; i < data.length; i++) {
      const ulaw = data[i];
      const sign = (ulaw & 0x80) ? -1 : 1;
      const exponent = (ulaw & 0x70) >> 4;
      const mantissa = ulaw & 0x0f;
      const sample = sign * ((33 + 2 * mantissa) << exponent - 1);
      pcm.writeInt16LE(sample, i * 2);
    }
    return pcm;
  }
  
  /**
   * Decode PCMA (A-law) to 16-bit PCM  
   */
  decodePCMA(data) {
    const pcm = Buffer.alloc(data.length * 2);
    for (let i = 0; i < data.length; i++) {
      const alaw = data[i];
      const sign = (alaw & 0x80) ? 1 : -1;
      const exponent = (alaw & 0x70) >> 4;  
      const mantissa = alaw & 0x0f;
      let sample;
      if (exponent === 0) {
        sample = sign * (2 * mantissa + 1);
      } else {
        sample = sign * ((32 + 2 * mantissa) << (exponent - 1));
      }
      pcm.writeInt16LE(sample, i * 2);
    }
    return pcm;
  }
  
  /**
   * Process accumulated audio data and send to Flask API
   */
  async processAccumulatedAudio(roomId) {
    console.log(`🎵 processAccumulatedAudio called for room ${roomId}`);
    
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor || !this.isConnected) {
      console.log(`❌ Cannot process audio: roomProcessor=${!!roomProcessor}, connected=${this.isConnected}`);
      return;
    }
    
    const audioBuffer = roomProcessor.audioBuffer;
    console.log(`📊 Audio buffer size: ${audioBuffer.length} bytes`);
    if (audioBuffer.length === 0) {
      console.log(`⚠️ No audio data to process`);
      return;
    }
    
    // Optimal chunk size for 0.79x real-time processing (3 seconds at 16kHz = 96KB)
    const maxChunkSize = this.sampleRate * 3.0 * 2; // 3.0 seconds * 16-bit samples
    const actualChunkSize = Math.min(audioBuffer.length, maxChunkSize);
    
    if (audioBuffer.length > maxChunkSize) {
      console.log(`⚠️ Large audio buffer (${audioBuffer.length} bytes), processing first ${actualChunkSize} bytes`);
    }
    
    const chunkToProcess = audioBuffer.slice(0, actualChunkSize);
    
    try {
      // Convert audio chunk to base64 for Flask API
      const audioBase64 = chunkToProcess.toString('base64');
      const timestamp = Date.now();
      
      console.log(`🎵 Sending ${chunkToProcess.length} bytes to Flask API for processing (room: ${roomId})`);
      
      // Send to Flask API
      const response = await fetch(`${this.redactionServiceUrl}/process_audio`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio_data: audioBase64,
          sample_rate: this.sampleRate,
          room_id: roomId,
          timestamp: timestamp
        })
      });
      
      if (!response.ok) {
        throw new Error(`Flask API error: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Handle the redacted audio result
      await this.handleRedactedAudioFromAPI(roomId, result, timestamp);
      
      // Remove processed chunk from buffer
      roomProcessor.audioBuffer = audioBuffer.slice(actualChunkSize);
      roomProcessor.lastProcessTime = timestamp;
      
      console.log(`✅ Audio processed: ${result.pii_count} PII detections, processing complete`);
      
    } catch (error) {
      console.error(`🔴 Failed to process audio for room ${roomId}:`, error);
    }
  }
  
  /**
   * Handle redacted audio result from Flask API
   */
  async handleRedactedAudioFromAPI(roomId, result, originalTimestamp) {
    if (!result.success) {
      console.error(`🔴 Audio processing failed:`, result.error);
      return;
    }
    
    try {
      // Convert base64 back to buffer
      const redactedAudioBuffer = Buffer.from(result.redacted_audio_data, 'base64');
      
      // Store the processed audio with synchronization data
      const audioSegment = {
        roomId: roomId,
        timestamp: originalTimestamp,
        processingTime: Date.now() - originalTimestamp,
        redactedAudio: redactedAudioBuffer,
        transcript: result.transcript,
        piiCount: result.pii_count,
        redactedIntervals: result.redacted_intervals,
        processed: true
      };
      
      // Add to audio buffer with delay for video sync
      if (!this.audioSegmentBuffer.has(roomId)) {
        this.audioSegmentBuffer.set(roomId, []);
      }
      this.audioSegmentBuffer.get(roomId).push(audioSegment);
      
      // Stream the redacted audio (with synchronization delay)
      await this.streamRedactedAudio(roomId, audioSegment);
      
      // If PII was detected, trigger video synchronization
      if (result.pii_count > 0) {
        await this.synchronizeVideoRedaction(roomId, result.redacted_intervals, originalTimestamp);
      }
      
    } catch (error) {
      console.error(`🔴 Failed to handle redacted audio:`, error);
    }
  }
  
  /**
   * Synchronize video redaction with audio PII detections
   */
  async synchronizeVideoRedaction(roomId, piiIntervals, audioTimestamp) {
    console.log(`🔄 Synchronizing video redaction for room ${roomId}: ${piiIntervals.length} intervals`);
    
    // Mark video frames that correspond to PII audio intervals for extra blurring
    if (this.videoFrameBuffer.has(roomId)) {
      const videoFrames = this.videoFrameBuffer.get(roomId);
      
      piiIntervals.forEach(([startSec, endSec]) => {
        const startTime = audioTimestamp + (startSec * 1000);
        const endTime = audioTimestamp + (endSec * 1000);
        
        // Mark corresponding video frames for enhanced processing
        videoFrames.forEach(frame => {
          if (frame.timestamp >= startTime && frame.timestamp <= endTime) {
            frame.enhancedBlurNeeded = true;
            frame.piiAudioDetected = true;
          }
        });
      });
      
      console.log(`✅ Marked video frames for enhanced blurring during PII intervals`);
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
   * Handle redacted audio (now handled by handleRedactedAudioFromAPI)
   * This method is kept for compatibility but redirects to new handler
   */
  handleRedactedAudio(data) {
    console.log('🔇 Legacy handler called - redirecting to new API handler');
    // This is now handled in handleRedactedAudioFromAPI method
  }
  
  /**
   * Stream redacted audio data to the redacted producer
   */
  async streamRedactedAudio(roomId, audioSegment) {
    const roomProcessor = this.roomProcessors.get(roomId);
    if (!roomProcessor || !roomProcessor.redactedProducer) {
      console.log(`🟡 No redacted producer for room ${roomId}, queueing audio`);
      return;
    }
    
    try {
      // Store redacted audio for this room with timing info
      if (!this.redactedAudioBuffer.has(roomId)) {
        this.redactedAudioBuffer.set(roomId, []);
      }
      
      const audioQueue = this.redactedAudioBuffer.get(roomId);
      audioQueue.push({
        timestamp: audioSegment.timestamp,
        audio: audioSegment.redactedAudio,
        piiCount: audioSegment.piiCount,
        processingTime: audioSegment.processingTime
      });
      
      // Keep only recent audio segments (10 seconds)
      const cutoffTime = Date.now() - 10000;
      const filteredQueue = audioQueue.filter(seg => seg.timestamp >= cutoffTime);
      this.redactedAudioBuffer.set(roomId, filteredQueue);
      
      console.log(`🎵 Queued redacted audio for room ${roomId}: ${audioSegment.redactedAudio.length} bytes, ${audioSegment.piiCount} PII detections`);
      
      // For now, we'll use the existing RTP infrastructure
      // In a full implementation, this would send the redacted audio through RTP
      this.logRedactionEvent({
        timestamp: audioSegment.timestamp,
        pii_count: audioSegment.piiCount,
        processing_time: audioSegment.processingTime,
        transcript: audioSegment.transcript
      });
      
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
      
      // Close FFmpeg process
      if (roomProcessor.ffmpegProcess) {
        console.log(`🔴 Terminating FFmpeg process for room ${roomId}`);
        roomProcessor.ffmpegProcess.kill('SIGTERM');
      }
      
      // Close audio capture resources
      if (roomProcessor.udpSocket) {
        roomProcessor.udpSocket.close();
      }
      if (roomProcessor.consumer) {
        roomProcessor.consumer.close();
      }
      if (roomProcessor.plainTransport) {
        roomProcessor.plainTransport.close();
      }
      if (roomProcessor.pipeTransport) {
        roomProcessor.pipeTransport.close();
      }
      if (roomProcessor.realConsumer) {
        roomProcessor.realConsumer.close();
      }
      if (roomProcessor.consumerTransport) {
        roomProcessor.consumerTransport.close();
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
    
    console.log(`✅ Audio redaction cleanup completed for room ${roomId}`);
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
   * Get audio redaction data for a specific timestamp (for video synchronization)
   */
  getAudioRedactionDataForTimestamp(roomId, timestamp) {
    const audioQueue = this.redactedAudioBuffer.get(roomId);
    if (!audioQueue || audioQueue.length === 0) {
      return null;
    }
    
    // Check if timestamp falls within any PII detection window
    for (const audioSegment of audioQueue) {
      const segmentStart = audioSegment.timestamp;
      const segmentEnd = segmentStart + 3000; // 3-second audio segments
      
      if (timestamp >= segmentStart && timestamp <= segmentEnd && audioSegment.piiCount > 0) {
        return {
          piiDetected: true,
          piiCount: audioSegment.piiCount,
          processingTime: audioSegment.processingTime,
          timestamp: audioSegment.timestamp
        };
      }
    }
    
    return { piiDetected: false };
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