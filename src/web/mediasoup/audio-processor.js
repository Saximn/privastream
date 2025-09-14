// audio-processor.js
const prism = require('prism-media');
const fetch = require('node-fetch');
const fs = require('fs');
const { EventEmitter } = require('events');

class AudioProcessor extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      redactionServiceUrl: options.redactionServiceUrl || 'http://localhost:5002',
      sampleRate: 48000,
      channels: 2,
      bufferDurationMs: 3000, // 3 seconds
      ...options
    };
    
    this.bufferSize = this.calculateBufferSize();
    this.pcmBuffer = Buffer.alloc(0);
    this.isProcessing = false;
    this.processedAudioQueue = [];
    
    console.log('[AUDIO-PROCESSOR] Initialized with options:', this.options);
    console.log('[AUDIO-PROCESSOR] Buffer size:', this.bufferSize, 'bytes');
  }
  
  calculateBufferSize() {
    // 48kHz, 2 channels, 16-bit PCM, 3 seconds
    return this.options.sampleRate * this.options.channels * 2 * (this.options.bufferDurationMs / 1000);
  }
  
  /**
   * Create and setup audio consumer for processing
   */
  async setupAudioConsumer(producer, router) {
    try {
      console.log('[AUDIO-PROCESSOR] Setting up audio consumer for producer:', producer.id);
      
      // Create a PlainTransport for RTP streams
      const plainTransport = await router.createPlainTransport({
        listenIp: '127.0.0.1',
        rtcpMux: false,
        comedia: true
      });
      
      console.log('[AUDIO-PROCESSOR] Created PlainTransport:', {
        id: plainTransport.id,
        tuple: plainTransport.tuple
      });
      
      // Create consumer on the PlainTransport
      const consumer = await plainTransport.consume({
        producerId: producer.id,
        rtpCapabilities: router.rtpCapabilities,
        paused: false
      });
      
      console.log('[AUDIO-PROCESSOR] Created consumer:', {
        id: consumer.id,
        kind: consumer.kind,
        rtpParameters: consumer.rtpParameters
      });
      
      // Set up RTP stream processing
      this.setupRtpStreaming(consumer, plainTransport);
      
      return { consumer, plainTransport };
      
    } catch (error) {
      console.error('[AUDIO-PROCESSOR] Failed to setup audio consumer:', error);
      throw error;
    }
  }
  
  /**
   * Setup RTP streaming and audio decoding pipeline
   */
  setupRtpStreaming(consumer, plainTransport) {
    console.log('[AUDIO-PROCESSOR] Setting up RTP streaming pipeline');
    
    // Create FFmpeg process to decode RTP stream
    const ffmpegArgs = [
      '-protocol_whitelist', 'pipe,udp,rtp',
      '-f', 'rtp',
      '-i', `rtp://127.0.0.1:${plainTransport.tuple.localPort}`,
      '-f', 's16le',
      '-ar', '48000',
      '-ac', '2',
      '-'
    ];
    
    const { spawn } = require('child_process');
    const ffmpeg = spawn('ffmpeg', ffmpegArgs);
    
    ffmpeg.stderr.on('data', (data) => {
      console.log('[AUDIO-PROCESSOR] FFmpeg:', data.toString().trim());
    });
    
    ffmpeg.on('error', (error) => {
      console.error('[AUDIO-PROCESSOR] FFmpeg error:', error);
    });
    
    // Process the decoded PCM stream
    ffmpeg.stdout.on('data', (chunk) => {
      this.processPcmChunk(chunk);
    });
    
    consumer.on('transportclose', () => {
      console.log('[AUDIO-PROCESSOR] Consumer transport closed, stopping FFmpeg');
      ffmpeg.kill();
    });
  }
  
  /**
   * Process incoming PCM chunks and buffer them
   */
  processPcmChunk(chunk) {
    try {
      // Add chunk to buffer
      this.pcmBuffer = Buffer.concat([this.pcmBuffer, chunk]);
      
      // Check if we have enough data for processing (3 seconds)
      if (this.pcmBuffer.length >= this.bufferSize && !this.isProcessing) {
        const audioChunk = this.pcmBuffer.slice(0, this.bufferSize);
        this.pcmBuffer = this.pcmBuffer.slice(this.bufferSize);
        
        console.log('[AUDIO-PROCESSOR] Processing audio chunk:', audioChunk.length, 'bytes');
        this.processAudioChunk(audioChunk);
      }
    } catch (error) {
      console.error('[AUDIO-PROCESSOR] Error processing PCM chunk:', error);
    }
  }
  
  /**
   * Send audio chunk to redaction service and get processed audio back
   */
  async processAudioChunk(audioBuffer) {
    if (this.isProcessing) {
      console.log('[AUDIO-PROCESSOR] Already processing, skipping chunk');
      return;
    }
    
    this.isProcessing = true;
    
    try {
      console.log('[AUDIO-PROCESSOR] Sending audio chunk to redaction service');
      
      // Convert PCM buffer to base64 for transmission
      const audioBase64 = audioBuffer.toString('base64');
      
      const response = await fetch(`${this.options.redactionServiceUrl}/process_audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_data: audioBase64,
          sample_rate: this.options.sampleRate
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Convert processed audio back to buffer
        const processedAudioBuffer = Buffer.from(result.redacted_audio_data, 'base64');
        
        console.log('[AUDIO-PROCESSOR] Received processed audio:', {
          originalSize: audioBuffer.length,
          processedSize: processedAudioBuffer.length,
          piiCount: result.pii_count,
          processingTime: result.processing_time
        });
        
        // Queue processed audio for encoding
        this.processedAudioQueue.push({
          buffer: processedAudioBuffer,
          metadata: result
        });
        
        // Process the queue
        this.processAudioQueue();
        
      } else {
        console.error('[AUDIO-PROCESSOR] Audio processing failed:', result.error);
        // Use original audio as fallback
        this.processedAudioQueue.push({
          buffer: audioBuffer,
          metadata: { success: false, error: result.error }
        });
        this.processAudioQueue();
      }
      
    } catch (error) {
      console.error('[AUDIO-PROCESSOR] Error calling redaction service:', error);
      // Use original audio as fallback
      this.processedAudioQueue.push({
        buffer: audioBuffer,
        metadata: { success: false, error: error.message }
      });
      this.processAudioQueue();
      
    } finally {
      this.isProcessing = false;
    }
  }
  
  /**
   * Process the queue of processed audio chunks
   */
  processAudioQueue() {
    while (this.processedAudioQueue.length > 0) {
      const { buffer, metadata } = this.processedAudioQueue.shift();
      
      // Emit processed audio event
      this.emit('processedAudio', {
        audioBuffer: buffer,
        metadata: metadata
      });
    }
  }
  
  /**
   * Create a new producer with processed audio
   */
  async createProcessedAudioProducer(router, processedAudioStream) {
    try {
      console.log('[AUDIO-PROCESSOR] Creating processed audio producer');
      
      // Create a PlainTransport for injecting processed audio
      const plainTransport = await router.createPlainTransport({
        listenIp: '127.0.0.1',
        rtcpMux: false,
        comedia: false
      });
      
      // Create producer with Opus codec
      const producer = await plainTransport.produce({
        kind: 'audio',
        rtpParameters: {
          codecs: [{
            mimeType: 'audio/opus',
            clockRate: 48000,
            channels: 2,
            payloadType: 111
          }],
          headerExtensions: [],
          encodings: [{ ssrc: 22222222 }],
          rtcp: { cname: 'processed-audio' }
        }
      });
      
      console.log('[AUDIO-PROCESSOR] Created processed audio producer:', producer.id);
      
      return { producer, plainTransport };
      
    } catch (error) {
      console.error('[AUDIO-PROCESSOR] Failed to create processed audio producer:', error);
      throw error;
    }
  }
}

/**
 * Main function to consume and process audio
 */
async function consumeAndProcessAudio(originalProducer, router, options = {}) {
  console.log('[AUDIO-PROCESSOR] Starting consumeAndProcessAudio for producer:', originalProducer.id);
  
  try {
    // Create audio processor
    const audioProcessor = new AudioProcessor(options);
    
    // Setup audio consumer
    const { consumer, plainTransport: consumerTransport } = await audioProcessor.setupAudioConsumer(originalProducer, router);
    
    // Create processed audio producer
    let processedProducer = null;
    let producerTransport = null;
    
    // Handle processed audio
    audioProcessor.on('processedAudio', async ({ audioBuffer, metadata }) => {
      try {
        console.log('[AUDIO-PROCESSOR] Received processed audio chunk:', audioBuffer.length, 'bytes');
        
        // Create processed producer if not exists
        if (!processedProducer) {
          const result = await audioProcessor.createProcessedAudioProducer(router, null);
          processedProducer = result.producer;
          producerTransport = result.plainTransport;
          
          console.log('[AUDIO-PROCESSOR] Processed audio producer created:', processedProducer.id);
        }
        
        // Here you would inject the processed audio into the producer
        // This requires setting up FFmpeg to encode PCM to Opus and stream via RTP
        
      } catch (error) {
        console.error('[AUDIO-PROCESSOR] Error handling processed audio:', error);
      }
    });
    
    // Clean up on producer close
    originalProducer.on('close', () => {
      console.log('[AUDIO-PROCESSOR] Original producer closed, cleaning up');
      consumer.close();
      consumerTransport.close();
      if (processedProducer) processedProducer.close();
      if (producerTransport) producerTransport.close();
    });
    
    return {
      processedProducer,
      consumer,
      audioProcessor
    };
    
  } catch (error) {
    console.error('[AUDIO-PROCESSOR] Failed to setup audio processing:', error);
    throw error;
  }
}

module.exports = {
  AudioProcessor,
  consumeAndProcessAudio
};