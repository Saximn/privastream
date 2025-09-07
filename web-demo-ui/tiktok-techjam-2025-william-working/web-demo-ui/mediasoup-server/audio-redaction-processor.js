// audio-redaction-processor.js
const fetch = require('node-fetch');
const { Transform } = require('stream');

class AudioRedactionProcessor {
  constructor(options = {}) {
    this.options = {
      redactionServiceUrl: options.redactionServiceUrl || 'http://localhost:5002',
      sampleRate: 16000, // Changed to 16kHz to match Vosk
      channels: 1,       // Changed to mono
      bufferDurationMs: 3000, // 3 seconds
      ...options
    };
    
    // Calculate buffer size for 3 seconds of audio
    // 16kHz * 1 channel * 2 bytes per sample * 3 seconds = 96,000 bytes
    this.bufferSize = this.options.sampleRate * this.options.channels * 2 * (this.options.bufferDurationMs / 1000);
    this.pcmBuffer = Buffer.alloc(0);
    this.processedProducers = new Map();
    
    console.log('[AUDIO-REDACTION-PROCESSOR] Initialized:', {
      bufferSize: this.bufferSize,
      sampleRate: this.options.sampleRate,
      channels: this.options.channels
    });
  }
  
  /**
   * Setup audio processing for a producer
   */
  async setupAudioProcessing(roomId, originalProducer, router) {
    try {
      console.log('[AUDIO-REDACTION-PROCESSOR] Setting up audio processing for room:', roomId);
      
      // Create a data consumer to receive audio data
      const audioConsumer = await this.createAudioConsumer(originalProducer, router);
      
      // Process audio and create new producer with processed audio
      const processedProducer = await this.createProcessedAudioProducer(roomId, audioConsumer, router);
      
      // Store the processed producer
      this.processedProducers.set(roomId, processedProducer);
      
      console.log('[AUDIO-REDACTION-PROCESSOR] Audio processing setup complete for room:', roomId);
      
      return processedProducer;
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Failed to setup audio processing:', error);
      throw error;
    }
  }
  
  /**
   * Create an audio consumer using PipeTransport for data extraction
   */
  async createAudioConsumer(producer, router) {
    try {
      // Create a PipeTransport for consuming audio data
      const pipeTransportConsumer = await router.createPipeTransport({
        listenIp: '127.0.0.1'
      });
      
      // Create consumer
      const consumer = await pipeTransportConsumer.consume({
        producerId: producer.id,
        rtpCapabilities: router.rtpCapabilities
      });
      
      console.log('[AUDIO-REDACTION-PROCESSOR] Created audio consumer:', consumer.id);
      
      return { consumer, transport: pipeTransportConsumer };
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Failed to create audio consumer:', error);
      throw error;
    }
  }
  
  /**
   * Create a processed audio producer
   */
  async createProcessedAudioProducer(roomId, audioConsumerData, router) {
    try {
      // Create a PipeTransport for producing processed audio
      const pipeTransportProducer = await router.createPipeTransport({
        listenIp: '127.0.0.1'
      });
      
      // For now, we'll use a simpler approach - create a producer that can be consumed by viewers
      // The actual audio processing will be handled via direct buffer processing
      console.log('[AUDIO-REDACTION-PROCESSOR] Created pipe transport for processed audio');
      
      return { transport: pipeTransportProducer, roomId };
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Failed to create processed audio producer:', error);
      throw error;
    }
  }
  
  /**
   * Process raw audio buffer through redaction service
   */
  async processAudioBuffer(audioBuffer) {
    try {
      console.log('[AUDIO-REDACTION-PROCESSOR] Processing audio buffer:', audioBuffer.length, 'bytes');
      
      // Convert audio buffer to base64 for transmission
      const audioBase64 = audioBuffer.toString('base64');
      
      // Send to redaction service
      const response = await fetch(`${this.options.redactionServiceUrl}/process_audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_data: audioBase64,
          sample_rate: this.options.sampleRate
        }),
        timeout: 10000 // 10 second timeout
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Convert processed audio back to buffer
        const processedBuffer = Buffer.from(result.redacted_audio_data, 'base64');
        
        console.log('[AUDIO-REDACTION-PROCESSOR] Audio processed successfully:', {
          piiCount: result.pii_count,
          processingTime: result.processing_time,
          transcript: result.transcript?.substring(0, 100) + '...'
        });
        
        return {
          success: true,
          processedAudio: processedBuffer,
          metadata: result
        };
      } else {
        console.error('[AUDIO-REDACTION-PROCESSOR] Processing failed:', result.error);
        return {
          success: false,
          processedAudio: audioBuffer, // Return original as fallback
          error: result.error
        };
      }
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Error processing audio:', error);
      return {
        success: false,
        processedAudio: audioBuffer, // Return original as fallback
        error: error.message
      };
    }
  }
  
  /**
   * Add audio chunk to buffer and process when ready
   */
  async addAudioChunk(roomId, audioChunk) {
    try {
      // Add to buffer
      this.pcmBuffer = Buffer.concat([this.pcmBuffer, audioChunk]);
      
      console.log(`[AUDIO-REDACTION-PROCESSOR] Buffer status: ${this.pcmBuffer.length}/${this.bufferSize} bytes (${(this.pcmBuffer.length/this.bufferSize*100).toFixed(1)}%)`);
      
      // Check if we have enough data (3 seconds worth)
      if (this.pcmBuffer.length >= this.bufferSize) {
        // Extract 3-second chunk
        const processChunk = this.pcmBuffer.slice(0, this.bufferSize);
        this.pcmBuffer = this.pcmBuffer.slice(this.bufferSize);
        
        console.log('[AUDIO-REDACTION-PROCESSOR] Processing 3-second audio chunk for room:', roomId);
        console.log('[AUDIO-REDACTION-PROCESSOR] Processing audio buffer:', processChunk.length, 'bytes');
        
        // Process the chunk
        const result = await this.processAudioBuffer(processChunk);
        
        if (result.success) {
          console.log('[AUDIO-REDACTION-PROCESSOR] Audio processed successfully:', {
            piiCount: result.metadata.pii_count,
            processingTime: result.metadata.processing_time,
            transcript: result.metadata.transcript ? result.metadata.transcript.substring(0, 50) + '...' : 'empty'
          });
        }
        
        return result;
      }
      
      return null; // Not enough data yet
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Error adding audio chunk:', error);
      return null;
    }
  }
  
  /**
   * Cleanup resources for a room
   */
  cleanup(roomId) {
    try {
      console.log('[AUDIO-REDACTION-PROCESSOR] Cleaning up room:', roomId);
      
      if (this.processedProducers.has(roomId)) {
        const producer = this.processedProducers.get(roomId);
        if (producer.transport) {
          producer.transport.close();
        }
        this.processedProducers.delete(roomId);
      }
      
      // Reset buffer for this room (in a real implementation, you'd have per-room buffers)
      this.pcmBuffer = Buffer.alloc(0);
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Error during cleanup:', error);
    }
  }
}

module.exports = { AudioRedactionProcessor };