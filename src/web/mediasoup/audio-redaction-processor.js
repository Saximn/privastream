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
    this.roomBuffers = new Map(); // roomId -> Buffer (per-room audio buffers)
    this.processedProducers = new Map();
    
    // Sliding window: store previous chunks per room for context processing
    this.previousChunks = new Map(); // roomId -> previous 3-second chunk (context only)
    
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
   * Process raw audio buffer through redaction service with sliding window
   */
  async processAudioBufferSlidingWindow(roomId, currentChunk, previousChunk = null) {
    try {
      let processingBuffer;
      let isFirstChunk = false;
      
      if (previousChunk) {
        // Combine previous + current for sliding window processing
        processingBuffer = Buffer.concat([previousChunk, currentChunk]);
        console.log(`[AUDIO-REDACTION-PROCESSOR] Sliding window: processing ${processingBuffer.length} bytes (previous: ${previousChunk.length} + current: ${currentChunk.length})`);
      } else {
        // First chunk - no previous data
        processingBuffer = currentChunk;
        isFirstChunk = true;
        console.log(`[AUDIO-REDACTION-PROCESSOR] First chunk: processing ${processingBuffer.length} bytes`);
      }
      
      // Convert audio buffer to base64 for transmission
      const audioBase64 = processingBuffer.toString('base64');
      
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
        const fullProcessedBuffer = Buffer.from(result.redacted_audio_data, 'base64');
        
        let outputBuffer;
        if (isFirstChunk) {
          // First chunk: return entire processed buffer (no previous context)
          outputBuffer = fullProcessedBuffer;
        } else {
          // Subsequent chunks: we sent [previous + current] but only want to output the current part
          // The API processed the combined audio, but we only output the second half (current chunk)
          const halfLength = Math.floor(fullProcessedBuffer.length / 2);
          outputBuffer = Buffer.from(fullProcessedBuffer.slice(halfLength));
          
          console.log(`[AUDIO-REDACTION-PROCESSOR] Context-only sliding window: extracted current chunk ${outputBuffer.length}/${fullProcessedBuffer.length} bytes`);
          console.log(`[AUDIO-REDACTION-PROCESSOR] Previous 3s used for context only - not modifying already-output audio`);
          console.log(`[AUDIO-REDACTION-PROCESSOR] PII detected: ${result.pii_count} regions, transcript: "${result.transcript?.substring(0, 50)}..."`);
        }
        
        console.log('[AUDIO-REDACTION-PROCESSOR] Audio processed successfully:', {
          piiCount: result.pii_count,
          processingTime: result.processing_time,
          transcript: result.transcript?.substring(0, 100) + '...',
          totalProcessed: fullProcessedBuffer.length,
          outputSize: outputBuffer.length,
          slidingWindow: !isFirstChunk
        });
        
        return {
          success: true,
          processedAudio: outputBuffer,
          metadata: {
            ...result,
            hadPreviousChunk: !isFirstChunk,
            slidingWindow: !isFirstChunk
          }
        };
      } else {
        console.error('[AUDIO-REDACTION-PROCESSOR] Processing failed:', result.error);
        return {
          success: false,
          processedAudio: currentChunk, // Return current chunk as fallback
          metadata: {
            hadPreviousChunk: !isFirstChunk,
            slidingWindow: !isFirstChunk
          },
          error: result.error
        };
      }
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Error processing audio:', error);
      return {
        success: false,
        processedAudio: currentChunk, // Return current chunk as fallback
        metadata: {
          hadPreviousChunk: !isFirstChunk,
          slidingWindow: !isFirstChunk
        },
        error: error.message
      };
    }
  }

  /**
   * Process raw audio buffer through redaction service (legacy method)
   */
  async processAudioBuffer(audioBuffer) {
    return this.processAudioBufferSlidingWindow('default', audioBuffer, null);
  }
  
  /**
   * Add audio chunk to buffer and process when ready with sliding window
   */
  async addAudioChunk(roomId, audioChunk) {
    try {
      // Get or create room-specific buffer
      if (!this.roomBuffers.has(roomId)) {
        this.roomBuffers.set(roomId, Buffer.alloc(0));
      }
      
      // Add to room-specific buffer
      const roomBuffer = this.roomBuffers.get(roomId);
      const updatedBuffer = Buffer.concat([roomBuffer, audioChunk]);
      this.roomBuffers.set(roomId, updatedBuffer);
      
      console.log(`[AUDIO-REDACTION-PROCESSOR] Room ${roomId} buffer status: ${updatedBuffer.length}/${this.bufferSize} bytes (${(updatedBuffer.length/this.bufferSize*100).toFixed(1)}%)`);
      
      // Check if we have enough data (3 seconds worth) for this room
      if (updatedBuffer.length >= this.bufferSize) {
        // Extract current 3-second chunk from room buffer
        const currentChunk = updatedBuffer.slice(0, this.bufferSize);
        const remainingBuffer = updatedBuffer.slice(this.bufferSize);
        this.roomBuffers.set(roomId, remainingBuffer);
        
        // Get previous chunk for sliding window
        const previousChunk = this.previousChunks.get(roomId);
        
        console.log('[AUDIO-REDACTION-PROCESSOR] Processing 3-second audio chunk for room:', roomId);
        console.log('[AUDIO-REDACTION-PROCESSOR] Current chunk:', currentChunk.length, 'bytes, Previous chunk:', previousChunk ? previousChunk.length : 0, 'bytes');
        
        // Process with sliding window
        const result = await this.processAudioBufferSlidingWindow(roomId, currentChunk, previousChunk);
        
        // Store current chunk as previous for next iteration
        this.previousChunks.set(roomId, currentChunk);
        
        if (result.success) {
          console.log('[AUDIO-REDACTION-PROCESSOR] Sliding window audio processed successfully:', {
            piiCount: result.metadata.pii_count,
            processingTime: result.metadata.processing_time,
            transcript: result.metadata.transcript ? result.metadata.transcript.substring(0, 50) + '...' : 'empty',
            outputSize: result.processedAudio.length,
            hadPreviousChunk: !!previousChunk
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
      
      // Clean up sliding window data for this room
      if (this.previousChunks.has(roomId)) {
        this.previousChunks.delete(roomId);
        console.log('[AUDIO-REDACTION-PROCESSOR] Cleared previous chunk context for room:', roomId);
      }
      
      // Clean up room-specific buffer
      if (this.roomBuffers.has(roomId)) {
        this.roomBuffers.delete(roomId);
        console.log('[AUDIO-REDACTION-PROCESSOR] Cleared audio buffer for room:', roomId);
      }
      
    } catch (error) {
      console.error('[AUDIO-REDACTION-PROCESSOR] Error during cleanup:', error);
    }
  }
}

module.exports = { AudioRedactionProcessor };