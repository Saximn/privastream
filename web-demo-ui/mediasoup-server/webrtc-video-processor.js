// webrtc-video-processor.js - In-memory WebRTC video processing pipeline
const fetch = require('node-fetch');
const { EventEmitter } = require('events');

class WebRTCVideoProcessor extends EventEmitter {
  constructor(options = {}) {
    super();
    this.options = {
      videoServiceUrl: options.videoServiceUrl || 'http://localhost:5001',
      processEveryNthFrame: 15, // Process every 15th frame (matches audio 3s chunks)
      bufferDurationMs: 3000, // 3 seconds to match audio
      frameWidth: 1280,
      frameHeight: 720,
      frameRate: 30,
      ...options
    };
    
    this.processedProducers = new Map(); // roomId -> processed producer
    this.frameProcessingState = new Map(); // roomId -> processing state
    this.videoConsumers = new Map(); // roomId -> consumer info
    
    console.log('[WEBRTC-VIDEO-PROCESSOR] 🚀 Initialized:', {
      processEveryNthFrame: this.options.processEveryNthFrame,
      bufferDurationMs: this.options.bufferDurationMs,
      frameSize: `${this.options.frameWidth}x${this.options.frameHeight}`
    });
  }
  
  /**
   * Setup WebRTC video processing pipeline for a room
   */
  async setupVideoProcessing(roomId, originalProducer, router) {
    try {
      console.log('[WEBRTC-VIDEO-PROCESSOR] 🎬 Setting up WebRTC video processing for room:', roomId);
      
      // Initialize processing state
      this.frameProcessingState.set(roomId, {
        frameCount: 0,
        lastDetection: null,
        lastProcessedFrame: 0,
        processingQueue: []
      });
      
      // Store reference to original producer for this room
      this.processedProducers.set(roomId, originalProducer);
      
      console.log('[WEBRTC-VIDEO-PROCESSOR] ✅ WebRTC video processing setup complete for room:', roomId);
      
      return originalProducer;
      
    } catch (error) {
      console.error('[WEBRTC-VIDEO-PROCESSOR] ❌ Failed to setup video processing:', error);
      throw error;
    }
  }
  
  /**
   * Process individual video frame through Python service
   */
  async processVideoFrame(roomId, frameBase64, frameCount) {
    const state = this.frameProcessingState.get(roomId);
    if (!state) {
      console.log(`[WEBRTC-VIDEO-PROCESSOR] ❌ No processing state found for room ${roomId}`);
      return;
    }
    
    console.log(`[WEBRTC-VIDEO-PROCESSOR] 📊 Processing frame ${frameCount} for room ${roomId}. Every ${this.options.processEveryNthFrame}th frame check: ${frameCount % this.options.processEveryNthFrame === 1}`);
    
    try {
      // Process every Nth frame for detection
      if (frameCount % this.options.processEveryNthFrame === 1) {
        console.log(`[WEBRTC-VIDEO-PROCESSOR] 🔍 DETECTION FRAME: Processing frame ${frameCount} for detection`);
        
        // Send to Python service for detection
        const detectionResult = await this.callPythonDetection(frameBase64, frameCount);
        
        if (detectionResult) {
          state.lastDetection = detectionResult;
          state.lastProcessedFrame = frameCount;
          
          console.log(`[WEBRTC-VIDEO-PROCESSOR] 🎯 Detection complete: ${detectionResult.boundingBoxes?.length || 0} regions found`);
          
          return {
            processedFrame: detectionResult.processedFrame,
            boundingBoxes: detectionResult.boundingBoxes,
            wasDetectionFrame: true
          };
        }
      } else if (state.lastDetection) {
        // Interpolate bounding boxes for intermediate frames
        const framesSinceDetection = frameCount - state.lastProcessedFrame;
        const interpolatedBoxes = this.interpolateBoundingBoxes(state.lastDetection.boundingBoxes, framesSinceDetection);
        
        console.log(`[WEBRTC-VIDEO-PROCESSOR] 📐 Frame ${frameCount}: Using interpolated bounding boxes (${framesSinceDetection} frames since detection)`);
        
        // Apply blur to current frame using interpolated boxes
        if (interpolatedBoxes && interpolatedBoxes.length > 0) {
          const blurredFrame = await this.applyBlurToFrame(frameBase64, interpolatedBoxes);
          // PRIVACY FIX: Check if blur failed and drop frame if needed
          if (blurredFrame === null) {
            console.log('[WEBRTC-VIDEO-PROCESSOR] 🔒 PRIVACY PROTECTION: Blur failed - dropping interpolated frame');
            return null;
          }
          return {
            processedFrame: blurredFrame,
            boundingBoxes: interpolatedBoxes,
            wasDetectionFrame: false
          };
        }
      }
      
      // PRIVACY FIX: Return null if no processing needed to force frame dropping
      console.log('[WEBRTC-VIDEO-PROCESSOR] 🔒 PRIVACY PROTECTION: No processing available - dropping frame');
      return null;
      
    } catch (error) {
      console.error('[WEBRTC-VIDEO-PROCESSOR] ❌ Frame processing error:', error);
      console.log('[WEBRTC-VIDEO-PROCESSOR] 🔒 PRIVACY PROTECTION: Error occurred - dropping frame');
      // PRIVACY FIX: Return null instead of original frame to prevent privacy leaks
      return null;
    }
  }
  
  /**
   * Call Python service for detection
   */
  async callPythonDetection(frameB64, frameCount) {
    try {
      console.log(`[WEBRTC-VIDEO-PROCESSOR] 📡 Calling Python detection service for frame ${frameCount}...`);
      console.log(`[WEBRTC-VIDEO-PROCESSOR] Service URL: ${this.options.videoServiceUrl}/process-frame`);
      console.log(`[WEBRTC-VIDEO-PROCESSOR] Frame data length: ${frameB64.length}`);
      
      const response = await fetch(`${this.options.videoServiceUrl}/process-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          frame: `data:image/jpeg;base64,${frameB64}`, 
          detect_only: true,
          frame_id: frameCount
        }),
        timeout: 5000
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log(`[WEBRTC-VIDEO-PROCESSOR] ✅ Python service responded successfully for frame ${frameCount}`);
        
        return {
          boundingBoxes: result.rectangles || [],
          polygons: result.polygons || [],
          processedFrame: result.frame
        };
      } else {
        console.error('[WEBRTC-VIDEO-PROCESSOR] ❌ Python service error:', response.status);
        return null;
      }
      
    } catch (error) {
      console.error('[WEBRTC-VIDEO-PROCESSOR] ❌ Python detection error:', error);
      return null;
    }
  }
  
  /**
   * Apply blur to frame using provided bounding boxes
   */
  async applyBlurToFrame(frameB64, boundingBoxes) {
    try {
      console.log(`[WEBRTC-VIDEO-PROCESSOR] 🔧 Applying blur to frame with ${boundingBoxes.length} regions`);
      
      const response = await fetch(`${this.options.videoServiceUrl}/process-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame: `data:image/jpeg;base64,${frameB64}`,
          blur_only: true,
          rectangles: boundingBoxes,
          polygons: []
        }),
        timeout: 3000
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('[WEBRTC-VIDEO-PROCESSOR] ✅ Blur applied successfully');
        return result.frame;
      } else {
        console.error('[WEBRTC-VIDEO-PROCESSOR] ❌ Blur service error:', response.status);
        console.log('[WEBRTC-VIDEO-PROCESSOR] 🔒 PRIVACY PROTECTION: Blur service failed - dropping frame');
        // PRIVACY FIX: Return null instead of original frame to prevent privacy leaks
        return null;
      }
      
    } catch (error) {
      console.error('[WEBRTC-VIDEO-PROCESSOR] ❌ Blur error:', error);
      console.log('[WEBRTC-VIDEO-PROCESSOR] 🔒 PRIVACY PROTECTION: Blur failed - dropping frame');
      // PRIVACY FIX: Return null instead of original frame to prevent privacy leaks
      return null;
    }
  }
  
  /**
   * Interpolate bounding boxes for intermediate frames
   */
  interpolateBoundingBoxes(originalBoxes, framesSinceDetection) {
    if (!originalBoxes || originalBoxes.length === 0) return [];
    
    // Simple interpolation: slightly expand boxes over time
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
  
  /**
   * Get processed producer for a room
   */
  getProcessedProducer(roomId) {
    return this.processedProducers.get(roomId);
  }
  
  /**
   * Cleanup resources for a room
   */
  cleanup(roomId) {
    try {
      console.log('[WEBRTC-VIDEO-PROCESSOR] 🧹 Cleaning up room:', roomId);
      
      // Close processed producer
      if (this.processedProducers.has(roomId)) {
        this.processedProducers.delete(roomId);
      }
      
      this.frameProcessingState.delete(roomId);
      
    } catch (error) {
      console.error('[WEBRTC-VIDEO-PROCESSOR] ❌ Error during cleanup:', error);
    }
  }
}

module.exports = { WebRTCVideoProcessor };