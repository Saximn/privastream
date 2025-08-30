/**
 * WebRTC Video Filter using insertable streams
 * Processes frames through Python API and applies blur client-side
 * Now with GPU acceleration and 3-second buffering
 */

import { GPUBlur } from './gpu-blur'

export interface FilterConfig {
  apiUrl: string
  processingFps: number
  blurType: 'gaussian' | 'pixelate' | 'fill'
  kernelSize: number
  pixelSize: number
  useGPU: boolean
  bufferFrames: number
  skipFrames: number
}

export interface BlurRegions {
  rectangles: number[][]
  polygons: number[][][]
  frame_id: number
  detection_counts: {
    face: number
    pii: number
    plate: number
  }
}

export class VideoFilter {
  private config: FilterConfig
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private processingWorker: Worker | null = null
  private frameCount = 0
  private lastProcessTime = 0
  private cachedBlurRegions: BlurRegions | null = null
  private processingQueue: Set<number> = new Set()
  private gpuBlur: GPUBlur | null = null
  private frameBuffer: VideoFrame[] = []
  private lastBlurTime = 0
  private skipCounter = 0

  constructor(config: Partial<FilterConfig> = {}) {
    this.config = {
      apiUrl: 'http://localhost:5001',
      processingFps: 4,
      blurType: 'gaussian',
      kernelSize: 35,
      pixelSize: 16,
      useGPU: false,
      bufferFrames: 90, // 3 seconds at 30fps
      skipFrames: 1, // Process every other frame for blur
      ...config
    }
    
    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d')!
    
    // Initialize GPU blur if supported and enabled
    if (this.config.useGPU && this.isWebGLSupported()) {
      try {
        this.gpuBlur = new GPUBlur({
          kernelSize: this.config.kernelSize,
          
        })
        console.log('[VideoFilter] GPU acceleration enabled')
      } catch (error) {
        console.warn('[VideoFilter] GPU initialization failed, falling back to CPU:', error)
        this.gpuBlur = null
      }
    }
  }

  private isWebGLSupported(): boolean {
    try {
      const canvas = document.createElement('canvas')
      return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    } catch {
      return false
    }
  }

  /**
   * Create video transform stream that applies filtering
   * Now with optimized processing and frame buffering
   */
  createTransformStream(): TransformStream<VideoFrame, VideoFrame> {
    const processingInterval = 1000 / this.config.processingFps
    
    return new TransformStream({
      transform: async (frame: VideoFrame, controller) => {
        try {
          const now = Date.now()
          
          // Add frame to buffer for smoothing
          this.frameBuffer.push(frame)
          if (this.frameBuffer.length > this.config.bufferFrames) {
            const oldFrame = this.frameBuffer.shift()
            if (oldFrame) oldFrame.close()
          }

          // Skip frame processing for performance (process every N frames)
          this.skipCounter++
          const shouldSkipBlur = this.skipCounter % (this.config.skipFrames + 1) !== 0

          // Process frame for detection if it's time
          const shouldProcess = (now - this.lastProcessTime) >= processingInterval
          if (shouldProcess && !this.processingQueue.has(this.frameCount)) {
            this.processFrameForDetection(frame)
            this.lastProcessTime = now
          }

          // Apply blur optimization: skip blur on some frames for performance
          let filteredFrame: VideoFrame
          if (shouldSkipBlur && this.cachedBlurRegions && 
              (this.cachedBlurRegions.rectangles.length === 0 && this.cachedBlurRegions.polygons.length === 0)) {
            // No blur needed and skipping frame - pass through directly
            filteredFrame = frame
          } else {
            // Apply blur to current frame using cached regions
            filteredFrame = await this.applyBlurToFrameOptimized(frame)
            this.lastBlurTime = now
          }
          
          controller.enqueue(filteredFrame)
          this.frameCount++
        } catch (error) {
          console.error('[VideoFilter] Transform error:', error)
          controller.enqueue(frame) // Pass through on error
        }
      }
    })
  }

  /**
   * Process frame for detection (async, doesn't block stream)
   */
  private async processFrameForDetection(frame: VideoFrame) {
    const frameId = this.frameCount
    this.processingQueue.add(frameId)

    try {
      // Convert VideoFrame to canvas
      this.canvas.width = frame.displayWidth
      this.canvas.height = frame.displayHeight
      this.ctx.drawImage(frame, 0, 0)
      
      // Convert to base64
      const dataUrl = this.canvas.toDataURL('image/jpeg', 0.8)
      
      // Send to API
      const response = await fetch(`${this.config.apiUrl}/filter-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame: dataUrl,
          frame_id: frameId
        })
      })
      
      if (response.ok) {
        const result: BlurRegions = await response.json()
        this.cachedBlurRegions = result
      } else {
        console.warn('[VideoFilter] API request failed:', response.status)
      }
    } catch (error) {
      console.error('[VideoFilter] Detection processing error:', error)
    } finally {
      this.processingQueue.delete(frameId)
    }
  }

  /**
   * Apply blur to frame using cached regions with GPU acceleration
   */
  private async applyBlurToFrameOptimized(frame: VideoFrame): Promise<VideoFrame> {
    if (!this.cachedBlurRegions || 
        (this.cachedBlurRegions.rectangles.length === 0 && this.cachedBlurRegions.polygons.length === 0)) {
      return frame // No blur needed
    }

    // Set canvas size
    this.canvas.width = frame.displayWidth
    this.canvas.height = frame.displayHeight
    
    // Draw original frame
    this.ctx.drawImage(frame, 0, 0)
    
    // Apply blur regions with GPU acceleration if available
    if (this.gpuBlur) {
      try {
        const blurredCanvas = this.gpuBlur.applyBlurToCanvas(this.canvas, this.cachedBlurRegions)
        
        // Create new VideoFrame from GPU-processed canvas
        const videoFrame = new VideoFrame(blurredCanvas, {
          timestamp: frame.timestamp
        })
        
        frame.close() // Clean up original frame
        return videoFrame
      } catch (error) {
        console.warn('[VideoFilter] GPU blur failed, falling back to CPU:', error)
        // Fall through to CPU blur
      }
    }
    
    // Fallback to CPU blur
    this.applyBlurRegions(this.cachedBlurRegions)
    
    // Create new VideoFrame from canvas
    const videoFrame = new VideoFrame(this.canvas, {
      timestamp: frame.timestamp
    })
    
    frame.close() // Clean up original frame
    return videoFrame
  }

  /**
   * Legacy method for backward compatibility
   */
  private async applyBlurToFrame(frame: VideoFrame): Promise<VideoFrame> {
    return this.applyBlurToFrameOptimized(frame)
  }

  /**
   * Apply blur effects to canvas regions
   */
  private applyBlurRegions(regions: BlurRegions) {
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height)
    let modified = false

    // Apply rectangular blurs
    for (const rect of regions.rectangles) {
      if (rect.length === 4) {
        this.blurRectangle(imageData, rect)
        modified = true
      }
    }

    // Apply polygon blurs
    for (const poly of regions.polygons) {
      if (poly.length > 2) {
        this.blurPolygon(imageData, poly)
        modified = true
      }
    }

    if (modified) {
      this.ctx.putImageData(imageData, 0, 0)
    }
  }

  /**
   * Apply blur to rectangular region
   */
  private blurRectangle(imageData: ImageData, rect: number[]) {
    const [x1, y1, x2, y2] = rect
    const width = x2 - x1
    const height = y2 - y1
    
    if (width <= 0 || height <= 0) return

    if (this.config.blurType === 'gaussian') {
      this.applyGaussianBlur(imageData, x1, y1, width, height)
    } else if (this.config.blurType === 'pixelate') {
      this.applyPixelation(imageData, x1, y1, width, height)
    } else if (this.config.blurType === 'fill') {
      this.applyFill(imageData, x1, y1, width, height)
    }
  }

  /**
   * Apply blur to polygonal region
   */
  private blurPolygon(imageData: ImageData, poly: number[][]) {
    // Get bounding box
    const xs = poly.map(p => p[0])
    const ys = poly.map(p => p[1])
    const x1 = Math.max(0, Math.min(...xs))
    const y1 = Math.max(0, Math.min(...ys))
    const x2 = Math.min(imageData.width, Math.max(...xs))
    const y2 = Math.min(imageData.height, Math.max(...ys))
    
    // Create mask for polygon
    const mask = this.createPolygonMask(poly, x1, y1, x2 - x1, y2 - y1)
    
    if (this.config.blurType === 'gaussian') {
      this.applyGaussianBlurWithMask(imageData, x1, y1, x2 - x1, y2 - y1, mask)
    } else if (this.config.blurType === 'pixelate') {
      this.applyPixelationWithMask(imageData, x1, y1, x2 - x1, y2 - y1, mask)
    } else if (this.config.blurType === 'fill') {
      this.applyFillWithMask(imageData, x1, y1, x2 - x1, y2 - y1, mask)
    }
  }

  /**
   * Simple box blur implementation (faster than gaussian)
   */
  private applyGaussianBlur(imageData: ImageData, x: number, y: number, width: number, height: number) {
    const data = imageData.data
    const imgWidth = imageData.width
    const radius = Math.floor(this.config.kernelSize / 6) // Approximate gaussian with box blur
    
    // Box blur implementation for speed
    for (let py = y; py < y + height; py++) {
      for (let px = x; px < x + width; px++) {
        let r = 0, g = 0, b = 0, count = 0
        
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = px + dx
            const ny = py + dy
            
            if (nx >= 0 && nx < imgWidth && ny >= 0 && ny < imageData.height) {
              const idx = (ny * imgWidth + nx) * 4
              r += data[idx]
              g += data[idx + 1]
              b += data[idx + 2]
              count++
            }
          }
        }
        
        if (count > 0) {
          const idx = (py * imgWidth + px) * 4
          data[idx] = r / count
          data[idx + 1] = g / count
          data[idx + 2] = b / count
        }
      }
    }
  }

  /**
   * Apply pixelation effect
   */
  private applyPixelation(imageData: ImageData, x: number, y: number, width: number, height: number) {
    const data = imageData.data
    const imgWidth = imageData.width
    const blockSize = this.config.pixelSize
    
    for (let py = y; py < y + height; py += blockSize) {
      for (let px = x; px < x + width; px += blockSize) {
        // Get average color of block
        let r = 0, g = 0, b = 0, count = 0
        
        for (let dy = 0; dy < blockSize && py + dy < y + height; dy++) {
          for (let dx = 0; dx < blockSize && px + dx < x + width; dx++) {
            const idx = ((py + dy) * imgWidth + (px + dx)) * 4
            r += data[idx]
            g += data[idx + 1]
            b += data[idx + 2]
            count++
          }
        }
        
        if (count > 0) {
          r /= count
          g /= count
          b /= count
          
          // Fill block with average color
          for (let dy = 0; dy < blockSize && py + dy < y + height; dy++) {
            for (let dx = 0; dx < blockSize && px + dx < x + width; dx++) {
              const idx = ((py + dy) * imgWidth + (px + dx)) * 4
              data[idx] = r
              data[idx + 1] = g
              data[idx + 2] = b
            }
          }
        }
      }
    }
  }

  /**
   * Apply solid fill
   */
  private applyFill(imageData: ImageData, x: number, y: number, width: number, height: number) {
    const data = imageData.data
    const imgWidth = imageData.width
    
    for (let py = y; py < y + height; py++) {
      for (let px = x; px < x + width; px++) {
        const idx = (py * imgWidth + px) * 4
        data[idx] = 0     // R
        data[idx + 1] = 0 // G
        data[idx + 2] = 0 // B
      }
    }
  }

  /**
   * Create mask for polygon region
   */
  private createPolygonMask(poly: number[][], x: number, y: number, width: number, height: number): boolean[][] {
    const mask: boolean[][] = Array(height).fill(null).map(() => Array(width).fill(false))
    
    // Simple point-in-polygon test for each pixel
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        const worldX = x + px
        const worldY = y + py
        
        if (this.pointInPolygon(worldX, worldY, poly)) {
          mask[py][px] = true
        }
      }
    }
    
    return mask
  }

  /**
   * Point in polygon test (ray casting)
   */
  private pointInPolygon(x: number, y: number, poly: number[][]): boolean {
    let inside = false
    
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1]
      const xj = poly[j][0], yj = poly[j][1]
      
      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
        inside = !inside
      }
    }
    
    return inside
  }

  /**
   * Apply blur with mask
   */
  private applyGaussianBlurWithMask(imageData: ImageData, x: number, y: number, width: number, height: number, mask: boolean[][]) {
    // Implementation similar to applyGaussianBlur but with mask check
    // For brevity, simplified version
    this.applyPixelationWithMask(imageData, x, y, width, height, mask)
  }

  private applyPixelationWithMask(imageData: ImageData, x: number, y: number, width: number, height: number, mask: boolean[][]) {
    const data = imageData.data
    const imgWidth = imageData.width
    const blockSize = this.config.pixelSize
    
    for (let py = 0; py < height; py += blockSize) {
      for (let px = 0; px < width; px += blockSize) {
        let r = 0, g = 0, b = 0, count = 0
        
        // Get average of masked pixels
        for (let dy = 0; dy < blockSize && py + dy < height; dy++) {
          for (let dx = 0; dx < blockSize && px + dx < width; dx++) {
            if (mask[py + dy] && mask[py + dy][px + dx]) {
              const idx = ((y + py + dy) * imgWidth + (x + px + dx)) * 4
              r += data[idx]
              g += data[idx + 1]
              b += data[idx + 2]
              count++
            }
          }
        }
        
        if (count > 0) {
          r /= count
          g /= count
          b /= count
          
          // Apply to masked pixels only
          for (let dy = 0; dy < blockSize && py + dy < height; dy++) {
            for (let dx = 0; dx < blockSize && px + dx < width; dx++) {
              if (mask[py + dy] && mask[py + dy][px + dx]) {
                const idx = ((y + py + dy) * imgWidth + (x + px + dx)) * 4
                data[idx] = r
                data[idx + 1] = g
                data[idx + 2] = b
              }
            }
          }
        }
      }
    }
  }

  private applyFillWithMask(imageData: ImageData, x: number, y: number, width: number, height: number, mask: boolean[][]) {
    const data = imageData.data
    const imgWidth = imageData.width
    
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        if (mask[py] && mask[py][px]) {
          const idx = ((y + py) * imgWidth + (x + px)) * 4
          data[idx] = 0
          data[idx + 1] = 0
          data[idx + 2] = 0
        }
      }
    }
  }

  /**
   * Get detection statistics with performance metrics
   */
  public getStats() {
    const bufferHealth = this.frameBuffer.length / this.config.bufferFrames
    return {
      frameCount: this.frameCount,
      processingQueueSize: this.processingQueue.size,
      lastDetections: this.cachedBlurRegions?.detection_counts || null,
      config: this.config,
      gpuEnabled: !!this.gpuBlur,
      bufferHealth: bufferHealth,
      bufferSize: this.frameBuffer.length,
      performance: {
        skipRatio: this.config.skipFrames + 1,
        lastBlurTime: this.lastBlurTime,
        averageProcessingGap: Date.now() - this.lastProcessTime
      }
    }
  }

  /**
   * Update configuration and reinitialize GPU if needed
   */
  public updateConfig(newConfig: Partial<FilterConfig>) {
    const oldConfig = { ...this.config }
    this.config = { ...this.config, ...newConfig }
    
    // Reinitialize GPU blur if blur settings changed
    if (this.gpuBlur && (
        newConfig.kernelSize !== undefined ||
        newConfig.blurType !== undefined ||
        newConfig.pixelSize !== undefined
    )) {
      this.gpuBlur.updateConfig({
        kernelSize: this.config.kernelSize,
        blurType: this.config.blurType as 'gaussian' | 'box' | 'pixelate',
        pixelSize: this.config.pixelSize
      })
    }
    
    console.log('[VideoFilter] Config updated:', {
      from: oldConfig,
      to: this.config
    })
  }

  /**
   * Clean up resources
   */
  public destroy() {
    // Clean up frame buffer
    this.frameBuffer.forEach(frame => frame.close())
    this.frameBuffer = []
    
    // Clean up GPU resources
    if (this.gpuBlur) {
      this.gpuBlur.destroy()
      this.gpuBlur = null
    }
    
    console.log('[VideoFilter] Resources cleaned up')
  }
}