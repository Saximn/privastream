/**
 * Buffered Video Player
 * Provides smooth playback with configurable buffer size
 */

export interface BufferedPlayerConfig {
  bufferSeconds: number
  targetLatency: number
  maxBufferSize: number
  enableSmoothing: boolean
}

export class BufferedVideoPlayer {
  private video: HTMLVideoElement
  private config: BufferedPlayerConfig
  private frameBuffer: VideoFrame[] = []
  private isPlaying = false
  private lastFrameTime = 0
  private playbackStartTime = 0
  private frameRate = 30
  private bufferedSourceBuffer?: SourceBuffer
  private mediaSource?: MediaSource
  
  constructor(videoElement: HTMLVideoElement, config: Partial<BufferedPlayerConfig> = {}) {
    this.video = videoElement
    this.config = {
      bufferSeconds: 3,
      targetLatency: 100, // ms
      maxBufferSize: 150, // max frames to buffer
      enableSmoothing: true,
      ...config
    }
    
    this.setupBufferedPlayback()
  }

  private setupBufferedPlayback() {
    // Enable smooth playback with buffer management
    this.video.addEventListener('loadedmetadata', () => {
      this.frameRate = 30 // Default, will be updated from stream
      console.log('[BufferedPlayer] Video loaded, setting up buffer')
    })

    this.video.addEventListener('timeupdate', () => {
      this.manageBufferHealth()
    })

    // Handle playback stalls
    this.video.addEventListener('waiting', () => {
      console.log('[BufferedPlayer] Playback stalled, checking buffer')
      this.handlePlaybackStall()
    })

    this.video.addEventListener('playing', () => {
      console.log('[BufferedPlayer] Playback resumed')
      this.isPlaying = true
    })
  }

  public setStream(stream: MediaStream) {
    if (!stream) {
      console.warn('[BufferedPlayer] No stream provided')
      return
    }
    
    // Add buffer management for incoming stream
    this.video.srcObject = stream
    
    // Configure video element for optimal buffering
    this.video.muted = true // Start muted for autoplay
    this.video.playsInline = true
    
    // Set buffer size hints
    if ('setVideoBufferSize' in this.video) {
      (this.video as any).setVideoBufferSize(this.config.bufferSeconds * 1000)
      console.log('[BufferedPlayer] Native buffer size set to', this.config.bufferSeconds * 1000)
    }
    
    // Start playback with buffer
    this.startBufferedPlayback()
  }

  private async startBufferedPlayback() {
    try {
      // Wait for sufficient buffer before starting
      await this.waitForBuffer()
      
      console.log('[BufferedPlayer] Starting playback with buffer')
      await this.video.play()
      this.isPlaying = true
      this.playbackStartTime = performance.now()
    } catch (error) {
      console.error('[BufferedPlayer] Failed to start buffered playback:', error)
    }
  }

  private async waitForBuffer(): Promise<void> {
    return new Promise((resolve) => {
      const checkBuffer = () => {
        if (this.video.readyState >= 3) { // HAVE_FUTURE_DATA
          const buffered = this.video.buffered
          if (buffered.length > 0) {
            const bufferEnd = buffered.end(buffered.length - 1)
            const bufferStart = buffered.start(0)
            const bufferDuration = bufferEnd - bufferStart
            
            console.log('[BufferedPlayer] Buffer status:', {
              duration: bufferDuration,
              target: this.config.bufferSeconds,
              readyState: this.video.readyState
            })
            
            if (bufferDuration >= Math.min(this.config.bufferSeconds, 1)) {
              resolve()
              return
            }
          }
        }
        
        setTimeout(checkBuffer, 100)
      }
      
      checkBuffer()
    })
  }

  private manageBufferHealth() {
    if (!this.isPlaying) return

    const buffered = this.video.buffered
    if (buffered.length === 0) return

    const currentTime = this.video.currentTime
    const bufferEnd = buffered.end(buffered.length - 1)
    const bufferAhead = bufferEnd - currentTime
    
    // Log buffer health
    if (performance.now() % 1000 < 16) { // Log roughly every second
      console.log('[BufferedPlayer] Buffer health:', {
        bufferAhead: bufferAhead.toFixed(2) + 's',
        target: this.config.bufferSeconds + 's',
        playbackRate: this.video.playbackRate
      })
    }

    // Adjust playback rate for smooth experience
    if (this.config.enableSmoothing) {
      this.adjustPlaybackRate(bufferAhead)
    }
  }

  private adjustPlaybackRate(bufferAhead: number) {
    const targetBuffer = this.config.bufferSeconds
    const tolerance = 0.5 // seconds
    
    let newRate = 1.0

    if (bufferAhead > targetBuffer + tolerance) {
      // Too much buffer, speed up slightly
      newRate = 1.02
    } else if (bufferAhead < targetBuffer - tolerance) {
      // Too little buffer, slow down slightly  
      newRate = 0.98
    }

    // Only adjust if change is significant
    if (Math.abs(this.video.playbackRate - newRate) > 0.01) {
      this.video.playbackRate = newRate
      console.log('[BufferedPlayer] Adjusted playback rate to:', newRate)
    }
  }

  private handlePlaybackStall() {
    if (!this.isPlaying) return

    console.log('[BufferedPlayer] Handling playback stall')
    
    // Pause briefly to allow buffer to build up
    this.video.pause()
    this.isPlaying = false
    
    setTimeout(async () => {
      try {
        if (this.video.readyState >= 3) {
          await this.video.play()
          this.isPlaying = true
          console.log('[BufferedPlayer] Resumed after stall')
        }
      } catch (error) {
        console.error('[BufferedPlayer] Failed to resume after stall:', error)
      }
    }, 500) // Wait 500ms for buffer
  }

  public getBufferStatus() {
    const buffered = this.video.buffered
    if (buffered.length === 0) {
      return { 
        duration: 0, 
        health: 'empty',
        playbackRate: 1.0,
        readyState: this.video.readyState || 0
      }
    }

    const currentTime = this.video.currentTime || 0
    const bufferEnd = buffered.end(buffered.length - 1)
    const bufferAhead = bufferEnd - currentTime
    
    let health = 'good'
    if (bufferAhead < 1) health = 'low'
    if (bufferAhead < 0.5) health = 'critical'
    if (bufferAhead > this.config.bufferSeconds + 1) health = 'high'

    return {
      duration: bufferAhead,
      health,
      playbackRate: this.video.playbackRate || 1.0,
      readyState: this.video.readyState || 0
    }
  }

  public updateConfig(newConfig: Partial<BufferedPlayerConfig>) {
    this.config = { ...this.config, ...newConfig }
    console.log('[BufferedPlayer] Config updated:', this.config)
  }

  public destroy() {
    this.isPlaying = false
    this.frameBuffer = []
  }
}