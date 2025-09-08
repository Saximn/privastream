'use client'

import { useState, useEffect, useRef } from 'react'
import { SocketManager } from '@/lib/socket'
import { MediasoupClient } from '@/lib/mediasoup-client'
import { io, Socket } from 'socket.io-client'

export default function Host() {
  const [roomId, setRoomId] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [viewerCount, setViewerCount] = useState(0)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('')

  // Video Filtering State
  const [isVideoFilterEnabled, setIsVideoFilterEnabled] = useState(false)
  const [videoFilterStats, setVideoFilterStats] = useState<any>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const socketRef = useRef<SocketManager | null>(null)
  const sfuSocketRef = useRef<Socket | null>(null)
  const mediasoupClientRef = useRef<MediasoupClient | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null)

  useEffect(() => {
    const initializeConnections = async () => {
      try {
        console.log('[DEBUG] Starting connection initialization...')
        setError('')
        setConnectionState('connecting to backend...')

        // Signaling socket
        socketRef.current = new SocketManager()
        await socketRef.current.connect()
        console.log('[DEBUG] Connected to backend signaling socket')

        setConnectionState('creating room...')
        
        // Create mediasoup room
        const roomResponse = await socketRef.current.createRoom()
        console.log('[DEBUG] Room created:', roomResponse)
        setRoomId(roomResponse.roomId)
        
        // Check if we have enrollment data and associate it with the new room
        const enrolledRoomId = sessionStorage.getItem('enrolledRoomId')
        if (enrolledRoomId) {
          console.log('[DEBUG] 🎯 Transferring enrolled face data from room:', enrolledRoomId, 'to mediasoup room:', roomResponse.roomId)
          
          // Transfer enrollment data from enrolledRoomId to actual roomId
          try {
            const transferResponse = await fetch('http://localhost:5001/transfer-embedding', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                from_room_id: enrolledRoomId,
                to_room_id: roomResponse.roomId
              })
            })
            
            const transferResult = await transferResponse.json()
            if (transferResult.success) {
              console.log('[DEBUG] ✅ Successfully transferred face embedding to streaming room')
              sessionStorage.setItem('mediasoupRoomId', roomResponse.roomId)
            } else {
              console.warn('[DEBUG] ⚠️ Failed to transfer face embedding:', transferResult.error)
            }
          } catch (error) {
            console.error('[DEBUG] ❌ Error transferring face embedding:', error)
          }
        } else {
          console.log('[DEBUG] ⚠️ No face enrollment found - proceeding without face recognition')
        }

        // SFU socket
        setConnectionState('connecting to mediasoup...')
        const sfuUrl = roomResponse.mediasoupUrl || 'http://localhost:3001'
        sfuSocketRef.current = io(sfuUrl, { transports: ['websocket'], reconnectionAttempts: 3 })

        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('SFU connection timeout')), 10000)
          sfuSocketRef.current!.on('connect', () => { clearTimeout(timeout); console.log('[DEBUG] Connected to SFU socket'); resolve() })
          sfuSocketRef.current!.on('connect_error', (err) => { clearTimeout(timeout); console.error('[DEBUG] SFU connection error:', err); reject(err) })
          sfuSocketRef.current!.on('disconnect', (reason) => console.warn('[DEBUG] SFU disconnected:', reason))
        })

        try {
          setConnectionState('initializing mediasoup client...')
          console.log('[DEBUG] Initializing Mediasoup client...')
          mediasoupClientRef.current = new MediasoupClient(roomResponse.roomId)
          await mediasoupClientRef.current.initialize(sfuSocketRef.current!)
          console.log('[DEBUG] Mediasoup client initialized')
        } catch(err) {
          console.error('[DEBUG] Mediasoup client initialization failed:', err)
        }

try {
  setConnectionState('creating SFU room...')
  console.log('[DEBUG] Creating SFU room via socket emit...')
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('SFU room creation timeout')), 5000)
    sfuSocketRef.current!.emit('create-room', { roomId: roomResponse.roomId }, (sfuResponse: any) => {
      clearTimeout(timeout)
      console.log('[DEBUG] SFU create-room callback fired', sfuResponse)
      if (sfuResponse.success) resolve()
      else reject(new Error(sfuResponse.error))
    })
  })
} catch(err) {
  console.error('[DEBUG] SFU room creation failed:', err)
}

        console.log('[DEBUG] All connections initialized, ready to stream')
        setConnectionState('ready')
      } catch (err) {
        console.error('[DEBUG] Initialization error:', err)
        setError(`Connection failed: ${err instanceof Error ? err.message : String(err)}`)
        setConnectionState('error')
      }
    }

    initializeConnections()

    return () => {
      // Cleanup
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current)
      }
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect()
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
      localStreamRef.current?.getTracks().forEach(track => track.stop())
      mediasoupClientRef.current?.stopProducing()
      socketRef.current?.disconnect()
      sfuSocketRef.current?.disconnect()
    }
  }, [])

  const startStreaming = async () => {
    try {
      setError('')
      setConnectionState('initializing')
      if (!mediasoupClientRef.current || !socketRef.current) throw new Error('Connections not initialized')

      console.log('[HOST] Requesting microphone and camera access...')
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720, frameRate: 30 }, 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      })
      
      console.log('[HOST] 🎤 Media access granted! Stream details:', {
        id: stream.id,
        active: stream.active,
        videoTracks: stream.getVideoTracks().length,
        audioTracks: stream.getAudioTracks().length
      })
      
      // Verify audio track
      const audioTrack = stream.getAudioTracks()[0]
      if (audioTrack) {
        console.log('[HOST] ✅ Audio track acquired:', {
          id: audioTrack.id,
          enabled: audioTrack.enabled,
          muted: audioTrack.muted,
          readyState: audioTrack.readyState
        })
      } else {
        console.error('[HOST] ❌ No audio track in stream!')
      }
      
      localStreamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream

      console.log('[DEBUG] Creating producer transport...')
      await mediasoupClientRef.current.createProducerTransport()
      console.log('[DEBUG] Producing stream with video filter:', isVideoFilterEnabled)
      await mediasoupClientRef.current.produce(stream, isVideoFilterEnabled)
      
      // Start frame processing for video filter
      if (isVideoFilterEnabled) {
        startFrameProcessing(stream)
      }
      
      // Start audio processing for redaction
      startAudioProcessing(stream)

      socketRef.current.getSocket().emit('sfu_streaming_started', { roomId })
      setIsStreaming(true)
      setConnectionState('streaming')
      console.log('[DEBUG] Streaming started successfully')
    } catch (err) {
      console.error('[DEBUG] Streaming error:', err)
      setError('Failed to start streaming. Check camera/microphone permissions.')
      setConnectionState('error')
    }
  }

  const stopStreaming = async () => {
    try {
      // Stop frame processing
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current)
        frameIntervalRef.current = null
      }
      
      // Stop audio processing
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect()
        audioProcessorRef.current = null
      }
      
      if (audioContextRef.current) {
        await audioContextRef.current.close()
        audioContextRef.current = null
      }
      
      await mediasoupClientRef.current?.stopProducing()
      localStreamRef.current?.getTracks().forEach(track => track.stop())
      localStreamRef.current = null
      if (videoRef.current) videoRef.current.srcObject = null

      socketRef.current?.getSocket().emit('sfu_streaming_stopped', { roomId })
      setIsStreaming(false)
      setConnectionState('ready')
      setViewerCount(0)
      console.log('[DEBUG] Streaming stopped')
    } catch (err) {
      console.error('[DEBUG] Stop streaming error:', err)
    }
  }
  
  const startFrameProcessing = (stream: MediaStream) => {
    console.log('[DEBUG] Starting frame processing for video filter')
    
    // Create canvas for frame extraction
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas')
    }
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')!
    
    // Create a video element for frame extraction
    const tempVideo = document.createElement('video')
    tempVideo.srcObject = stream
    tempVideo.muted = true
    tempVideo.playsInline = true
    tempVideo.play()
    
    // Process frames at 4 FPS (every 250ms)
    frameIntervalRef.current = setInterval(() => {
      try {
        if (tempVideo.videoWidth > 0 && tempVideo.videoHeight > 0) {
          // Set canvas size to match video
          canvas.width = tempVideo.videoWidth
          canvas.height = tempVideo.videoHeight
          
          // Draw current video frame to canvas
          ctx.drawImage(tempVideo, 0, 0)
          
          // Convert to base64
          const frameData = canvas.toDataURL('image/jpeg', 0.7)
          const frameId = Date.now()
          
          // Send frame to MediaSoup server for processing
          if (sfuSocketRef.current) {
            sfuSocketRef.current.emit('video-frame', {
              frame: frameData,
              frameId: frameId,
              roomId: roomId
            })
            
            console.log('[DEBUG] Sent video frame for processing:', frameId)
          }
        }
      } catch (error) {
        console.error('[DEBUG] Frame processing error:', error)
      }
    }, 250) // 4 FPS
    
    console.log('[DEBUG] Frame processing started')
  }

  const startAudioProcessing = (stream: MediaStream) => {
    console.log('[DEBUG] Starting audio processing for redaction')
    
    const audioTrack = stream.getAudioTracks()[0]
    if (!audioTrack) {
      console.error('[DEBUG] No audio track available for processing')
      return
    }
    
    try {
      // Create audio context
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      const audioContext = audioContextRef.current
      
      console.log('[DEBUG] Audio context created:', {
        sampleRate: audioContext.sampleRate,
        state: audioContext.state
      })
      
      // Create media stream source
      const source = audioContext.createMediaStreamSource(stream)
      
      // Create script processor node (deprecated but still works)
      const bufferSize = 4096 // Buffer size for processing
      const processor = audioContext.createScriptProcessor(bufferSize, 2, 2) // Stereo
      audioProcessorRef.current = processor
      
      // Process audio data
      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer
        const outputBuffer = event.outputBuffer
        
        // Get left and right channel data
        const leftChannel = inputBuffer.getChannelData(0)
        const rightChannel = inputBuffer.getChannelData(1)
        
        // Mix stereo to mono by averaging channels
        const monoData = new Float32Array(bufferSize)
        for (let i = 0; i < bufferSize; i++) {
          monoData[i] = (leftChannel[i] + rightChannel[i]) / 2
        }
        
        // Downsample from 48kHz to 16kHz (3:1 ratio)
        const downsampleRatio = 3
        const outputSamples = Math.floor(bufferSize / downsampleRatio)
        const downsampledData = new Float32Array(outputSamples)
        
        for (let i = 0; i < outputSamples; i++) {
          // Simple downsampling - take every 3rd sample
          downsampledData[i] = monoData[i * downsampleRatio]
        }
        
        // Convert float32 to int16 PCM data (16kHz mono)
        const pcmData = new Int16Array(outputSamples)
        for (let i = 0; i < outputSamples; i++) {
          const sample = Math.max(-1, Math.min(1, downsampledData[i]))
          pcmData[i] = sample * 0x7FFF
        }
        
        // Send PCM data to server for processing
        if (sfuSocketRef.current) {
          sfuSocketRef.current.emit('audio-data', Array.from(pcmData))
        }
        
        // Copy input to output but muted to avoid feedback
        for (let channel = 0; channel < outputBuffer.numberOfChannels; channel++) {
          const outputData = outputBuffer.getChannelData(channel)
          outputData.fill(0) // Fill with silence to prevent feedback
        }
      }
      
      // Connect audio processing chain
      source.connect(processor)
      processor.connect(audioContext.destination) // Connect to ensure processing happens
      
      console.log('[DEBUG] Audio processing pipeline connected')
      
    } catch (error) {
      console.error('[DEBUG] Audio processing setup error:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">SFU Host Stream</h1>
          {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">{error}</div>}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Room ID</h3>
              <code className="bg-gray-200 px-2 py-1 rounded text-sm font-mono">{roomId || 'Generating...'}</code>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Viewers</h3>
              <div className="text-2xl font-bold text-green-600">{viewerCount}</div>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Status</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${isStreaming ? 'bg-green-500' : 'bg-red-500'}`} />
                {isStreaming ? 'Live (SFU)' : 'Offline'}
              </div>
            </div>
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Video Filter</h3>
              <button
                onClick={() => setIsVideoFilterEnabled(!isVideoFilterEnabled)}
                disabled={isStreaming}
                className={`text-white text-sm px-3 py-1 rounded disabled:opacity-50 ${isVideoFilterEnabled ? 'bg-purple-600' : 'bg-gray-600'}`}
              >
                {isVideoFilterEnabled ? 'Enabled' : 'Disabled'}
              </button>
            </div>
          </div>

          <div className="flex gap-4 justify-center mb-6">
            {!isStreaming ? (
              <button onClick={startStreaming} disabled={connectionState !== 'ready'} className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50">
                {connectionState === 'ready' ? 'Start SFU Streaming' : 'Initializing...'}
              </button>
            ) : (
              <button onClick={stopStreaming} className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg transition-colors">
                Stop Streaming
              </button>
            )}
          </div>

          <div className="bg-black rounded-lg overflow-hidden">
            <video ref={videoRef} autoPlay muted playsInline className="w-full h-auto max-h-96 object-contain" style={{ backgroundColor: '#000' }} />
          </div>
        </div>
      </div>
    </div>
  )
}
