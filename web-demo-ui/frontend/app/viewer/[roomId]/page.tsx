'use client'

import { useState, useEffect, useRef } from 'react'
import { useParams } from 'next/navigation'
import { SocketManager } from '@/lib/socket'
import { MediasoupClient } from '@/lib/mediasoup-client'
import { BufferedVideoPlayer } from '@/lib/buffered-video-player'
import { io, Socket } from 'socket.io-client'

export default function Viewer() {
  const params = useParams()
  const roomId = params.roomId as string

  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState('')
  const [connectionState, setConnectionState] = useState('connecting')
  const [streamAvailable, setStreamAvailable] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [piiDetectionActive, setPiiDetectionActive] = useState(false)
  const [recentPIIAlert, setRecentPIIAlert] = useState<string | null>(null)
  const [bufferStatus, setBufferStatus] = useState<any>(null)
  const [performanceStats, setPerformanceStats] = useState<any>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const socketRef = useRef<SocketManager | null>(null)
  const sfuSocketRef = useRef<Socket | null>(null)
  const mediasoupClientRef = useRef<MediasoupClient | null>(null)
  const bufferedPlayerRef = useRef<BufferedVideoPlayer | null>(null)
  const remoteStreamsRef = useRef<Map<string, MediaStream>>(new Map())
  const consumersRef = useRef<Set<string>>(new Set())
  const pendingProducersRef = useRef<Array<{producerId: string, kind: string}>>([])

  const handleUnmute = async () => {
    if (videoRef.current) {
      try {
        console.log('🔊 [VIEWER] Attempting to unmute...')
        console.log('🔊 [VIEWER] Current srcObject:', videoRef.current.srcObject)
        
        if (videoRef.current.srcObject) {
          const stream = videoRef.current.srcObject as MediaStream
          console.log('🔊 [VIEWER] Audio tracks in stream:', stream.getAudioTracks().length)
          console.log('🔊 [VIEWER] Video tracks in stream:', stream.getVideoTracks().length)
          
          stream.getAudioTracks().forEach((track, index) => {
            console.log(`🔊 [VIEWER] Audio track ${index}:`, {
              id: track.id,
              kind: track.kind,
              enabled: track.enabled,
              muted: track.muted,
              readyState: track.readyState
            })
          })
        }
        
        videoRef.current.muted = false
        setIsMuted(false)
        console.log('🔊 [VIEWER] Video element unmuted successfully')
        console.log('🔊 [VIEWER] Video element muted property:', videoRef.current.muted)
        console.log('🔊 [VIEWER] Video element volume:', videoRef.current.volume)
      } catch (error) {
        console.error('Failed to unmute video:', error)
      }
    }
  };

  useEffect(() => {
    const initializeViewer = async () => {
      try {
        setError('')
        setConnectionState('connecting')
        console.log('🔵 [VIEWER] === STEP 1: Initializing viewer for room:', roomId)

        // Initialize main signaling socket (Flask backend)
        console.log('🔵 [VIEWER] === STEP 2: Connecting to Flask backend')
        socketRef.current = new SocketManager()
        await socketRef.current.connect()
        console.log('🔵 [VIEWER] === STEP 2 SUCCESS: Connected to Flask backend')
        
        // Join room
        console.log('🔵 [VIEWER] === STEP 3: Joining Flask backend room:', roomId)
        const response = await socketRef.current.joinRoom(roomId)
        console.log('🔵 [VIEWER] === STEP 3 SUCCESS: Joined Flask backend room successfully. Response:', response)
        
        // Initialize SFU connection (Mediasoup server)
        const sfuUrl = response.mediasoupUrl || 'http://localhost:3001'
        console.log('🔵 [VIEWER] === STEP 4: Connecting to SFU server:', sfuUrl)
        sfuSocketRef.current = io(sfuUrl)
        
        await new Promise((resolve) => {
          sfuSocketRef.current!.on('connect', () => {
            console.log('🔵 [VIEWER] === STEP 4 SUCCESS: Connected to SFU server')
            resolve(undefined)
          })
        })

        // Initialize Mediasoup client FIRST (needed for event handlers)
        console.log('🔵 [VIEWER] === STEP 5: Initializing Mediasoup client')
        mediasoupClientRef.current = new MediasoupClient(roomId)
        await mediasoupClientRef.current.initialize(sfuSocketRef.current!)
        console.log('🔵 [VIEWER] === STEP 5 SUCCESS: Mediasoup client initialized')
        
        // Set up SFU event handlers BEFORE joining to catch producer notifications
        console.log('🔵 [VIEWER] === STEP 6: Setting up SFU event handlers')
        setupSFUEventHandlers()
        setupEventHandlers() // Add PII detection event handlers
        console.log('🔵 [VIEWER] === STEP 6 SUCCESS: SFU event handlers set up')

        // Join room in SFU server (will trigger immediate producer notifications)
        console.log('🔵 [VIEWER] === STEP 7: Joining SFU room:', roomId)
        const sfuJoinResponse = await new Promise((resolve, reject) => {
          sfuSocketRef.current!.emit('join-room', { roomId }, (response: any) => {
            console.log('🔵 [VIEWER] === STEP 7 RESPONSE: SFU join-room response:', JSON.stringify(response, null, 2))
            if (response.success) {
              resolve(response)
            } else {
              console.error('🔵 [VIEWER] === STEP 7 ERROR: SFU join failed:', response.error)
              reject(new Error(response.error))
            }
          })
        })
        console.log('🔵 [VIEWER] === STEP 7 SUCCESS: Joined SFU room - producer notifications should be received')

        // Create consumer transport (now authorized)
        console.log('🔵 [VIEWER] === STEP 8: Creating consumer transport')
        try {
          await mediasoupClientRef.current.createConsumerTransport()
          console.log('🔵 [VIEWER] === STEP 8 SUCCESS: Consumer transport created')
        } catch (transportError) {
          console.error('🔵 [VIEWER] === STEP 8 ERROR: Failed to create consumer transport:', transportError)
          throw transportError
        }

        // Process any pending producers that were queued during join
        console.log('🔵 [VIEWER] === STEP 9: Processing pending producers:', pendingProducersRef.current.length)
        const hadPendingProducers = pendingProducersRef.current.length > 0
        for (const producer of pendingProducersRef.current) {
          console.log(`🔵 [VIEWER] Processing queued producer: ${producer.producerId} (${producer.kind})`)
          await processProducer(producer.producerId, producer.kind)
        }
        pendingProducersRef.current = [] // Clear the queue
        
        // If we processed any producers or have streams, streaming is available
        if (hadPendingProducers || remoteStreamsRef.current.size > 0) {
          console.log('🔵 [VIEWER] === STEP 10: Setting streamAvailable to true (producers processed)')
          setStreamAvailable(true)
        }

        console.log('🔵 [VIEWER] Viewer initialized successfully')
        setIsConnected(true)
        setConnectionState('connected')

        // Set up main signaling event handlers AFTER successful connection
        console.log('🔵 [VIEWER] Setting up main signaling event handlers')
        setupMainEventHandlers()

      } catch (err) {
        setError(`Failed to join room: ${err instanceof Error ? err.message : 'Unknown error'}`)
        setConnectionState('error')
        console.error('Viewer initialization error:', err)
      }
    }

    const setupMainEventHandlers = () => {
      if (!socketRef.current) return

      // Main signaling socket handlers
      socketRef.current.onStreamingStarted(() => {
        console.log('🔴 [VIEWER] Host started streaming - setStreamAvailable(true)')
        setStreamAvailable(true)
      })

      socketRef.current.onStreamingStopped(() => {
        console.log('🔴 [VIEWER] Host stopped streaming - setStreamAvailable(false)')
        setStreamAvailable(false)
        // Clean up streams
        remoteStreamsRef.current.clear()
        if (videoRef.current) {
          videoRef.current.srcObject = null
        }
      })

      socketRef.current.onHostDisconnected(() => {
        setError('Host disconnected')
        setStreamAvailable(false)
        setConnectionState('disconnected')
      })
    };

    const processProducer = async (producerId: string, kind: string) => {
      try {
        // Check if we already have this consumer
        if (consumersRef.current.has(producerId)) {
          console.log(`🟡 [VIEWER] Already consuming ${producerId}`)
          return
        }

        // Check if consumer transport is ready
        if (!mediasoupClientRef.current?.hasConsumerTransport()) {
          console.log(`🟡 [VIEWER] Consumer transport not ready, queueing producer: ${producerId} (${kind})`)
          pendingProducersRef.current.push({ producerId, kind })
          return
        }

        console.log(`🟢 [VIEWER] Starting to consume ${kind} producer: ${producerId}`)
        try {
          const stream = await mediasoupClientRef.current!.consume(producerId, kind)
          if (stream) {
            consumersRef.current.add(producerId)
            
            // Store the stream in our collection
            remoteStreamsRef.current.set(producerId, stream)
            console.log(`🔵 [VIEWER] Stored ${kind} stream:`, producerId)
            
            // Update the video element with combined streams
            updateVideoElement()
            
            function updateVideoElement() {
              if (!videoRef.current) return
              
              const allStreams = Array.from(remoteStreamsRef.current.values())
              const videoTracks: MediaStreamTrack[] = []
              const audioTracks: MediaStreamTrack[] = []
              
              // Collect all video and audio tracks
              allStreams.forEach(stream => {
                videoTracks.push(...stream.getVideoTracks())
                audioTracks.push(...stream.getAudioTracks())
              })
              
              console.log('🔵 [VIEWER] Creating combined stream with:')
              console.log('🔵 [VIEWER] - Video tracks:', videoTracks.length)
              console.log('🔵 [VIEWER] - Audio tracks:', audioTracks.length)
              
              // Create combined stream
              const combinedStream = new MediaStream([...videoTracks, ...audioTracks])
              
              // Initialize buffered player if not already done
              if (!bufferedPlayerRef.current && videoRef.current) {
                bufferedPlayerRef.current = new BufferedVideoPlayer(videoRef.current, {
                  bufferSeconds: 3,
                  targetLatency: 150,
                  maxBufferSize: 180,
                  enableSmoothing: true
                })
                console.log('🔵 [VIEWER] BufferedVideoPlayer initialized')
              }
              
              // Set stream through buffered player
              if (bufferedPlayerRef.current) {
                bufferedPlayerRef.current.setStream(combinedStream)
                console.log('🔵 [VIEWER] Stream set through BufferedVideoPlayer')
              } else {
                // Fallback to direct assignment
                videoRef.current.srcObject = combinedStream
                console.log('🔵 [VIEWER] Stream set directly to video element')
              }
            }
          }
        } catch (consumeError) {
          console.error(`Failed to consume ${kind} producer ${producerId}:`, consumeError)
          // If consume fails because transport not ready, queue it for later
          if (consumeError.message.includes('transport') || consumeError.message.includes('device not ready')) {
            console.log(`🟡 [VIEWER] Queueing producer due to transport error: ${producerId} (${kind})`)
            pendingProducersRef.current.push({ producerId, kind })
          }
        }
      } catch (err) {
        console.error(`Failed to consume ${kind} producer ${producerId}:`, err)
      }
    };

    const setupEventHandlers = () => {
      if (!socketRef.current) return

      // Set up PII detection event handlers
      socketRef.current.onPIIDetected((data) => {
        console.log('PII detected in stream:', data.entities)
        setRecentPIIAlert(`Sensitive content detected and redacted (${data.entities.length} items)`)
        
        // Clear the alert after 5 seconds
        setTimeout(() => {
          setRecentPIIAlert(null)
        }, 5000)
      })

      socketRef.current.onAudioProcessingStarted(() => {
        setPiiDetectionActive(true)
      })

      socketRef.current.onAudioProcessingStopped(() => {
        setPiiDetectionActive(false)
      })
    };

    const setupSFUEventHandlers = () => {
      if (!sfuSocketRef.current || !mediasoupClientRef.current) return

      // SFU socket handlers
      sfuSocketRef.current!.on('new-producer', async (data) => {
        const { producerId, kind } = data
        console.log(`🟢 [VIEWER] New producer available: ${producerId} (${kind})`)
        await processProducer(producerId, kind)
      })

      sfuSocketRef.current!.on('producer-closed', (data) => {
        const { consumerId } = data
        console.log(`Producer closed: ${consumerId}`)
        
        // Clean up consumer
        consumersRef.current.delete(consumerId)
        remoteStreamsRef.current.delete(consumerId)
        
        // If no more streams, clear video element
        if (remoteStreamsRef.current.size === 0 && videoRef.current) {
          videoRef.current.srcObject = null
        }
      })

      sfuSocketRef.current!.on('host-disconnected', () => {
        setError('Host disconnected from SFU')
        setStreamAvailable(false)
        setConnectionState('disconnected')
      })
    };

    if (roomId) {
      initializeViewer()
    }

    // Set up performance monitoring
    const performanceInterval = setInterval(() => {
      try {
        if (bufferedPlayerRef.current) {
          const bufferStatus = bufferedPlayerRef.current.getBufferStatus()
          setBufferStatus(bufferStatus)
        }
        
        if (mediasoupClientRef.current) {
          const stats = mediasoupClientRef.current.getVideoFilterStats()
          setPerformanceStats(stats)
        }
      } catch (error) {
        console.warn('[VIEWER] Performance monitoring error:', error)
      }
    }, 2000)

    return () => {
      // Cleanup performance monitoring
      clearInterval(performanceInterval)
      
      // Cleanup buffered player
      if (bufferedPlayerRef.current) {
        bufferedPlayerRef.current.destroy()
        bufferedPlayerRef.current = null
      }
      
      // Cleanup
      if (mediasoupClientRef.current) {
        mediasoupClientRef.current.stopConsuming()
      }
      
      if (socketRef.current) {
        socketRef.current.disconnect()
      }
      
      if (sfuSocketRef.current) {
        sfuSocketRef.current.disconnect()
      }
      
      remoteStreamsRef.current.clear()
      consumersRef.current.clear()
    }
  }, [roomId])

  const getConnectionStatusColor = () => {
    switch (connectionState) {
      case 'connected': return 'bg-green-500'
      case 'connecting': return 'bg-yellow-500'
      case 'error':
      case 'disconnected': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getConnectionStatusText = () => {
    switch (connectionState) {
      case 'connected': return streamAvailable ? 'Watching Live Stream (SFU)' : 'Connected - Waiting for Stream'
      case 'connecting': return 'Connecting to SFU...'
      case 'error': return 'Connection Error'
      case 'disconnected': return 'Disconnected'
      default: return 'Unknown'
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
            SFU Viewer
          </h1>
          
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Room ID</h3>
              <code className="bg-gray-200 px-2 py-1 rounded text-sm font-mono">
                {roomId}
              </code>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Status</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${getConnectionStatusColor()}`} />
                {getConnectionStatusText()}
              </div>
            </div>

            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Privacy Protection</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${
                  piiDetectionActive ? 'bg-green-500' : 'bg-gray-400'
                }`} />
                {piiDetectionActive ? 'Active' : 'Inactive'}
              </div>
            </div>

            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Buffer Health</h3>
              <div className="text-sm">
                <div className={`inline-block w-2 h-2 rounded-full mr-2 ${
                  !bufferStatus ? 'bg-gray-400' :
                  bufferStatus.health === 'good' ? 'bg-green-500' :
                  bufferStatus.health === 'low' ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                {bufferStatus ? 
                  `${(bufferStatus.duration || 0).toFixed(1)}s (${bufferStatus.health || 'unknown'})` : 
                  'Initializing'
                }
              </div>
              {bufferStatus && bufferStatus.playbackRate && Math.abs(bufferStatus.playbackRate - 1.0) > 0.01 && (
                <div className="text-xs text-gray-600 mt-1">
                  Rate: {bufferStatus.playbackRate.toFixed(2)}x
                </div>
              )}
            </div>
          </div>

          {/* Performance Stats */}
          {performanceStats && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <h3 className="font-semibold text-blue-800 mb-3">🚀 Performance Stats</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-blue-700 font-medium">GPU Acceleration:</div>
                  <div className={performanceStats.gpuEnabled ? 'text-green-600' : 'text-yellow-600'}>
                    {performanceStats.gpuEnabled ? '✅ Enabled' : '⚠️ CPU Fallback'}
                  </div>
                </div>
                <div>
                  <div className="text-blue-700 font-medium">Frame Processing:</div>
                  <div className="text-blue-600">
                    Skip Ratio: 1:{performanceStats.performance?.skipRatio || 1}
                  </div>
                </div>
                <div>
                  <div className="text-blue-700 font-medium">Buffer Status:</div>
                  <div className="text-blue-600">
                    {performanceStats.bufferSize || 0}/{performanceStats.config?.bufferFrames || 90} frames
                    ({((performanceStats.bufferHealth || 0) * 100).toFixed(0)}%)
                  </div>
                </div>
              </div>
              {performanceStats.lastDetections && (
                <div className="mt-2 text-xs text-blue-600">
                  Last Detection: Face({performanceStats.lastDetections.face || 0}), 
                  PII({performanceStats.lastDetections.pii || 0}), 
                  Plate({performanceStats.lastDetections.plate || 0})
                </div>
              )}
            </div>
          )}

          {/* PII Alert */}
          {recentPIIAlert && (
            <div className="bg-blue-100 border border-blue-300 text-blue-800 px-4 py-3 rounded mb-4">
              <div className="flex items-center">
                <span className="text-blue-600 mr-2">🛡️</span>
                {recentPIIAlert}
              </div>
            </div>
          )}

          <div className="text-sm text-gray-600 text-center">
            <p><strong>SFU Mode:</strong> Optimized streaming via Mediasoup server</p>
            <p>Low latency, high quality viewing experience</p>
            <p><strong>Privacy Protected:</strong> Sensitive content automatically redacted</p>
          </div>
        </div>

        <div className="bg-black rounded-lg overflow-hidden relative">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            controls
            className="w-full h-auto object-contain"
            style={{ backgroundColor: '#000', minHeight: '400px' }}
            onLoadedData={async () => {
              console.log('Video loaded and ready to play')
              // Always start muted and playing
              if (videoRef.current) {
                try {
                  videoRef.current.muted = true
                  await videoRef.current.play()
                  console.log('Video autoplay successful (muted)')
                } catch (error) {
                  console.log('Autoplay failed:', error)
                }
              }
            }}
            onError={(e) => {
              console.error('Video error:', e)
            }}
          />
          
          {isMuted && streamAvailable && (
            <div 
              className="absolute bottom-4 right-4 bg-black bg-opacity-75 text-white p-3 rounded-lg cursor-pointer z-10 hover:bg-opacity-90 transition-all duration-200"
              onClick={handleUnmute}
            >
              <div className="flex items-center space-x-2">
                <div className="text-2xl">🔊</div>
                <div className="text-sm font-medium">Unmute</div>
              </div>
            </div>
          )}
          
          {!streamAvailable && connectionState === 'connected' && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">⏳</div>
                <div>Waiting for host to start streaming...</div>
                <div className="text-sm mt-2 opacity-75">Connected to SFU server</div>
              </div>
            </div>
          )}
          
          {connectionState === 'connecting' && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">🔄</div>
                <div>Connecting to SFU server...</div>
                <div className="text-sm mt-2 opacity-75">Please wait</div>
              </div>
            </div>
          )}
          
          {(connectionState === 'error' || connectionState === 'disconnected') && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">❌</div>
                <div>Connection failed</div>
                <div className="text-sm mt-2 opacity-75">Please refresh the page</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}