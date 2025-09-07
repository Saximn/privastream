'use client'

import { io, Socket } from 'socket.io-client'
import * as mediasoupClient from 'mediasoup-client'

export class MediasoupClient {
  private roomId: string
  private socket: Socket | null = null
  private device: mediasoupClient.Device | null = null
  private producerTransport: mediasoupClient.types.Transport | null = null
  private videoProducer: mediasoupClient.types.Producer | null = null

  constructor(roomId: string) {
    this.roomId = roomId
  }

  async initialize(socket: Socket) {
    console.log('[MediasoupClient] Initializing device...')
    this.socket = socket

    try {
      this.device = new mediasoupClient.Device()
      console.log('[MediasoupClient] Device created:', this.device)

      // Request router RTP capabilities
      const routerRtpCapabilities = await new Promise<any>((resolve, reject) => {
        console.log('[MediasoupClient] Requesting router RTP capabilities for room:', this.roomId)

        this.socket!.emit('getRouterRtpCapabilities', { roomId: this.roomId }, (response: any) => {
          console.log('[MediasoupClient] getRouterRtpCapabilities response:', response)

          if (!response || response.error) {
            return reject(new Error(response?.error || 'No RTP capabilities received'))
          }
          resolve(response.rtpCapabilities)
        })
      })

      console.log('[MediasoupClient] Received router RTP capabilities:', routerRtpCapabilities)
      await this.device.load({ routerRtpCapabilities })
      console.log('[MediasoupClient] Device loaded with RTP capabilities')

    } catch (err) {
      console.error('[MediasoupClient] Device initialization failed:', err)
      throw err
    }
  }

  async createProducerTransport() {
    if (!this.socket) throw new Error('Socket not initialized')
    console.log('[MediasoupClient] Creating producer transport...')

    const transportParams = await new Promise<any>((resolve, reject) => {
      this.socket!.emit('createProducerTransport', { roomId: this.roomId }, (response: any) => {
        console.log('[MediasoupClient] createProducerTransport response:', response)
        if (!response || response.error) return reject(new Error(response?.error || 'Failed to create transport'))
        resolve(response)
      })
    })

    console.log('[MediasoupClient] Transport params received:', transportParams)
    this.producerTransport = this.device!.createSendTransport(transportParams)

    this.producerTransport.on('connect', ({ dtlsParameters }, callback, errback) => {
      console.log('[MediasoupClient] Transport connecting with DTLS params:', dtlsParameters)
      this.socket!.emit('connectProducerTransport', { dtlsParameters, roomId: this.roomId }, (response: any) => {
        console.log('[MediasoupClient] connectProducerTransport response:', response)
        if (response.error) return errback(new Error(response.error))
        callback()
      })
    })

    this.producerTransport.on('produce', async ({ kind, rtpParameters }, callback, errback) => {
      console.log('[MediasoupClient] Producing track:', kind, rtpParameters)
      this.socket!.emit('produce', { kind, rtpParameters, roomId: this.roomId }, (response: any) => {
        console.log('[MediasoupClient] produce response:', response)
        if (response.error) return errback(new Error(response.error))
        callback({ id: response.id })
      })
    })

    console.log('[MediasoupClient] Producer transport created')
  }

  async produce(stream: MediaStream, enableVideoFilter: boolean = false) {
    if (!this.producerTransport) throw new Error('Producer transport not initialized')
    console.log('[MediasoupClient] Starting to produce tracks from local stream...')
    
    // Log stream details
    console.log('[MediasoupClient] Stream details:', {
      id: stream.id,
      active: stream.active,
      videoTracks: stream.getVideoTracks().length,
      audioTracks: stream.getAudioTracks().length
    })

    const videoTrack = stream.getVideoTracks()[0]
    if (videoTrack) {
      console.log('[MediasoupClient] Video track details:', {
        id: videoTrack.id,
        kind: videoTrack.kind,
        enabled: videoTrack.enabled,
        muted: videoTrack.muted,
        readyState: videoTrack.readyState,
        settings: videoTrack.getSettings()
      })
      this.videoProducer = await this.producerTransport.produce({ track: videoTrack })
      console.log('[MediasoupClient] Video track produced:', this.videoProducer.id)
    } else {
      console.warn('[MediasoupClient] No video track found in stream')
    }

    const audioTrack = stream.getAudioTracks()[0]
    if (audioTrack) {
      console.log('[MediasoupClient] Audio track details:', {
        id: audioTrack.id,
        kind: audioTrack.kind,
        enabled: audioTrack.enabled,
        muted: audioTrack.muted,
        readyState: audioTrack.readyState,
        settings: audioTrack.getSettings()
      })
      const audioProducer = await this.producerTransport.produce({ track: audioTrack })
      console.log('[MediasoupClient] Audio track produced:', audioProducer.id)
      console.log('[MediasoupClient] 🎤 MICROPHONE AUDIO IS NOW BEING SENT TO MEDIASOUP SERVER')
    } else {
      console.error('[MediasoupClient] ❌ NO AUDIO TRACK FOUND IN STREAM - MICROPHONE ACCESS FAILED')
    }

    console.log('[MediasoupClient] All tracks are now being sent to SFU')
  }

  async stopProducing() {
    if (this.videoProducer) {
      this.videoProducer.close()
      console.log('[MediasoupClient] Video producer closed')
      this.videoProducer = null
    }
    if (this.producerTransport) {
      this.producerTransport.close()
      console.log('[MediasoupClient] Producer transport closed')
      this.producerTransport = null
    }
  }

  getVideoFilterStats() {
    // Placeholder for video filter stats
    return {
      frameCount: 0,
      processingQueueSize: 0,
      lastDetections: { face: 0, pii: 0, plate: 0 }
    }
  }
}
