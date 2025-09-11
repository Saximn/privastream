import { io, Socket } from 'socket.io-client'
import API_CONFIG from './config'

export interface PIIEntity {
  label: string
  text: string
  confidence: number
  start: number
  end: number
}

export interface PIIDetectionData {
  room_id: string
  timestamp: number
  pii_detected: boolean
  entities: PIIEntity[]
  redaction_intervals: [number, number][]
  text: string
  enable_audio_redaction: boolean
  enable_mouth_blur: boolean
  redaction_type: string
}

export interface AudioProcessingConfig {
  whisper_model_size?: 'tiny' | 'base' | 'small' | 'medium' | 'large'
  buffer_duration?: number
  redaction_type?: 'beep' | 'silence' | 'reverse'
  min_confidence?: number
  enable_audio_redaction?: boolean
  enable_mouth_blur?: boolean
}

export class SocketManager {
  private socket: Socket
  private serverUrl: string

  constructor(serverUrl?: string) {
    this.serverUrl = serverUrl || API_CONFIG.BACKEND_URL;

    this.socket = io(this.serverUrl, {
      transports: ['websocket', 'polling'],
      path: '/backend/socket.io'
    })
  }

  connect(): Promise<string> {
    return new Promise((resolve) => {
      this.socket.on('connected', (data) => {
        resolve(data.userId)
      })
    })
  }

  createRoom(): Promise<{ roomId: string; mediasoupUrl?: string }> {
    return new Promise((resolve) => {
      this.socket.emit('create_room')
      this.socket.on('room_created', (data) => {
        resolve(data)
      })
    })
  }

  joinRoom(roomId: string): Promise<{ roomId: string; mediasoupUrl?: string }> {
    return new Promise((resolve, reject) => {
      this.socket.emit('join_room', { roomId })
      this.socket.on('joined_room', (data) => {
        resolve(data)
      })
      this.socket.on('error', (error) => {
        reject(error.message)
      })
    })
  }

  checkRoom(roomId: string): Promise<{ exists: boolean; viewerCount?: number }> {
    return new Promise((resolve) => {
      this.socket.emit('get_room_info', { roomId })
      this.socket.on('room_info', (data) => {
        resolve(data)
      })
    })
  }

  // Audio Processing Methods
  startAudioProcessing(roomId: string, config?: AudioProcessingConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      this.socket.emit('start_audio_processing', { 
        room_id: roomId, 
        config: config || {
          whisper_model_size: 'base',
          min_confidence: 0.7,
          redaction_type: 'beep',
          enable_audio_redaction: true,
          enable_mouth_blur: true
        }
      })
      
      this.socket.once('audio_processing_started', () => resolve())
      this.socket.once('audio_processing_error', (error) => reject(error.error))
    })
  }

  stopAudioProcessing(roomId: string): Promise<void> {
    return new Promise((resolve) => {
      this.socket.emit('stop_audio_processing', { room_id: roomId })
      this.socket.once('audio_processing_stopped', () => resolve())
    })
  }

  sendAudioChunk(roomId: string, audioData: string, timestamp: number, format?: any) {
    this.socket.emit('audio_chunk', {
      room_id: roomId,
      audio_data: audioData,
      timestamp,
      format: format || { sample_rate: 16000, channels: 1, bit_depth: 16 }
    })
  }

  updateAudioConfig(roomId: string, config: AudioProcessingConfig): Promise<void> {
    return new Promise((resolve, reject) => {
      this.socket.emit('update_audio_config', { room_id: roomId, config })
      this.socket.once('audio_config_updated', () => resolve())
      this.socket.once('audio_processing_error', (error) => reject(error.error))
    })
  }

  getAudioStatus(roomId?: string): Promise<any> {
    return new Promise((resolve) => {
      this.socket.emit('get_audio_status', roomId ? { room_id: roomId } : {})
      this.socket.once('audio_status', resolve)
      this.socket.once('all_audio_status', resolve)
    })
  }

  // Audio Processing Event Handlers
  onPIIDetected(callback: (data: PIIDetectionData) => void) {
    this.socket.on('pii_detected', callback)
  }

  onAudioProcessingStarted(callback: (data: { room_id: string; status: string }) => void) {
    this.socket.on('audio_processing_started', callback)
  }

  onAudioProcessingStopped(callback: (data: { room_id: string; status: string }) => void) {
    this.socket.on('audio_processing_stopped', callback)
  }

  onAudioProcessingStatus(callback: (data: any) => void) {
    this.socket.on('audio_processing_status', callback)
  }

  onAudioProcessingError(callback: (error: { error: string }) => void) {
    this.socket.on('audio_processing_error', callback)
  }

  // Existing WebRTC methods
  sendOffer(offer: RTCSessionDescriptionInit) {
    this.socket.emit('offer', { offer })
  }

  sendAnswer(answer: RTCSessionDescriptionInit) {
    this.socket.emit('answer', { answer })
  }

  sendIceCandidate(candidate: RTCIceCandidate) {
    this.socket.emit('ice_candidate', { candidate })
  }

  onOffer(callback: (data: { offer: RTCSessionDescriptionInit; from: string }) => void) {
    this.socket.on('offer', callback)
  }

  onAnswer(callback: (data: { answer: RTCSessionDescriptionInit; from: string }) => void) {
    this.socket.on('answer', callback)
  }

  onIceCandidate(callback: (data: { candidate: RTCIceCandidateInit; from: string }) => void) {
    this.socket.on('ice_candidate', callback)
  }

  onViewerJoined(callback: (data: { userId: string; viewerCount?: number }) => void) {
    this.socket.on('viewer_joined', callback)
  }

  onViewerLeft(callback: (data: { userId: string; viewerCount?: number }) => void) {
    this.socket.on('viewer_left', callback)
  }

  onHostDisconnected(callback: () => void) {
    this.socket.on('host_disconnected', callback)
  }

  onStreamingStarted(callback: () => void) {
    this.socket.on('streaming_started', callback)
  }

  onStreamingStopped(callback: () => void) {
    this.socket.on('streaming_stopped', callback)
  }

  disconnect() {
    this.socket.disconnect()
  }

  getSocket(): Socket {
    return this.socket
  }
}
