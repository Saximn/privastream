export interface WebRTCConfig {
  iceServers: RTCIceServer[]
}

export const defaultWebRTCConfig: WebRTCConfig = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' }
  ]
}

export class WebRTCPeer {
  public peerConnection: RTCPeerConnection
  private localStream?: MediaStream
  private remoteStream?: MediaStream
  private onRemoteStreamCallback?: (stream: MediaStream) => void
  private onIceCandidateCallback?: (candidate: RTCIceCandidate) => void

  constructor(config: WebRTCConfig = defaultWebRTCConfig) {
    this.peerConnection = new RTCPeerConnection(config)
    this.setupPeerConnection()
  }

  private setupPeerConnection() {
    this.peerConnection.onicecandidate = (event) => {
      if (event.candidate && this.onIceCandidateCallback) {
        this.onIceCandidateCallback(event.candidate)
      }
    }

    this.peerConnection.ontrack = (event) => {
      this.remoteStream = event.streams[0]
      if (this.onRemoteStreamCallback) {
        this.onRemoteStreamCallback(this.remoteStream)
      }
    }
  }

  async initializeLocalStream(video = true, audio = true): Promise<MediaStream> {
    try {
      this.localStream = await navigator.mediaDevices.getUserMedia({ video, audio })
      this.localStream.getTracks().forEach(track => {
        this.peerConnection.addTrack(track, this.localStream!)
      })
      return this.localStream
    } catch (error) {
      console.error('Error accessing media devices:', error)
      throw error
    }
  }

  async createOffer(): Promise<RTCSessionDescriptionInit> {
    const offer = await this.peerConnection.createOffer()
    await this.peerConnection.setLocalDescription(offer)
    return offer
  }

  async createAnswer(offer: RTCSessionDescriptionInit): Promise<RTCSessionDescriptionInit> {
    await this.peerConnection.setRemoteDescription(offer)
    const answer = await this.peerConnection.createAnswer()
    await this.peerConnection.setLocalDescription(answer)
    return answer
  }

  async handleAnswer(answer: RTCSessionDescriptionInit) {
    await this.peerConnection.setRemoteDescription(answer)
  }

  async addIceCandidate(candidate: RTCIceCandidateInit) {
    try {
      await this.peerConnection.addIceCandidate(candidate)
    } catch (error) {
      console.error('Error adding ICE candidate:', error)
    }
  }

  onRemoteStream(callback: (stream: MediaStream) => void) {
    this.onRemoteStreamCallback = callback
  }

  onIceCandidate(callback: (candidate: RTCIceCandidate) => void) {
    this.onIceCandidateCallback = callback
  }

  getLocalStream(): MediaStream | undefined {
    return this.localStream
  }

  getRemoteStream(): MediaStream | undefined {
    return this.remoteStream
  }

  close() {
    if (this.localStream) {
      this.localStream.getTracks().forEach(track => track.stop())
    }
    this.peerConnection.close()
  }

  getConnectionState(): RTCPeerConnectionState {
    return this.peerConnection.connectionState
  }
}