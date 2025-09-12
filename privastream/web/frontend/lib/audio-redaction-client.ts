import { Socket, io } from 'socket.io-client';

export interface PIIDetection {
  text: string;
  pii_type: string;
  confidence: number;
  start_time: number;
  end_time: number;
}

export interface RedactionResult {
  segment_id: string;
  original_transcription: string;
  redacted_transcription: string;
  pii_count: number;
  processing_time: number;
  timestamp: number;
  pii_detections?: PIIDetection[];
  audio_data?: ArrayBuffer;
}

export interface RedactionStats {
  processed_segments: number;
  total_pii_detections: number;
  total_processing_time: number;
  average_processing_time: number;
}

export interface AudioRedactionClientConfig {
  serviceUrl: string;
  onRedactedAudio?: (result: RedactionResult) => void;
  onPIIDetected?: (result: RedactionResult) => void;
  onConnectionStatusChange?: (connected: boolean) => void;
  onError?: (error: string) => void;
}

export class AudioRedactionClient {
  private socket: Socket | null = null;
  private config: AudioRedactionClientConfig;
  private isConnected: boolean = false;
  private currentRoomId: string | null = null;
  private currentRole: 'host' | 'viewer' | null = null;

  constructor(config: AudioRedactionClientConfig) {
    this.config = config;
  }

  async connect(): Promise<void> {
    try {
      this.socket = io(this.config.serviceUrl, {
        transports: ['websocket'],
        upgrade: true,
        rememberUpgrade: true
      });

      return new Promise((resolve, reject) => {
        if (!this.socket) {
          reject(new Error('Failed to create socket'));
          return;
        }

        this.socket.on('connect', () => {
          console.log('✅ Connected to audio redaction service');
          this.isConnected = true;
          this.config.onConnectionStatusChange?.(true);
          resolve();
        });

        this.socket.on('disconnect', () => {
          console.log('❌ Disconnected from audio redaction service');
          this.isConnected = false;
          this.config.onConnectionStatusChange?.(false);
        });

        this.socket.on('connect_error', (error) => {
          console.error('🔴 Connection error:', error);
          this.config.onError?.(`Connection failed: ${error.message}`);
          reject(error);
        });

        this.socket.on('error', (error) => {
          console.error('🔴 Socket error:', error);
          this.config.onError?.(error);
        });

        this.socket.on('redacted_audio', (data: RedactionResult) => {
          console.log('🔇 Received redacted audio:', {
            segment_id: data.segment_id,
            pii_count: data.pii_count,
            processing_time: data.processing_time
          });
          this.config.onRedactedAudio?.(data);
        });

        this.socket.on('pii_detections', (data: RedactionResult) => {
          console.log('🚨 PII detected:', {
            segment_id: data.segment_id,
            pii_count: data.pii_count,
            detections: data.pii_detections?.map(d => ({
              type: d.pii_type,
              confidence: d.confidence
            }))
          });
          this.config.onPIIDetected?.(data);
        });

        this.socket.on('joined_redaction_room', (data) => {
          console.log('✅ Joined redaction room:', data);
          this.currentRoomId = data.room_id;
          this.currentRole = data.role;
        });

        // Set connection timeout
        setTimeout(() => {
          if (!this.isConnected) {
            reject(new Error('Connection timeout'));
          }
        }, 10000);
      });
    } catch (error) {
      console.error('🔴 Failed to connect:', error);
      throw error;
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnected = false;
    this.currentRoomId = null;
    this.currentRole = null;
  }

  async joinRoom(roomId: string, role: 'host' | 'viewer' = 'viewer'): Promise<void> {
    if (!this.socket || !this.isConnected) {
      throw new Error('Not connected to redaction service');
    }

    return new Promise((resolve, reject) => {
      this.socket!.emit('join_redaction_room', {
        room_id: roomId,
        role: role
      });

      // Wait for confirmation
      const timeout = setTimeout(() => {
        reject(new Error('Join room timeout'));
      }, 5000);

      this.socket!.once('joined_redaction_room', (data) => {
        clearTimeout(timeout);
        if (data.room_id === roomId) {
          resolve();
        } else {
          reject(new Error('Room ID mismatch'));
        }
      });

      this.socket!.once('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
  }

  sendAudioData(audioData: ArrayBuffer): void {
    if (!this.socket || !this.isConnected) {
      console.warn('⚠️ Not connected to redaction service, skipping audio data');
      return;
    }

    if (!this.currentRoomId) {
      console.warn('⚠️ Not in a redaction room, skipping audio data');
      return;
    }

    try {
      // Convert ArrayBuffer to Buffer for transmission
      const buffer = new Uint8Array(audioData);
      
      this.socket.emit('audio_data', {
        audio: buffer
      });
    } catch (error) {
      console.error('🔴 Failed to send audio data:', error);
      this.config.onError?.(`Failed to send audio: ${error}`);
    }
  }

  async getStats(): Promise<RedactionStats> {
    if (!this.socket || !this.isConnected) {
      throw new Error('Not connected to redaction service');
    }

    return new Promise((resolve, reject) => {
      this.socket!.emit('get_redaction_stats');

      const timeout = setTimeout(() => {
        reject(new Error('Stats request timeout'));
      }, 5000);

      this.socket!.once('redaction_stats', (stats: RedactionStats) => {
        clearTimeout(timeout);
        resolve(stats);
      });
    });
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  getCurrentRoom(): string | null {
    return this.currentRoomId;
  }

  getCurrentRole(): 'host' | 'viewer' | null {
    return this.currentRole;
  }
}

// Audio capture utility for browser
export class AudioCaptureManager {
  private mediaStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private redactionClient: AudioRedactionClient | null = null;
  private isCapturing: boolean = false;

  constructor(redactionClient: AudioRedactionClient) {
    this.redactionClient = redactionClient;
  }

  async startCapture(stream: MediaStream): Promise<void> {
    try {
      this.mediaStream = stream;
      
      // Create audio context
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      // Create media stream source
      const source = this.audioContext.createMediaStreamSource(stream);
      
      // Create processor for audio data
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      this.processor.onaudioprocess = (event) => {
        if (!this.isCapturing || !this.redactionClient) return;
        
        const inputBuffer = event.inputBuffer;
        const outputBuffer = event.outputBuffer;
        
        // Get audio data
        const inputData = inputBuffer.getChannelData(0);
        
        // Convert to 16-bit PCM
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          pcmData[i] = Math.round(inputData[i] * 32767);
        }
        
        // Send to redaction service
        this.redactionClient.sendAudioData(pcmData.buffer);
        
        // Copy input to output (pass-through)
        for (let channel = 0; channel < outputBuffer.numberOfChannels; channel++) {
          const outputData = outputBuffer.getChannelData(channel);
          outputData.set(inputData);
        }
      };
      
      // Connect the audio graph
      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);
      
      this.isCapturing = true;
      console.log('✅ Audio capture started');
      
    } catch (error) {
      console.error('🔴 Failed to start audio capture:', error);
      throw error;
    }
  }

  stopCapture(): void {
    this.isCapturing = false;
    
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    this.mediaStream = null;
    console.log('🛑 Audio capture stopped');
  }

  isActive(): boolean {
    return this.isCapturing;
  }
}

// React hook for audio redaction
export const useAudioRedaction = () => {
  const [client, setClient] = React.useState<AudioRedactionClient | null>(null);
  const [isConnected, setIsConnected] = React.useState(false);
  const [captureManager, setCaptureManager] = React.useState<AudioCaptureManager | null>(null);
  const [redactionResults, setRedactionResults] = React.useState<RedactionResult[]>([]);
  const [piiDetections, setPIIDetections] = React.useState<RedactionResult[]>([]);
  const [stats, setStats] = React.useState<RedactionStats | null>(null);

  const initializeClient = React.useCallback((serviceUrl: string) => {
    const newClient = new AudioRedactionClient({
      serviceUrl,
      onConnectionStatusChange: setIsConnected,
      onRedactedAudio: (result) => {
        setRedactionResults(prev => [...prev.slice(-19), result]);
      },
      onPIIDetected: (result) => {
        setPIIDetections(prev => [...prev.slice(-19), result]);
      },
      onError: (error) => {
        console.error('Audio redaction error:', error);
      }
    });

    setClient(newClient);
    setCaptureManager(new AudioCaptureManager(newClient));
    
    return newClient;
  }, []);

  const connect = React.useCallback(async () => {
    if (client) {
      await client.connect();
    }
  }, [client]);

  const disconnect = React.useCallback(() => {
    captureManager?.stopCapture();
    client?.disconnect();
  }, [client, captureManager]);

  const joinRoom = React.useCallback(async (roomId: string, role: 'host' | 'viewer' = 'viewer') => {
    if (client) {
      await client.joinRoom(roomId, role);
    }
  }, [client]);

  const startAudioCapture = React.useCallback(async (stream: MediaStream) => {
    if (captureManager) {
      await captureManager.startCapture(stream);
    }
  }, [captureManager]);

  const stopAudioCapture = React.useCallback(() => {
    captureManager?.stopCapture();
  }, [captureManager]);

  const updateStats = React.useCallback(async () => {
    if (client && isConnected) {
      try {
        const currentStats = await client.getStats();
        setStats(currentStats);
      } catch (error) {
        console.error('Failed to get stats:', error);
      }
    }
  }, [client, isConnected]);

  // Auto-update stats every 5 seconds
  React.useEffect(() => {
    if (isConnected) {
      const interval = setInterval(updateStats, 5000);
      return () => clearInterval(interval);
    }
  }, [isConnected, updateStats]);

  return {
    client,
    isConnected,
    redactionResults,
    piiDetections,
    stats,
    initializeClient,
    connect,
    disconnect,
    joinRoom,
    startAudioCapture,
    stopAudioCapture,
    updateStats
  };
};

// Import React for the hook
import React from 'react';