import { Device } from 'mediasoup-client';
import { Socket } from 'socket.io-client';
import { VideoFilter } from './video-filter';

export interface MediasoupTransportOptions {
  id: string;
  iceParameters: any;
  iceCandidates: any[];
  dtlsParameters: any;
}

export interface ConsumerOptions {
  id: string;
  producerId: string;
  kind: 'audio' | 'video';
  rtpParameters: any;
}

export class MediasoupClient {
  private device?: Device;
  private sfuSocket?: Socket;
  private producerTransport?: any;
  private consumerTransport?: any;
  private producers = new Map<string, any>();
  private consumers = new Map<string, any>();
  private roomId: string;
  private videoFilter?: VideoFilter;
  private originalStream?: MediaStream;
  private filteredStream?: MediaStream;

  constructor(roomId: string) {
    this.roomId = roomId;
  }

  async initialize(sfuSocket: Socket): Promise<void> {
    this.sfuSocket = sfuSocket;
    this.device = new Device();

    // Get router RTP capabilities from SFU server
    const rtpCapabilities = await this.request('get-router-rtp-capabilities', {});
    
    // Load device with router RTP capabilities
    await this.device.load({ routerRtpCapabilities: rtpCapabilities });
    
    console.log('Mediasoup device initialized');
  }

  async createProducerTransport(): Promise<void> {
    if (!this.device || !this.sfuSocket) {
      throw new Error('Device not initialized');
    }

    const transportOptions = await this.request('create-producer-transport', {
      roomId: this.roomId
    });

    this.producerTransport = this.device.createSendTransport(transportOptions);

    // Handle transport events
    this.producerTransport.on('connect', async ({ dtlsParameters }: any, callback: any, errback: any) => {
      try {
        await this.request('connect-transport', {
          transportId: this.producerTransport.id,
          dtlsParameters
        });
        callback();
      } catch (error) {
        errback(error);
      }
    });

    this.producerTransport.on('produce', async ({ kind, rtpParameters }: any, callback: any, errback: any) => {
      try {
        const { producerId } = await this.request('produce', {
          roomId: this.roomId,
          transportId: this.producerTransport.id,
          kind,
          rtpParameters
        });
        callback({ id: producerId });
      } catch (error) {
        errback(error);
      }
    });

    this.producerTransport.on('connectionstatechange', (state: any) => {
      console.log('Producer transport state:', state);
      if (state === 'closed' || state === 'failed' || state === 'disconnected') {
        console.log('Producer transport closed');
      }
    });
  }

  async createConsumerTransport(): Promise<void> {
    if (!this.device || !this.sfuSocket) {
      throw new Error('Device not initialized');
    }

    console.log('🟢 [MEDIASOUP-CLIENT] Requesting consumer transport creation for room:', this.roomId);
    let transportOptions;
    try {
      transportOptions = await this.request('create-consumer-transport', {
        roomId: this.roomId
      });
      console.log('🟢 [MEDIASOUP-CLIENT] Consumer transport options received:', JSON.stringify(transportOptions, null, 2));
    } catch (requestError) {
      console.error('🟢 [MEDIASOUP-CLIENT] Failed to request consumer transport:', requestError);
      throw requestError;
    }

    this.consumerTransport = this.device.createRecvTransport(transportOptions);

    // Handle transport events
    this.consumerTransport.on('connect', async ({ dtlsParameters }: any, callback: any, errback: any) => {
      try {
        await this.request('connect-transport', {
          transportId: this.consumerTransport.id,
          dtlsParameters
        });
        callback();
      } catch (error) {
        errback(error);
      }
    });

    this.consumerTransport.on('connectionstatechange', (state: any) => {
      console.log('Consumer transport state:', state);
      if (state === 'closed' || state === 'failed' || state === 'disconnected') {
        console.log('Consumer transport closed');
      }
    });
  }

  hasConsumerTransport(): boolean {
    return !!this.consumerTransport;
  }

  async produce(stream: MediaStream, enableFiltering: boolean = false): Promise<void> {
    if (!this.producerTransport) {
      throw new Error('Producer transport not created');
    }

    this.originalStream = stream;
    let streamToUse = stream;

    // Apply video filtering if enabled
    if (enableFiltering) {
      streamToUse = await this.applyVideoFiltering(stream);
      this.filteredStream = streamToUse;
    }

    const tracks = streamToUse.getTracks();
    
    for (const track of tracks) {
      try {
        const producer = await this.producerTransport.produce({ track });
        this.producers.set(track.kind, producer);
        
        producer.on('transportclose', () => {
          console.log(`Producer transport closed: ${producer.id}`);
        });

        producer.on('trackended', () => {
          console.log(`Producer track ended: ${producer.id}`);
        });

        console.log(`Producer created: ${producer.id} (${track.kind}) - Filtering: ${enableFiltering}`);
      } catch (error) {
        console.error(`Error creating producer for ${track.kind}:`, error);
      }
    }
  }

  private async applyVideoFiltering(stream: MediaStream): Promise<MediaStream> {
    // Initialize video filter with optimized settings
    this.videoFilter = new VideoFilter({
      apiUrl: 'http://localhost:5001',
      processingFps: 4,
      blurType: 'gaussian',
      kernelSize: 35,
      pixelSize: 16,
      useGPU: false,
      bufferFrames: 90, // 3 seconds at 30fps
      skipFrames: 1 // Process every other frame for better performance
    });

    // Get video track
    const videoTrack = stream.getVideoTracks()[0];
    if (!videoTrack) {
      console.warn('No video track found for filtering');
      return stream;
    }

    // Check if browser supports insertable streams
    if (!('MediaStreamTrackProcessor' in window)) {
      console.warn('Browser does not support video filtering');
      return stream;
    }

    try {
      // Create transform stream
      const processor = new (window as any).MediaStreamTrackProcessor({ track: videoTrack });
      const generator = new (window as any).MediaStreamTrackGenerator({ kind: 'video' });
      
      // Apply filtering transform
      const transformStream = this.videoFilter.createTransformStream();
      processor.readable.pipeThrough(transformStream).pipeTo(generator.writable);
      
      // Create new stream with filtered video + original audio
      const filteredStream = new MediaStream();
      filteredStream.addTrack(generator);
      
      // Add audio tracks from original stream
      stream.getAudioTracks().forEach(track => {
        filteredStream.addTrack(track);
      });

      console.log('Video filtering applied successfully');
      return filteredStream;
    } catch (error) {
      console.error('Failed to apply video filtering:', error);
      return stream; // Fallback to original stream
    }
  }

  async consume(producerId: string, kind: 'audio' | 'video'): Promise<MediaStream | null> {
    if (!this.consumerTransport || !this.device) {
      console.error('🔴 [MEDIASOUP-CLIENT] consume() failed - missing transport or device:', {
        hasTransport: !!this.consumerTransport,
        hasDevice: !!this.device,
        transportState: this.consumerTransport?.connectionState
      });
      throw new Error('Consumer transport or device not ready');
    }

    console.log('🟢 [MEDIASOUP-CLIENT] consume() attempting to consume:', {
      producerId,
      kind,
      transportState: this.consumerTransport.connectionState,
      deviceReady: this.device.loaded
    });

    try {
      const response = await this.request('consume', {
        roomId: this.roomId,
        transportId: this.consumerTransport.id,
        producerId,
        rtpCapabilities: this.device.rtpCapabilities
      });

      const consumerOptions = response.consumerOptions;
      const consumer = await this.consumerTransport.consume(consumerOptions);
      this.consumers.set(consumer.id, consumer);

      // Resume the consumer
      await this.request('resume-consumer', {
        consumerId: consumer.id
      });

      consumer.on('transportclose', () => {
        console.log(`Consumer transport closed: ${consumer.id}`);
      });

      consumer.on('producerclose', () => {
        console.log(`Consumer producer closed: ${consumer.id}`);
      });

      // Create media stream from consumer track
      const stream = new MediaStream();
      stream.addTrack(consumer.track);

      console.log(`Consumer created: ${consumer.id} (${kind})`);
      return stream;
    } catch (error) {
      console.error(`Error consuming ${kind}:`, error);
      return null;
    }
  }

  async stopProducing(): Promise<void> {
    // Close all producers
    for (const producer of this.producers.values()) {
      producer.close();
    }
    this.producers.clear();

    // Clean up video filter
    if (this.videoFilter) {
      this.videoFilter.destroy();
      this.videoFilter = undefined;
    }

    // Stop original and filtered streams
    if (this.originalStream) {
      this.originalStream.getTracks().forEach(track => track.stop());
      this.originalStream = undefined;
    }
    
    if (this.filteredStream) {
      this.filteredStream.getTracks().forEach(track => track.stop());
      this.filteredStream = undefined;
    }

    // Close producer transport
    if (this.producerTransport) {
      this.producerTransport.close();
      this.producerTransport = null;
    }
  }

  async stopConsuming(): Promise<void> {
    // Close all consumers
    for (const consumer of this.consumers.values()) {
      consumer.close();
    }
    this.consumers.clear();

    // Close consumer transport
    if (this.consumerTransport) {
      this.consumerTransport.close();
      this.consumerTransport = null;
    }
  }

  isProducerReady(): boolean {
    return !!this.producerTransport && this.producers.size > 0;
  }

  isConsumerReady(): boolean {
    return !!this.consumerTransport;
  }

  getProducers(): Map<string, any> {
    return this.producers;
  }

  getConsumers(): Map<string, any> {
    return this.consumers;
  }

  getVideoFilterStats() {
    return this.videoFilter?.getStats() || null;
  }

  updateFilterConfig(config: any) {
    if (this.videoFilter) {
      this.videoFilter.updateConfig(config);
    }
  }

  private async request(method: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.sfuSocket) {
        reject(new Error('SFU socket not connected'));
        return;
      }

      this.sfuSocket.emit(method, data, (response: any) => {
        if (response.success) {
          // Return the appropriate data based on the response structure
          if (response.rtpCapabilities) {
            resolve(response.rtpCapabilities);
          } else if (response.transportOptions) {
            resolve(response.transportOptions);
          } else if (response.consumerOptions) {
            resolve(response);
          } else if (response.producerId) {
            resolve({ producerId: response.producerId });
          } else {
            resolve(response);
          }
        } else {
          reject(new Error(response.error || 'Request failed'));
        }
      });
    });
  }
}