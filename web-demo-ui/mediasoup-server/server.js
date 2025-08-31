const express = require('express');
const { Server } = require('socket.io');
const http = require('http');
const cors = require('cors');
const mediasoup = require('mediasoup');

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json());

const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Mediasoup configuration
const mediaCodecs = [
  {
    kind: 'audio',
    mimeType: 'audio/opus',
    clockRate: 48000,
    channels: 2,
  },
  {
    kind: 'video',
    mimeType: 'video/VP8',
    clockRate: 90000,
    parameters: {
      'x-google-start-bitrate': 1000,
    },
  },
];

const webRtcTransportOptions = {
  listenIps: [
    {
      ip: '0.0.0.0',
      announcedIp: '127.0.0.1', // Replace with your server's public IP in production
    },
  ],
  enableUdp: true,
  enableTcp: true,
  preferUdp: true,
};

// Global variables
let worker;
let router;
const rooms = new Map();
const transports = new Map();
const producers = new Map();
const consumers = new Map();

async function createWorker() {
  worker = await mediasoup.createWorker({
    rtcMinPort: 10000,
    rtcMaxPort: 10100,
  });
  
  worker.on('died', error => {
    console.error('mediasoup worker died:', error);
    setTimeout(() => process.exit(1), 2000);
  });

  return worker;
}

async function createRouter() {
  router = await worker.createRouter({ mediaCodecs });
  return router;
}

// Initialize mediasoup
async function init() {
  try {
    await createWorker();
    await createRouter();
    console.log('Mediasoup initialized successfully');
  } catch (error) {
    console.error('Failed to initialize mediasoup:', error);
    process.exit(1);
  }
}

// Room management
class Room {
  constructor(roomId) {
    this.roomId = roomId;
    this.host = null;
    this.viewers = new Set();
    this.hostTransports = new Map(); // producerTransport, consumerTransport
    this.hostProducers = new Map(); // video, audio
  }

  addViewer(socketId) {
    this.viewers.add(socketId);
  }

  removeViewer(socketId) {
    this.viewers.delete(socketId);
  }

  setHost(socketId) {
    this.host = socketId;
  }

  getViewerCount() {
    return this.viewers.size;
  }
}

// Socket.IO handlers
io.on('connection', async (socket) => {
  console.log(`Client connected: ${socket.id}`);

  socket.on('create-room', async (data, callback) => {
    try {
      const roomId = data.roomId;
      const room = new Room(roomId);
      room.setHost(socket.id);
      rooms.set(roomId, room);
      
      socket.join(roomId);
      
      callback({ success: true, roomId });
      console.log(`Room created: ${roomId} by ${socket.id}`);
    } catch (error) {
      console.error('Error creating room:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('join-room', async (data, callback) => {
    try {
      const { roomId } = data;
      const room = rooms.get(roomId);
      
      if (!room) {
        callback({ success: false, error: 'Room not found' });
        return;
      }

      room.addViewer(socket.id);
      socket.join(roomId);
      
      // Notify host about new viewer
      socket.to(room.host).emit('viewer-joined', { 
        viewerId: socket.id,
        viewerCount: room.getViewerCount()
      });
      
      // Send existing producers to the new viewer
      room.hostProducers.forEach((producer, kind) => {
        console.log(`Notifying new viewer about existing ${kind} producer: ${producer.id}`);
        socket.emit('new-producer', {
          producerId: producer.id,
          kind: producer.kind
        });
      });
      
      callback({ success: true, roomId });
      console.log(`Viewer ${socket.id} joined room ${roomId} (${room.hostProducers.size} producers available)`);
    } catch (error) {
      console.error('Error joining room:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('get-router-rtp-capabilities', async (data, callback) => {
    try {
      const rtpCapabilities = router.rtpCapabilities;
      callback({ success: true, rtpCapabilities });
    } catch (error) {
      console.error('Error getting RTP capabilities:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('create-producer-transport', async (data, callback) => {
    try {
      const { roomId } = data;
      const room = rooms.get(roomId);
      
      if (!room || room.host !== socket.id) {
        callback({ success: false, error: 'Unauthorized or room not found' });
        return;
      }

      const transport = await router.createWebRtcTransport(webRtcTransportOptions);
      
      transport.on('dtlsstatechange', dtlsState => {
        if (dtlsState === 'closed') {
          transport.close();
        }
      });

      transport.on('close', () => {
        console.log('Producer transport closed');
      });

      room.hostTransports.set('producer', transport);
      transports.set(transport.id, transport);

      callback({
        success: true,
        transportOptions: {
          id: transport.id,
          iceParameters: transport.iceParameters,
          iceCandidates: transport.iceCandidates,
          dtlsParameters: transport.dtlsParameters,
        },
      });
    } catch (error) {
      console.error('Error creating producer transport:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('create-consumer-transport', async (data, callback) => {
    try {
      const { roomId } = data;
      const room = rooms.get(roomId);
      
      if (!room || !room.viewers.has(socket.id)) {
        callback({ success: false, error: 'Unauthorized or room not found' });
        return;
      }

      const transport = await router.createWebRtcTransport(webRtcTransportOptions);
      
      transport.on('dtlsstatechange', dtlsState => {
        if (dtlsState === 'closed') {
          transport.close();
        }
      });

      transport.on('close', () => {
        console.log('Consumer transport closed');
      });

      transports.set(transport.id, transport);

      callback({
        success: true,
        transportOptions: {
          id: transport.id,
          iceParameters: transport.iceParameters,
          iceCandidates: transport.iceCandidates,
          dtlsParameters: transport.dtlsParameters,
        },
      });
    } catch (error) {
      console.error('Error creating consumer transport:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('connect-transport', async (data, callback) => {
    try {
      const { transportId, dtlsParameters } = data;
      const transport = transports.get(transportId);
      
      if (!transport) {
        callback({ success: false, error: 'Transport not found' });
        return;
      }

      await transport.connect({ dtlsParameters });
      callback({ success: true });
    } catch (error) {
      console.error('Error connecting transport:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('produce', async (data, callback) => {
    try {
      const { roomId, transportId, kind, rtpParameters } = data;
      const room = rooms.get(roomId);
      
      if (!room || room.host !== socket.id) {
        callback({ success: false, error: 'Unauthorized or room not found' });
        return;
      }

      const transport = transports.get(transportId);
      if (!transport) {
        callback({ success: false, error: 'Transport not found' });
        return;
      }

      const producer = await transport.produce({ kind, rtpParameters });
      room.hostProducers.set(kind, producer);
      producers.set(producer.id, producer);

      producer.on('transportclose', () => {
        console.log(`Producer transport closed: ${producer.id}`);
      });

      // Notify all viewers that a new stream is available
      room.viewers.forEach(viewerId => {
        socket.to(viewerId).emit('new-producer', {
          producerId: producer.id,
          kind: producer.kind
        });
      });

      callback({ success: true, producerId: producer.id });
      console.log(`Producer created: ${producer.id} (${kind}) for room ${roomId}`);
    } catch (error) {
      console.error('Error creating producer:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('consume', async (data, callback) => {
    try {
      const { roomId, transportId, producerId, rtpCapabilities } = data;
      const room = rooms.get(roomId);
      
      if (!room || !room.viewers.has(socket.id)) {
        callback({ success: false, error: 'Unauthorized or room not found' });
        return;
      }

      const transport = transports.get(transportId);
      const producer = producers.get(producerId);
      
      if (!transport || !producer) {
        callback({ success: false, error: 'Transport or producer not found' });
        return;
      }

      if (!router.canConsume({ producerId, rtpCapabilities })) {
        callback({ success: false, error: 'Cannot consume' });
        return;
      }

      const consumer = await transport.consume({
        producerId,
        rtpCapabilities,
        paused: true,
      });

      consumers.set(consumer.id, consumer);

      consumer.on('transportclose', () => {
        console.log(`Consumer transport closed: ${consumer.id}`);
      });

      consumer.on('producerclose', () => {
        console.log(`Consumer producer closed: ${consumer.id}`);
        socket.emit('producer-closed', { consumerId: consumer.id });
      });

      callback({
        success: true,
        consumerOptions: {
          id: consumer.id,
          producerId,
          kind: consumer.kind,
          rtpParameters: consumer.rtpParameters,
        },
      });
    } catch (error) {
      console.error('Error creating consumer:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('resume-consumer', async (data, callback) => {
    try {
      const { consumerId } = data;
      const consumer = consumers.get(consumerId);
      
      if (!consumer) {
        callback({ success: false, error: 'Consumer not found' });
        return;
      }

      await consumer.resume();
      callback({ success: true });
    } catch (error) {
      console.error('Error resuming consumer:', error);
      callback({ success: false, error: error.message });
    }
  });

  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
    
    // Clean up resources
    for (const [roomId, room] of rooms.entries()) {
      if (room.host === socket.id) {
        // Host disconnected, clean up room
        room.viewers.forEach(viewerId => {
          io.to(viewerId).emit('host-disconnected');
        });
        rooms.delete(roomId);
      } else if (room.viewers.has(socket.id)) {
        // Viewer disconnected
        room.removeViewer(socket.id);
        if (room.host) {
          io.to(room.host).emit('viewer-left', { 
            viewerId: socket.id,
            viewerCount: room.getViewerCount()
          });
        }
      }
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', mediasoup: 'ready' });
});

const PORT = process.env.PORT || 3001;

// Initialize and start server
init().then(() => {
  server.listen(PORT, () => {
    console.log(`Mediasoup SFU server running on port ${PORT}`);
  });
}).catch(error => {
  console.error('Failed to start server:', error);
  process.exit(1);
});