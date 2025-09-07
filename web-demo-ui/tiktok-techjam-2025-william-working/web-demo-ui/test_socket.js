// Simple Node.js script to test socket connection
const io = require('socket.io-client');

console.log('Testing socket connection to backend...');

const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('✅ Connected to backend socket');
  console.log('Socket ID:', socket.id);
});

socket.on('connected', (data) => {
  console.log('✅ Got connected event:', data);
  
  // Test creating a room
  socket.emit('create_room', {}, (response) => {
    console.log('Create room response:', response);
    process.exit(0);
  });
});

socket.on('connect_error', (error) => {
  console.error('❌ Connection error:', error);
  process.exit(1);
});

socket.on('disconnect', (reason) => {
  console.log('Disconnected:', reason);
});

// Exit after 10 seconds if nothing happens
setTimeout(() => {
  console.log('❌ Test timeout');
  process.exit(1);
}, 10000);