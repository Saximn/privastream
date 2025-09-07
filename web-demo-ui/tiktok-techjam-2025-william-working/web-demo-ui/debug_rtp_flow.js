#!/usr/bin/env node
/**
 * Debug script to check if RTP packets are flowing
 */

const dgram = require('dgram');

// Create a simple UDP server to listen for RTP packets
function createRTPMonitor(port) {
    return new Promise((resolve, reject) => {
        const socket = dgram.createSocket('udp4');
        
        socket.on('listening', () => {
            const address = socket.address();
            console.log(`🔍 RTP Monitor listening on ${address.address}:${address.port}`);
            resolve(socket);
        });
        
        socket.on('message', (msg, rinfo) => {
            console.log(`📡 RTP packet received: ${msg.length} bytes from ${rinfo.address}:${rinfo.port}`);
            console.log(`📊 First 16 bytes: ${msg.slice(0, 16).toString('hex')}`);
            
            // Parse basic RTP header
            if (msg.length >= 12) {
                const version = (msg[0] & 0xC0) >> 6;
                const padding = (msg[0] & 0x20) >> 5;
                const extension = (msg[0] & 0x10) >> 4;
                const csrcCount = msg[0] & 0x0F;
                const marker = (msg[1] & 0x80) >> 7;
                const payloadType = msg[1] & 0x7F;
                const sequenceNumber = msg.readUInt16BE(2);
                const timestamp = msg.readUInt32BE(4);
                const ssrc = msg.readUInt32BE(8);
                
                console.log(`📊 RTP Header: v=${version}, pt=${payloadType}, seq=${sequenceNumber}, ts=${timestamp}, ssrc=${ssrc}`);
            }
        });
        
        socket.on('error', (err) => {
            console.error(`❌ RTP Monitor error:`, err);
            reject(err);
        });
        
        socket.bind(port, '127.0.0.1');
    });
}

async function main() {
    console.log('🔍 RTP FLOW DEBUG TOOL');
    console.log('='.repeat(50));
    
    console.log('\n📋 Instructions:');
    console.log('1. Start this script');
    console.log('2. Start MediaSoup server');
    console.log('3. Start streaming and SPEAK INTO MICROPHONE');
    console.log('4. Look for RTP packets below');
    console.log('');
    
    // Monitor common RTP port range
    const ports = [10010, 10020, 10030, 10040, 10050];
    const monitors = [];
    
    for (const port of ports) {
        try {
            const monitor = await createRTPMonitor(port);
            monitors.push(monitor);
            console.log(`✅ Monitoring port ${port}`);
        } catch (error) {
            console.log(`⚠️ Could not monitor port ${port}: ${error.message}`);
        }
    }
    
    console.log('\n🎤 Now start streaming and SPEAK INTO THE MICROPHONE...');
    console.log('📊 Waiting for RTP packets...\n');
    
    // Keep running
    process.on('SIGINT', () => {
        console.log('\n🛑 Shutting down monitors...');
        monitors.forEach(monitor => monitor.close());
        process.exit(0);
    });
}

if (require.main === module) {
    main().catch(console.error);
}