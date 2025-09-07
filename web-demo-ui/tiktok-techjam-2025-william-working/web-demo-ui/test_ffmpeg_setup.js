#!/usr/bin/env node
/**
 * Test script to verify FFmpeg audio consumption setup
 */

console.log('🎬 TESTING FFMPEG AUDIO CONSUMPTION');
console.log('='.repeat(50));

// 1. Test if the audio redaction plugin is working
console.log('\n1️⃣ Testing Audio Redaction Plugin Import...');
try {
    const { AudioRedactionPlugin } = require('./mediasoup-server/audio-redaction-plugin');
    console.log('   ✅ Audio redaction plugin imported successfully');
    
    // Test plugin initialization
    const plugin = new AudioRedactionPlugin({
        redactionServiceUrl: 'http://localhost:5002',
        enabled: true,
        sampleRate: 16000
    });
    
    // Wait a moment for connection
    setTimeout(() => {
        console.log('   📊 Plugin Status:', {
            enabled: plugin.isEnabled,
            connected: plugin.isConnected,
            serviceUrl: plugin.redactionServiceUrl
        });
        
        if (plugin.isConnected) {
            console.log('   ✅ Plugin connected to audio redaction service');
        } else {
            console.log('   ❌ Plugin not connected to audio redaction service');
        }
    }, 2000);
    
} catch (error) {
    console.log('   ❌ Failed to import audio redaction plugin:', error.message);
}

// 2. Test FFmpeg command construction
console.log('\n2️⃣ Testing FFmpeg Command...');
const testPort = 12345;
const ffmpegArgs = [
    '-protocol_whitelist', 'file,udp,rtp',
    '-f', 'rtp',
    '-i', `rtp://127.0.0.1:${testPort}`,
    '-f', 'wav',
    '-acodec', 'pcm_s16le', 
    '-ar', '16000',
    '-ac', '1',
    '-loglevel', 'info',
    'pipe:1'
];

console.log('   🎬 FFmpeg command would be:');
console.log('   ', 'ffmpeg', ffmpegArgs.join(' '));

// 3. Test FFmpeg availability
console.log('\n3️⃣ Testing FFmpeg Process Start...');
try {
    const { spawn } = require('child_process');
    
    // Test FFmpeg with version command
    const testFFmpeg = spawn('ffmpeg', ['-version']);
    
    let versionOutput = '';
    testFFmpeg.stdout.on('data', (data) => {
        versionOutput += data.toString();
    });
    
    testFFmpeg.on('close', (code) => {
        if (code === 0) {
            const version = versionOutput.split('\n')[0];
            console.log('   ✅ FFmpeg available:', version);
        } else {
            console.log('   ❌ FFmpeg failed with exit code:', code);
        }
    });
    
    testFFmpeg.on('error', (error) => {
        console.log('   ❌ FFmpeg error:', error.message);
    });
    
} catch (error) {
    console.log('   ❌ Failed to spawn FFmpeg:', error.message);
}

// 4. Check system requirements
console.log('\n4️⃣ Checking System Requirements...');
console.log('   📋 Node.js version:', process.version);
console.log('   📋 Platform:', process.platform);
console.log('   📋 Architecture:', process.arch);

// 5. Instructions
setTimeout(() => {
    console.log('\n' + '='.repeat(50));
    console.log('🎯 NEXT STEPS TO TEST:');
    console.log('');
    console.log('1. Restart MediaSoup server:');
    console.log('   cd mediasoup-server');
    console.log('   node server.js');
    console.log('');
    console.log('2. Look for these logs when starting streaming:');
    console.log('   🎬 Setting up FFmpeg audio consumption for room [roomId]');
    console.log('   ✅ Created PlainTransport for FFmpeg: [transportId]');
    console.log('   🎬 Starting FFmpeg process for room [roomId]...');
    console.log('   ✅ PlainTransport connected - RTP data should now flow to FFmpeg');
    console.log('');
    console.log('3. Then check if debug_audio folder is created:');
    console.log('   cd audio-william');  
    console.log('   ls -la debug_audio/');
    console.log('');
    console.log('4. If still no audio files, the issue is likely:');
    console.log('   - PlainTransport connection timing');
    console.log('   - RTP port binding');
    console.log('   - FFmpeg RTP consumption format');
}, 3000);