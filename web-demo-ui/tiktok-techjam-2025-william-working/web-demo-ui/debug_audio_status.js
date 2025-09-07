#!/usr/bin/env node
/**
 * Debug script to check audio redaction status
 */

const http = require('http');

// Simple fetch implementation
function fetch(url, options = {}) {
    return new Promise((resolve, reject) => {
        const urlObj = new URL(url);
        const protocol = urlObj.protocol === 'https:' ? require('https') : http;
        
        const requestOptions = {
            hostname: urlObj.hostname,
            port: urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80),
            path: urlObj.pathname + urlObj.search,
            method: options.method || 'GET',
            headers: options.headers || {}
        };

        if (options.body) {
            requestOptions.headers['Content-Length'] = Buffer.byteLength(options.body);
        }

        const req = protocol.request(requestOptions, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
                resolve({
                    ok: res.statusCode >= 200 && res.statusCode < 300,
                    status: res.statusCode,
                    json: () => Promise.resolve(JSON.parse(data)),
                    text: () => Promise.resolve(data)
                });
            });
        });

        req.on('error', reject);

        if (options.body) {
            req.write(options.body);
        }
        req.end();
    });
}

async function checkAudioStatus() {
    console.log('🔍 AUDIO REDACTION DEBUG STATUS');
    console.log('='.repeat(50));
    
    // 1. Check MediaSoup Server
    try {
        console.log('\n1️⃣ Checking MediaSoup Server...');
        const response = await fetch('http://localhost:3001/health');
        if (response.ok) {
            const data = await response.json();
            console.log('   ✅ MediaSoup Server:', data.status);
        } else {
            console.log('   ❌ MediaSoup Server not responding');
        }
    } catch (error) {
        console.log('   ❌ MediaSoup Server error:', error.message);
    }
    
    // 2. Check Audio Redaction Server
    try {
        console.log('\n2️⃣ Checking Audio Redaction Server...');
        const response = await fetch('http://localhost:5002/health');
        if (response.ok) {
            const data = await response.json();
            console.log('   ✅ Audio Redaction Server:', data.status);
            console.log('   📊 Models:');
            console.log('      - Vosk ASR:', data.models_loaded?.vosk ? '✅' : '❌');
            console.log('      - BERT NER:', data.models_loaded?.bert ? '✅' : '❌');
            console.log('   🖥️  Device:', data.models?.ner || 'N/A');
        } else {
            console.log('   ❌ Audio Redaction Server not responding');
        }
    } catch (error) {
        console.log('   ❌ Audio Redaction Server error:', error.message);
    }
    
    // 3. Check FFmpeg availability
    console.log('\n3️⃣ Checking FFmpeg...');
    try {
        const { spawn } = require('child_process');
        const ffmpeg = spawn('ffmpeg', ['-version']);
        
        ffmpeg.on('close', (code) => {
            if (code === 0) {
                console.log('   ✅ FFmpeg is available');
            } else {
                console.log('   ❌ FFmpeg failed with code:', code);
            }
        });
        
        ffmpeg.on('error', (error) => {
            console.log('   ❌ FFmpeg not available:', error.message);
        });
        
    } catch (error) {
        console.log('   ❌ FFmpeg check error:', error.message);
    }
    
    // 4. Check debug audio folder
    console.log('\n4️⃣ Checking Debug Audio Folder...');
    const fs = require('fs');
    const path = require('path');
    
    const debugPath = path.join(__dirname, 'audio-william', 'debug_audio');
    try {
        if (fs.existsSync(debugPath)) {
            const files = fs.readdirSync(debugPath);
            console.log('   ✅ Debug audio folder exists');
            console.log('   📁 Files:', files.length, 'files');
            if (files.length > 0) {
                const recentFiles = files.slice(-3);
                console.log('   📄 Recent files:', recentFiles.join(', '));
            }
        } else {
            console.log('   ❌ Debug audio folder does not exist');
            console.log('   💡 This means no audio has been processed yet');
        }
    } catch (error) {
        console.log('   ❌ Error checking debug folder:', error.message);
    }
    
    // 5. Test simple audio processing
    console.log('\n5️⃣ Testing Audio Processing...');
    try {
        // Generate test audio
        const sampleRate = 16000;
        const duration = 2;
        const samples = sampleRate * duration;
        const audioData = Buffer.alloc(samples * 2);
        
        // Generate sine wave
        for (let i = 0; i < samples; i++) {
            const sample = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.3;
            const intSample = Math.round(sample * 32767);
            audioData.writeInt16LE(Math.max(-32767, Math.min(32767, intSample)), i * 2);
        }
        
        const audioBase64 = audioData.toString('base64');
        
        const response = await fetch('http://localhost:5002/process_audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                audio_data: audioBase64,
                sample_rate: sampleRate
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('   ✅ Audio processing test successful');
            console.log('   📊 Result:', {
                success: result.success,
                transcript: `"${result.transcript}"`,
                piiCount: result.pii_count,
                processingTime: `${result.processing_time?.toFixed(3)}s`
            });
        } else {
            console.log('   ❌ Audio processing test failed:', response.status);
        }
    } catch (error) {
        console.log('   ❌ Audio processing test error:', error.message);
    }
    
    console.log('\n' + '='.repeat(50));
    console.log('🎯 DIAGNOSIS:');
    console.log('   If debug_audio folder is missing, the FFmpeg pipeline is not working');
    console.log('   Check MediaSoup server logs for FFmpeg startup messages');
    console.log('   Look for: "🎬 Setting up FFmpeg audio consumption"');
}

if (require.main === module) {
    checkAudioStatus().catch(console.error);
}