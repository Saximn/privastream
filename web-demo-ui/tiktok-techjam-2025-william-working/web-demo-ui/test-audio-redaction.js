// test-audio-redaction.js
const { default: fetch } = require('node-fetch');

async function testAudioRedaction() {
  console.log('Testing audio redaction server...');
  
  try {
    // Test health endpoint
    console.log('Testing health endpoint...');
    const healthResponse = await fetch('http://localhost:5002/health');
    const healthData = await healthResponse.json();
    console.log('Health check:', healthData);
    
    // Create test audio data (3 seconds of sine wave at 16kHz)
    const sampleRate = 16000;
    const duration = 3; // 3 seconds
    const samples = sampleRate * duration;
    const frequency = 440; // A4 note
    
    console.log('Generating test audio data...');
    const audioBuffer = new Int16Array(samples);
    for (let i = 0; i < samples; i++) {
      const t = i / sampleRate;
      const sample = Math.sin(2 * Math.PI * frequency * t) * 0.5;
      audioBuffer[i] = Math.round(sample * 32767);
    }
    
    // Convert to base64
    const audioBase64 = Buffer.from(audioBuffer.buffer).toString('base64');
    
    console.log('Sending audio data to redaction service...');
    const startTime = Date.now();
    
    const response = await fetch('http://localhost:5002/process_audio', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_data: audioBase64,
        sample_rate: sampleRate
      })
    });
    
    const responseTime = Date.now() - startTime;
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    console.log('\n✅ Audio redaction test results:');
    console.log(`Response time: ${responseTime}ms`);
    console.log(`Success: ${result.success}`);
    console.log(`Transcript: "${result.transcript}"`);
    console.log(`PII count: ${result.pii_count}`);
    console.log(`Processing time: ${result.processing_time}s`);
    console.log(`Speed ratio: ${result.speed_ratio}x real-time`);
    console.log(`Device used: ${result.device_used}`);
    
    if (result.redacted_audio_data) {
      const processedAudioSize = Buffer.from(result.redacted_audio_data, 'base64').length;
      console.log(`Processed audio size: ${processedAudioSize} bytes`);
      console.log('✅ Audio redaction server is working correctly!');
    } else {
      console.log('❌ No processed audio data returned');
    }
    
  } catch (error) {
    console.error('❌ Audio redaction test failed:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      console.log('\n💡 Make sure the audio redaction server is running:');
      console.log('cd audio-william && python audio_redaction_server_vosk.py');
    }
  }
}

// Run the test
testAudioRedaction();