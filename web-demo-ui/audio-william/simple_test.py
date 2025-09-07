#!/usr/bin/env python3
"""
Simple test for the Audio Redaction Flask API
"""

import requests
import numpy as np
import base64
import json
import time

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    
    try:
        response = requests.get("http://localhost:5002/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("PASS: Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   BERT Model: {'OK' if data['models_loaded']['bert'] else 'FAIL'}")
            print(f"   Vosk Model: {'OK' if data['models_loaded']['vosk'] else 'FAIL'}")
            return True
        else:
            print(f"FAIL: Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Health check error: {e}")
        return False

def generate_test_audio(duration=2.0, sample_rate=16000):
    """Generate test audio with speech-like patterns"""
    print(f"Generating test audio: {duration}s at {sample_rate}Hz")
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create speech-like audio with multiple frequency components
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +      # Base frequency
        0.2 * np.sin(2 * np.pi * 880 * t) +      # First harmonic
        0.1 * np.sin(2 * np.pi * 220 * t)        # Sub-harmonic
    )
    
    # Add variation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio = audio * envelope
    
    # Add noise
    noise = np.random.normal(0, 0.05, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio, sample_rate

def test_json_endpoint():
    """Test the JSON-based audio processing endpoint"""
    print("\nTesting JSON endpoint (/process_audio)...")
    
    # Generate test audio
    audio_data, sample_rate = generate_test_audio(duration=2.0)
    
    # Convert to 16-bit PCM and base64 encode
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Prepare request data
    request_data = {
        "audio_data": audio_b64,
        "sample_rate": sample_rate
    }
    
    try:
        print(f"Sending {len(audio_bytes)} bytes of audio data...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5002/process_audio",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        processing_time = time.time() - start_time
        print(f"Processing took {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print("PASS: JSON endpoint test passed")
            print(f"   Success: {data.get('success', False)}")
            print(f"   PII Count: {data.get('pii_count', 0)}")
            print(f"   Transcript: '{data.get('transcript', 'N/A')[:50]}...'")
            print(f"   Redacted intervals: {len(data.get('redacted_intervals', []))}")
            print(f"   Redacted audio size: {len(base64.b64decode(data.get('redacted_audio_data', '')))} bytes")
            return True
        else:
            print(f"FAIL: JSON endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"FAIL: JSON endpoint error: {e}")
        return False

def main():
    """Run tests"""
    print("Audio Redaction API Test")
    print("=" * 40)
    
    results = []
    
    # Test 1: Health Check
    results.append(test_health_check())
    
    # Test 2: JSON Endpoint
    results.append(test_json_endpoint())
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Results:")
    
    tests = ["Health Check", "JSON Endpoint"]
    for i, (test_name, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"   {i+1}. {test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The API is working correctly.")
        return True
    else:
        print("Some tests failed. Check the API server.")
        return False

if __name__ == "__main__":
    main()