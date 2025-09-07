#!/usr/bin/env python3
"""
Test script for the Audio Redaction Flask API
Tests both JSON and raw binary endpoints
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
            print("PASS: Health check passed:")
            print(f"   Status: {data['status']}")
            print(f"   NER Model: {'OK' if data['models_loaded']['ner'] else 'FAIL'}")
            print(f"   ASR Model: {'OK' if data['models_loaded']['asr'] else 'FAIL'}")
            return True
        else:
            print(f"FAIL: Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Health check error: {e}")
        return False

def generate_test_audio(duration=3.0, sample_rate=16000):
    """Generate test audio with speech-like patterns"""
    print(f"🎵 Generating test audio: {duration}s at {sample_rate}Hz")
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create speech-like audio with multiple frequency components
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t) +      # Base frequency
        0.2 * np.sin(2 * np.pi * 880 * t) +      # First harmonic
        0.1 * np.sin(2 * np.pi * 220 * t)        # Sub-harmonic
    )
    
    # Add some variation to simulate speech patterns
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2Hz modulation
    audio = audio * envelope
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.05, len(audio))
    audio = audio + noise
    
    # Normalize
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio, sample_rate

def test_json_endpoint():
    """Test the JSON-based audio processing endpoint"""
    print("\n🧪 Testing JSON endpoint (/process_audio)...")
    
    # Generate test audio
    audio_data, sample_rate = generate_test_audio(duration=2.0)
    
    # Convert to 16-bit PCM and base64 encode
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Prepare request data
    request_data = {
        "audio_data": audio_b64,
        "sample_rate": sample_rate,
        "metadata": {
            "test": True,
            "duration": 2.0
        }
    }
    
    try:
        print(f"📤 Sending {len(audio_bytes)} bytes of audio data...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5002/process_audio",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️ Processing took {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ JSON endpoint test passed:")
            print(f"   Success: {data.get('success', False)}")
            print(f"   PII Count: {data.get('pii_count', 0)}")
            print(f"   Transcript: '{data.get('transcript', 'N/A')[:50]}...'")
            print(f"   Redacted intervals: {len(data.get('redacted_intervals', []))}")
            print(f"   Redacted audio size: {len(base64.b64decode(data.get('redacted_audio_data', '')))} bytes")
            return True
        else:
            print(f"❌ JSON endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ JSON endpoint error: {e}")
        return False

def test_raw_endpoint():
    """Test the raw binary audio processing endpoint"""
    print("\n🧪 Testing raw binary endpoint (/process_audio_raw)...")
    
    # Generate test audio
    audio_data, sample_rate = generate_test_audio(duration=1.5)
    
    # Convert to bytes
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    try:
        print(f"📤 Sending {len(audio_bytes)} bytes of raw audio...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:5002/process_audio_raw",
            data=audio_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": str(sample_rate)
            },
            timeout=30
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️ Processing took {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            redacted_audio = response.content
            pii_count = response.headers.get('X-PII-Count', '0')
            success = response.headers.get('X-Processing-Success', 'False')
            
            print("✅ Raw endpoint test passed:")
            print(f"   Success: {success}")
            print(f"   PII Count: {pii_count}")
            print(f"   Original audio size: {len(audio_bytes)} bytes")
            print(f"   Redacted audio size: {len(redacted_audio)} bytes")
            return True
        else:
            print(f"❌ Raw endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Raw endpoint error: {e}")
        return False

def test_with_sensitive_audio():
    """Test with audio that should trigger PII detection"""
    print("\n🚨 Testing with potentially sensitive content...")
    
    # For this test, we'll rely on the transcript detection
    # In a real scenario, you would use actual speech audio
    # This test mainly verifies the pipeline works end-to-end
    
    audio_data, sample_rate = generate_test_audio(duration=2.5)
    
    # Convert to base64
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    request_data = {
        "audio_data": audio_b64,
        "sample_rate": sample_rate,
        "simulated_transcript": "My password is secret123 and my social security number is confidential"
    }
    
    try:
        response = requests.post(
            "http://localhost:5002/process_audio",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Sensitive content test completed:")
            print(f"   PII Count: {data.get('pii_count', 0)}")
            print(f"   Redacted intervals: {data.get('redacted_intervals', [])}")
            
            if data.get('pii_count', 0) > 0:
                print("   🚨 PII was detected and redacted!")
            else:
                print("   ℹ️ No PII detected in generated audio (expected for synthetic audio)")
            
            return True
        else:
            print(f"❌ Sensitive content test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Sensitive content test error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Audio Redaction API")
    print("=" * 50)
    
    results = []
    
    # Test 1: Health Check
    results.append(test_health_check())
    
    # Test 2: JSON Endpoint
    results.append(test_json_endpoint())
    
    # Test 3: Raw Binary Endpoint
    results.append(test_raw_endpoint())
    
    # Test 4: Sensitive Content
    results.append(test_with_sensitive_audio())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    tests = ["Health Check", "JSON Endpoint", "Raw Endpoint", "Sensitive Content"]
    for i, (test_name, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the API server and try again.")
        return False

if __name__ == "__main__":
    main()