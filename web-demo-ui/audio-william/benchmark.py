#!/usr/bin/env python3
"""
Benchmark script for audio redaction processing speed
"""
import requests
import numpy as np
import time
import base64

def test_processing_speed():
    print('🎯 Audio Redaction Speed Test')
    print('=' * 40)
    
    base_url = 'http://localhost:5002'
    
    # Test if server is running
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        if response.status_code != 200:
            print('❌ Server not responding')
            return
        print('✅ Server is healthy')
    except:
        print('❌ Cannot connect to server')
        return
    
    # Test different chunk sizes
    test_chunks = [
        (0.5, 'Short (0.5s)'),
        (1.0, 'Medium (1.0s)'), 
        (1.5, 'Target (1.5s)'),
        (2.0, 'Long (2.0s)')
    ]
    
    print('\nTesting processing speeds...')
    print('-' * 40)
    
    for duration, label in test_chunks:
        try:
            # Generate test audio
            sample_rate = 16000
            samples = int(duration * sample_rate)
            audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)
            audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
            
            # Time the request
            print(f'{label:15} | ', end='', flush=True)
            start_time = time.time()
            
            response = requests.post(f'{base_url}/process_audio', 
                                   json={
                                       'audio_data': audio_base64, 
                                       'sample_rate': sample_rate
                                   },
                                   timeout=30)
            
            end_time = time.time()
            processing_time = end_time - start_time
            speed_ratio = processing_time / duration
            
            if response.status_code == 200:
                result = response.json()
                status = f'{processing_time:.2f}s ({speed_ratio:.2f}x real-time)'
                if result.get('success'):
                    pii_count = result.get('pii_count', 0)
                    status += f' | {pii_count} PII'
                print(f'{status}')
            else:
                print(f'❌ ERROR {response.status_code}')
                if response.status_code == 500:
                    print(f'   GPU Memory Issue: {response.json().get("error", "Unknown")[:100]}...')
                break
                
        except Exception as e:
            print(f'❌ ERROR: {e}')
            break
    
    print('\n🏁 Speed test complete')
    print('\n📊 Recommendations:')
    print('- Target processing: <1.5x real-time for 1.5s chunks')
    print('- If >3x real-time: Consider Vosk (faster but less accurate)')
    print('- If OOM errors: Need better memory management')

if __name__ == '__main__':
    test_processing_speed()