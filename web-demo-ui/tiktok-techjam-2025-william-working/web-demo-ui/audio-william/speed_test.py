import requests
import time
import base64
import numpy as np

print("Audio Processing Speed Test")
print("=" * 30)

# Test different chunk sizes
test_sizes = [
    (0.5, "0.5s chunk"),
    (1.0, "1.0s chunk"), 
    (1.5, "1.5s chunk"),
    (2.0, "2.0s chunk"),
    (2.5, "2.5s chunk"),
    (3.0, "3.0s chunk")

]

for duration, label in test_sizes:
    try:
        # Generate test audio
        samples = int(duration * 16000)
        audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode()
        
        # Time the processing
        print(f"{label:12} | ", end="", flush=True)
        start_time = time.time()
        
        response = requests.post('http://localhost:5002/process_audio', 
                               json={
                                   'audio_data': audio_base64, 
                                   'sample_rate': 16000
                               }, 
                               timeout=30)
        
        processing_time = time.time() - start_time
        speed_ratio = processing_time / duration
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"{processing_time:.2f}s processing ({speed_ratio:.2f}x real-time)")
            else:
                print(f"FAILED: {result.get('error', 'Unknown error')}")
                break
        else:
            print(f"ERROR {response.status_code}")
            if response.status_code == 500:
                try:
                    error_detail = response.json().get('error', '')
                    if 'out of memory' in error_detail.lower():
                        print("   --> GPU OUT OF MEMORY")
                    else:
                        print(f"   --> {error_detail[:100]}")
                except:
                    print(f"   --> {response.text[:100]}")
            break
            
    except Exception as e:
        print(f"ERROR: {e}")
        break

print("\nSummary:")
print("- Target: <2.0x real-time for good performance")
print("- If >3.0x: Consider faster model (Vosk)")
print("- If OOM: Need memory management fixes")