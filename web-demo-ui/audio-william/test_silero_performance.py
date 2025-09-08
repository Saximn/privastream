#!/usr/bin/env python3
"""
Test script to compare Silero VAD vs Python VAD performance
"""

import requests
import base64
import numpy as np
import librosa
import time
import json
from pathlib import Path

def test_vad_comparison():
    """Compare Silero VAD vs Python VAD performance"""
    
    audio_file = Path("input.wav")
    if not audio_file.exists():
        print("❌ input.wav not found. Please provide a test audio file.")
        return
    
    print("🆚 VAD PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Load test audio
    try:
        audio_data, sample_rate = librosa.load(str(audio_file), sr=16000)
        audio_duration = len(audio_data) / sample_rate
        print(f"📁 Audio: {len(audio_data)} samples ({audio_duration:.2f}s) at {sample_rate}Hz")
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return
    
    # Prepare test data
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_base64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
    
    test_payload = {
        "audio_data": audio_base64,
        "sample_rate": sample_rate
    }
    
    # Test endpoints
    endpoints = [
        ("Python VAD", "http://localhost:5004/process_audio"),
        ("Silero VAD", "http://localhost:5005/process_audio")
    ]
    
    results = {}
    
    for vad_type, url in endpoints:
        try:
            print(f"\\n🧪 Testing {vad_type}...")
            
            # Check health first
            health_url = url.replace('/process_audio', '/health')
            health_response = requests.get(health_url, timeout=5)
            
            if health_response.status_code != 200:
                print(f"❌ {vad_type}: Service not healthy")
                continue
            
            # Process audio
            start_time = time.time()
            response = requests.post(url, json=test_payload, timeout=45)
            api_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                if result["success"]:
                    processing_time = result.get('processing_time', 0)
                    chunks_processed = result.get('chunks_processed', 0)
                    pii_count = result.get('pii_count', 0)
                    speed_ratio = result.get('speed_ratio', 0)
                    chunk_results = result.get('chunk_results', [])
                    
                    # Calculate VAD efficiency
                    silent_chunks = len([c for c in chunk_results if c.get('was_silent')])
                    speech_chunks = chunks_processed - silent_chunks
                    vad_efficiency = (silent_chunks / chunks_processed * 100) if chunks_processed > 0 else 0
                    
                    # Calculate timing breakdown
                    if chunk_results:
                        successful_chunks = [c for c in chunk_results if c.get('success')]
                        if successful_chunks:
                            avg_vad_time = np.mean([c.get('timing_breakdown', {}).get('vad_time', 0) for c in successful_chunks])
                            avg_whisper_time = np.mean([c.get('timing_breakdown', {}).get('whisper_time', 0) for c in successful_chunks])
                            avg_bert_time = np.mean([c.get('timing_breakdown', {}).get('bert_time', 0) for c in successful_chunks])
                        else:
                            avg_vad_time = avg_whisper_time = avg_bert_time = 0
                    else:
                        avg_vad_time = avg_whisper_time = avg_bert_time = 0
                    
                    results[vad_type] = {
                        'processing_time': processing_time,
                        'api_time': api_time,
                        'speed_ratio': speed_ratio,
                        'pii_count': pii_count,
                        'chunks_processed': chunks_processed,
                        'silent_chunks': silent_chunks,
                        'speech_chunks': speech_chunks,
                        'vad_efficiency': vad_efficiency,
                        'avg_vad_time': avg_vad_time,
                        'avg_whisper_time': avg_whisper_time,
                        'avg_bert_time': avg_bert_time,
                        'transcript': result.get('transcript', ''),
                        'vad_type': result.get('configuration', {}).get('vad_type', 'unknown')
                    }
                    
                    print(f"✅ {vad_type}: {processing_time:.3f}s ({speed_ratio:.2f}x) | {pii_count} PII")
                    print(f"   📊 VAD efficiency: {vad_efficiency:.1f}% ({silent_chunks}/{chunks_processed} chunks skipped)")
                    print(f"   ⏱️  VAD time: {avg_vad_time*1000:.1f}ms avg per chunk")
                    
                else:
                    print(f"❌ {vad_type}: Processing failed - {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ {vad_type}: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"⚠️  {vad_type}: Service not running")
        except Exception as e:
            print(f"❌ {vad_type}: Error - {e}")
    
    # Display detailed comparison
    if len(results) >= 2:
        print(f"\\n📊 DETAILED VAD COMPARISON:")
        print(f"{'Metric':<25} | {'Python VAD':<12} | {'Silero VAD':<12} | {'Winner'}")
        print("-" * 70)
        
        python_result = results.get("Python VAD", {})
        silero_result = results.get("Silero VAD", {})
        
        # Speed comparison
        python_speed = python_result.get('speed_ratio', float('inf'))
        silero_speed = silero_result.get('speed_ratio', float('inf'))
        speed_winner = "Silero VAD" if silero_speed < python_speed else "Python VAD"
        print(f"{'Processing Speed':<25} | {python_speed:.2f}x        | {silero_speed:.2f}x        | {speed_winner}")
        
        # VAD efficiency comparison
        python_vad_eff = python_result.get('vad_efficiency', 0)
        silero_vad_eff = silero_result.get('vad_efficiency', 0)
        vad_eff_winner = "Silero VAD" if silero_vad_eff > python_vad_eff else "Python VAD"
        print(f"{'VAD Efficiency':<25} | {python_vad_eff:.1f}%         | {silero_vad_eff:.1f}%         | {vad_eff_winner}")
        
        # VAD speed comparison
        python_vad_time = python_result.get('avg_vad_time', 0) * 1000
        silero_vad_time = silero_result.get('avg_vad_time', 0) * 1000
        vad_speed_winner = "Python VAD" if python_vad_time < silero_vad_time else "Silero VAD"
        print(f"{'VAD Speed (ms/chunk)':<25} | {python_vad_time:.1f}ms        | {silero_vad_time:.1f}ms        | {vad_speed_winner}")
        
        # PII detection comparison
        python_pii = python_result.get('pii_count', 0)
        silero_pii = silero_result.get('pii_count', 0)
        pii_winner = "Same" if python_pii == silero_pii else ("Silero VAD" if silero_pii >= python_pii else "Python VAD")
        print(f"{'PII Detection':<25} | {python_pii:<12} | {silero_pii:<12} | {pii_winner}")
        
        print("\\n🏆 OVERALL ASSESSMENT:")
        
        # Calculate overall scores
        python_score = 0
        silero_score = 0
        
        if python_speed <= silero_speed:
            python_score += 1
        else:
            silero_score += 1
            
        if python_vad_eff >= silero_vad_eff:
            python_score += 1
        else:
            silero_score += 1
            
        if python_vad_time <= silero_vad_time:
            python_score += 1
        else:
            silero_score += 1
        
        if silero_score > python_score:
            print("🥇 WINNER: Silero VAD")
            print("   💡 Silero provides more accurate VAD with neural networks")
            print("   🚀 Better at detecting speech vs silence")
            print("   🎯 Recommended for production TikTok livestream")
        elif python_score > silero_score:
            print("🥇 WINNER: Python VAD")
            print("   ⚡ Python VAD is faster and more lightweight")
            print("   💻 Lower resource usage")
            print("   🔧 Good for systems with limited resources")
        else:
            print("🤝 TIE: Both VAD methods perform similarly")
            print("   ⚖️  Choice depends on your specific requirements")
        
        print(f"\\n📈 PERFORMANCE SUMMARY:")
        print(f"   🐍 Python VAD: {python_speed:.2f}x speed, {python_vad_eff:.1f}% efficiency")
        print(f"   🧠 Silero VAD: {silero_speed:.2f}x speed, {silero_vad_eff:.1f}% efficiency")
        
        # Speed target assessment
        target_speed = 1.5  # 1.5x real-time target
        print(f"\\n🎯 TARGET ASSESSMENT (≤{target_speed}x):")
        
        python_meets_target = python_speed <= target_speed
        silero_meets_target = silero_speed <= target_speed
        
        print(f"   Python VAD: {'✅ MEETS TARGET' if python_meets_target else '❌ EXCEEDS TARGET'}")
        print(f"   Silero VAD: {'✅ MEETS TARGET' if silero_meets_target else '❌ EXCEEDS TARGET'}")
    
    # Save comparison results
    try:
        comparison_log = {
            "timestamp": time.time(),
            "audio_duration": audio_duration,
            "results": results,
            "recommendation": "silero_vad" if silero_score > python_score else "python_vad"
        }
        
        with open("vad_comparison_results.json", "w") as f:
            json.dump(comparison_log, f, indent=2)
        
        print(f"\\n💾 Comparison results saved: vad_comparison_results.json")
        
    except Exception as e:
        print(f"⚠️  Could not save comparison results: {e}")

if __name__ == "__main__":
    print("🧠 VAD COMPARISON TEST SUITE")
    print("🆚 Python VAD vs Silero Neural VAD")
    print("=" * 60)
    
    test_vad_comparison()
    
    print(f"\\n✅ VAD comparison complete!")
