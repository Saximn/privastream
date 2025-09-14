#!/usr/bin/env python3
"""
GPU Memory Monitoring Script for Audio Redaction Server
Monitors GPU memory usage and can trigger cleanup
"""

import requests
import time
import json
from datetime import datetime

def check_server_status():
    """Check if the Flask server is running"""
    try:
        response = requests.get("http://localhost:5002/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_memory_info():
    """Get memory info from server"""
    try:
        response = requests.get("http://localhost:5002/health", timeout=5)
        if response.ok:
            data = response.json()
            return data.get('gpu_info')
        return None
    except Exception as e:
        print(f"Error getting memory info: {e}")
        return None

def trigger_cleanup():
    """Trigger manual cleanup on server"""
    try:
        response = requests.post("http://localhost:5002/cleanup", timeout=10)
        if response.ok:
            return response.json()
        return None
    except Exception as e:
        print(f"Error triggering cleanup: {e}")
        return None

def monitor_memory(interval=5, cleanup_threshold=85):
    """
    Monitor GPU memory usage continuously
    
    Args:
        interval: Check interval in seconds
        cleanup_threshold: Memory usage % to trigger cleanup
    """
    print(f"🎮 GPU Memory Monitor Started")
    print(f"   Check interval: {interval} seconds")
    print(f"   Cleanup threshold: {cleanup_threshold}%")
    print("   Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            if not check_server_status():
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Server not responding")
                time.sleep(interval)
                continue
            
            memory_info = get_memory_info()
            
            if memory_info is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Running on CPU (no GPU monitoring)")
                time.sleep(interval * 2)  # Check less frequently for CPU
                continue
            
            allocated = memory_info['memory_allocated']
            reserved = memory_info['memory_reserved'] 
            total_gb = float(memory_info['memory_total'].replace(' GB', ''))
            total_mb = total_gb * 1000
            
            # Use allocated memory for actual usage (reserved can be higher due to caching)
            actual_used = min(float(allocated), total_mb)
            usage_percent = min(100, (actual_used / total_mb) * 100)
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            status_icon = "🟢" if usage_percent < 50 else "🟡" if usage_percent < 80 else "🔴"
            
            print(f"[{timestamp}] {status_icon} GPU: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved ({usage_percent:.1f}%)")
            
            # Trigger cleanup if usage is high
            if usage_percent > cleanup_threshold:
                print(f"[{timestamp}] 🧹 HIGH MEMORY USAGE! Triggering cleanup...")
                cleanup_result = trigger_cleanup()
                
                if cleanup_result:
                    if cleanup_result['status'] == 'cleaned':
                        freed = cleanup_result.get('freed_mb', 0)
                        print(f"[{timestamp}] ✅ Cleanup complete: freed {freed:.1f}MB")
                    else:
                        print(f"[{timestamp}] ⚠️ Cleanup result: {cleanup_result['status']}")
                else:
                    print(f"[{timestamp}] ❌ Cleanup failed")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

def show_current_status():
    """Show current memory status once"""
    print("GPU Current GPU Memory Status:")
    print("-" * 30)
    
    if not check_server_status():
        print("X Server not responding")
        return
    
    memory_info = get_memory_info()
    
    if memory_info is None:
        print("Running on CPU (no GPU info)")
        return
    
    print(f"Device: {memory_info['device_name']}")
    print(f"Total Memory: {memory_info['memory_total']}")
    print(f"Allocated: {memory_info['memory_allocated']:.1f} MB")
    print(f"Reserved: {memory_info['memory_reserved']:.1f} MB")
    
    total_gb = float(memory_info['memory_total'].replace(' GB', ''))
    total_mb = total_gb * 1000
    allocated_mb = float(memory_info['memory_allocated'])
    
    # Use allocated memory for actual usage
    actual_used = min(allocated_mb, total_mb)
    usage_percent = min(100, (actual_used / total_mb) * 100)
    
    print(f"Usage: {usage_percent:.1f}%")
    
    if usage_percent > 80:
        print("! HIGH USAGE - Consider cleanup")
    elif usage_percent > 50:
        print("MODERATE USAGE")
    else:
        print("LOW USAGE")

def manual_cleanup():
    """Trigger manual cleanup"""
    print("Triggering manual cleanup...")
    
    if not check_server_status():
        print("X Server not responding")
        return
    
    result = trigger_cleanup()
    
    if result:
        print("Cleanup result:")
        print(json.dumps(result, indent=2))
    else:
        print("X Cleanup failed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "monitor":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 85
            monitor_memory(interval, threshold)
        elif command == "status":
            show_current_status()
        elif command == "cleanup":
            manual_cleanup()
        else:
            print("Usage:")
            print("  python monitor_gpu.py monitor [interval] [cleanup_threshold]")
            print("  python monitor_gpu.py status")
            print("  python monitor_gpu.py cleanup")
    else:
        print("🎮 GPU Memory Monitor for Audio Redaction Server")
        print()
        print("Usage:")
        print("  python monitor_gpu.py monitor     # Start monitoring (5s interval, 85% threshold)")
        print("  python monitor_gpu.py monitor 3 75  # Custom interval & threshold") 
        print("  python monitor_gpu.py status     # Show current memory status")
        print("  python monitor_gpu.py cleanup    # Trigger manual cleanup")
        print()
        print("Examples:")
        print("  python monitor_gpu.py monitor    # Monitor with defaults")
        print("  python monitor_gpu.py status     # Quick status check")