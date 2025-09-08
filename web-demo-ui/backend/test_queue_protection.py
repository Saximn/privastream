#!/usr/bin/env python3
"""
Test script for queue protection functionality.
Sends multiple requests to test timestamp-based dropping and concurrent request limits.
"""

import requests
import base64
import cv2
import numpy as np
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

def create_simple_test_image():
    """Create a simple test image."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST FRAME", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img

def send_request_with_timestamp(frame_data, frame_id, timestamp_offset_ms=0):
    """Send a single request with specific timestamp offset."""
    api_base = "http://localhost:5001"
    
    # Create timestamp (current time + offset)
    request_timestamp = int(time.time() * 1000) + timestamp_offset_ms
    
    payload = {
        "frame": frame_data,
        "frame_id": frame_id,
        "timestamp": request_timestamp,
        "detect_only": True
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{api_base}/process-frame", json=payload, timeout=10)
        end_time = time.time()
        
        result = {
            "frame_id": frame_id,
            "status_code": response.status_code,
            "processing_time_ms": int((end_time - start_time) * 1000),
            "timestamp_offset": timestamp_offset_ms
        }
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = data.get("success", False)
            result["dropped"] = data.get("dropped", False)
            result["rectangles"] = data.get("regions_processed", 0)
        elif response.status_code in [429, 503]:
            data = response.json()
            result["success"] = False
            result["dropped"] = True
            result["reason"] = data.get("reason", "unknown")
            result["error"] = data.get("error", "unknown")
        else:
            result["success"] = False
            result["error"] = response.text
        
        return result
    
    except Exception as e:
        return {
            "frame_id": frame_id,
            "status_code": 0,
            "success": False,
            "error": str(e),
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "timestamp_offset": timestamp_offset_ms
        }

def test_queue_protection():
    """Test queue protection functionality."""
    print("🧪 Testing Queue Protection Functionality")
    print("=" * 60)
    
    # Create test image
    test_img = create_simple_test_image()
    _, buffer = cv2.imencode('.jpg', test_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_b64}"
    
    api_base = "http://localhost:5001"
    
    # 1. Check initial queue status
    print("\n🔍 Checking initial queue status...")
    try:
        response = requests.get(f"{api_base}/queue-status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Queue Status:")
            print(f"   - Active requests: {status['active_requests']}")
            print(f"   - Max concurrent: {status['max_concurrent']}")
            print(f"   - Can accept new: {status['can_accept_new']}")
            print(f"   - Max request age: {status['max_request_age_ms']}ms")
            print(f"   - Request dropping: {status['request_dropping_enabled']}")
            print(f"   - Current time: {status['current_time_ms']}")
        else:
            print(f"❌ Failed to get queue status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting queue status: {e}")
    
    # 2. Test stale request dropping
    print("\n🗑️ Testing stale request dropping...")
    stale_results = []
    
    test_cases = [
        ("Current time", 0),
        ("500ms old", -500),
        ("1.5s old (should drop)", -1500),
        ("3s old (should drop)", -3000),
    ]
    
    for description, offset in test_cases:
        print(f"   Sending {description} request...")
        result = send_request_with_timestamp(img_data_url, len(stale_results) + 100, offset)
        stale_results.append(result)
        
        status_icon = "✅" if result["success"] else "🗑️" if result.get("dropped") else "❌"
        print(f"   {status_icon} {description}: {result['status_code']} - {result.get('reason', 'processed')}")
    
    # 3. Test concurrent request limits
    print("\n🚫 Testing concurrent request limits...")
    print("   Sending 6 concurrent requests (max is typically 3)...")
    
    def send_concurrent_request(frame_id):
        return send_request_with_timestamp(img_data_url, frame_id + 200, 0)
    
    # Send requests concurrently
    with ThreadPoolExecutor(max_workers=6) as executor:
        concurrent_futures = [executor.submit(send_concurrent_request, i) for i in range(6)]
        concurrent_results = [future.result() for future in concurrent_futures]
    
    # Analyze results
    processed = sum(1 for r in concurrent_results if r["success"])
    dropped = sum(1 for r in concurrent_results if r.get("dropped"))
    errors = len(concurrent_results) - processed - dropped
    
    print(f"   Results: {processed} processed, {dropped} dropped, {errors} errors")
    
    for i, result in enumerate(concurrent_results):
        status_icon = "✅" if result["success"] else "🚫" if result.get("dropped") else "❌"
        reason = result.get("reason", "processed" if result["success"] else "error")
        print(f"   {status_icon} Request {i+1}: {result['status_code']} - {reason}")
    
    # 4. Check final queue status
    print("\n📊 Checking final queue status...")
    try:
        response = requests.get(f"{api_base}/queue-status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Final Queue Status:")
            print(f"   - Active requests: {status['active_requests']}")
            print(f"   - Can accept new: {status['can_accept_new']}")
        else:
            print(f"❌ Failed to get final queue status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting final queue status: {e}")
    
    # 5. Summary
    print("\n📋 TEST SUMMARY:")
    print("=" * 60)
    
    stale_dropped = sum(1 for r in stale_results if r.get("dropped") and r.get("reason") == "stale_request")
    concurrent_dropped = sum(1 for r in concurrent_results if r.get("dropped") and r.get("reason") == "overloaded")
    
    print(f"✅ Stale requests dropped: {stale_dropped}/2 (expected 2)")
    print(f"✅ Concurrent requests dropped: {concurrent_dropped} (expected 3+)")
    print(f"🛡️ Queue protection is {'WORKING' if stale_dropped >= 1 and concurrent_dropped >= 1 else 'NEEDS ATTENTION'}")
    
    print("\n🎯 Test Complete!")
    print("📈 Check server logs for detailed queue monitoring messages")

if __name__ == "__main__":
    test_queue_protection()