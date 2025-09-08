#!/usr/bin/env python3
"""
Test script for debug image functionality.
Creates a sample image and sends it to the video filter API to test debug output.
"""

import requests
import base64
import cv2
import numpy as np
import json

def create_test_image_with_text():
    """Create a test image with some text that should be detected as PII."""
    # Create a white background image
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    
    # Add some test text that should be detected as PII
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add various types of text
    cv2.putText(img, "John Smith", (100, 150), font, 2, (0, 0, 0), 3)
    cv2.putText(img, "Phone: 555-123-4567", (100, 250), font, 1, (0, 0, 0), 2)
    cv2.putText(img, "Email: john@example.com", (100, 350), font, 1, (0, 0, 0), 2)
    cv2.putText(img, "SSN: 123-45-6789", (100, 450), font, 1, (0, 0, 0), 2)
    cv2.putText(img, "Address: 123 Main St, City", (100, 550), font, 1, (0, 0, 0), 2)
    
    # Add a rectangle that might be detected as a license plate
    cv2.rectangle(img, (800, 300), (1100, 380), (100, 100, 100), -1)
    cv2.putText(img, "ABC-1234", (850, 350), font, 1.5, (255, 255, 255), 2)
    
    return img

def test_debug_functionality():
    """Test the debug image functionality."""
    print("🧪 Testing Debug Image Functionality")
    print("=" * 50)
    
    # Create test image
    print("📸 Creating test image with PII content...")
    test_img = create_test_image_with_text()
    
    # Encode image as base64
    _, buffer = cv2.imencode('.jpg', test_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_b64}"
    
    # Test API endpoints
    api_base = "http://localhost:5001"
    
    # 1. Check debug status
    print("\n🔍 Checking debug status...")
    try:
        response = requests.get(f"{api_base}/debug-status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Debug Status:")
            print(f"   - Enabled: {status['debug_enabled']}")
            print(f"   - Output Directory: {status['output_directory']}")
            print(f"   - Save Input: {status['save_input']}")
            print(f"   - Save Output: {status['save_output']}")
            print(f"   - Image Counts: {status['image_counts']}")
        else:
            print(f"❌ Failed to get debug status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting debug status: {e}")
    
    # 2. Process test frame
    print("\n🔄 Processing test frame...")
    try:
        payload = {
            "frame": img_data_url,
            "frame_id": 12345,
            "detect_only": False
        }
        
        response = requests.post(f"{api_base}/process-frame", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Frame processed successfully!")
            print(f"   - Frame ID: {result.get('frame_id', 'N/A')}")
            print(f"   - Rectangles detected: {result.get('regions_processed', 0)}")
            print(f"   - Processing mode: {result.get('processing_mode', 'N/A')}")
        else:
            print(f"❌ Frame processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
    
    # 3. Check updated debug status
    print("\n📊 Checking debug status after processing...")
    try:
        response = requests.get(f"{api_base}/debug-status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Updated Image Counts: {status['image_counts']}")
            print(f"   Debug images should now be saved in: {status['output_directory']}")
        else:
            print(f"❌ Failed to get updated debug status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting updated debug status: {e}")
    
    print("\n🎯 Test Complete!")
    print("📁 Check the 'debug_images' directory for:")
    print("   - input/: Original frames with detection boxes")
    print("   - output/: Processed frames with PII blurred")
    print("   - comparison/: Input frames with green bounding boxes")

if __name__ == "__main__":
    test_debug_functionality()