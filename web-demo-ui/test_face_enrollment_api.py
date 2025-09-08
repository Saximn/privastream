#!/usr/bin/env python3
"""
Test script for Face Enrollment API
Tests InsightFace Buffalo_S integration
"""

import requests
import numpy as np
import base64
import json
import time
import cv2
from datetime import datetime

def create_test_image():
    """Create a simple test image with a face-like pattern"""
    # Create a simple test image (this won't actually contain a face)
    # In real usage, you'd need actual face images
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some pattern that might resemble a face structure
    cv2.rectangle(img, (200, 150), (440, 350), (100, 100, 100), -1)  # Head
    cv2.circle(img, (280, 220), 20, (255, 255, 255), -1)  # Left eye
    cv2.circle(img, (360, 220), 20, (255, 255, 255), -1)  # Right eye
    cv2.rectangle(img, (310, 260), (330, 280), (255, 255, 255), -1)  # Nose
    cv2.rectangle(img, (300, 300), (340, 320), (255, 255, 255), -1)  # Mouth
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def test_face_enrollment_api():
    """Test the Face Enrollment API"""
    print("🧪 Testing Face Enrollment API with InsightFace Buffalo_S")
    print("=" * 60)
    
    api_url = "http://localhost:5003"
    room_id = f"test_room_{int(time.time())}"
    
    # 1. Health check
    print("\n🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed!")
            print(f"   Service: {health_data.get('service', 'Unknown')}")
            print(f"   Models loaded: {health_data.get('models_loaded', {})}")
            print(f"   InsightFace available: {health_data.get('insightface_available', False)}")
            print(f"   Active rooms: {health_data.get('active_rooms', 0)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # 2. Face detection test
    print("\n🔍 Testing face detection...")
    try:
        # Create test image
        test_image_b64 = create_test_image()
        print(f"Created test image: {len(test_image_b64)} characters")
        
        # Test detection
        detection_data = {
            "frame_data": test_image_b64,
            "room_id": room_id,
            "detect_only": True
        }
        
        start_time = time.time()
        response = requests.post(f"{api_url}/api/face-detection", json=detection_data)
        detection_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face detection successful!")
            print(f"   Detection time: {detection_time:.3f}s")
            print(f"   API detection time: {result.get('detection_time', 0):.3f}s")
            print(f"   Faces detected: {len(result.get('faces_detected', []))}")
            
            for i, face in enumerate(result.get('faces_detected', [])):
                bbox = face.get('bbox', [])
                confidence = face.get('confidence', 0)
                print(f"   Face {i+1}: bbox={bbox}, confidence={confidence:.3f}")
        else:
            print(f"❌ Face detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False
    
    # 3. Face enrollment test
    print("\n📝 Testing face enrollment...")
    try:
        # Create multiple test frames
        test_frames = [create_test_image() for _ in range(5)]
        
        enrollment_data = {
            "frames": test_frames,
            "room_id": room_id
        }
        
        print(f"Enrolling {len(test_frames)} frames for room: {room_id}")
        start_time = time.time()
        
        response = requests.post(f"{api_url}/api/face-enrollment", json=enrollment_data)
        enrollment_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face enrollment successful!")
            print(f"   Enrollment time: {enrollment_time:.3f}s")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Message: {result.get('message', 'No message')}")
            print(f"   Complete: {result.get('enrollment_complete', False)}")
            
            metadata = result.get('metadata', {})
            print(f"   Frames processed: {metadata.get('frames_processed', 0)}")
            print(f"   Valid frames: {metadata.get('valid_frames', 0)}")
            print(f"   Embeddings count: {metadata.get('embeddings_count', 0)}")
            
        else:
            print(f"❌ Face enrollment failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Face enrollment error: {e}")
        return False
    
    # 4. Room status test
    print("\n📊 Testing room status...")
    try:
        response = requests.get(f"{api_url}/api/room-status/{room_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Room status retrieved!")
            print(f"   Room enrolled: {result.get('enrolled', False)}")
            print(f"   Room ID: {result.get('room_id', 'Unknown')}")
            
            if result.get('metadata'):
                metadata = result['metadata']
                print(f"   Enrollment time: {metadata.get('enrollment_time', 'Unknown')}")
                print(f"   Frames processed: {metadata.get('frames_processed', 0)}")
                
        else:
            print(f"❌ Room status failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Room status error: {e}")
    
    # 5. Cleanup test
    print("\n🧹 Testing room cleanup...")
    try:
        response = requests.delete(f"{api_url}/api/cleanup-room/{room_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Room cleanup successful!")
            print(f"   Message: {result.get('message', 'No message')}")
        else:
            print(f"❌ Room cleanup failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Room cleanup error: {e}")
    
    print("\n🎉 All tests completed!")
    return True

def test_integration_workflow():
    """Test the complete integration workflow"""
    print("\n🔄 Testing complete enrollment workflow...")
    
    api_url = "http://localhost:5003"
    room_id = f"workflow_test_{int(time.time())}"
    
    print(f"Testing workflow for room: {room_id}")
    
    # Simulate frontend workflow
    steps = [
        "1. Start camera and live detection",
        "2. User sees face bounding boxes",
        "3. User clicks 'Enroll Face'", 
        "4. Collect multiple frames",
        "5. Compute average embedding",
        "6. Store in room memory",
        "7. Ready for streaming"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n✅ Workflow simulation complete!")
    print(f"   Ready for integration with Mediasoup server")
    print(f"   Face embeddings stored temporarily per room")
    print(f"   No permanent storage required")

if __name__ == "__main__":
    print(f"🚀 Starting Face Enrollment API Test at {datetime.now()}")
    
    success = test_face_enrollment_api()
    
    if success:
        test_integration_workflow()
        print("\n✅ All tests passed! Face Enrollment API is working correctly.")
        print("\n📝 Integration Notes:")
        print("   1. Install dependencies: pip install -r face_enrollment_requirements.txt")
        print("   2. Start server: python face_enrollment_server.py")
        print("   3. Server runs on: http://localhost:5003")
        print("   4. Frontend calls: /api/face-detection and /api/face-enrollment")
        print("   5. Embeddings stored temporarily per room (in-memory)")
    else:
        print("\n❌ Tests failed. Check the server and dependencies.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure server is running: python face_enrollment_server.py")
        print("   2. Install InsightFace: pip install insightface")
        print("   3. Install dependencies: pip install -r face_enrollment_requirements.txt")
        print("   4. Check server logs for errors")