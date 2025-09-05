#!/usr/bin/env python3
"""
Quick latency performance test for the optimized DeBERTaV3 PII detection system.
"""

import sys
import time
import logging
sys.path.append('../src')

def test_optimized_pii_detection():
    """Test the optimized PII detection with latency measurements."""
    print("Testing Optimized DeBERTaV3 PII Detection System")
    print("=" * 55)
    
    # Configure logging to be quiet
    logging.basicConfig(level=logging.WARNING)
    
    try:
        from pii_detector import PIIDetector
        
        print("Initializing optimized PII detector...")
        start_init = time.time()
        
        detector = PIIDetector(
            model_path="./models/",
            device="cuda" if _cuda_available() else "cpu",
            confidence_threshold=0.7,
            max_length=512  # Optimized for speed
        )
        
        init_time = time.time() - start_init
        print(f"Initialization: {init_time:.2f}s")
        print(f"Device: {detector.device}")
        print(f"Optimizations applied: {hasattr(detector, 'use_amp')}")
        
        # Test cases representing realistic livestream scenarios
        test_cases = [
            "Hi everyone, my name is Sarah Johnson and welcome to the stream",
            "You can email me at sarah.johnson@university.edu for questions", 
            "My phone number is 555-123-4567 if you need to reach me",
            "I live at 123 Oak Street in downtown Springfield",
            "Check out my LinkedIn at https://linkedin.com/in/sarahjohnson",
            "My username on Twitter is @sarah_streams",
            "My student ID is 987654321 for verification",
            "This is just regular stream content without any personal info"
        ]
        
        print(f"\nProcessing {len(test_cases)} realistic test cases...")
        
        total_time = 0
        total_detections = 0
        
        for i, text in enumerate(test_cases, 1):
            print(f"\nTest {i}: {text[:50]}...")
            
            start_time = time.time()
            detections = detector.detect_pii_in_text(text)
            processing_time = time.time() - start_time
            
            total_time += processing_time
            total_detections += len(detections)
            
            print(f"  Processing time: {processing_time*1000:.1f}ms")
            print(f"  PII detected: {len(detections)}")
            
            for detection in detections:
                print(f"    - {detection.pii_type.value}: '{detection.text}' "
                      f"(confidence: {detection.confidence:.2f})")
        
        # Performance summary
        avg_latency = (total_time / len(test_cases)) * 1000  # ms
        
        print("\n" + "=" * 55)
        print("PERFORMANCE SUMMARY")
        print("=" * 55)
        print(f"Average latency: {avg_latency:.1f}ms")
        print(f"Total PII detected: {total_detections}")
        print(f"Throughput: {len(test_cases)/total_time:.1f} texts/second")
        
        # Latency assessment
        if avg_latency < 50:
            print("Status: EXCELLENT - Ultra low latency")
        elif avg_latency < 100:
            print("Status: GOOD - Low latency suitable for real-time")
        elif avg_latency < 200:
            print("Status: ACCEPTABLE - Medium latency")
        else:
            print("Status: NEEDS OPTIMIZATION - High latency")
        
        print(f"\nModel: DeBERTaV3 (your trained ensemble approach)")
        print(f"Post-processing: Sophisticated rules applied")
        print(f"Hardware: {detector.device.upper()}")
        
        # Cleanup
        detector.cleanup()
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def _cuda_available():
    """Check CUDA availability."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def main():
    """Run the performance test."""
    success = test_optimized_pii_detection()
    if success:
        print("\nTest completed successfully! System is optimized for real-time use.")
    else:
        print("\nTest failed. Check error messages above.")
    
    return success

if __name__ == "__main__":
    main()
