"""
Quick test to verify fixed FPS functionality.
Runs for 10 seconds to demonstrate the fixed frame rate processing.
"""
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from unified_bbox_test import UnifiedBoundingBoxTester
    import cv2
    
    def test_fixed_fps():
        """Test fixed FPS functionality with webcam."""
        print("="*60)
        print("FIXED FPS FUNCTIONALITY TEST")
        print("="*60)
        
        # Initialize tester
        config = {
            'face': {'embed_path': 'face_blur/whitelist/creator_embedding.json'},
            'pii': {'classifier_path': 'pii_blur/pii_clf.joblib'},
            'plate': {'weights_path': 'plate_blur/best.pt'}
        }
        
        tester = UnifiedBoundingBoxTester(config)
        
        if len(tester.models) == 0:
            print("[ERROR] No models available for testing!")
            return
        
        print(f"Initialized {len(tester.models)} models: {list(tester.models.keys())}")
        
        # Test with webcam at 2 FPS for 10 seconds
        print("\nTesting fixed 2 FPS processing...")
        print("Will run for 10 seconds, then exit automatically.")
        print("Watch the frame rate in the display!")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return
        
        target_fps = 2.0
        frame_interval = 1.0 / target_fps
        start_time = time.time()
        last_process_time = time.time()
        frame_id = 0
        processed_frames = 0
        
        try:
            while time.time() - start_time < 10.0:  # Run for 10 seconds
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Check if it's time to process
                if current_time - last_process_time >= frame_interval:
                    # Process frame
                    results = tester.process_frame(frame, frame_id)
                    vis_frame = tester.visualize_results(frame, results)
                    vis_frame = tester.add_info_overlay(vis_frame, results, target_fps, 30.0)
                    
                    # Add test info
                    elapsed = current_time - start_time
                    cv2.putText(vis_frame, f"Test Time: {elapsed:.1f}s / 10s", 
                               (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"Processed Frames: {processed_frames}", 
                               (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    last_process_time = current_time
                    frame_id += 1
                    processed_frames += 1
                else:
                    # Show waiting frame
                    vis_frame = frame.copy()
                    cv2.putText(vis_frame, f"Fixed FPS Test: {target_fps} FPS", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    elapsed = current_time - start_time
                    cv2.putText(vis_frame, f"Test Time: {elapsed:.1f}s / 10s", 
                               (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"Processed Frames: {processed_frames}", 
                               (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Fixed FPS Test - Auto-Exit in 10s", vis_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                time.sleep(0.01)  # Small sleep to prevent high CPU usage
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print results
            elapsed_total = time.time() - start_time
            expected_frames = int(elapsed_total * target_fps)
            
            print(f"\nTEST RESULTS:")
            print(f"  Duration: {elapsed_total:.1f}s")
            print(f"  Target FPS: {target_fps}")
            print(f"  Expected frames: {expected_frames}")
            print(f"  Actual processed frames: {processed_frames}")
            print(f"  Actual FPS: {processed_frames / elapsed_total:.2f}")
            print(f"  Accuracy: {(processed_frames / expected_frames * 100):.1f}%")
            
            if abs(processed_frames / elapsed_total - target_fps) < 0.3:
                print("  ✅ FIXED FPS WORKING CORRECTLY!")
            else:
                print("  ⚠️  Fixed FPS may need adjustment")
            
            print("\nFixed FPS test completed successfully!")
    
    if __name__ == "__main__":
        test_fixed_fps()
        
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure all dependencies are installed.")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
