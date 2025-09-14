"""
Demonstration script showing the difference between fixed FPS and variable FPS processing.
Creates a side-by-side comparison or sequential demonstration.
"""
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from unified_bbox_test import UnifiedBoundingBoxTester
    import cv2
    import numpy as np
    
    def demonstrate_fps_modes():
        """Demonstrate both fixed and variable FPS modes."""
        print("="*60)
        print("FPS MODES DEMONSTRATION")
        print("="*60)
        
        # Check if webcam is available
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[INFO] No webcam available, using synthetic frames")
            use_synthetic = True
        else:
            use_synthetic = False
            cap.release()
        
        # Initialize tester with minimal config for speed
        config = {
            'face': {'embed_path': 'face_blur/whitelist/creator_embedding.json', 'det_size': 640},
            'pii': {'classifier_path': 'pii_blur/pii_clf.joblib', 'conf_thresh': 0.4},
            'plate': {'weights_path': 'plate_blur/best.pt', 'conf_thresh': 0.3}
        }
        
        tester = UnifiedBoundingBoxTester(config)
        
        if len(tester.models) == 0:
            print("[ERROR] No models available!")
            return
        
        print(f"Using {len(tester.models)} models: {list(tester.models.keys())}")
        
        def get_frame(frame_id):
            """Get frame from webcam or generate synthetic."""
            if use_synthetic:
                # Create synthetic frame with changing content
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                # Add some consistent elements
                cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), -1)
                cv2.putText(frame, f"Frame {frame_id}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return True, frame
            else:
                return cap.read()
        
        def run_fps_test(mode_name, target_fps=None, duration=8):
            """Run FPS test in either fixed or variable mode."""
            print(f"\n{mode_name} MODE - Running for {duration} seconds...")
            
            if not use_synthetic:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            start_time = time.time()
            last_process_time = time.time()
            frame_id = 0
            processed_frames = 0
            
            # Fixed FPS parameters
            if target_fps:
                frame_interval = 1.0 / target_fps
            
            try:
                while time.time() - start_time < duration:
                    if use_synthetic:
                        ret, frame = get_frame(frame_id)
                    else:
                        ret, frame = cap.read()
                        
                    if not ret:
                        break
                    
                    current_time = time.time()
                    should_process = False
                    
                    if target_fps:  # Fixed FPS mode
                        if current_time - last_process_time >= frame_interval:
                            should_process = True
                            last_process_time = current_time
                    else:  # Variable FPS mode
                        should_process = True
                    
                    if should_process:
                        # Process frame
                        results = tester.process_frame(frame, frame_id)
                        vis_frame = tester.visualize_results(frame, results)
                        
                        # Add mode info
                        if target_fps:
                            vis_frame = tester.add_info_overlay(vis_frame, results, target_fps, 30.0)
                            mode_text = f"FIXED FPS: {target_fps}"
                        else:
                            vis_frame = tester.add_info_overlay(vis_frame, results, 0.0, 30.0)
                            mode_text = "VARIABLE FPS"
                        
                        # Add test info
                        elapsed = current_time - start_time
                        cv2.rectangle(vis_frame, (0, vis_frame.shape[0]-100), 
                                     (vis_frame.shape[1], vis_frame.shape[0]), (0, 0, 0), -1)
                        cv2.putText(vis_frame, mode_text, (10, vis_frame.shape[0] - 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"Time: {elapsed:.1f}s / {duration}s", 
                                   (10, vis_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(vis_frame, f"Processed: {processed_frames}", 
                                   (10, vis_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        processed_frames += 1
                        frame_id += 1
                    else:
                        # Show waiting frame for fixed FPS
                        vis_frame = frame.copy()
                        cv2.putText(vis_frame, f"FIXED FPS: {target_fps} - Waiting...", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        elapsed = current_time - start_time
                        cv2.putText(vis_frame, f"Time: {elapsed:.1f}s / {duration}s", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow(f"FPS Demo - {mode_name}", vis_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    if not target_fps:
                        time.sleep(0.01)  # Small delay for variable FPS to prevent overwhelming
                    else:
                        time.sleep(0.005)  # Smaller delay for fixed FPS
                        
            except KeyboardInterrupt:
                print(f"\n{mode_name} interrupted")
            finally:
                if not use_synthetic:
                    cap.release()
                cv2.destroyAllWindows()
                
                # Print results
                elapsed_total = time.time() - start_time
                actual_fps = processed_frames / elapsed_total if elapsed_total > 0 else 0
                
                print(f"{mode_name} RESULTS:")
                print(f"  Duration: {elapsed_total:.1f}s")
                print(f"  Processed frames: {processed_frames}")
                print(f"  Actual FPS: {actual_fps:.2f}")
                
                if target_fps:
                    expected_frames = int(elapsed_total * target_fps)
                    accuracy = (processed_frames / expected_frames * 100) if expected_frames > 0 else 0
                    print(f"  Target FPS: {target_fps}")
                    print(f"  Expected frames: {expected_frames}")
                    print(f"  Accuracy: {accuracy:.1f}%")
                
                return processed_frames, actual_fps
        
        # Run demonstrations
        print("\nThis demo will show:")
        print("1. Variable FPS mode (processes every frame as fast as possible)")
        print("2. Fixed 2 FPS mode (processes exactly 2 frames per second)")
        print("3. Fixed 3 FPS mode (processes exactly 3 frames per second)")
        print("\nPress 'q' in any window to skip to next test")
        
        input("\nPress Enter to start Variable FPS test...")
        var_frames, var_fps = run_fps_test("VARIABLE FPS", None, 6)
        
        input("\nPress Enter to start Fixed 2 FPS test...")
        fixed2_frames, fixed2_fps = run_fps_test("FIXED 2 FPS", 2.0, 8)
        
        input("\nPress Enter to start Fixed 3 FPS test...")
        fixed3_frames, fixed3_fps = run_fps_test("FIXED 3 FPS", 3.0, 8)
        
        # Summary
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"Variable FPS:  {var_frames} frames  ({var_fps:.2f} FPS)")
        print(f"Fixed 2 FPS:   {fixed2_frames} frames  ({fixed2_fps:.2f} FPS)")
        print(f"Fixed 3 FPS:   {fixed3_frames} frames  ({fixed3_fps:.2f} FPS)")
        print("\nKey Benefits of Fixed FPS:")
        print("✅ Predictable processing load")
        print("✅ Consistent resource usage")
        print("✅ Better for performance testing")
        print("✅ Suitable for real-time applications")
        print("="*60)
    
    if __name__ == "__main__":
        demonstrate_fps_modes()
        
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure all dependencies are installed.")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
