"""
Test script for the refactored face detector.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add face_blur directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from face_detector import FaceDetector
    
    def test_face_detector():
        """Test the face detector with webcam."""
        print("Testing Face Detector...")
        
        # Initialize detector
        detector = FaceDetector(
            embed_path="whitelist/creator_embedding.json",
            gpu_id=0,
            det_size=960,
            threshold=0.35
        )
        
        # Print model info
        info = detector.get_model_info()
        print("Model Info:", info)
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam, using dummy frame")
            # Create a dummy frame for testing
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_id, rectangles = detector.process_frame(frame, 0)
            print(f"Frame {frame_id}: {len(rectangles)} rectangles to blur")
            for i, rect in enumerate(rectangles):
                print(f"  Rectangle {i}: {rect}")
            return
        
        frame_id = 0
        print("Press 'q' to quit, 'p' to toggle panic mode, 'r' to reload embedding")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            returned_frame_id, rectangles = detector.process_frame(frame, frame_id)
            
            # Visualize
            vis_frame = frame.copy()
            for rect in rectangles:
                x1, y1, x2, y2 = rect
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_frame, "BLUR", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.putText(vis_frame, f"Frame: {frame_id}, Blur regions: {len(rectangles)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Face Detector Test", vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                detector.set_panic_mode(not detector.panic_mode)
            elif key == ord('r'):
                detector.reload_embedding()
            
            frame_id += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("Face detector test completed")

    if __name__ == "__main__":
        test_face_detector()
        
except ImportError as e:
    print(f"Face detector not available: {e}")
    print("Make sure InsightFace is installed: pip install insightface")
