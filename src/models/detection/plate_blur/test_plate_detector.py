"""
Test script for the refactored plate detector.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add plate_blur directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from plate_detector import PlateDetector
    
    def test_plate_detector():
        """Test the plate detector with webcam."""
        print("Testing Plate Detector...")
        
        # Initialize detector
        detector = PlateDetector(
            weights_path="best.pt",
            imgsz=960,
            conf_thresh=0.25,
            iou_thresh=0.5,
            pad=4
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
            
            # Test with metadata
            frame_id, detection_data = detector.process_frame_with_metadata(frame, 0)
            print(f"Frame {frame_id}: {len(detection_data)} detections with metadata")
            for i, data in enumerate(detection_data):
                print(f"  Detection {i}: confidence={data['confidence']:.3f}, class={data['class_id']}")
            return
        
        frame_id = 0
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with metadata
            returned_frame_id, detection_data = detector.process_frame_with_metadata(frame, frame_id)
            
            # Visualize
            vis_frame = frame.copy()
            for data in detection_data:
                rect = data["rectangle"]
                conf = data["confidence"]
                x1, y1, x2, y2 = rect
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_frame, f"PLATE {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.putText(vis_frame, f"Frame: {frame_id}, Plates: {len(detection_data)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Plate Detector Test", vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_id += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("Plate detector test completed")

    if __name__ == "__main__":
        test_plate_detector()
        
except ImportError as e:
    print(f"Plate detector not available: {e}")
    print("Make sure required packages are installed: pip install torch ultralytics")
