"""
Test script for the refactored PII detector.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add pii_blur directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from pii_detector import PIIDetector
    
    def test_pii_detector():
        """Test the PII detector with webcam."""
        print("Testing PII Detector...")
        
        # Initialize detector
        detector = PIIDetector(
            classifier_path="pii_clf.joblib",
            conf_thresh=0.35,
            min_area=80,
            K_confirm=2,
            K_hold=8
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
            frame_id, polygons = detector.process_frame(frame, 0)
            print(f"Frame {frame_id}: {len(polygons)} polygons to blur")
            for i, poly in enumerate(polygons):
                print(f"  Polygon {i}: {poly.shape} points")
            return
        
        frame_id = 0
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            returned_frame_id, polygons = detector.process_frame(frame, frame_id)
            
            # Visualize
            vis_frame = frame.copy()
            for poly in polygons:
                cv2.polylines(vis_frame, [poly], True, (0, 255, 0), 2)
                if len(poly) > 0:
                    cv2.putText(vis_frame, "PII", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(vis_frame, f"Frame: {frame_id}, PII regions: {len(polygons)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("PII Detector Test", vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_id += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("PII detector test completed")

    if __name__ == "__main__":
        test_pii_detector()
        
except ImportError as e:
    print(f"PII detector not available: {e}")
    print("Make sure required packages are installed: pip install torch doctr easyocr scikit-learn")
