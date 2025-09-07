"""
Quick test script to verify the unified bounding box tester setup.
Creates a synthetic test image and runs all detectors on it.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from unified_bbox_test import UnifiedBoundingBoxTester
    
    def create_test_image():
        """Create a synthetic test image with various elements."""
        # Create a 800x600 image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 128  # Gray background
        
        # Add some colored rectangles to simulate objects
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)   # Blue rectangle
        cv2.rectangle(img, (200, 100), (300, 200), (0, 255, 0), -1) # Green rectangle
        cv2.rectangle(img, (400, 150), (500, 250), (0, 0, 255), -1) # Red rectangle
        
        # Add some text that might be detected as PII
        cv2.putText(img, "123 Main Street", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "ABC-1234", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Phone: 555-1234", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add some shapes that might look like plates
        cv2.rectangle(img, (550, 300), (650, 350), (255, 255, 255), -1)  # White rectangle
        cv2.putText(img, "ABC123", (560, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return img
    
    def quick_test():
        """Run a quick test of the unified detector."""
        print("="*60)
        print("UNIFIED BOUNDING BOX TESTER - QUICK TEST")
        print("="*60)
        
        # Create test configuration
        config = {
            'face': {
                'embed_path': 'face_blur/whitelist/creator_embedding.json',
                'threshold': 0.35
            },
            'pii': {
                'classifier_path': 'pii_blur/pii_clf.joblib',
                'conf_thresh': 0.3  # Lower threshold for test
            },
            'plate': {
                'weights_path': 'plate_blur/best.pt',
                'conf_thresh': 0.2  # Lower threshold for test
            }
        }
        
        # Initialize tester
        print("Initializing unified tester...")
        tester = UnifiedBoundingBoxTester(config)
        
        if len(tester.models) == 0:
            print("[ERROR] No models available!")
            print("This may be normal if dependencies are not installed.")
            print("Available models:", list(tester.models.keys()))
            return
        
        print(f"Successfully initialized {len(tester.models)} models: {list(tester.models.keys())}")
        
        # Create test image
        print("\nCreating synthetic test image...")
        test_image = create_test_image()
        
        # Save test image
        cv2.imwrite("test_image_synthetic.jpg", test_image)
        print("Saved test image: test_image_synthetic.jpg")
        
        # Process test image
        print("\nProcessing test image...")
        results = tester.process_frame(test_image, 0)
        
        # Print results
        print("\nDetection Results:")
        print("-" * 40)
        total_detections = 0
        
        for model_name, model_results in results.get('models', {}).items():
            if 'error' in model_results:
                print(f"{model_name.upper()}: ERROR - {model_results['error']}")
            else:
                count = model_results.get('count', 0)
                timing = results.get('timing', {}).get(model_name, 0) * 1000
                total_detections += count
                print(f"{model_name.upper()}: {count} detections ({timing:.1f}ms)")
        
        print(f"\nTOTAL DETECTIONS: {total_detections}")
        
        # Visualize results
        print("\nGenerating visualization...")
        vis_image = tester.visualize_results(test_image, results)
        vis_image = tester.add_info_overlay(vis_image, results, 0.0)
        
        # Save visualization
        cv2.imwrite("test_results_visualization.jpg", vis_image)
        print("Saved visualization: test_results_visualization.jpg")
        
        # Show results if possible
        try:
            print("\nDisplaying results (close window to continue)...")
            cv2.imshow("Unified Test Results", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Could not display image: {e}")
        
        # Print statistics
        tester.print_statistics()
        
        print("\n" + "="*60)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nTo run the full interactive test, use:")
        print("  python unified_bbox_test.py --mode webcam")
        print("  python unified_bbox_test.py --mode image --image <path>")
        print("\nOr run the batch file:")
        print("  run_unified_test.bat")
    
    if __name__ == "__main__":
        quick_test()

except ImportError as e:
    print(f"Error importing unified tester: {e}")
    print("\nMake sure the required dependencies are installed:")
    print("  Face detection: pip install insightface")
    print("  PII detection: pip install torch doctr easyocr scikit-learn") 
    print("  Plate detection: pip install torch ultralytics")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
