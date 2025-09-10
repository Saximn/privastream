"""
Unified bounding box test for all three detection models.
Tests face detection, PII detection, and plate detection simultaneously,
visualizing all bounding boxes/polygons with different colors.
"""
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

# Add model directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "face_blur"))
sys.path.append(str(current_dir / "pii_blur"))
sys.path.append(str(current_dir / "plate_blur"))

# Import models with error handling
models_available = {}

try:
    from face_detector import FaceDetector
    models_available['face'] = True
    print("[INFO] Face detector available")
except ImportError as e:
    models_available['face'] = False
    print(f"[WARN] Face detector not available: {e}")

try:
    from pii_detector import PIIDetector
    models_available['pii'] = True
    print("[INFO] PII detector available")
except ImportError as e:
    models_available['pii'] = False
    print(f"[WARN] PII detector not available: {e}")

try:
    from plate_detector import PlateDetector
    models_available['plate'] = True
    print("[INFO] Plate detector available")
except ImportError as e:
    models_available['plate'] = False
    print(f"[WARN] Plate detector not available: {e}")


class UnifiedBoundingBoxTester:
    """
    Unified tester for all three detection models.
    Visualizes detections with different colors and provides performance metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified tester.
        
        Args:
            config: Configuration dictionary for each model
        """
        self.config = config or {}
        self.models = {}
        self.colors = {
            'face': (0, 0, 255),    # Red for faces
            'pii': (0, 255, 0),     # Green for PII
            'plate': (255, 0, 0)    # Blue for plates
        }
        self.stats = {
            'face': {'frames': 0, 'detections': 0, 'total_time': 0},
            'pii': {'frames': 0, 'detections': 0, 'total_time': 0},
            'plate': {'frames': 0, 'detections': 0, 'total_time': 0}
        }
        
        self._init_models()
    
    def _init_models(self):
        """Initialize available detection models."""
        # Face detector
        if models_available.get('face', False):
            try:
                face_config = self.config.get('face', {})
                self.models['face'] = FaceDetector(
                    embed_path=face_config.get('embed_path', 'face_blur/whitelist/creator_embedding.json'),
                    gpu_id=face_config.get('gpu_id', 0),
                    det_size=face_config.get('det_size', 960),
                    threshold=face_config.get('threshold', 0.35),
                    dilate_px=face_config.get('dilate_px', 12),
                    smooth_ms=face_config.get('smooth_ms', 300)
                )
                print("[UnifiedTester] Face detector initialized")
            except Exception as e:
                print(f"[UnifiedTester][ERROR] Face detector initialization failed: {e}")
                models_available['face'] = False
        
        # PII detector
        if models_available.get('pii', False):
            try:
                pii_config = self.config.get('pii', {})
                self.models['pii'] = PIIDetector(
                    classifier_path=pii_config.get('classifier_path', 'pii_blur/pii_clf.joblib'),
                    conf_thresh=pii_config.get('conf_thresh', 0.35),
                    min_area=pii_config.get('min_area', 80),
                    K_confirm=pii_config.get('K_confirm', 2),
                    K_hold=pii_config.get('K_hold', 8)
                )
                print("[UnifiedTester] PII detector initialized")
            except Exception as e:
                print(f"[UnifiedTester][ERROR] PII detector initialization failed: {e}")
                models_available['pii'] = False
        
        # Plate detector
        if models_available.get('plate', False):
            try:
                plate_config = self.config.get('plate', {})
                self.models['plate'] = PlateDetector(
                    weights_path=plate_config.get('weights_path', 'plate_blur/best.pt'),
                    imgsz=plate_config.get('imgsz', 960),
                    conf_thresh=plate_config.get('conf_thresh', 0.25),
                    iou_thresh=plate_config.get('iou_thresh', 0.5),
                    pad=plate_config.get('pad', 4)
                )
                print("[UnifiedTester] Plate detector initialized")
            except Exception as e:
                print(f"[UnifiedTester][ERROR] Plate detector initialization failed: {e}")
                models_available['plate'] = False
        
        print(f"[UnifiedTester] Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def draw_rectangle(self, img: np.ndarray, rect: List[int], color: Tuple[int, int, int], 
                      label: str, confidence: float = 1.0, thickness: int = 2):
        """Draw a rectangle with label and confidence."""
        x1, y1, x2, y2 = rect
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if confidence < 1.0:
            text = f"{label} {confidence:.2f}"
        else:
            text = label
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), 
                     (x1 + text_w + 10, y1), color, -1)
        
        # Draw text
        cv2.putText(img, text, (x1 + 5, y1 - baseline - 5), 
                   font, font_scale, (255, 255, 255), text_thickness)
    
    def draw_polygon(self, img: np.ndarray, poly: np.ndarray, color: Tuple[int, int, int], 
                    label: str, thickness: int = 2):
        """Draw a polygon with label."""
        if len(poly) < 3:
            return
        
        # Draw polygon
        cv2.polylines(img, [poly], True, color, thickness)
        
        # Draw label at first point
        if len(poly) > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            x, y = poly[0]
            # Draw background rectangle for text
            cv2.rectangle(img, (x, y - text_h - baseline - 5), 
                         (x + text_w + 10, y), color, -1)
            
            # Draw text
            cv2.putText(img, label, (x + 5, y - baseline - 5), 
                       font, font_scale, (255, 255, 255), text_thickness)
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """
        Process a frame with all available models and return detection results.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            Dictionary containing results from all models with timing info
        """
        results = {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'models': {},
            'timing': {}
        }
        
        # Process with face detector
        if 'face' in self.models:
            try:
                start_time = time.time()
                face_frame_id, face_rectangles = self.models['face'].process_frame(frame, frame_id)
                process_time = time.time() - start_time
                
                results['models']['face'] = {
                    'frame_id': face_frame_id,
                    'rectangles': face_rectangles,
                    'count': len(face_rectangles),
                    'type': 'rectangles'
                }
                results['timing']['face'] = process_time
                
                # Update stats
                self.stats['face']['frames'] += 1
                self.stats['face']['detections'] += len(face_rectangles)
                self.stats['face']['total_time'] += process_time
                
            except Exception as e:
                print(f"[UnifiedTester][ERROR] Face detection failed: {e}")
                results['models']['face'] = {'error': str(e)}
        
        # Process with PII detector
        if 'pii' in self.models:
            try:
                start_time = time.time()
                pii_frame_id, pii_rectangles = self.models['pii'].process_frame(frame, frame_id)
                process_time = time.time() - start_time
                
                results['models']['pii'] = {
                    'frame_id': pii_frame_id,
                    'rectangles': pii_rectangles,
                    'count': len(pii_rectangles),
                    'type': 'rectangles'
                }
                results['timing']['pii'] = process_time
                
                # Update stats
                self.stats['pii']['frames'] += 1
                self.stats['pii']['detections'] += len(pii_rectangles)
                self.stats['pii']['total_time'] += process_time
                
            except Exception as e:
                print(f"[UnifiedTester][ERROR] PII detection failed: {e}")
                results['models']['pii'] = {'error': str(e)}
        
        # Process with plate detector
        if 'plate' in self.models:
            try:
                start_time = time.time()
                plate_frame_id, plate_data = self.models['plate'].process_frame_with_metadata(frame, frame_id)
                process_time = time.time() - start_time
                
                results['models']['plate'] = {
                    'frame_id': plate_frame_id,
                    'detection_data': plate_data,
                    'count': len(plate_data),
                    'type': 'rectangles_with_confidence'
                }
                results['timing']['plate'] = process_time
                
                # Update stats
                self.stats['plate']['frames'] += 1
                self.stats['plate']['detections'] += len(plate_data)
                self.stats['plate']['total_time'] += process_time
                
            except Exception as e:
                print(f"[UnifiedTester][ERROR] Plate detection failed: {e}")
                results['models']['plate'] = {'error': str(e)}
        
        return results
    
    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Visualize detection results on the frame.
        
        Args:
            frame: Input frame
            results: Detection results from process_frame
            
        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        # Draw face rectangles (red)
        face_data = results.get('models', {}).get('face', {})
        if 'rectangles' in face_data:
            for i, rect in enumerate(face_data['rectangles']):
                self.draw_rectangle(vis_frame, rect, self.colors['face'], 
                                  f"FACE_{i+1}", thickness=2)
        
        # Draw PII polygons (green)
        pii_data = results.get('models', {}).get('pii', {})
        if 'rectangles' in pii_data:
            for i, rect in enumerate(pii_data['rectangles']):
                self.draw_rectangle(vis_frame, rect, self.colors['pii'], 
                                f"PII_{i+1}", thickness=2)
        
        # Draw plate rectangles (blue) with confidence
        plate_data = results.get('models', {}).get('plate', {})
        if 'detection_data' in plate_data:
            for i, data in enumerate(plate_data['detection_data']):
                rect = data['rectangle']
                conf = data['confidence']
                self.draw_rectangle(vis_frame, rect, self.colors['plate'], 
                                  f"PLATE_{i+1}", conf, thickness=2)
        
        return vis_frame
    
    def add_info_overlay(self, frame: np.ndarray, results: Dict[str, Any], 
                        target_fps: float = 0.0, display_fps: float = 0.0) -> np.ndarray:
        """Add information overlay to the frame."""
        info_frame = frame.copy()
        
        # Count total detections
        total_detections = 0
        face_count = results.get('models', {}).get('face', {}).get('count', 0)
        pii_count = results.get('models', {}).get('pii', {}).get('count', 0) 
        plate_count = results.get('models', {}).get('plate', {}).get('count', 0)
        total_detections = face_count + pii_count + plate_count
        
        # Get timing info
        timing = results.get('timing', {})
        face_time = timing.get('face', 0) * 1000  # Convert to ms
        pii_time = timing.get('pii', 0) * 1000
        plate_time = timing.get('plate', 0) * 1000
        total_time = sum(timing.values()) * 1000
        
        # Prepare info text
        frame_id = results.get('frame_id', 0)
        
        if target_fps > 0:
            # Fixed FPS mode
            info_lines = [
                f"Frame: {frame_id} | Target: {target_fps} FPS",
                f"Display: {display_fps:.1f} FPS",
                f"Total Detections: {total_detections}",
                f"Face: {face_count} ({face_time:.1f}ms)",
                f"PII: {pii_count} ({pii_time:.1f}ms)", 
                f"Plate: {plate_count} ({plate_time:.1f}ms)",
                f"Processing: {total_time:.1f}ms"
            ]
        else:
            # Original variable FPS mode
            info_lines = [
                f"Frame: {frame_id} | FPS: {display_fps:.1f}",
                f"Total Detections: {total_detections}",
                f"Face: {face_count} ({face_time:.1f}ms)",
                f"PII: {pii_count} ({pii_time:.1f}ms)", 
                f"Plate: {plate_count} ({plate_time:.1f}ms)",
                f"Total Time: {total_time:.1f}ms"
            ]
        
        # Draw info background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_height = 25
        margin = 10
        
        # Calculate background size
        max_width = 0
        for line in info_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_w)
        
        bg_height = len(info_lines) * line_height + margin * 2
        bg_width = max_width + margin * 2
        
        # Draw semi-transparent background
        overlay = info_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + bg_width, 10 + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, info_frame, 0.3, 0, info_frame)
        
        # Draw info text
        y_offset = 35
        for line in info_lines:
            cv2.putText(info_frame, line, (20, y_offset), font, font_scale, (255, 255, 255), thickness)
            y_offset += line_height
        
        return info_frame
    
    def print_statistics(self):
        """Print performance statistics."""
        print("\n" + "="*60)
        print("UNIFIED DETECTION STATISTICS")
        print("="*60)
        
        for model_name in ['face', 'pii', 'plate']:
            stats = self.stats[model_name]
            if stats['frames'] > 0:
                avg_time = (stats['total_time'] / stats['frames']) * 1000
                avg_detections = stats['detections'] / stats['frames']
                fps = stats['frames'] / stats['total_time'] if stats['total_time'] > 0 else 0
                
                print(f"{model_name.upper()} DETECTOR:")
                print(f"  Processed frames: {stats['frames']}")
                print(f"  Total detections: {stats['detections']}")
                print(f"  Avg detections/frame: {avg_detections:.2f}")
                print(f"  Avg processing time: {avg_time:.1f}ms")
                print(f"  Model FPS: {fps:.1f}")
                print()
        
        # Overall stats
        total_frames = max(stats['frames'] for stats in self.stats.values())
        if total_frames > 0:
            total_time = sum(stats['total_time'] for stats in self.stats.values())
            total_detections = sum(stats['detections'] for stats in self.stats.values())
            
            print("OVERALL:")
            print(f"  Total frames processed: {total_frames}")
            print(f"  Total detections: {total_detections}")
            print(f"  Combined processing time: {total_time:.2f}s")
            print(f"  Overall FPS: {total_frames/total_time:.1f}")
        
        print("="*60)
    
    def run_webcam_test(self, camera_id: int = 0, width: int = 1280, height: int = 720, target_fps: float = 3.0):
        """Run unified test with webcam at fixed frame rate."""
        print(f"\n[UnifiedTester] Starting webcam test (camera {camera_id})")
        print(f"Target processing rate: {target_fps} FPS")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame and results")
        print("  'p' - Toggle face detector panic mode")
        print("  'r' - Reload face detector embedding")
        print("  'i' - Print current statistics")
        print("  '+' - Increase target FPS by 0.5")
        print("  '-' - Decrease target FPS by 0.5")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        frame_id = 0
        last_process_time = time.time()
        frame_interval = 1.0 / target_fps  # Time between processing frames
        
        # For display FPS calculation
        last_display_time = time.time()
        display_fps = 0.0
        
        print(f"[UnifiedTester] Webcam test started. Processing every {frame_interval:.3f}s. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break
                
                current_time = time.time()
                
                # Check if it's time to process a new frame
                if current_time - last_process_time >= frame_interval:
                    # Calculate display FPS (how often we show frames)
                    display_fps = 1.0 / max(current_time - last_display_time, 0.001)
                    last_display_time = current_time
                    
                    # Process frame with all models
                    results = self.process_frame(frame, frame_id)
                    
                    # Visualize results
                    vis_frame = self.visualize_results(frame, results)
                    vis_frame = self.add_info_overlay(vis_frame, results, target_fps, display_fps)
                    
                    last_process_time = current_time
                    frame_id += 1
                else:
                    # Use the last processed frame or current frame without processing
                    if 'vis_frame' not in locals():
                        vis_frame = frame.copy()
                        # Add simple overlay for unprocessed frames
                        cv2.putText(vis_frame, f"Waiting for next processing cycle...", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(vis_frame, f"Target: {target_fps} FPS | Display: {display_fps:.1f} FPS", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Show frame
                cv2.imshow("Unified Bounding Box Test - Fixed FPS", vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and 'results' in locals():
                    # Save current frame and results
                    timestamp = int(time.time())
                    frame_filename = f"unified_test_frame_{timestamp}.jpg"
                    results_filename = f"unified_test_results_{timestamp}.txt"
                    
                    cv2.imwrite(frame_filename, vis_frame)
                    
                    with open(results_filename, 'w') as f:
                        f.write(f"Unified Detection Results - Frame {frame_id}\n")
                        f.write(f"Timestamp: {results['timestamp']}\n")
                        f.write(f"Target FPS: {target_fps}\n")
                        f.write(f"Display FPS: {display_fps:.1f}\n\n")
                        
                        for model_name, model_results in results.get('models', {}).items():
                            f.write(f"{model_name.upper()} RESULTS:\n")
                            if 'error' in model_results:
                                f.write(f"  Error: {model_results['error']}\n")
                            else:
                                f.write(f"  Count: {model_results.get('count', 0)}\n")
                                f.write(f"  Type: {model_results.get('type', 'unknown')}\n")
                                if model_name in results.get('timing', {}):
                                    f.write(f"  Processing time: {results['timing'][model_name]*1000:.1f}ms\n")
                            f.write("\n")
                    
                    print(f"[UnifiedTester] Saved frame: {frame_filename}")
                    print(f"[UnifiedTester] Saved results: {results_filename}")
                    
                elif key == ord('p') and 'face' in self.models:
                    # Toggle face detector panic mode
                    current_panic = self.models['face'].panic_mode
                    self.models['face'].set_panic_mode(not current_panic)
                    
                elif key == ord('r') and 'face' in self.models:
                    # Reload face detector embedding
                    self.models['face'].reload_embedding()
                    
                elif key == ord('i'):
                    # Print current statistics
                    self.print_statistics()
                    
                elif key == ord('+') or key == ord('='):
                    # Increase target FPS
                    target_fps = min(10.0, target_fps + 0.5)
                    frame_interval = 1.0 / target_fps
                    print(f"[UnifiedTester] Target FPS increased to {target_fps}")
                    
                elif key == ord('-') or key == ord('_'):
                    # Decrease target FPS
                    target_fps = max(0.5, target_fps - 0.5)
                    frame_interval = 1.0 / target_fps
                    print(f"[UnifiedTester] Target FPS decreased to {target_fps}")
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[UnifiedTester] Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_statistics()
            print("[UnifiedTester] Webcam test completed")
    
    def run_image_test(self, image_path: str):
        """Run unified test with a single image."""
        print(f"[UnifiedTester] Testing with image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot load image: {image_path}")
            return
        
        print(f"[UnifiedTester] Image loaded: {frame.shape}")
        
        # Process image
        results = self.process_frame(frame, 0)
        
        # Visualize results
        vis_frame = self.visualize_results(frame, results)
        vis_frame = self.add_info_overlay(vis_frame, results, 0.0)  # No FPS for single image
        
        # Display results
        print("\nDetection Results:")
        for model_name, model_results in results.get('models', {}).items():
            if 'error' in model_results:
                print(f"  {model_name.upper()}: Error - {model_results['error']}")
            else:
                count = model_results.get('count', 0)
                timing = results.get('timing', {}).get(model_name, 0) * 1000
                print(f"  {model_name.upper()}: {count} detections ({timing:.1f}ms)")
        
        # Show image
        cv2.imshow("Unified Detection Test - Image", vis_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Unified Bounding Box Test for All Detection Models")
    parser.add_argument('--mode', choices=['webcam', 'image'], default='webcam',
                       help='Test mode: webcam or single image')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam mode')
    parser.add_argument('--image', type=str,
                       help='Path to image file for image mode')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width for webcam mode')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height for webcam mode')
    parser.add_argument('--fps', type=float, default=3.0,
                       help='Target processing FPS (default: 3.0)')
    parser.add_argument('--config', type=str,
                       help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"[UnifiedTester] Loaded config from {args.config}")
        except Exception as e:
            print(f"[WARN] Failed to load config: {e}")
    
    # Initialize tester
    print("[UnifiedTester] Initializing unified bounding box tester...")
    tester = UnifiedBoundingBoxTester(config)
    
    if len(tester.models) == 0:
        print("[ERROR] No models available for testing!")
        print("Make sure you have the required dependencies installed:")
        print("  Face: pip install insightface")
        print("  PII: pip install torch doctr easyocr scikit-learn")
        print("  Plate: pip install torch ultralytics")
        return
    
    # Run appropriate test
    if args.mode == 'webcam':
        tester.run_webcam_test(args.camera, args.width, args.height, args.fps)
    else:  # image mode
        if not args.image:
            print("[ERROR] Image path required for image mode. Use --image <path>")
            return
        tester.run_image_test(args.image)


if __name__ == "__main__":
    main()
