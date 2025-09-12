import cv2
import time
from plate_detector import PlateDetector  # your class
import numpy as np

# Initialize detector
detector = PlateDetector(weights_path="best.engine", imgsz=640)

# --- Choose one: image or video ---
# Option 1: Test with a single image

frame = cv2.imread("images (1).jpeg")
if frame is not None:
    start = time.perf_counter()
    frame_id, rectangles = detector.process_frame(frame, frame_id=0)
    end = time.perf_counter()
    print(f"[Image] Frame ID: {frame_id}")
    print(f"Detected rectangles: {rectangles}")
    print(f"Processing time: {end - start:.4f} seconds")

    # Draw rectangles
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = "test_output.jpeg"
    cv2.imwrite(output_path, frame)
else:
    print("Image not found.")


