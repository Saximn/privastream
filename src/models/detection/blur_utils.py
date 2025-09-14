"""
Utility functions for applying blur to detected regions.
Supports different blur types and handles both rectangles and polygons.
"""
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional


def clamp(v: float, lo: int, hi: int) -> int:
    """Clamp value between bounds."""
    return max(lo, min(hi, int(v)))


def blur_rectangle(img: np.ndarray, 
                   rect: List[int], 
                   blur_type: str = "gaussian",
                   kernel_size: int = 35,
                   pixel_size: int = 16,
                   fill_color: Tuple[int, int, int] = (0, 0, 0),
                   pad: int = 0) -> np.ndarray:
    """
    Apply blur to a rectangular region.
    
    Args:
        img: Input image (BGR format)
        rect: Rectangle as [x1, y1, x2, y2]
        blur_type: Type of blur ("gaussian", "pixelate", "fill")
        kernel_size: Gaussian blur kernel size (must be odd)
        pixel_size: Pixelation block size
        fill_color: Fill color for "fill" mode
        pad: Additional padding around rectangle
    
    Returns:
        Image with blurred region
    """
    if len(rect) != 4:
        return img
    
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    
    # Apply padding
    x1 = clamp(x1 - pad, 0, w - 1)
    y1 = clamp(y1 - pad, 0, h - 1)  
    x2 = clamp(x2 + pad, 0, w - 1)
    y2 = clamp(y2 + pad, 0, h - 1)
    
    if x2 <= x1 or y2 <= y1:
        return img
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    
    if blur_type == "gaussian":
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        img[y1:y2, x1:x2] = blurred_roi
        
    elif blur_type == "pixelate":
        roi_h, roi_w = roi.shape[:2]
        if roi_h > 0 and roi_w > 0:
            # Downscale
            small_h = max(1, roi_h // pixel_size)
            small_w = max(1, roi_w // pixel_size)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            # Upscale with nearest neighbor for pixelated effect
            pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            img[y1:y2, x1:x2] = pixelated
            
    elif blur_type == "fill":
        img[y1:y2, x1:x2] = fill_color
    
    return img


def blur_polygon(img: np.ndarray,
                 poly: np.ndarray,
                 blur_type: str = "gaussian", 
                 kernel_size: int = 35,
                 pixel_size: int = 16,
                 fill_color: Tuple[int, int, int] = (0, 0, 0),
                 pad: int = 3) -> np.ndarray:
    """
    Apply blur to a polygonal region.
    
    Args:
        img: Input image (BGR format)
        poly: Polygon as numpy array of points [[x1,y1], [x2,y2], ...]
        blur_type: Type of blur ("gaussian", "pixelate", "fill")
        kernel_size: Gaussian blur kernel size (must be odd)
        pixel_size: Pixelation block size
        fill_color: Fill color for "fill" mode
        pad: Additional padding around polygon bounding box
    
    Returns:
        Image with blurred region
    """
    if len(poly) < 3:  # Need at least 3 points for a polygon
        return img
    
    # Get bounding box of polygon
    xs, ys = poly[:, 0], poly[:, 1]
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    
    # Apply padding
    h, w = img.shape[:2]
    x1 = clamp(x1 - pad, 0, w - 1)
    y1 = clamp(y1 - pad, 0, h - 1)
    x2 = clamp(x2 + pad, 0, w - 1)
    y2 = clamp(y2 + pad, 0, h - 1)
    
    if x2 <= x1 or y2 <= y1:
        return img
    
    # Create mask for the polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    
    # Extract ROI
    roi = img[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]
    
    if roi.size == 0:
        return img
    
    if blur_type == "gaussian":
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        # Apply only to masked area
        roi_mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        roi_mask_norm = roi_mask_3ch.astype(np.float32) / 255.0
        result_roi = roi * (1 - roi_mask_norm) + blurred_roi * roi_mask_norm
        img[y1:y2, x1:x2] = result_roi.astype(np.uint8)
        
    elif blur_type == "pixelate":
        roi_h, roi_w = roi.shape[:2]
        if roi_h > 0 and roi_w > 0:
            # Downscale
            small_h = max(1, roi_h // pixel_size)
            small_w = max(1, roi_w // pixel_size)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            # Upscale with nearest neighbor
            pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            # Apply only to masked area
            roi_mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            roi_mask_norm = roi_mask_3ch.astype(np.float32) / 255.0
            result_roi = roi * (1 - roi_mask_norm) + pixelated * roi_mask_norm
            img[y1:y2, x1:x2] = result_roi.astype(np.uint8)
            
    elif blur_type == "fill":
        # Fill only the masked area
        img[mask == 255] = fill_color
    
    return img


def apply_blur_regions(img: np.ndarray,
                       rectangles: Optional[List[List[int]]] = None,
                       polygons: Optional[List[np.ndarray]] = None,
                       blur_type: str = "gaussian",
                       kernel_size: int = 35,
                       pixel_size: int = 16,
                       fill_color: Tuple[int, int, int] = (0, 0, 0),
                       rect_pad: int = 0,
                       poly_pad: int = 3) -> np.ndarray:
    """
    Apply blur to multiple regions (rectangles and polygons).
    
    Args:
        img: Input image (BGR format)
        rectangles: List of rectangles [[x1,y1,x2,y2], ...]
        polygons: List of polygons [poly1, poly2, ...] where each poly is np.ndarray
        blur_type: Type of blur ("gaussian", "pixelate", "fill")
        kernel_size: Gaussian blur kernel size
        pixel_size: Pixelation block size  
        fill_color: Fill color for "fill" mode
        rect_pad: Padding for rectangles
        poly_pad: Padding for polygons
    
    Returns:
        Image with all regions blurred
    """
    result = img.copy()
    
    # Apply rectangle blurs
    if rectangles:
        for rect in rectangles:
            result = blur_rectangle(
                result, rect, blur_type, kernel_size, pixel_size, fill_color, rect_pad
            )
    
    # Apply polygon blurs
    if polygons:
        for poly in polygons:
            result = blur_polygon(
                result, poly, blur_type, kernel_size, pixel_size, fill_color, poly_pad
            )
    
    return result


def demo_blur_utility():
    """Demonstrate the blur utility functions."""
    # Create a test image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to see the blur effect
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue
    cv2.rectangle(img, (300, 150), (400, 250), (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (450, 200), (550, 300), (0, 0, 255), -1)  # Red
    
    # Define test regions
    rectangles = [
        [100, 100, 200, 200],  # Blue rectangle
        [450, 200, 550, 300]   # Red rectangle
    ]
    
    polygons = [
        np.array([[300, 150], [400, 150], [400, 250], [300, 250]], dtype=np.int32)  # Green rectangle as polygon
    ]
    
    print("Testing blur utility...")
    print("Original image created with colored rectangles")
    
    # Test different blur types
    blur_types = ["gaussian", "pixelate", "fill"]
    
    for blur_type in blur_types:
        print(f"Testing {blur_type} blur...")
        
        blurred = apply_blur_regions(
            img.copy(),
            rectangles=rectangles,
            polygons=polygons,
            blur_type=blur_type,
            kernel_size=31,
            pixel_size=12,
            fill_color=(128, 128, 128)  # Gray
        )
        
        # Show result
        cv2.imshow(f"Blur Test - {blur_type.title()}", blurred)
        cv2.waitKey(2000)  # Show for 2 seconds
    
    cv2.destroyAllWindows()
    print("Blur utility demo completed")


if __name__ == "__main__":
    demo_blur_utility()
