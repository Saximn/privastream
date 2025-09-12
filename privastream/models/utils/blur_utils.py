"""
Utility functions for applying blur effects to detected regions.
Moved from detection directory for better organization.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


def apply_blur_regions(frame: np.ndarray, 
                      rectangles: List[List[int]], 
                      polygons: List[np.ndarray],
                      kernel_size: int = 35) -> np.ndarray:
    """
    Apply Gaussian blur to specified regions in a frame.
    
    Args:
        frame: Input frame (BGR format)
        rectangles: List of rectangles [x1, y1, x2, y2] 
        polygons: List of polygon regions
        kernel_size: Gaussian blur kernel size
        
    Returns:
        Frame with blurred regions
    """
    result = frame.copy()
    
    # Apply blur to rectangular regions
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        if x2 > x1 and y2 > y1:  # Valid rectangle
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                result[y1:y2, x1:x2] = blurred_roi
    
    # Apply blur to polygon regions
    for poly in polygons:
        if len(poly) > 2:  # Valid polygon
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            
            # Apply blur only to the polygon region
            blurred = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
            result = np.where(mask[..., None], blurred, result)
    
    return result


def create_privacy_overlay(frame: np.ndarray, 
                          text: str = "PRIVACY FILTER ON",
                          position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Add privacy filter watermark to frame.
    
    Args:
        frame: Input frame
        text: Watermark text
        position: Text position (x, y)
        
    Returns:
        Frame with watermark
    """
    result = frame.copy()
    
    # Add semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, (position[0] - 5, position[1] - 25), 
                 (position[0] + len(text) * 12, position[1] + 5), 
                 (0, 0, 0), -1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    
    # Add text
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    return result