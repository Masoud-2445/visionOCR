import cv2
import numpy as np
from typing import Tuple, Optional, List

def detect_document_contour(image: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Enhanced document contour detection with border reduction strategies.
    
    Args:
        image: Input BGR image
        debug: If True, returns debug visualization
        
    Returns:
        Tuple of (corners, debug_image) where corners is (4,2) array of document corners
    """
    try:
        debug_image = image.copy() if debug else None
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing to better separate document from dark background
        gray_filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Calculate adaptive parameters based on image size
        min_kernel_size = max(5, min(width, height) // 100)
        max_kernel_size = max(15, min(width, height) // 50)
        
        methods = []
        
        # Method 1: Enhanced black-hat transform with better thresholding
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_kernel_size, min_kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        
        # More aggressive thresholding for black-hat to avoid dark areas
        blackhat_mean = np.mean(blackhat)
        blackhat_std = np.std(blackhat)
        thresh_val = max(10, min(40, blackhat_mean + 1.5 * blackhat_std))  # Increased threshold
        
        _, thresh_blackhat = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)
        methods.append(('blackhat', thresh_blackhat))
        
        # Method 2: Document-focused adaptive thresholding
        # Use mean thresholding which works better for documents on dark backgrounds
        thresh_adapt = cv2.adaptiveThreshold(
            gray_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 15, 10  # Increased block size and C value
        )
        
        # Apply morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel)
        thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel)
        
        edges_adapt = cv2.Canny(thresh_adapt, 50, 150, apertureSize=3)
        methods.append(('adaptive', edges_adapt))
        
        # Method 3: Otsu with morphological operations
        _, thresh_otsu = cv2.threshold(gray_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
        
        edges_otsu = cv2.Canny(thresh_otsu, 50, 150, apertureSize=3)
        methods.append(('otsu', edges_otsu))
        
        # Method 4: Gradient-based edge detection
        grad_x = cv2.Sobel(gray_filtered, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_filtered, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = np.uint8(gradient / gradient.max() * 255)
        
        _, thresh_grad = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(('gradient', thresh_grad))
        
        # Method 5: Hough line detection for rectangular documents
        hough_edges = detect_document_by_hough_lines(gray_filtered, width, height)
        if hough_edges is not None:
            methods.append(('hough', hough_edges))
        
        # Find best contour across all methods with improved filtering
        candidates = []
        min_area_threshold = 0.02 * width * height  # Increased minimum area
        max_area_threshold = 0.85 * width * height  # Decreased maximum area
        
        for method_name, edges in methods:
            # Reduced dilation to avoid including background
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:5]:
                area = cv2.contourArea(contour)
                if area < min_area_threshold or area > max_area_threshold:
                    continue
                
                # More conservative approximation tolerances
                for tolerance in [0.015, 0.02, 0.01, 0.025]:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, tolerance * peri, True)
                    
                    if len(approx) == 4:
                        corners = approx.reshape(4, 2)
                        quality = evaluate_contour_quality(corners, width, height)
                        
                        # Additional check for dark background rejection
                        bg_score = check_background_brightness(image, corners)
                        combined_quality = quality * bg_score
                        
                        candidates.append((corners, area, combined_quality, method_name))
        
        # Select best candidate and apply border reduction
        if candidates:
            best_corners = select_best_contour(candidates, width, height)
            if best_corners is not None:
                # Apply border reduction to move corners inward
                best_corners = reduce_border_effect(best_corners, width, height)
        else:
            print("No valid contours found with any method")
            # Fallback with small margin
            margin = min(width, height) * 0.02
            best_corners = np.array([
                [margin, margin], 
                [width - margin, margin], 
                [width - margin, height - margin], 
                [margin, height - margin]
            ], dtype=np.float32)
        
        if best_corners is not None:
            best_corners = order_points(best_corners)
            
            if debug:
                # Enhanced debug visualization
                cv2.drawContours(debug_image, [best_corners.astype(int)], -1, (0, 255, 0), 3)
                for i, (x, y) in enumerate(best_corners):
                    cv2.circle(debug_image, (int(x), int(y)), 8, (0, 0, 255), -1)
                    cv2.putText(debug_image, str(i), (int(x+10), int(y+10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return best_corners, debug_image
        
    except Exception as e:
        print(f"Error during document contour detection: {e}")
        return None, None

def check_background_brightness(image: np.ndarray, corners: np.ndarray) -> float:
    """
    Check if the detected area is significantly brighter than the background.
    Returns a score between 0 and 1, where 1 means good document detection.
    """
    try:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create masks for inside and outside the detected area
        mask_inside = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask_inside, [corners.astype(int)], 255)
        
        # Sample background from corners and edges
        mask_outside = np.ones((height, width), dtype=np.uint8) * 255
        cv2.fillPoly(mask_outside, [corners.astype(int)], 0)
        
        # Calculate mean brightness inside and outside
        inside_mean = cv2.mean(gray, mask_inside)[0]
        outside_mean = cv2.mean(gray, mask_outside)[0]
        
        # If inside is significantly brighter than outside, it's likely a document
        brightness_ratio = inside_mean / (outside_mean + 1e-6)
        
        # Score based on brightness difference
        if brightness_ratio > 1.5:  # Document should be brighter than background
            return 1.0
        elif brightness_ratio > 1.2:
            return 0.8
        elif brightness_ratio > 1.0:
            return 0.6
        else:
            return 0.3
            
    except Exception as e:
        return 0.5

def reduce_border_effect(corners: np.ndarray, width: int, height: int, reduction_factor: float = 0.98) -> np.ndarray:
    """
    Reduce the detected contour slightly to avoid including background borders.
    """
    try:
        # Calculate the center of the detected rectangle
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        center = np.array([center_x, center_y])
        
        # Move each corner slightly toward the center
        reduced_corners = np.zeros_like(corners)
        for i, corner in enumerate(corners):
            direction = corner - center
            reduced_corners[i] = center + direction * reduction_factor
            
        # Ensure corners stay within image bounds
        reduced_corners[:, 0] = np.clip(reduced_corners[:, 0], 0, width - 1)
        reduced_corners[:, 1] = np.clip(reduced_corners[:, 1], 0, height - 1)
        
        return reduced_corners.astype(np.float32)
        
    except Exception as e:
        print(f"Error reducing border effect: {e}")
        return corners

def detect_document_by_hough_lines(gray: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
    """
    Detect document edges using Hough line detection with improved filtering.
    """
    try:
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Use more conservative parameters for line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                               minLineLength=min(width, height)//6, maxLineGap=30)
        
        if lines is None or len(lines) < 4:
            return None
        
        # Group lines by orientation with tighter tolerances
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 10 or abs(angle) > 170:  # Horizontal (tighter tolerance)
                horizontal_lines.append(line[0])
            elif abs(angle - 90) < 10 or abs(angle + 90) < 10:  # Vertical (tighter tolerance)
                vertical_lines.append(line[0])
        
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Create binary image from detected lines with thinner lines
            line_image = np.zeros_like(gray)
            for line in horizontal_lines + vertical_lines:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)  # Thinner lines
            
            return line_image
        
        return None
        
    except Exception as e:
        print(f"Error in Hough line detection: {e}")
        return None

def evaluate_contour_quality(corners: np.ndarray, width: int, height: int) -> float:
    """
    Evaluate the quality of detected document corners with improved metrics.
    """
    try:
        # Check if corners form a reasonable rectangle
        ordered_corners = order_points(corners)
        
        # Calculate side lengths
        side_lengths = []
        for i in range(4):
            p1 = ordered_corners[i]
            p2 = ordered_corners[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            side_lengths.append(length)
        
        # Check aspect ratio reasonableness (not too extreme)
        width_ratio = side_lengths[0] / side_lengths[1] if side_lengths[1] > 0 else 0
        height_ratio = side_lengths[1] / side_lengths[0] if side_lengths[0] > 0 else 0
        aspect_score = min(width_ratio, height_ratio)
        
        # Penalize extremely elongated rectangles
        if aspect_score < 0.1:
            aspect_score *= 0.5
        
        # Check corner angles (should be close to 90 degrees)
        angle_score = calculate_angle_score(ordered_corners)
        
        # Check if corners are within image bounds with tighter margins
        bound_score = calculate_boundary_score(ordered_corners, width, height)
        
        # Check if the rectangle is reasonably sized (not too small)
        area = cv2.contourArea(ordered_corners)
        size_score = min(1.0, area / (0.1 * width * height))
        
        # Overall quality score with adjusted weights
        quality = (aspect_score * 0.25 + angle_score * 0.35 + bound_score * 0.25 + size_score * 0.15)
        
        return quality
        
    except Exception as e:
        print(f"Error evaluating contour quality: {e}")
        return 0.0

def calculate_angle_score(corners: np.ndarray) -> float:
    """Calculate how close the corners are to 90-degree angles with improved scoring."""
    try:
        angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue
                
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(abs(angle - 90))
        
        if not angles:
            return 0.0
        
        # Score based on how close to 90 degrees with stricter requirements
        avg_deviation = np.mean(angles)
        max_deviation = max(angles)
        
        # Penalize if any angle is too far from 90 degrees
        if max_deviation > 30:
            return 0.0
        
        return max(0, 1 - avg_deviation / 30)  # Stricter tolerance
        
    except Exception as e:
        return 0.0

def calculate_boundary_score(corners: np.ndarray, width: int, height: int) -> float:
    """Calculate how well corners fit within image boundaries with tighter margins."""
    try:
        margin = 0.02  # Reduced to 2% margin
        score = 1.0
        
        for x, y in corners:
            if x < -margin * width or x > width * (1 + margin):
                score *= 0.3  # Harsher penalty
            if y < -margin * height or y > height * (1 + margin):
                score *= 0.3  # Harsher penalty
                
        return score
        
    except Exception as e:
        return 0.0

def select_best_contour(candidates: List, width: int, height: int) -> np.ndarray:
    """Select the best contour from candidates with improved selection criteria."""
    try:
        # Sort by quality score first, then by area
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        best_corners, best_area, best_quality, best_method = candidates[0]
        print(f"Selected contour from method: {best_method}, Quality: {best_quality:.3f}")
        
        return best_corners
        
    except Exception as e:
        print(f"Error selecting best contour: {e}")
        return None

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders points as: top-left, top-right, bottom-right, bottom-left.
    """
    try:
        if pts.dtype != np.float32:
            pts = pts.astype(np.float32)
            
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect
        
    except Exception as e:
        print(f"Error ordering points: {e}")
        return np.zeros((4, 2), dtype=np.float32)
