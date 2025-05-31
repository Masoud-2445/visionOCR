import cv2
import numpy as np

def detect_document_contour(image, debug=False):
    """
    Detects the contour of a document within the given image using multiple edge detection strategies.

    Args:
        image (np.ndarray): Input image in which to detect the document.
        debug (bool, optional): If True, returns an image with visual debug annotations. Defaults to False.

    Returns:
        tuple: 
            - best_corners (np.ndarray): Ordered coordinates of the detected document's corners.
            - debug_image (np.ndarray | None): Debug image with contour visualization if debug=True, otherwise None.
    """
    try:
        debug_image = image.copy() if debug else None
        height, width = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        methods = []

        thresh_adapt = cv2.adaptiveThreshold(
            gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        edges_adapt = cv2.Canny(thresh_adapt, 50, 150, apertureSize=3)
        methods.append(edges_adapt)
        
        _, thresh_otsu = cv2.threshold(gray_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges_otsu = cv2.Canny(thresh_otsu, 50, 150, apertureSize=3)
        methods.append(edges_otsu)
        
        edges_direct = cv2.Canny(gray_filtered, 30, 200, apertureSize=3)
        methods.append(edges_direct)
        
        edges_tight = cv2.Canny(gray_filtered, 70, 200, apertureSize=3)
        methods.append(edges_tight)
        
        max_area = 0
        best_corners = None
        min_area_threshold = 0.01 * width * height  
        max_area_threshold = 0.95 * width * height  

        for i, edges in enumerate(methods):
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:5]:
                area = cv2.contourArea(contour)

                if area < min_area_threshold or area > max_area_threshold:
                    continue
                    
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    best_corners = approx.reshape(4, 2)

            if best_corners is not None:
                break

        if best_corners is None:
            combined_edges = np.zeros_like(methods[0])
            for edges in methods:
                combined_edges = cv2.bitwise_or(combined_edges, edges)

            kernel = np.ones((7, 7), np.uint8)
            dilated = cv2.dilate(combined_edges, kernel, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > min_area_threshold:
                    hull = cv2.convexHull(largest_contour)
                    peri = cv2.arcLength(hull, True)
                    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
                    
                    if len(approx) > 4:
                        rect = cv2.minAreaRect(approx)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        best_corners = box
                    elif len(approx) == 4:
                        best_corners = approx.reshape(4, 2)
        
        if best_corners is None:
            best_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        
        if best_corners is not None:
            best_corners = order_points(best_corners)
            
            if debug:
                for i, (x, y) in enumerate(best_corners):
                    cv2.circle(debug_image, (int(x), int(y)), 10, (0, 255, 0), -1)
                
                cv2.line(debug_image, tuple(best_corners[0]), tuple(best_corners[1]), (0, 255, 0), 2)
                cv2.line(debug_image, tuple(best_corners[1]), tuple(best_corners[2]), (0, 255, 0), 2)
                cv2.line(debug_image, tuple(best_corners[2]), tuple(best_corners[3]), (0, 255, 0), 2)
                cv2.line(debug_image, tuple(best_corners[3]), tuple(best_corners[0]), (0, 255, 0), 2)

        return best_corners, debug_image

    except Exception as e:
        print("Error during document contour detection:")
        print(f"Exception: {e}")
        return None, None



def order_points(pts):
    """
    Orders a set of four points in the order: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (np.ndarray): Array of shape (4, 2) containing the coordinates of the points.

    Returns:
        np.ndarray: Array of shape (4, 2) with points ordered as [top-left, top-right, bottom-right, bottom-left].
    """
    try:
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect.astype(np.int32)

    except Exception as e:
        print("Error during ordering points:")
        print(f"Exception: {e}")
        return np.zeros((4, 2), dtype=np.int32)

    
















