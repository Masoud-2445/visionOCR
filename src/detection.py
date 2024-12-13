import cv2
import numpy as np


def contour_detection(image: np.ndarray, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE, minimum_area = 1000) -> list | None :
    """
    Detects the largest contour in the given edge-detected image using OpenCV.

    This function identifies contours from a binary or edge-detected image using OpenCV's
    `cv2.findContours`. It filters the detected contours based on the specified `minimum_area`
    and returns the largest valid contour.

    Args:
        image (np.ndarray): The input image (edge-detected or binary).
        mode (int, optional): Contour retrieval mode. Default is `cv2.RETR_EXTERNAL`,
            which retrieves only the outermost contours.
        method (int, optional): Contour approximation method. Default is `cv2.CHAIN_APPROX_SIMPLE`,
            which compresses horizontal, vertical, and diagonal segments.
        minimum_area (int, optional): Minimum contour area for a contour to be considered valid (default: 1000).

    Returns:
        list: The largest valid contour if found.
        None: If no valid contour is detected or an unexpected error occurs.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        ValueError: If `minimum_area` is less than or equal to zero.
        Exception: For unexpected runtime errors during contour detection.
    """
    try:
        if not isinstance(image, np.ndarray):
            raise TypeError("Error: Image must be a numpy array.")
        contours= cv2.findContours(image, mode, method)[0]
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > minimum_area]
        if not filtered_contours :
            print("No contours found")
            return None
        biggest_contour = max(filtered_contours, key=cv2.contourArea)
        return biggest_contour

    except Exception as e:
        print("Error: Unable to detect contours.")
        print(f"Error: {e}")
        return None


def paper_contour_detection(image: np.ndarray,) -> np.ndarray | None :
    """
    Detects the largest paper-like contour in an edge-detected image.

    This function detects the largest contour in the input image, approximates its shape,
    and checks if it forms a quadrilateral. If the largest contour has four vertices,
    it returns the approximated contour. If not, it falls back to returning a bounding
    rectangle around the largest contour.

    Args:
        image (np.ndarray): The input edge-detected or binary image as a NumPy array.

    Returns:
        np.ndarray: A NumPy array of points representing the detected paper contour
                    (either a quadrilateral or a bounding rectangle).
        None: If no valid contour is detected or if an unexpected error occurs.

    Raises:
        ValueError: If no contours are detected or if the detected contour is too small.
        TypeError: If the input image is not a valid NumPy array.
        Exception: For unexpected runtime errors during contour detection.
    """
    try :
        largest_contour = contour_detection(image, minimum_area = 1000)
        if largest_contour is None :
            raise ValueError("Error : No contours detected.")
        if cv2.contourArea(largest_contour) < 1000:
            raise ValueError("Error : contours are small compared to a paper.")
        epsilon = 0.05 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4 :
            return approx
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    except Exception as e:
        print(f"Error in detecting paper contour: {e}")
        return None