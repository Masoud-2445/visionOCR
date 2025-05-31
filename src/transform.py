import numpy as np
import cv2



def resize_image(image: np.ndarray, target_size: tuple, aspect_ratio= True, method= cv2.INTER_LINEAR) -> np.ndarray | None :
    """
    Resize an image to the given target size, optionally preserving the aspect ratio.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        target_size (tuple): Target dimensions as (width, height). Must contain two positive integers.
        aspect_ratio (bool, optional): Whether to preserve the original aspect ratio. Defaults to True.
        method (int, optional): Interpolation method to use (e.g., cv2.INTER_LINEAR). Defaults to cv2.INTER_LINEAR.

    Returns:
        np.ndarray | None: The resized image as a NumPy array, or None if an error occurs.

    Raises:
        TypeError: If the input image is not a NumPy array.
        Prints error messages for invalid input or resizing failure.
    """
    try :
        if not isinstance(image, np.ndarray) :
            raise TypeError('image must be numpy array')
        get_size = image.shape[:2]
        if not (isinstance(target_size, tuple) and len(target_size) == 2) and all(isinstance(i, int) and i > 0 for i in target_size) :
            print('Error : target size must be 2 dimensional and positive integer')
            return None

        if aspect_ratio :
            target_aspect_ratio = get_size[1] / get_size[0]
            if get_size[0] > get_size[1] :
                new_height = target_size[1]
                new_width = int(new_height * target_aspect_ratio)
            else :
                new_width = target_size[0]
                new_height = int(new_width / target_aspect_ratio)
            target_size = (new_width, new_height)
            resize_conversion = cv2.resize(image, target_size, interpolation= method)
            return resize_conversion

    except Exception as e :
        print('Error : unable to resize image')
        print(f'Error : {e}')
        return None



def deskew(image, delta=1, limit=5):
    """
    Corrects the skew of an image using the Hough Line Transform to detect lines and rotate accordingly.

    Args:
        image (np.ndarray): The input image (grayscale or BGR).
        delta (int, optional): Not used directly, reserved for future adjustments. Defaults to 1.
        limit (int, optional): Angle threshold to filter out extreme skew angles. Defaults to 5.

    Returns:
        np.ndarray: The deskewed image.
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if np.max(gray) > 1:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thresh = gray

        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi

            if abs(angle) < limit:
                angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    except Exception as e:
        print("Error during deskewing:")
        print(f"Exception: {e}")
        return image





def warp_perspective(image, corners):
    """
    Applies a perspective transformation to the input image using the provided corner points.

    Args:
        image (np.ndarray): Input image to be transformed.
        corners (np.ndarray): Array of four corner points (shape: (4, 2)) defining the source quadrilateral.

    Returns:
        np.ndarray: The warped image after applying the perspective transform.
    """
    try:
        corners = corners.astype(np.float32)
        
        width_a = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + ((corners[1][1] - corners[0][1]) ** 2))
        width_b = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + ((corners[2][1] - corners[3][1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + ((corners[3][1] - corners[0][1]) ** 2))
        height_b = np.sqrt(((corners[2][0] - corners[1][0]) ** 2) + ((corners[2][1] - corners[1][1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        max_width = max(max_width, 1)
        max_height = max(max_height, 1)

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

    except Exception as e:
        print("Error during perspective warp:")
        print(f"Exception: {e}")
        return image
