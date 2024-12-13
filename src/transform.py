import numpy as np
import cv2
from preprocessing import convert_to_grayscale


def resize_image(image: np.ndarray, target_size: tuple, aspect_ratio= True, method= cv2.INTER_LINEAR) -> np.ndarray | None :
    """
    Resizes an image to the target size, preserving aspect ratio if specified.

    Args:
        image (np.ndarray): Input image in numpy array format.
        target_size (tuple): Desired width and height (w, h).
        aspect_ratio (bool): Whether to preserve the aspect ratio.
        method (int): Interpolation method for resizing (default: cv2.INTER_LINEAR).

    Returns:
        np.ndarray: Resized image.
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


def deskew_image(image: np.ndarray) -> np.ndarray | None :
    """
    Corrects the skew of the input image using its rotation angle.

    This function calculates the image's skew angle using its binary mask and
    applies rotation to deskew the image.

    Args:
        image (np.ndarray): The input image (binary or grayscale).

    Returns:
        np.ndarray: The deskewed image.
        None: If the deskewing process fails or if the input is invalid.

    Raises:
        TypeError: If the input is not a valid NumPy array.
        Exception: For unexpected runtime errors.
    """
    try:
        if not isinstance(image, np.ndarray) :
            raise TypeError("Error : image must be an numpy array.")

        if len(image.shape) == 3 :
            image_gray = convert_to_grayscale(image)
        elif len(image.shape) == 2 :
            image_gray = image
        else:
            raise TypeError("Error : image must be an numpy array(unsupported format).")
        image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(image_binary > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle += 90

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    except Exception as e:
        print("Error: Unable to deskew image.")
        print(f"Error: {e}")
        return None


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray | None:
    """
    Rotates the input image by a specified angle.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        angle (float): The rotation angle in degrees (positive for counter-clockwise).

    Returns:
        np.ndarray: The rotated image.
        None: If rotation fails or if the input is invalid.

    Raises:
        TypeError: If the input is not a valid NumPy array.
        ValueError: If the angle is not a valid float or integer.
        Exception: For unexpected runtime errors.
    """
    try:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")
        if not isinstance(angle, (int, float)):
            raise ValueError("Angle must be a float or integer.")

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    except Exception as e:
        print(f"Error in rotating the image: {e}")
        return None

def prespective_transformation(image: np.ndarray, contour: np.ndarray, output_size= (2480, 3508), to_a4 = True) -> np.ndarray | None :
    """
    Applies a perspective transformation to the detected contour in the input image.

    This function warps the detected paper contour into a flat, rectangular perspective.
    If `to_a4` is set to `True`, the output size is fixed to A4 dimensions (2480x3508 pixels).
    Otherwise, the function dynamically calculates the output size while maintaining
    the detected contour's aspect ratio.

    Steps:
        1. Verify that the contour has exactly 4 points.
        2. Sort the contour points in the correct order (top-left, top-right, bottom-right, bottom-left).
        3. Calculate the target size:
            - If `to_a4` is `True`, use the specified `output_size` (default: A4).
            - If `to_a4` is `False`, calculate the width and height dynamically.
        4. Compute the perspective transform matrix and apply the warp.

    Args:
        image (np.ndarray): The original image containing the detected paper.
        contour (np.ndarray): The detected paper contour with 4 corner points.
        output_size (tuple, optional): The target output size (width, height).
            Default is (2480, 3508) for A4 size.
        to_a4 (bool, optional): If `True`, fixes the output to A4 size. If `False`,
            calculates the size dynamically based on the detected contour's dimensions.

    Returns:
        np.ndarray: The warped image with the applied perspective transformation.
        None: If the transformation fails or if an invalid input is provided.

    Raises:
        ValueError: If the contour does not have exactly 4 points.
        TypeError: If the input image or contour is not a valid NumPy array.
        Exception: For unexpected runtime errors during the transformation.
    """
    try :
        if contour.shape[0] != 4 :
            raise ValueError("Error : contour must be an array with 4 values")
        def sort_points(pts):
            pts = pts.reshape((4, 2))
            # Sort by x-coordinates
            x_sorted = sorted(pts, key=lambda x: x[0])
            # Top-left and bottom-left
            left = sorted(x_sorted[:2], key=lambda x: x[1])
            # Top-right and bottom-right
            right = sorted(x_sorted[2:], key=lambda x: x[1])
            return np.array([left[0], right[0], right[1], left[1]], dtype="float32")

        sorted_points = sort_points(contour)

        if to_a4 :
            max_width, max_height = output_size
        else:

            width_top = np.linalg.norm(sorted_points[0] - sorted_points[1])
            width_bottom = np.linalg.norm(sorted_points[3] - sorted_points[2])
            height_left = np.linalg.norm(sorted_points[0] - sorted_points[3])
            height_right = np.linalg.norm(sorted_points[1] - sorted_points[2])

            max_width = int(max(width_top, width_bottom))
            max_height = int(max(height_left, height_right))

        width, height = max_width, max_height
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(sorted_points, dst_points)
        warped_image = cv2.warpPerspective(image, matrix, (width, height))
        return warped_image

    except Exception as e:
        print(f"Error in applying perspective transform: {e}")
        return None