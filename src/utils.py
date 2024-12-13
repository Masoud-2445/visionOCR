import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray | None :
    """
    Loads an image from the specified file path using OpenCV.

    This function attempts to read an image file from the given path using OpenCV's
    `cv2.imread` method in color mode. If the image is successfully loaded, it returns
    the image as a NumPy array. If loading fails (e.g., file not found or path incorrect),
    it raises an IOError and returns None.

    Args:
        image_path (str): The file path of the image to be loaded.

    Returns:
        np.ndarray: The loaded image as a NumPy array if successful.
        None: If the image could not be loaded due to an error.

    Raises:
        IOError: If the image could not be loaded (e.g., invalid file path).
        Exception: If any other unexpected error occurs during loading.
    """
    try :
        import_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if import_image is None :
            raise IOError(f"Unable to load the image {image_path}")
        return import_image

    except Exception as e :
        print(f"Error while loading the image : {e}")
        return None