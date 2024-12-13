import numpy as np
import cv2
from numpy import ndarray


def convert_to_grayscale(image: np.ndarray) -> np.ndarray | None :
    """
    Converts the given image to grayscale using OpenCV.

    This function checks if the input image is a valid NumPy array and converts it
    to a grayscale image if it has three color channels (BGR). If the image is
    already in grayscale (single channel), it returns the image as is. If the input
    is not a valid NumPy array or has an unsupported format, it raises a TypeError.

    Args:
        image (np.ndarray): The input image as a NumPy array (BGR or grayscale).

    Returns:
        np.ndarray: The grayscale version of the image if conversion is successful.
        None: If the conversion fails due to an invalid input or an unexpected error.

    Raises:
        TypeError: If the input is not a valid NumPy array or has an unsupported format.
        Exception: If any other unexpected error occurs during conversion.
    """
    try :
        if not isinstance(image, np.ndarray) :
            raise TypeError('Error : image must be an numpy array')
        if len(image.shape) == 3 :
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2 :
            grayscale_image = image
        else :
            raise TypeError('Error : image must be an numpy array (unsupported format)')
        return grayscale_image

    except Exception as e :
        print('Error : unable to gray scale image')
        print(f'Error : {e}')
        return None


def apply_threshold(image: np.ndarray, threshold_method: str = 'otsu', block_size: int = 11, constant: int = 2) -> np.ndarray |None :
    """
    Applies the specified thresholding method to the given image using OpenCV.

    This function applies thresholding techniques to the input image to create a binary
    image. It supports three thresholding methods: 'binary', 'otsu', and 'adaptive'.
    The parameters `block_size` and `constant` are used only when `threshold_method`
    is set to 'adaptive'.

    Args:
        image (np.ndarray): The input image in grayscale format.
        threshold_method (str): The thresholding method to apply. Supported methods are:
            - 'binary': Applies basic binary thresholding.
            - 'otsu': Applies Otsu's thresholding method (automatic).
            - 'adaptive': Applies adaptive thresholding with a Gaussian filter.
        block_size (int, optional): Size of the neighborhood for adaptive thresholding.
            Must be an odd integer greater than 1 (default: 11).
        constant (int, optional): Constant subtracted from the mean in adaptive
            thresholding (default: 2).

    Returns:
        np.ndarray: The thresholded image if successful.
        None: If the input is invalid or an unexpected error occurs.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        TypeError: If `block_size` is not an odd integer or <= 1 when using 'adaptive'.
        ValueError: If `threshold_method` is not one of 'binary', 'otsu', or 'adaptive'.
        Exception: For unexpected runtime errors during thresholding.
    """
    try :
         if not isinstance(image, np.ndarray) :
             raise TypeError('Error : image must be an numpy array')
         if threshold_method == 'adaptive' and (block_size % 2 == 0 or block_size <= 1) :
             raise TypeError('Error : block_size must be an odd integer and grater than 1')
         methods = {
             "binary": lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
             "otsu": lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
             "adaptive": lambda img: cv2.adaptiveThreshold(
                 img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
             )
         }
         if threshold_method not in methods:
             raise ValueError("Unsupported thresholding method. Choose 'binary', 'otsu', or 'adaptive'.")
         thresholded = methods[threshold_method](image)
         return thresholded

    except Exception as e :
        print('Error : unable to apply threshold to image')
        print(f'Error : {e}')
        return None


def edge_detection(image: np.ndarray, threshold_low = 50, threshold_high = 150) -> ndarray | None :
    """
    Performs edge detection on a given image using the Canny edge detection algorithm.

    This function applies OpenCV's Canny edge detection algorithm to identify edges in the input image.
    It uses the provided threshold values for the hysteresis procedure to determine strong and weak edges.

    Args:
        image (np.ndarray): The input image in grayscale format (as a NumPy array).
        threshold_low (int, optional): Lower threshold for edge detection (default: 50).
        threshold_high (int, optional): Upper threshold for edge detection (default: 150).

    Returns:
        np.ndarray: A binary image highlighting detected edges if successful.
        None: If the input is invalid or an unexpected error occurs.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        ValueError: If the threshold values are invalid or out of range.
        Exception: For unexpected runtime errors during edge detection.
    """
    try :
        if not isinstance(image, np.ndarray) :
            raise TypeError('Error : image must be an numpy array')
        edges = cv2.Canny(image, threshold_low, threshold_high)
        return edges

    except Exception as e :
        print('Error : unable to do edge detection image')
        print(f'Error : {e}')
        return None


def denoise_image(image: np.ndarray, method = "gaussian",  kernel = (5,5), sigmax = 1) -> np.ndarray | None :
    """
    Applies noise reduction to the input image using various filtering methods.

    This function reduces noise in the input image using one of three supported
    denoising methods: 'median', 'gaussian', or 'bilateral'. The corresponding
    OpenCV functions are used based on the selected method. The default method
    is Gaussian blur.

    Args:
        image (np.ndarray): The input image as a NumPy array (grayscale or color).
        method (str, optional): The denoising method to apply. Supported methods:
            - 'median': Applies median blurring.
            - 'gaussian': Applies Gaussian blur (default).
            - 'bilateral': Applies bilateral filtering for edge-preserving smoothing.
        kernel (tuple, optional): Kernel size for Gaussian and median blurs (default: (5, 5)).
            For median blur, only the first value of the tuple is used.
        sigmax (int, optional): Standard deviation for Gaussian blur along the x-axis (default: 1).
            This parameter is ignored for other methods.

    Returns:
        np.ndarray: The denoised image if successful.
        None: If the input is invalid or an unexpected error occurs.

    Raises:
        ValueError: If an unsupported denoising method is provided.
        TypeError: If the input image is not a valid NumPy array.
        Exception: For unexpected runtime errors during the denoising process.
    """
    try :
        methods = {
            "median" : lambda img: cv2.medianBlur(img,kernel[0]),
            "gaussian" : lambda img: cv2.GaussianBlur(img, kernel, sigmax),
            "bilateral" : lambda img: cv2.bilateralFilter(img, d=9, sigmaColor = 0, sigmaSpace = 0)
        }
        if method not in methods :
             raise ValueError("Unsupported method (use gaussian or median or bilateral.)")
        denoised_image = methods[method](image)
        return denoised_image

    except Exception as e :
        print("Error: Unable to denoise image.")
        print(f"Error: {e}")
        return None


def adjust_brightness(image: np.ndarray, brightness: int = 0) -> np.ndarray | None :
    """
    Adjusts the brightness of an image.

    Args:
        image (np.ndarray): The input image.
        brightness (int): Value to adjust brightness (-255 to 255).

    Returns:
        np.ndarray: The brightness-adjusted image.
    """
    try :
        if not isinstance(image, np.ndarray) :
            raise TypeError("Error : image must be an numpy array.")
        brightness_level = max(-255, min(255, brightness))
        final_lv = cv2.convertScaleAbs(image, beta=brightness_level)
        return final_lv

    except Exception as e :
        print("Error : unable to adjust brightness.")
        print(f"Error: {e}")
        return None


def adjust_contrast(image: np.ndarray, contrast: int = 0) -> np.ndarray | None :
    """
    Adjusts the contrast of an image.

    This function adjusts the contrast of the input image using a scaling factor `alpha`.
    The contrast value must be between -127 and 127. A value of 0 means no change.
    Positive values increase contrast, while negative values decrease it.

    Args:
        image (np.ndarray): The input image as a NumPy array (grayscale or color).
        contrast (int, optional): The contrast adjustment value (-127 to 127).
            Default is 0 (no change).

    Returns:
        np.ndarray: The contrast-adjusted image if successful.
        None: If an error occurs or if the input is invalid.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        ValueError: If the contrast value is outside the allowed range.
        Exception: For unexpected runtime errors during contrast adjustment.
    """
    try :
        if not isinstance(image, np.ndarray) :
            raise TypeError("Error : image must be an numpy array.")
        contrast = max(-127, min(127, contrast))
        if contrast != 0:
            alpha = 1 + (contrast / 127.0)
            adjusted = cv2.convertScaleAbs(image, alpha=alpha)
        else:
            adjusted = image.copy()
        return adjusted

    except Exception as e:
        print("Error: Unable to adjust contrast.")
        print(f"Error: {e}")
        return None


def image_erosion(image: np.ndarray, kernel = np.ones((5,5), np.uint8)) -> np.ndarray | None :
    """
    Applies morphological erosion to the input image.

    This function performs erosion on the input image using the specified kernel.
    Erosion removes pixels on object boundaries, shrinking white regions in binary
    images or reducing noise in edge-detected images.

    Args:
        image (np.ndarray): The input image as a NumPy array (binary, grayscale, or color).
        kernel (np.ndarray, optional): The structuring element used for erosion.
            Default is a 5x5 rectangular kernel of type `np.uint8`.

    Returns:
        np.ndarray: The eroded image if the operation is successful.
        None: If an error occurs or if the input is invalid.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        Exception: For unexpected runtime errors during the erosion process.
    """
    try :
        if not isinstance(image, np.ndarray) :
            raise TypeError("Error : image must be an numpy array.")
        img_erode = cv2.erode(image, kernel, iterations = 1)
        return img_erode

    except Exception as e:
        print("Error: Unable to erode image.")
        print(f"Error: {e}")
        return None


def image_dilation(image: np.ndarray, kernel = np.ones((5,5), np.uint8)) -> np.ndarray | None :
    """
    Applies morphological dilation to the input image.

    This function performs dilation on the input image using the specified kernel.
    Dilation expands the white regions in binary images, filling small holes and
    connecting disconnected components. It is commonly used after edge detection
    or thresholding.

    Args:
        image (np.ndarray): The input image as a NumPy array (binary, grayscale, or color).
        kernel (np.ndarray, optional): The structuring element used for dilation.
            Default is a 5x5 rectangular kernel of type `np.uint8`.

    Returns:
        np.ndarray: The dilated image if the operation is successful.
        None: If an error occurs or if the input is invalid.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        Exception: For unexpected runtime errors during the dilation process.
    """
    try:
        if not isinstance(image, np.ndarray) :
            raise TypeError("Error : image must be an numpy array.")
        img_dilate = cv2.dilate(image, kernel, iterations = 2)
        return img_dilate

    except Exception as e:
        print("Error: Unable to dilate image.")
        print(f"Error: {e}")
        return None


def get_morphological(image: np.ndarray, kernel = (5,5)) -> np.ndarray | None :
    """
    Applies morphological closing to the input image.

    This function performs the morphological closing operation on the input image
    using the specified kernel size. Closing is useful for filling small holes
    and connecting broken edges in binary or edge-detected images.

    Args:
        image (np.ndarray): The input image as a NumPy array (binary, grayscale, or edge-detected).
        kernel (tuple, optional): The kernel size for the morphological operation.
            Default is a 5x5 rectangular kernel.

    Returns:
        np.ndarray: The processed image after applying morphological closing.
        None: If an error occurs or if the input is invalid.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        Exception: For unexpected runtime errors during the morphological operation.
    """
    try :
        if not isinstance(image, np.ndarray) :
            raise TypeError("Error : image must be an numpy array.")
        struct_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        open_edges = cv2.morphologyEx(image, cv2.MORPH_CLOSE, struct_kernel)
        return open_edges

    except Exception as e:
        print("Error: Unable to get morphological operations.")
        print(f"Error: {e}")
        return None


def preprocess_image(image: np.ndarray) -> np.ndarray | None :
    """
    Preprocesses the input image for edge detection and contour extraction.

    This function applies a sequence of preprocessing steps to prepare the input image
    for tasks like edge detection and contour extraction. The steps include converting
    the image to grayscale, reducing noise, applying thresholding, detecting edges,
    and performing morphological dilation to strengthen the detected edges.

    Preprocessing Steps:
        1. Convert to Grayscale: Simplifies the image for processing.
        2. Denoising: Removes noise using Gaussian blur.
        3. Thresholding: Binarizes the image using Otsu's method.
        4. Edge Detection: Detects edges using Canny edge detection.
        5. Dilation: Strengthens edges to close gaps.

    Args:
        image (np.ndarray): The input image as a NumPy array (color or grayscale).

    Returns:
        np.ndarray: The preprocessed edge-detected image.
        None: If an error occurs or if the input is invalid.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        Exception: For unexpected runtime errors during the preprocessing steps.
    """
    try:
        gray_image = convert_to_grayscale(image)
        denoised_image = denoise_image(gray_image)
        thresholded_image = apply_threshold(denoised_image, threshold_method="otsu")
        edges = edge_detection(thresholded_image)
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        return edges_dilated

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def preprocess_ocr(image: np.ndarray) -> np.ndarray | None :
    """
    Preprocesses the input image for OCR (Optical Character Recognition).

    This function prepares an input image by applying a sequence of preprocessing steps
    designed to enhance text visibility and improve OCR accuracy. The steps include:

    1. **Grayscale Conversion:** Simplifies the image for further processing.
    2. **Contrast Enhancement:** Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
       to improve text visibility.
    3. **Thresholding:** Applies Otsu's binarization to create a binary image.
    4. **Noise Removal:** Uses morphological closing to clean up small artifacts.

    Args:
        image (np.ndarray): The input image as a NumPy array (grayscale or color).

    Returns:
        np.ndarray: The preprocessed binary image ready for OCR.
        None: If an error occurs or if the input is invalid.

    Raises:
        TypeError: If the input image is not a valid NumPy array.
        Exception: For unexpected runtime errors during the preprocessing steps.
    """
    try:
        gray_image = convert_to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray_image)
        thresholded_image = apply_threshold(enhanced_image, threshold_method="otsu")
        clean_image = get_morphological(thresholded_image, kernel=(3,3))
        return clean_image

    except Exception as e:
        print(f"Error in preprocessing for ocr: {e}")
        return None