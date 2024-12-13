import cv2
import numpy as np
import pytesseract


def extract_text(image: np.ndarray) -> str:
    """
    Extracts text from the preprocessed image using Tesseract OCR.

    Args:
        image (np.ndarray): The preprocessed binary image.

    Returns:
        str: The extracted text from the image.
    """
    try:
        text = pytesseract.image_to_string(image, lang='eng')
        return text.strip()
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        return ""