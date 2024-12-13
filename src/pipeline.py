import cv2
import numpy as np
import pytesseract
from src.preprocessing import preprocess_ocr
from src.transform import (
    prespective_transformation,
    deskew_image,
    rotate_image
)
from src.detection import  paper_contour_detection
from src.utils import load_image


def ocr_pipeline(image_path: str, rotate_angle: float = 0) -> str | None:
    """
    Processes an input image and extracts text using the complete OCR pipeline.

    Steps:
        1. Load the image.
        2. Preprocess the image (grayscale, threshold, denoise).
        3. Detect the paper contour.
        4. Apply perspective transformation.
        5. Deskew the image (auto-alignment).
        6. Apply manual rotation if specified.
        7. Extract text using Tesseract OCR.

    Args:
        image_path (str): Path to the input image file.
        rotate_angle (float, optional): Manual rotation angle in degrees (default: 0).

    Returns:
        str: Extracted text from the image.
        None: If any processing step fails or an error occurs.
    """
    try:
        # Step 1: Load the image
        image = load_image(image_path)
        if image is None:
            raise ValueError("Image loading failed.")

        # Step 2: Preprocess the image for OCR
        preprocessed = preprocess_ocr(image)
        if preprocessed is None:
            raise ValueError("Preprocessing failed.")

        # Step 3: Detect the paper contour
        paper_contour = paper_contour_detection(preprocessed)
        if paper_contour is None:
            raise ValueError("Paper detection failed.")

        # Step 4: Apply perspective transformation
        warped = prespective_transformation(image, paper_contour)
        if warped is None:
            raise ValueError("Perspective transformation failed.")

        # Step 5: Deskew the image
        deskewed = deskew_image(warped)
        if deskewed is None:
            raise ValueError("Deskewing failed.")

        # Step 6: Apply manual rotation (if needed)
        if rotate_angle != 0:
            rotated = rotate_image(deskewed, rotate_angle)
        else:
            rotated = deskewed

        # Step 7: Extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(rotated, lang='eng')
        return extracted_text.strip()

    except Exception as e:
        print(f"Error in OCR pipeline: {e}")
        return None