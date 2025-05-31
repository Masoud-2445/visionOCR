"""
OCR Engine Module

This module provides functionality for text detection and extraction from images
using pytesseract and OpenCV. It is designed to work with preprocessed images
and return structured text data along with visualizations.

Requirements:
- pytesseract
- opencv-python (cv2)
- numpy

Usage:
    from ocr_engine import OCREngine
    
    # Create OCR engine instance
    ocr = OCREngine(language='eng')
    
    # Process an image
    results = ocr.extract_text(preprocessed_image)
    
    # Visualize results
    annotated_image = ocr.draw_text_boxes(original_image, results)
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple, Union, Optional


class OCREngine:
    """
    OCR Engine for text detection and extraction from images.
    
    This class provides methods to extract text from images using pytesseract,
    filter results based on confidence, and visualize the detected text.
    """
    
    def __init__(self, language: str = 'eng', 
                 min_confidence: float = 60.0,
                 tesseract_config: str = ''):
        """
        Initialize the OCR Engine with the specified parameters.
        
        Args:
            language: Language code for OCR. Use 'eng' for English, 'fas' for Persian,
                     or 'eng+fas' for both. Default is 'eng'.
            min_confidence: Minimum confidence threshold (0-100) for text detection.
                           Text with confidence below this value will be filtered out.
            tesseract_config: Additional configuration parameters for pytesseract.
        """
        self.language = language
        self.min_confidence = min_confidence
        self.tesseract_config = tesseract_config
        
        # Check if pytesseract is properly installed and configured
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            print(f"Error: pytesseract is not properly installed or configured. {str(e)}")
            print("Please ensure Tesseract OCR is installed and the path is correctly set.")
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from a preprocessed image and return structured data.
        
        Args:
            image: Preprocessed OpenCV image as NumPy array
            
        Returns:
            List of dictionaries with text detection data, where each dictionary contains:
            - 'text': The detected text string
            - 'bbox': Bounding box coordinates as (left, top, width, height)
            - 'confidence': Confidence score (0-100)
        """
        # Ensure image is in the correct format for pytesseract
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Use pytesseract to extract text data
        config = f'-l {self.language} --oem 1 --psm 11 {self.tesseract_config}'
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
        
        # Process the results
        num_boxes = len(data['text'])
        results = []
        
        for i in range(num_boxes):
            # Skip empty text
            if not data['text'][i].strip():
                continue
                
            # Get confidence and filter low-confidence results
            confidence = float(data['conf'][i])
            if confidence < self.min_confidence:
                continue
                
            # Extract text and bounding box
            text = data['text'][i]
            left = data['left'][i]
            top = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]
            
            # Create result dictionary
            result = {
                'text': text,
                'bbox': (left, top, width, height),
                'confidence': confidence
            }
            
            results.append(result)
            
        return results
    
    def draw_text_boxes(self, image: np.ndarray, results: List[Dict], 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       text_color: Tuple[int, int, int] = (255, 0, 0),
                       thickness: int = 2,
                       font_scale: float = 0.5) -> np.ndarray:
        """
        Draw bounding boxes and text on the original image.
        
        Args:
            image: Original image as NumPy array
            results: List of text detection results from extract_text method
            color: Color for bounding boxes (B, G, R)
            text_color: Color for text display (B, G, R)
            thickness: Line thickness for bounding boxes
            font_scale: Scale factor for text font size
            
        Returns:
            Image with bounding boxes and text drawn on it
        """
        # Create a copy of the original image to avoid modifying it
        output_image = image.copy()
        
        for result in results:
            # Extract bounding box coordinates
            left, top, width, height = result['bbox']
            
            # Draw rectangle around the text
            cv2.rectangle(output_image, (left, top), (left + width, top + height), color, thickness)
            
            # Prepare text with confidence score
            display_text = f"{result['text']} ({result['confidence']:.1f}%)"
            
            # Calculate text position (just above the bounding box)
            text_position = (left, max(top - 10, 10))
            
            # Add text to the image
            cv2.putText(output_image, display_text, text_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
        return output_image
    
    def get_text_overlay(self, image: np.ndarray, results: List[Dict], 
                        font_scale: float = 0.5,
                        font_thickness: int = 1,
                        text_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Create a transparent text overlay on the original image.
        
        This function creates a transparent overlay with just the text (no boxes)
        positioned exactly where it was detected in the original image.
        
        Args:
            image: Original image as NumPy array
            results: List of text detection results from extract_text method
            font_scale: Scale factor for text font size
            font_thickness: Thickness of font strokes
            text_color: Color for text display (B, G, R)
            
        Returns:
            Image with transparent text overlay
        """
        # Create a copy of the original image
        overlay_image = image.copy()
        
        for result in results:
            # Extract bounding box coordinates and text
            left, top, width, height = result['bbox']
            text = result['text']
            
            # Position text at bottom-left corner of bounding box
            text_position = (left, top + height)
            
            # Add text to the overlay image
            cv2.putText(overlay_image, text, text_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            
        return overlay_image
    
    def prepare_for_export(self, results: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """
        Prepare OCR results for export to document formats.
        
        This function organizes the OCR results into a structured format that can
        be used by export modules to create searchable documents.
        
        Args:
            results: List of text detection results from extract_text method
            image_shape: Shape of the original image (height, width)
            
        Returns:
            Dictionary with organized text data suitable for export
        """
        # Sort results by vertical position (top coordinate)
        sorted_results = sorted(results, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        # Group text items by approximate lines
        line_tolerance = int(image_shape[0] * 0.02)  # 2% of image height
        lines = []
        current_line = []
        current_line_y = None
        
        for result in sorted_results:
            # Get the vertical position of this text item
            _, top, _, _ = result['bbox']
            
            if current_line_y is None:
                # This is the first text item
                current_line_y = top
                current_line.append(result)
            elif abs(top - current_line_y) <= line_tolerance:
                # This text item is on the same line
                current_line.append(result)
            else:
                # This text item is on a new line
                # Sort the current line by horizontal position
                current_line.sort(key=lambda x: x['bbox'][0])
                lines.append(current_line)
                
                # Start a new line
                current_line = [result]
                current_line_y = top
        
        # Don't forget to add the last line
        if current_line:
            current_line.sort(key=lambda x: x['bbox'][0])
            lines.append(current_line)
        
        # Create the export data
        export_data = {
            'lines': [],
            'image_size': image_shape
        }
        
        # Process each line
        for line_items in lines:
            line_text = ' '.join(item['text'] for item in line_items)
            line_data = {
                'text': line_text,
                'items': line_items
            }
            export_data['lines'].append(line_data)
        
        return export_data


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.
    
    This function applies common preprocessing techniques to enhance text visibility
    and improve OCR results. Note that this is a basic implementation and may need
    to be adjusted based on specific image characteristics.
    
    Args:
        image: Original image as NumPy array
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply noise reduction
    processed = cv2.medianBlur(binary, 3)
    
    return processed


# Example usage
if __name__ == "__main__":
    # This is a simple example of how to use the OCREngine class
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        sys.exit(1)
    
    # Preprocess the image (in a real application, you'd have more sophisticated preprocessing)
    preprocessed_image = preprocess_image_for_ocr(image)
    
    # Create OCR engine instance
    ocr = OCREngine(language='eng+fas')
    
    # Extract text
    results = ocr.extract_text(preprocessed_image)
    
    # Display results
    print(f"Found {len(results)} text regions:")
    for i, result in enumerate(results):
        print(f"{i+1}. Text: '{result['text']}', Confidence: {result['confidence']:.1f}%")
    
    # Draw text boxes on the image
    annotated_image = ocr.draw_text_boxes(image, results)
    
    # Display the annotated image
    cv2.imshow("OCR Results", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()