import cv2
import numpy as np
import utils
import detection
import proccess
import transform
from ocr_engine import OCREngine
import cv2
import numpy as np
from export import overlay_ocr_to_pdf, export_ocr_to_word

def upload_image() -> np.ndarray | None:
    try:
        img_path = utils.select_image_path()
        img = utils.load_image(img_path)
        c_point, c_img = detection.detect_document_contour(img)
        warped = transform.warp_perspective(img, c_point)
        deskew_img = transform.deskew(warped)
        return deskew_img
    except Exception as e:
        print(f"Error while uploading image : {e}")
        return None
    

def enhance_img(image: np.ndarray) -> np.ndarray | None:
    try:
        if len(image.shape) ==3:
            gray_img = proccess.convert_to_grayscale(image)
        else :
            gray_img = image.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        threshold_img = proccess.apply_threshold(enhanced)
        return threshold_img
    except Exception as e:
        print(f"Error in enhancing image : {e}")
        return None
        

def preform_ocr(image: np.ndarray):
    try:
        img_enhanced = enhance_img(image)
        ocr = OCREngine(language='eng+fas')
        results = ocr.extract_text(img_enhanced)
        annotated_img = ocr.draw_text_boxes(image, results)
        utils.show_image(annotated_img)
        return results
    except Exception as e:
        print(f"Error in performing ocr on image : {e}")
        return None
    

def export_ready_data(results: dict, image: np.ndarray):
    ocr = OCREngine('eng+fas')
    height, width, _ = image.shape
    image_shape = (height, width)
    ocr_ready_data = ocr.prepare_for_export(results, image_shape)
    return ocr_ready_data


def export_pdf(data, image, output_path: str): 
    try:
        overlay_ocr_to_pdf(image, data, output_path)
    except Exception as e : 
        print(f"Error in exporting PDF : {e}")


def export_word(data,output_path: str):
    try:
        export_ocr_to_word(data, output_path)
    except Exception as e:
        print(f"Error in exporting Word : {e}")


if __name__ == "__main__" :
    img_holder = upload_image()
    utils.show_image(img_holder)
    result = preform_ocr(img_holder)
    ready_data = export_ready_data(result, img_holder)
