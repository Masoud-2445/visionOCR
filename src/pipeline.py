import cv2
import numpy as np
from PIL import Image
import utils
import detection
import proccess
import transform
import PIL
import layout_detector as ld
import json
from constructor import docx_constructor
import os
import uuid
from layout_process import post_process_layout



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
        if len(image.shape) == 3:
            gray_img = proccess.convert_to_grayscale(image)
        else:
            gray_img = image.copy()

        # 1. Apply a more powerful but edge-preserving denoising filter
        denoised = cv2.fastNlMeansDenoising(gray_img, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # 2. Apply Adaptive Thresholding
        threshold_img = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15, # Use a slightly larger block size for noisy images
            4
        )
        
        return threshold_img
    except Exception as e:
        print(f"Error in enhancing image : {e}")
        return None
        



def resize_before(pil_image, target_height = 1920):

    original_width, original_height = pil_image.size

    aspect_ratio = original_width / original_height
    
    target_width = int(target_height * aspect_ratio)
    
    new_size = (target_width, target_height)
    resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    return resized_image


def image_scale(image: Image.Image, json_path: str):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)

        # Scale up the layout coordinates
        original_width, original_height = image.size
        gpt_width = layout_data["analyzed_image_size"]["width"]
        gpt_height = layout_data["analyzed_image_size"]["height"]

        resized_image = image.resize((gpt_width, gpt_height), Image.Resampling.LANCZOS)

        scale_x = original_width / gpt_width
        scale_y = original_height / gpt_height
        resize_width, resize_height = resized_image.size
        print(f"Original size: {original_width} x {original_height}")
        print(f"GPT size: {gpt_width} x {gpt_height}")
        print(f"Resized size: {resize_width} x {resize_height}")
        print(f"Scale factors → X: {scale_x:.3f}, Y: {scale_y:.3f}")




        return resized_image
    except Exception as e:
        print(f"Error in scaling up results: {e}")
        return None
    


    except Exception as e:
        print(f"Error in constructing pipeline: {e}")


def load_json_data(json_path):
    """Loads and parses the JSON file from a previous API call."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            print(f"Loading layout data from: {json_path}")
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: JSON file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse the JSON file. It may be malformed.")
        return None

def main(original_image_path):
    """
    Runs the full document reconstruction pipeline using a saved image and JSON file.
    """
    try:
        
        saved_json_path = "C:/Users/masoo/Documents/OCR-project/layout_results.json"
        output_docx_path = "C:/Users/masoo/Documents/OCR-project/Final_Reconstructed_Document.docx"


        print("Loading original image...")
        original_cv_image = cv2.imread(original_image_path)
        if original_cv_image is None:
            print(f"Image loading failed. Check the path: {original_image_path}")
            return

        
        rich_layout_data = load_json_data(saved_json_path)
        if not rich_layout_data:
            print("Layout data loading failed. Exiting.")
            return


        print("\nStarting the document reconstruction process...")
        docx_constructor(rich_layout_data, original_cv_image, output_docx_path)

        print("\n Pipeline test finished successfully!")

    except Exception as e:
        print(f"\n A critical error occurred in the pipeline: {e}")



def save_pillow(pill_image, temp_folder_path):
    temp_file_path = None
    try :
        os.makedirs(temp_folder_path, exist_ok=True)
        uuid_name = f"{uuid.uuid4()}.png"
        temp_file_path = os.path.join(temp_folder_path, uuid_name)

        pill_image.save(temp_file_path)
        return temp_file_path

    except Exception as e :
        print("failed to save temp image")
        return None



def doc_construct():
    try :
        img = upload_image()
        pil_img = ld.img_convert(img)
        image = pil_img.copy()

        layout = ld.get_layout(image)

        


        output_file = "C:/Users/masoo/Documents/OCR-project/layout_results.json"
        ld.save_results_to_json(layout, output_file)

        resize_construct_image = image_scale(image, output_file)
        temp_image = save_pillow(resize_construct_image, "C:/Users/masoo/Documents/OCR-project/temp_images")
        main(temp_image)
        print("successful construct")
    except Exception as e :
        print("failed to make construct of document ")
        exit(1)

if __name__ == "__main__" :


    img = upload_image()

    pil_img = ld.img_convert(img)
    image = pil_img.copy()
    resize_image = resize_before(image)
    print(resize_image.size)
    layout = ld.get_layout(image)


    output_file = "C:/Users/masoo/Documents/OCR-project/layout_results.json"
    ld.save_results_to_json(layout, output_file)



# Post-process
    layout_data = post_process_layout(output_file)

    resize_construct_image = image_scale(image, output_file)
    temp_image = save_pillow(resize_construct_image, "C:/Users/masoo/Documents/OCR-project/temp_images")
    main(temp_image)

    # enhanced = enhance_img(img)
    # utils.show_image(enhanced, "enhanced image")
