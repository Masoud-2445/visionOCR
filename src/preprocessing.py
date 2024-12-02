from PIL import Image
import numpy as np
import cv2



def load_image(image_path: str):
    """Loads the target image from the given path."""

    try :
        image =  cv2.imread(image_path,cv2.IMREAD_COLOR)  # open image using opencv
        if image is not None :
            return image
    except IOError :
        print('Error : unable to load image')
        return None


def resize_image(image: np.ndarray, target_size: tuple, aspect_ratio= True, method= cv2.INTER_LINEAR) -> np.ndarray:
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


    try :           #handeling error (just for caution)
        if not isinstance(image, np.ndarray) :
            raise TypeError('image must be numpy array')

        get_size = image.shape[:2]

        #check for acceptable format
        if not (isinstance(target_size, tuple) and len(target_size) == 2) and all(isinstance(i, int) and i > 0 for i in target_size) :
            print('Error : target size must be 2 dimensional and positive integer')
            return None

        #new size based on aspect ratio
        if aspect_ratio :
            target_aspect_ratio = get_size[1] / get_size[0]

            if get_size[0] > get_size[1] :      #handeling vertical images
                new_height = target_size[1]
                new_width = int(new_height * target_aspect_ratio)

            else :                              #handeling horizontal or equal images
                new_width = target_size[0]
                new_height = int(new_width / target_aspect_ratio)

            target_size = (new_width, new_height)      #calculated target size
            resize_conversion = cv2.resize(image, target_size, interpolation= method)  #resizing method

            return resize_conversion
    except Exception as e :
        print('Error : unable to resize image')
        print(f'Error : {e}')



def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts the given image to grayscale.
    take both PIL and Opencv format and return PIL format image"""

    try :
        #check image type and do conversion if needed
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



def apply_threshold(image: np.ndarray, threshold_method: str = 'binary', block_size: int = 9, constant: int = 2) -> np.ndarray:
    """
      Applies the specified thresholding method to the given image.

      Parameters:
          image (np.ndarray): Input image in OpenCV format.
          threshold_method (str): Thresholding method to apply ('binary', 'otsu', or 'adaptive').
          block_size (int, optional): Size of the neighborhood for adaptive thresholding (default: 11).
          constant (int, optional): Constant to subtract in adaptive thresholding (default: 2).

      Returns:
          np.ndarray: Thresholded image.

      Raises:
          TypeError: If the input image is not a valid NumPy array or has an unsupported format.
          ValueError: If block_size or constant is invalid for adaptive thresholding.
    """


    try :           # check for correct input
         if not isinstance(image, np.ndarray) :
             raise TypeError('Error : image must be an numpy array')
                    # grayscale conversion
         if len(image.shape) == 3 :
             gray_image = convert_to_grayscale(image)
         elif len(image.shape) == 2 :
             gray_image = image
         else :
             raise TypeError('Error : image must be an numpy array or grayscale (unsupported format)')


                    # check for valid inputs for adaptive threshold
         if threshold_method == 'adaptive' and (block_size % 2 == 0 or block_size <= 1) :
             raise TypeError('Error : block_size must be an odd integer and grater than 1')

                    # methods compact in dict (old code use if-=elif repetitive)
         methods = {
             "binary": lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
             "otsu": lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
             "adaptive": lambda img: cv2.adaptiveThreshold(
                 img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
             )
         }
                # raise for invalid threshold method
         if threshold_method not in methods:
             raise ValueError("Unsupported thresholding method. Choose 'binary', 'otsu', or 'adaptive'.")


         thresholded = methods[threshold_method](gray_image)
         return thresholded

    except Exception as e :
        print('Error : unable to apply threshold to image')
        print(f'Error : {e}')




def edge_detection(image: np.ndarray, threshold_low = 50, threshold_high = 150) -> np.ndarray:
    """
      Performs edge detection on a given image using the Canny algorithm.

      Parameters:
          image: Input image in OpenCV format (NumPy array).
          threshold_low: Lower threshold for the hysteresis procedure (default: 50).
          threshold_high: Upper threshold for the hysteresis procedure (default: 150).

      Returns:
          edges: Binary image showing detected edges (NumPy array).

      Raises:
          TypeError: If the input image is not a valid NumPy array or has unsupported format.
          ValueError: If thresholds are invalid.
      """

    try :           # check for correct input
        if not isinstance(image, np.ndarray) :
            raise TypeError('Error : image must be an numpy array')
                    # make sure image is grayscale
        if len(image.shape) == 3 :
            grayscale_image = convert_to_grayscale(image)
        elif len(image.shape) == 2 :
            grayscale_image = image
        else :
            raise TypeError('Error : image must be an numpy array or grayscale image (unsupported format)')
        # main part --> apply edge detection using canny (opencv)
        edges = cv2.Canny(grayscale_image, threshold_low, threshold_high)
        return edges


    except Exception as e :
        print('Error : unable to do edge detection image')
        print(f'Error : {e}')





def contour_detection(image: np.ndarray, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE, minimum_area = 1000) -> list:
    """
    Detects contours in the given edge-detected image.

    Args:
        image (np.ndarray): Edge-detected image (grayscale or binary).
        mode (int): Contour retrieval mode (default is cv2.RETR_EXTERNAL).
        method (int): Contour approximation method (default is cv2.CHAIN_APPROX_SIMPLE).
        minimum_area (int): Minimum area for a contour to be considered valid.

    Returns:
        list: List of filtered contours that meet the area criteria.
    """
    try:            #check for valid input
        if not isinstance(image, np.ndarray):
            raise TypeError("Error: Image must be a numpy array.")

        # Ensure the image is binary for contour detection
        if len(image.shape) == 3:
            raise ValueError("Error: Image must be grayscale or binary.")

        # Find contours
        contours, _ = cv2.findContours(image, mode, method)

        # Filter contours by area
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > minimum_area]

        return filtered_contours

    except Exception as e:
        print("Error: Unable to detect contours.")
        print(f"Error: {e}")
        return []








im_path = "C:/Users/masoo/Desktop/photo_2024-11-18_19-01-15.jpg"
im = load_image(im_path)



m = contour_detection(apply_threshold(im))
cv2.drawContours(im, m, -1, (255, 0, 0), 4)

cv2.imshow("Image", cv2.drawContours(im, m, -1, (255, 0, 0), 4))
cv2.waitKey(0)
cv2.destroyAllWindows()



print(m)