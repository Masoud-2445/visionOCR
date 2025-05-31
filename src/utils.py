import cv2
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

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

def select_image_path() -> str | None:
    """
    Opens a graphical file dialog allowing the user to select an image file.

    This function utilizes the tkinter library's `filedialog.askopenfilename`
    to present a native OS window for file selection. The dialog is configured
    to filter for common image file types (png, jpg, jpeg, bmp, tiff). If a
    file is successfully selected by the user, its absolute path is returned
    as a string. If the user cancels the operation or if an internal error
    occurs (e.g., related to the graphical environment), None is returned.

    Args:
        None

    Returns:
        str | None: The absolute file path of the selected image as a string
                    if the selection is successful. Returns None if the user
                    cancels the dialog or an error prevents completion.

    Raises:
        ImportError: If the required `tkinter` library cannot be imported
                     (typically indicates an issue with the Python installation
                     or environment). Other exceptions related to GUI operations
                     are generally caught internally, resulting in a None return.
    """
    try:
        root = Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("image files", ".png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )

        if not file_path:
            print("No file selected")
            return None
        print(f"selected image file: {file_path}")
        return file_path

    except Exception as e :
        print(f"Error while selecting image path : {e}")
        return None

def show_image(img, title='Image', figsize=(10, 10)):
    """
    Displays an image using Matplotlib with appropriate color handling.

    This function takes an image (as a NumPy array) and displays it in a
    Matplotlib figure. It automatically detects if the image is color (3 channels)
    or grayscale (2 dimensions) and displays it accordingly. For color images,
    it assumes the input is in BGR format (common in OpenCV) and converts it
    to RGB for correct display with Matplotlib. The axes are turned off, and
    a custom title and figure size can be provided. Errors during display
    are caught, printed, and the function returns None in case of failure.

    Args:
        img (np.ndarray): The image to display, expected as a NumPy array.
                          Should be in BGR format if color.
        title (str, optional): The title to display above the image.
                               Defaults to 'Image'.
        figsize (tuple, optional): The size of the Matplotlib figure (width, height)
                                   in inches. Defaults to (10, 10).

    Returns:
        None: This function does not return a value upon successful execution;
              its primary purpose is the side effect of displaying the image.
              Returns None explicitly if an exception occurs during plotting.

    Raises:
        AttributeError: If `img` does not have a `shape` attribute (i.e., not
                        a NumPy-like array).
        TypeError: If `img` data type is incompatible with `plt.imshow`.
        Exception: Catches and prints any other exceptions that might occur
                   during the plotting process (e.g., issues with Matplotlib
                   backend, invalid color conversion).
    """

    try:

        plt.figure(figsize=figsize)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error while showing image : {e}")
        return None