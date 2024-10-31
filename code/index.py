import pytesseract
from PIL import Image


#Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Create a sample image or use a simple test image
image = Image.new('RGB', (100, 50), color = (73, 109, 137))
image.save("sample.png")


# Try reading the text from the image
print(pytesseract.image_to_string("sample.png"))
