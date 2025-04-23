import pytesseract
from PIL import Image
import os

# Print PyTesseract version and path
print(f"PyTesseract version: {pytesseract.__version__}")
print(f"Tesseract command: {pytesseract.pytesseract.tesseract_cmd}")

# Test if tesseract is available
try:
    print("Testing Tesseract availability...")
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
    print("Tesseract is working!")
except Exception as e:
    print(f"Error with Tesseract: {e}")

# Create a simple test image with text
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create a white image
width, height = 400, 100
img = Image.new('RGB', (width, height), color='white')
d = ImageDraw.Draw(img)

# Add text to the image
text = "Arsenal vs Manchester United - BTTS"
d.text((10, 40), text, fill='black')

# Save the test image
test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_image.png')
img.save(test_image_path)
print(f"Created test image at: {test_image_path}")

# Try OCR on the test image
try:
    text = pytesseract.image_to_string(test_image_path)
    print(f"OCR result: {text.strip()}")
    
    if "Arsenal" in text and "Manchester United" in text:
        print("OCR test successful!")
    else:
        print("OCR detected text but not the expected content.")
except Exception as e:
    print(f"OCR failed: {e}")