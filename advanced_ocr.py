import pytesseract
from PIL import Image
import cv2
import numpy as np
import easyocr
import os

# Initialize EasyOCR reader (only do this once)
reader = None

def get_easyocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader

def preprocess_image_advanced(image_path):
    """Apply multiple preprocessing techniques to enhance OCR accuracy"""
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different processing techniques
    # 1. Simple binary threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # 2. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # 3. Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Blur and sharpen
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # Save all processed versions
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    processed_paths = {
        'binary': os.path.join(output_dir, f"{base_name}_binary.jpg"),
        'adaptive': os.path.join(output_dir, f"{base_name}_adaptive.jpg"),
        'otsu': os.path.join(output_dir, f"{base_name}_otsu.jpg"),
        'sharpened': os.path.join(output_dir, f"{base_name}_sharpened.jpg")
    }
    
    cv2.imwrite(processed_paths['binary'], binary)
    cv2.imwrite(processed_paths['adaptive'], adaptive)
    cv2.imwrite(processed_paths['otsu'], otsu)
    cv2.imwrite(processed_paths['sharpened'], sharpened)
    
    return processed_paths

def extract_text_with_multiple_engines(image_path):
    """Extract text using multiple OCR engines and techniques"""
    processed_paths = preprocess_image_advanced(image_path)
    
    results = {}
    
    # Tesseract OCR on each processed version
    for version, path in processed_paths.items():
        try:
            text = pytesseract.image_to_string(Image.open(path))
            results[f"tesseract_{version}"] = text.strip()
        except Exception as e:
            results[f"tesseract_{version}"] = f"Error: {str(e)}"
    
    # EasyOCR on original and processed versions
    try:
        reader = get_easyocr_reader()
        original_result = reader.readtext(image_path)
        results["easyocr_original"] = "\n".join([text for _, text, _ in original_result])
        
        # Also try EasyOCR on sharpened version
        sharpened_result = reader.readtext(processed_paths['sharpened'])
        results["easyocr_sharpened"] = "\n".join([text for _, text, _ in sharpened_result])
    except Exception as e:
        results["easyocr"] = f"Error: {str(e)}"
    
    return results