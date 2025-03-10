import cv2
import pytesseract
from PIL import Image
import os

def preprocess_image(image_path):
    """Preprocess the image to improve OCR results."""
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get image with only black and white
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Save the preprocessed image
    preprocessed_path = f"{os.path.splitext(image_path)[0]}_preprocessed.jpg"
    cv2.imwrite(preprocessed_path, thresh)
    
    return preprocessed_path

def extract_text_from_image(image_path):
    """Extract text from image using Tesseract OCR."""
    # Preprocess the image
    preprocessed_path = preprocess_image(image_path)
    
    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(Image.open(preprocessed_path))
    
    # Clean up the extracted text
    text = text.strip()
    
    return text

def parse_prescription(text):
    """
    Parse prescription text to extract key information.
    This is a simplified version. In a real system, you would use
    more sophisticated NLP techniques or Gemini API.
    """
    # Split text into lines
    lines = text.split('\n')
    
    # Basic parsing - actual implementation would be more robust
    prescription_info = {
        'patient_name': None,
        'medications': [],
        'dosages': [],
        'instructions': []
    }
    
    medication_keywords = ['tablet', 'capsule', 'mg', 'ml', 'injection']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # This is a very simplified parsing logic
        if any(keyword in line.lower() for keyword in medication_keywords):
            prescription_info['medications'].append(line)
            
            # Extract dosage (very simplified)
            dosage_parts = [part for part in line.split() if part.isdigit() or 'mg' in part or 'ml' in part]
            if dosage_parts:
                prescription_info['dosages'].append(' '.join(dosage_parts))
            else:
                prescription_info['dosages'].append('Unknown dosage')
                
    return prescription_info
