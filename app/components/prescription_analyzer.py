# app/components/prescription_analyzer.py
import cv2
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple

class PrescriptionAnalyzer:
    def __init__(self):
        """Initialize the prescription analyzer."""
        # Configure Tesseract for better results with handwriting
        self.custom_config = r'--oem 3 --psm 6'
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to improve text extraction."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(cleaned)
        
        return enhanced
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew."""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from the preprocessed image."""
        try:
            text = pytesseract.image_to_string(image, config=self.custom_config)
            return text
        except Exception as e:
            print(f"Error in text extraction: {e}")
            return ""

    def analyze(self, image: np.ndarray) -> Dict:
        """Main analysis function."""
        # Create a copy of the image
        processed = image.copy()
        
        # Apply preprocessing
        processed = self.preprocess_image(processed)
        
        # Deskew if needed
        processed = self.deskew(processed)
        
        # Extract text
        text = self.extract_text(processed)
        
        return {
            'processed_image': processed,
            'extracted_text': text,
            'preprocessing_steps': [
                'Grayscale conversion',
                'Adaptive thresholding',
                'Noise removal',
                'Contrast enhancement',
                'Deskew correction'
            ]
        }

# app/main.py
import streamlit as st
import cv2
import numpy as np
from components.prescription_analyzer import PrescriptionAnalyzer
import os

def load_image(image_path: str) -> np.ndarray:
    """Load and return an image."""
    return cv2.imread(image_path)

def main():
    st.title("Illegible Prescription Analysis System")
    
    # Initialize analyzer
    analyzer = PrescriptionAnalyzer()
    
    # File upload option
    st.header("Upload Prescription")
    uploaded_file = st.file_uploader("Choose a prescription image", type=['jpg', 'jpeg', 'png'])
    
    # Dataset path input
    st.header("Or Use Dataset Images")
    dataset_path = st.text_input("Enter dataset path", "path/to/dataset")
    
    if uploaded_file is not None:
        # Handle uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(uploaded_file, caption='Original Prescription', use_column_width=True)
        
        if st.button('Analyze Uploaded Prescription'):
            analyze_and_display(image, analyzer)
            
    elif os.path.exists(dataset_path):
        # Handle dataset images
        images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        selected_image = st.selectbox("Select an image from dataset", images)
        
        if selected_image:
            image_path = os.path.join(dataset_path, selected_image)
            image = load_image(image_path)
            st.image(image, caption='Selected Prescription', use_column_width=True)
            
            if st.button('Analyze Selected Prescription'):
                analyze_and_display(image, analyzer)

def analyze_and_display(image: np.ndarray, analyzer: PrescriptionAnalyzer):
    """Analyze image and display results."""
    with st.spinner('Analyzing prescription...'):
        results = analyzer.analyze(image)
        
        # Display processed image
        st.subheader("Processed Image")
        st.image(results['processed_image'], caption='Processed Image', use_column_width=True)
        
        # Display preprocessing steps
        st.subheader("Preprocessing Steps Applied")
        for step in results['preprocessing_steps']:
            st.write(f"✓ {step}")
        
        # Display extracted text
        st.subheader("Extracted Text")
        st.text_area("", results['extracted_text'], height=200)

if __name__ == '__main__':
    main()

