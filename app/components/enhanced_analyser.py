# app/components/enhanced_analyzer.py
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import pandas as pd
from typing import Dict, List, Tuple
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class EnhancedPrescriptionAnalyzer:
    def __init__(self, drug_database_path: str = None):
        """
        Initialize the enhanced prescription analyzer.
        
        Args:
            drug_database_path: Path to CSV file containing drug database
        """
        # Load drug database if provided
        self.drug_db = None
        self.vectorizer = None
        
        if drug_database_path and os.path.exists(drug_database_path):
            self.load_drug_database(drug_database_path)
        
        # Initialize analysis history
        self.analysis_history = []
    
    def load_drug_database(self, database_path: str):
        """Load and prepare drug database."""
        try:
            self.drug_db = pd.read_csv(database_path)
            # Ensure required column exists
            if 'drug_name' not in self.drug_db.columns:
                # If not found, assume first column is drug name
                self.drug_db = self.drug_db.rename(columns={self.drug_db.columns[0]: 'drug_name'})
            
            # Prepare TF-IDF vectorizer for drug matching
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.drug_vectors = self.vectorizer.fit_transform(self.drug_db['drug_name'].astype(str))
        except Exception as e:
            print(f"Error loading drug database: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> Dict:
        """
        Apply multiple preprocessing techniques and return all variants.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary of processed images with technique names
        """
        results = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results['grayscale'] = gray
        
        # Apply adaptive thresholding
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        results['adaptive_threshold'] = binary_adaptive
        
        # Apply Otsu's thresholding
        _, binary_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        results['otsu_threshold'] = binary_otsu
        
        # Noise removal with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph_cleaned = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)
        results['morphological_cleaning'] = morph_cleaned
        
        # Deskew image
        try:
            coords = np.column_stack(np.where(binary_otsu > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = 90 + angle
                (h, w) = binary_otsu.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                deskewed = cv2.warpAffine(
                    gray, M, (w, h), 
                    flags=cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REPLICATE
                )
                results['deskewed'] = deskewed
        except Exception as e:
            print(f"Deskew error: {e}")
        
        # Edge enhancement
        edges = cv2.Canny(gray, 100, 200)
        results['edge_enhanced'] = edges
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
        results['contrast_enhanced'] = contrast_enhanced
        
        return results
    
    def extract_text_from_variants(self, processed_variants: Dict) -> Dict:
        """
        Extract text from all processed image variants.
        
        Args:
            processed_variants: Dictionary of processed images
            
        Returns:
            Dictionary mapping processing technique to extracted text
        """
        results = {}
        
        # Configure Tesseract for better results with handwriting
        configs = [
            '--oem 3 --psm 6',  # Assume single uniform block of text
            '--oem 3 --psm 4',  # Assume single column of text
            '--oem 3 --psm 3',  # Fully automatic page segmentation
        ]
        
        # Try different combinations of images and configs
        for variant_name, img in processed_variants.items():
            for config in configs:
                key = f"{variant_name}_{config.replace(' ', '_')}"
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    results[key] = text.strip()
                except Exception as e:
                    print(f"Error extracting text from {key}: {e}")
        
        return results
    
    def select_best_text(self, text_variants: Dict) -> str:
        """
        Select the best text variant based on heuristics.
        
        Args:
            text_variants: Dictionary of text variants
            
        Returns:
            Best text variant
        """
        # Simple heuristic: Choose the variant with the most words
        best_text = ""
        max_word_count = 0
        
        for variant, text in text_variants.items():
            word_count = len(text.split())
            if word_count > max_word_count:
                max_word_count = word_count
                best_text = text
        
        return best_text
    
    def extract_medication_info(self, text: str) -> List[Dict]:
        """
        Extract medication information from text.
        
        Args:
            text: Extracted text from prescription
            
        Returns:
            List of dictionaries containing medication information
        """
        medications = []
        
        if not text or not self.drug_db is not None or not self.vectorizer:
            return medications
        
        # Split text into lines and process each line
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 3:  # Skip very short lines
                continue
            
            # Find potential drug matches
            try:
                # Vectorize the line
                line_vector = self.vectorizer.transform([line])
                # Calculate similarities
                similarities = cosine_similarity(line_vector, self.drug_vectors)[0]
                # Get top match
                best_match_idx = similarities.argmax()
                similarity_score = similarities[best_match_idx]
                
                # Only consider matches above threshold
                if similarity_score > 0.3:
                    drug_name = self.drug_db.iloc[best_match_idx]['drug_name']
                    
                    # Extract dosage information
                    dosage_patterns = [
                        r'\d+\s*mg',
                        r'\d+\s*ml',
                        r'\d+\s*tab',
                        r'\d+\s*capsule',
                        r'\d+\s*times?\s*(?:per|a)\s*day',
                    ]
                    
                    dosages = []
                    for pattern in dosage_patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        dosages.extend(match.group() for match in matches)
                    
                    medications.append({
                        'drug_name': drug_name,
                        'dosage': dosages if dosages else ["Not specified"],
                        'confidence': float(similarity_score),
                        'original_text': line
                    })
            except Exception as e:
                print(f"Error matching drug: {e}")
        
        return medications
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Analyze a prescription image comprehensively.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Analysis results dictionary
        """
        # Apply multiple preprocessing techniques
        processed_variants = self.preprocess_image(image)
        
        # Extract text from all variants
        text_variants = self.extract_text_from_variants(processed_variants)
        
        # Select best text
        best_text = self.select_best_text(text_variants)
        
        # Extract medication information
        medications = self.extract_medication_info(best_text)
        
        # Prepare result
        result = {
            'extracted_text': best_text,
            'medications': medications,
            'processed_images': processed_variants,
            'text_variants': text_variants,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to history
        self.analysis_history.append({
            'timestamp': result['timestamp'],
            'medications_count': len(medications),
            'text_length': len(best_text)
        })
        
        return result
    
    def get_history_data(self) -> pd.DataFrame:
        """Get analysis history as DataFrame for dashboard."""
        return pd.DataFrame(self.analysis_history)


# app/components/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from typing import Dict, List

class PrescriptionDashboard:
    def __init__(self):
        """Initialize the prescription dashboard."""
        self.history = []
        
    def add_analysis_result(self, result: Dict):
        """Add analysis result to dashboard history."""
        if 'timestamp' in result and 'medications' in result:
            entry = {
                'timestamp': result['timestamp'],
                'num_medications': len(result['medications']),
                'text_length': len(result['extracted_text']),
                'highest_confidence': max([med['confidence'] for med in result['medications']]) if result['medications'] else 0,
            }
            self.history.append(entry)
    
    def render_dashboard(self):
        """Render the dashboard in Streamlit."""
        st.header("Prescription Analysis Dashboard")
        
        if not self.history:
            st.info("No analysis data available yet. Analyze some prescriptions to populate the dashboard.")
            return
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        
        # Display basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Prescriptions Analyzed", len(df))
        with col2:
            st.metric("Avg. Medications per Prescription", f"{df['num_medications'].mean():.1f}")
        with col3:
            st.metric("Avg. Confidence Score", f"{df['highest_confidence'].mean():.2f}")
        
        # Time series analysis
        if len(df) > 1:
            st.subheader("Analysis History")
            
            # Create time series chart
            chart = alt.Chart(df).mark_line().encode(
                x='timestamp:T',
                y='num_medications:Q',
                tooltip=['timestamp', 'num_medications', 'highest_confidence']
            ).properties(
                width=600,
                height=300,
                title='Medications Detected Over Time'
            )
            
            st.altair_chart(chart, use_container_width=True)
        
        # Display latest results
        st.subheader("Recent Analysis Results")
        st.dataframe(df.tail(10).sort_values('timestamp', ascending=False))

# app/main.py
import streamlit as st
import cv2
import numpy as np
from components.enhanced_analyzer import EnhancedPrescriptionAnalyzer
from components.dashboard import PrescriptionDashboard
import os
import pandas as pd
from PIL import Image
import io

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = EnhancedPrescriptionAnalyzer()
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = PrescriptionDashboard()
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'history' not in st.session_state:
    st.session_state.history = []

def main():
    st.title("Aushadhi: Enhanced Prescription Analysis System")
    
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Analyze Prescription", "Dashboard", "Settings"])
    
    if page == "Analyze Prescription":
        show_analysis_page()
    elif page == "Dashboard":
        show_dashboard_page()
    else:
        show_settings_page()

def show_analysis_page():
    st.header("Prescription Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a prescription image", type=['png', 'jpg', 'jpeg'])
    
    # Display sample image option
    use_sample = st.checkbox("Use sample image instead")
    
    if uploaded_file is not None and not use_sample:
        # Process uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display original image
        st.image(uploaded_file, caption='Uploaded Prescription', use_column_width=True)
        
        if st.button('Analyze Prescription'):
            process_image(image)
    
    elif use_sample:
        # Use sample image
        sample_path = os.path.join("sample_data", "prescription_sample.jpg")
        
        # Check if sample exists
        if os.path.exists(sample_path):
            image = cv2.imread(sample_path)
            st.image(sample_path, caption='Sample Prescription', use_column_width=True)
            
            if st.button('Analyze Sample Prescription'):
                process_image(image)
        else:
            st.error(f"Sample image not found at {sample_path}")
    
    # Display current result if available
    if st.session_state.current_result:
        display_result(st.session_state.current_result)

def process_image(image):
    """Process image and update state."""
    with st.spinner('Analyzing prescription...'):
        result = st.session_state.analyzer.analyze(image)
        st.session_state.current_result = result
        
        # Update dashboard
        st.session_state.dashboard.add_analysis_result(result)
        st.session_state.history.append(result)

def display_result(result):
    """Display analysis result."""
    st.subheader("Analysis Results")
    
    # Display tabs for different result views
    tab1, tab2, tab3 = st.tabs(["Medications", "Processed Images", "Raw Text"])
    
    with tab1:
        if result['medications']:
            st.write(f"Detected {len(result['medications'])} medications:")
            
            for i, med in enumerate(result['medications']):
                with st.expander(f"{i+1}. {med['drug_name']} ({med['confidence']:.2f} confidence)"):
                    st.write(f"**Drug:** {med['drug_name']}")
                    st.write(f"**Dosage:** {', '.join(med['dosage'])}")
                    st.write(f"**Confidence:** {med['confidence']:.2f}")
                    st.write(f"**Original Text:** {med['original_text']}")
        else:
            st.info("No medications detected. Try adjusting the image or using a clearer prescription.")
    
    with tab2:
        # Display image processing variants
        cols = st.columns(2)
        
        for i, (name, img) in enumerate(result['processed_images'].items()):
            with cols[i % 2]:
                st.image(img, caption=name, use_column_width=True)
    
    with tab3:
        st.subheader("Extracted Text")
        st.text_area("", result['extracted_text'], height=200)
        
        with st.expander("View all text variants"):
            for technique, text in result['text_variants'].items():
                st.write(f"**{technique}:**")
                st.text(text)

def show_dashboard_page():
    """Display the analytics dashboard."""
    st.session_state.dashboard.render_dashboard()

def show_settings_page():
    """Display settings page."""
    st.header("Settings")
    
    # Drug database upload
    st.subheader("Drug Database")
    db_file = st.file_uploader("Upload drug database (CSV format)", type=['csv'])
    
    if db_file:
        try:
            # Save temporarily
            with open("temp_drug_db.csv", "wb") as f:
                f.write(db_file.getbuffer())
            
            # Load database
            st.session_state.analyzer.load_drug_database("temp_drug_db.csv")
            st.success("Drug database loaded successfully!")
            
            # Preview
            db = pd.read_csv("temp_drug_db.csv")
            st.write("Database Preview:")
            st.dataframe(db.head())
        except Exception as e:
            st.error(f"Error loading database: {e}")
    
    # OCR settings
    st.subheader("OCR Settings")
    tesseract_path = st.text_input("Tesseract Path", value=pytesseract.pytesseract.tesseract_cmd)
    
    if st.button("Update Tesseract Path"):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        st.success(f"Tesseract path updated to: {tesseract_path}")

if __name__ == "__main__":
    main()