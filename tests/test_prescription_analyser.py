import pytest
import cv2
import numpy as np
import os
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import tempfile

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.components.enhanced_analyzer import EnhancedPrescriptionAnalyzer

class TestEnhancedPrescriptionAnalyzer:
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a 300x300 white image with black text
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        # Add some text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Amoxicillin 500mg', (50, 50), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, 'Take 1 tablet 3 times daily', (50, 100), font, 0.5, (0, 0, 0), 1)
        return img
    
    @pytest.fixture
    def sample_drug_db(self):
        """Create a sample drug database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write("drug_name,dosage_form\n")
            f.write("amoxicillin,tablet\n")
            f.write("metformin,tablet\n")
            f.write("lisinopril,tablet\n")
            f.write("atorvastatin,tablet\n")
            path = f.name
        return path
    
    @pytest.fixture
    def analyzer(self, sample_drug_db):
        """Create an analyzer instance with the sample drug database"""
        return EnhancedPrescriptionAnalyzer(sample_drug_db)
    
    def test_initialization(self, analyzer, sample_drug_db):
        """Test that the analyzer initializes correctly"""
        assert analyzer is not None
        assert analyzer.drug_db is not None
        assert len(analyzer.drug_db) == 4
        assert 'drug_name' in analyzer.drug_db.columns
        assert analyzer.vectorizer is not None
        
        # Clean up
        os.unlink(sample_drug_db)
    
    def test_preprocess_image(self, analyzer, sample_image):
        """Test image preprocessing functions"""
        preprocessed = analyzer.preprocess_image(sample_image)
        
        # Verify the expected preprocessing variants are present
        assert 'grayscale' in preprocessed
        assert 'adaptive_threshold' in preprocessed
        assert 'otsu_threshold' in preprocessed
        assert 'morphological_cleaning' in preprocessed
        assert 'contrast_enhanced' in preprocessed
        
        # Check that each result is a valid image
        for name, img in preprocessed.items():
            assert img is not None
            assert isinstance(img, np.ndarray)
            assert img.shape[0] > 0 and img.shape[1] > 0
    
    @patch('pytesseract.image_to_string')
    def test_extract_text_from_variants(self, mock_image_to_string, analyzer, sample_image):
        """Test text extraction from image variants"""
        # Setup mock to return predefined text
        mock_image_to_string.return_value = "Amoxicillin 500mg\nTake 1 tablet 3 times daily"
        
        preprocessed = analyzer.preprocess_image(sample_image)
        text_variants = analyzer.extract_text_from_variants(preprocessed)
        
        # Verify that we have text variants
        assert len(text_variants) > 0
        
        # Check that pytesseract was called for each variant
        assert mock_image_to_string.call_count > 0
        
        # Verify text content in variants
        for variant, text in text_variants.items():
            assert "Amoxicillin" in text
            assert "500mg" in text
    
    def test_select_best_text(self, analyzer):
        """Test selection of best text variant"""
        variants = {
            'variant1': "Short text",
            'variant2': "This is a longer text that should be selected",
            'variant3': "Medium text here"
        }
        
        best_text = analyzer.select_best_text(variants)
        assert best_text == "This is a longer text that should be selected"
    
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    def test_extract_medication_info(self, mock_cosine_similarity, analyzer):
        """Test medication extraction from text"""
        # Setup mock for cosine similarity
        mock_cosine_similarity.return_value = np.array([[0.8, 0.3, 0.2, 0.1]])
        
        text = "Patient should take Amoxicillin 500mg three times a day"
        medications = analyzer.extract_medication_info(text)
        
        # Verify that medications were extracted
        assert len(medications) > 0
        assert medications[0]['drug_name'] == 'amoxicillin'
        assert '500mg' in str(medications[0]['dosage'])
        assert medications[0]['confidence'] > 0.5
    
    @patch.object(EnhancedPrescriptionAnalyzer, 'preprocess_image')
    @patch.object(EnhancedPrescriptionAnalyzer, 'extract_text_from_variants')
    @patch.object(EnhancedPrescriptionAnalyzer, 'select_best_text')
    @patch.object(EnhancedPrescriptionAnalyzer, 'extract_medication_info')
    def test_analyze(self, mock_extract_info, mock_select_text, mock_extract_variants, 
                     mock_preprocess, analyzer, sample_image):
        """Test the full analysis pipeline"""
        # Setup mocks
        mock_preprocess.return_value = {'grayscale': np.zeros((100, 100))}
        mock_extract_variants.return_value = {'variant1': "Amoxicillin 500mg"}
        mock_select_text.return_value = "Amoxicillin 500mg"
        mock_extract_info.return_value = [
            {
                'drug_name': 'amoxicillin',
                'dosage': ['500mg'],
                'confidence': 0.8,
                'original_text': "Amoxicillin 500mg"
            }
        ]
        
        # Run analysis
        result = analyzer.analyze(sample_image)
        
        # Verify result structure
        assert 'extracted_text' in result
        assert 'medications' in result
        assert 'processed_images' in result
        assert 'text_variants' in result
        assert 'timestamp' in result
        
        # Verify medications
        assert len(result['medications']) == 1
        assert result['medications'][0]['drug_name'] == 'amoxicillin'
        assert result['medications'][0]['dosage'] == ['500mg']
        assert result['medications'][0]['confidence'] == 0.8
        
        # Verify that all mocks were called
        mock_preprocess.assert_called_once()
        mock_extract_variants.assert_called_once()
        mock_select_text.assert_called_once()
        mock_extract_info.assert_called_once()
        
        # Verify history was updated
        assert len(analyzer.analysis_history) == 1
        assert analyzer.analysis_history[0]['medications_count'] == 1

# tests/test_dashboard.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.components.dashboard import PrescriptionDashboard

class TestPrescriptionDashboard:
    
    @pytest.fixture
    def dashboard(self):
        """Create a dashboard instance"""
        return PrescriptionDashboard()
    
    @pytest.fixture
    def sample_result(self):
        """Create a sample analysis result"""
        return {
            'timestamp': '2025-02-24 10:15:30',
            'extracted_text': 'Amoxicillin 500mg three times daily',
            'medications': [
                {
                    'drug_name': 'amoxicillin',
                    'dosage': ['500mg'],
                    'confidence': 0.85,
                    'original_text': 'Amoxicillin 500mg'
                }
            ],
            'processed_images': {},
            'text_variants': {}
        }
    
    def test_initialization(self, dashboard):
        """Test dashboard initialization"""
        assert dashboard is not None
        assert dashboard.history == []
    
    def test_add_analysis_result(self, dashboard, sample_result):
        """Test adding results to dashboard"""
        dashboard.add_analysis_result(sample_result)
        
        # Verify history was updated
        assert len(dashboard.history) == 1
        assert dashboard.history[0]['num_medications'] == 1
        assert dashboard.history[0]['highest_confidence'] == 0.85
        
        # Add another result
        sample_result2 = sample_result.copy()
        sample_result2['timestamp'] = '2025-02-24 11:30:45'
        sample_result2['medications'].append({
            'drug_name': 'lisinopril',
            'dosage': ['10mg'],
            'confidence': 0.75,
            'original_text': 'Lisinopril 10mg'
        })
        
        dashboard.add_analysis_result(sample_result2)
        
        # Verify history was updated again
        assert len(dashboard.history) == 2
        assert dashboard.history[1]['num_medications'] == 2
        assert dashboard.history[1]['highest_confidence'] == 0.85  # The highest of both medications
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.header')
    @patch('streamlit.subheader')
    @patch('streamlit.altair_chart')
    @patch('streamlit.dataframe')
    @patch('streamlit.info')
    def test_render_dashboard_with_data(self, mock_info, mock_dataframe, mock_chart, 
                                        mock_subheader, mock_header, mock_metric, 
                                        mock_columns, dashboard, sample_result):
        """Test dashboard rendering with data"""
        # Add sample data
        dashboard.add_analysis_result(sample_result)
        
        # Mock columns
        col_mocks = [MagicMock(), MagicMock(), MagicMock()]
        mock_columns.return_value = col_mocks
        
        # Render dashboard
        dashboard.render_dashboard()
        
        # Verify calls
        mock_header.assert_called_once()
        assert mock_metric.call_count >= 3  # At least 3 metrics should be displayed
        mock_dataframe.assert_called_once()  # The history dataframe
    
    @patch('streamlit.info')
    @patch('streamlit.header')
    def test_render_dashboard_without_data(self, mock_header, mock_info, dashboard):
        """Test dashboard rendering without data"""
        dashboard.render_dashboard()
        
        mock_header.assert_called_once()
        mock_info.assert_called_once()

# tests/test_main.py
import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import app.main as main

class TestMain:
    
    @patch('streamlit.sidebar.radio')
    def test_main_navigation(self, mock_radio):
        """Test main navigation functionality"""
        # Test each navigation option
        for page in ["🔍 Analyze Prescription", "📊 Dashboard", "⚙️ Settings", "❓ Help"]:
            mock_radio.return_value = page
            
            # Patch the corresponding page function
            with patch.object(main, f'show_{page.split()[1].lower()}_page') as mock_page:
                main.main()
                mock_page.assert_called_once()
    
    @patch('streamlit.file_uploader')
    @patch('streamlit.image')
    @patch('streamlit.button')
    @patch('cv2.imdecode')
    @patch.object(main, 'process_image')
    def test_show_analysis_page_with_upload(self, mock_process, mock_imdecode, 
                                           mock_button, mock_image, mock_uploader):
        """Test analysis page with file upload"""
        # Mock uploaded file
        mock_file = MagicMock()
        mock_uploader.return_value = mock_file
        
        # Mock button click
        mock_button.return_value = True
        
        # Mock image decoding
        mock_image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imdecode.return_value = mock_image_array
        
        # Call the function
        main.show_analysis_page()
        
        # Verify image processing
        mock_process.assert_called_once()
        # Verify imdecode was called to process the uploaded file
        mock_imdecode.assert_called_once()
    
    @patch('streamlit.sidebar.radio')
    @patch('streamlit.session_state')
    @patch.object(main, 'show_dashboard_page')
    def test_dashboard_display(self, mock_dashboard_page, mock_session_state, mock_radio):
        """Test dashboard display"""
        # Set up navigation
        mock_radio.return_value = "📊 Dashboard"
        
        # Mock session state
        mock_session_state.dashboard = MagicMock()
        mock_session_state.history = [{'sample': 'data'}]
        
        # Call main
        main.main()
        
        # Verify dashboard page was shown
        mock_dashboard_page.assert_called_once()

# tests/test_image_processing.py
import pytest
import cv2
import numpy as np
import os
import sys

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestImageProcessing:
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample prescription image"""
        # Create a 400x300 white image
        img = np.ones((400, 300, 3), dtype=np.uint8) * 255
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Dr. Smith', (50, 50), font, 0.8, (0, 0, 0), 2)
        cv2.putText(img, 'Rx', (30, 100), font, 1, (0, 0, 0), 2)
        cv2.putText(img, 'Amoxicillin 500mg', (50, 150), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img, 'Take 1 tablet 3 times daily', (50, 200), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img, 'for 7 days', (50, 230), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img, 'Signature', (150, 300), font, 0.6, (0, 0, 0), 2)
        
        return img
    
    def test_grayscale_conversion(self, sample_image):
        """Test grayscale conversion"""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        
        # Check image dimensions and type
        assert gray.ndim == 2
        assert gray.shape[0] == sample_image.shape[0]
        assert gray.shape[1] == sample_image.shape[1]
    
    def test_adaptive_thresholding(self, sample_image):
        """Test adaptive thresholding"""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Check image properties
        assert binary.ndim == 2
        assert binary.shape == gray.shape
        assert np.min(binary) == 0  # Should contain black pixels
        assert np.max(binary) == 255  # Should contain white pixels
    
    def test_otsu_thresholding(self, sample_image):
        """Test Otsu's thresholding"""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Check image properties
        assert binary.ndim == 2
        assert binary.shape == gray.shape
        assert np.min(binary) == 0  # Should contain black pixels
        assert np.max(binary) == 255  # Should contain white pixels
    
    def test_morphological_operations(self, sample_image):
        """Test morphological operations"""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        morph_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Check results
        assert morph_cleaned.shape == binary.shape
        
        # The morphological operation should reduce noise
        # Count the number of transitions between 0 and 255
        transitions_binary = np.sum(np.abs(np.diff(binary.flatten())))
        transitions_morph = np.sum(np.abs(np.diff(morph_cleaned.flatten())))
        
        # Morphological cleaning should reduce the number of transitions
        assert transitions_morph <= transitions_binary
    
    def test_contrast_enhancement(self, sample_image):
        """Test contrast enhancement"""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Check results
        assert enhanced.shape == gray.shape
        
        # Calculate histogram for both images
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_enhanced = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
        
        # The enhanced image should have a more spread out histogram
        std_gray = np.std(hist_gray)
        std_enhanced = np.std(hist_enhanced)
        
        # The standard deviation of the enhanced histogram should be higher
        assert std_enhanced >= std_gray * 0.9  # Allow for some tolerance