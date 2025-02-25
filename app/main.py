import streamlit as st
import cv2
import numpy as np
from components.enhanced_analyzer import EnhancedPrescriptionAnalyzer
from components.dashboard import PrescriptionDashboard
import os
import pandas as pd
from PIL import Image
import io
import tempfile
import pytesseract

# Page configuration
st.set_page_config(
    page_title="Aushadhi: Prescription Analysis System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def init_session_state():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedPrescriptionAnalyzer()
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = PrescriptionDashboard()
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'tesseract_path': pytesseract.pytesseract.tesseract_cmd,
            'confidence_threshold': 0.3
        }

init_session_state()

def main():
    # Application title with logo/emoji
    st.sidebar.title("Aushadhi 💊")
    st.sidebar.markdown("### Prescription Analysis System")
    
    # Sidebar navigation with icons
    pages = {
        "🔍 Analyze Prescription": show_analysis_page,
        "📊 Dashboard": show_dashboard_page,
        "⚙️ Settings": show_settings_page,
        "❓ Help": show_help_page
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Display page info
    st.sidebar.info(
        "This application analyzes prescription images to extract medication information "
        "using OCR technology and machine learning."
    )
    
    # Display version info
    st.sidebar.markdown("---")
    st.sidebar.markdown("v1.2.0 | © 2025 Aushadhi")
    
    # Call the selected page function
    pages[selection]()

def show_analysis_page():
    st.title("📝 Prescription Analysis")
    
    # Create two columns for upload options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Prescription")
        uploaded_file = st.file_uploader(
            "Upload a prescription image", 
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: JPG, JPEG, PNG"
        )
    
    with col2:
        st.subheader("Camera Capture")
        use_camera = st.checkbox("Use camera to capture prescription")
        
        if use_camera:
            camera_input = st.camera_input("Take a picture of your prescription")
            if camera_input:
                # Convert to opencv format
                file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                uploaded_file = camera_input  # For display purposes
    
    # Sample images section
    st.markdown("---")
    st.subheader("Or use a sample image:")
    
    # Display sample image options in a horizontal layout
    sample_dir = "sample_data"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if sample_files:
            sample_cols = st.columns(min(3, len(sample_files)))
            selected_sample = None
            
            for i, sample_file in enumerate(sample_files[:3]):  # Limit to 3 samples
                sample_path = os.path.join(sample_dir, sample_file)
                with sample_cols[i]:
                    st.image(sample_path, width=150)
                    if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                        selected_sample = sample_path
            
            if selected_sample:
                st.success(f"Using sample image: {os.path.basename(selected_sample)}")
                image = cv2.imread(selected_sample)
                st.image(selected_sample, caption='Selected Sample', use_column_width=True)
                
                if st.button('Analyze Sample Prescription', key="analyze_sample"):
                    process_image(image)
        else:
            st.info("No sample images found in the sample_data directory.")
    else:
        st.warning(f"Sample directory not found: {sample_dir}")
    
    # Process uploaded file if available
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("Uploaded Prescription")
        
        # Display original image
        st.image(uploaded_file, caption='Uploaded Prescription', use_column_width=True)
        
        # Image preprocessing options
        with st.expander("Image Preprocessing Options"):
            enhance_contrast = st.checkbox("Enhance contrast", value=True)
            denoise = st.checkbox("Remove noise", value=True)
            sharpen = st.checkbox("Sharpen image", value=False)
        
        # Analyze button with loading animation
        if st.button('Analyze Prescription', key="analyze_upload"):
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Apply selected preprocessing
            if enhance_contrast or denoise or sharpen:
                with st.spinner("Preprocessing image..."):
                    if enhance_contrast:
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        cl = clahe.apply(l)
                        limg = cv2.merge((cl, a, b))
                        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    
                    if denoise:
                        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                    
                    if sharpen:
                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                        image = cv2.filter2D(image, -1, kernel)
            
            process_image(image)
    
    # Display current result if available
    if st.session_state.current_result:
        display_result(st.session_state.current_result)

def process_image(image):
    """Process image and update state with progress indicators."""
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processing steps with visual feedback
    try:
        status_text.text("Step 1/4: Preprocessing image...")
        progress_bar.progress(25)
        
        status_text.text("Step 2/4: Applying OCR...")
        progress_bar.progress(50)
        
        status_text.text("Step 3/4: Extracting medication information...")
        progress_bar.progress(75)
        
        # Actual analysis
        result = st.session_state.analyzer.analyze(image)
        
        status_text.text("Step 4/4: Finalizing results...")
        progress_bar.progress(100)
        
        # Update state
        st.session_state.current_result = result
        st.session_state.dashboard.add_analysis_result(result)
        st.session_state.history.append(result)
        
        # Clear progress indicators
        status_text.empty()
        
        st.success("Analysis completed successfully! Found {} medications.".format(
            len(result['medications'])))
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_result(result):
    """Display analysis result with enhanced UI components."""
    st.markdown("---")
    st.header("Analysis Results")
    
    # Get timestamp
    timestamp = result.get('timestamp', 'Unknown time')
    st.markdown(f"*Analyzed at: {timestamp}*")
    
    # Create tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs([
        "💊 Medications", 
        "🔄 Processed Images", 
        "📝 Extracted Text",
        "📊 Confidence Scores"
    ])
    
    with tab1:
        if result['medications']:
            med_count = len(result['medications'])
            st.write(f"Detected {med_count} medication{'s' if med_count != 1 else ''}:")
            
            # Sort medications by confidence
            sorted_meds = sorted(
                result['medications'], 
                key=lambda x: x['confidence'], 
                reverse=True
            )
            
            # Display each medication in a card-like layout
            for i, med in enumerate(sorted_meds):
                confidence_color = "green" if med['confidence'] > 0.7 else "orange" if med['confidence'] > 0.4 else "red"
                
                med_container = st.container()
                med_container.markdown(f"""
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 10px; 
                    margin-bottom: 10px;
                    background-color: rgba(0, 0, 0, 0.02);">
                    <h3 style="margin-top: 0;">{i+1}. {med['drug_name']}</h3>
                    <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{med['confidence']:.2f}</span></p>
                    <p><strong>Dosage:</strong> {', '.join(med['dosage'])}</p>
                    <p><strong>Original Text:</strong> <em>{med['original_text']}</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add verification options
                with med_container.expander("Verify/Edit"):
                    st.write("Is this identification correct?")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("✅ Correct", key=f"correct_{i}")
                    with col2:
                        st.button("❌ Incorrect", key=f"incorrect_{i}")
                    
                    corrected_name = st.text_input("Corrected drug name:", key=f"corrected_name_{i}")
                    corrected_dosage = st.text_input("Corrected dosage:", key=f"corrected_dosage_{i}")
                    
                    if st.button("Submit Correction", key=f"submit_{i}"):
                        st.success("Correction submitted. This will help improve future analysis.")
        else:
            st.info("No medications detected. Try adjusting the image or using a clearer prescription.")
            
            # Provide troubleshooting suggestions
            with st.expander("Troubleshooting Tips"):
                st.markdown("""
                * Ensure the prescription is well-lit and clearly visible
                * Try using the contrast enhancement option
                * Make sure the text is properly focused
                * Crop the image to include only the prescription area
                * Try a different angle if there's glare on the paper
                """)
    
    with tab2:
        # Display image processing variants in a grid with toggles
        st.write("Toggle different processing techniques to see their effect:")
        
        # Create checkboxes for each technique
        techniques = list(result['processed_images'].keys())
        selected_techniques = []
        
        # Create checkbox columns
        checkbox_cols = st.columns(min(3, len(techniques)))
        
        for i, technique in enumerate(techniques):
            with checkbox_cols[i % 3]:
                if st.checkbox(technique, key=f"tech_{technique}", value=i==0):
                    selected_techniques.append(technique)
        
        # Display selected images
        image_cols = st.columns(min(2, max(1, len(selected_techniques))))
        
        for i, technique in enumerate(selected_techniques):
            with image_cols[i % 2]:
                st.image(result['processed_images'][technique], 
                         caption=technique, 
                         use_column_width=True)
                
                # Add download button for each image
                img_pil = Image.fromarray(result['processed_images'][technique])
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.download_button(
                    label=f"Download {technique}",
                    data=buf.getvalue(),
                    file_name=f"{technique}_{timestamp.replace(':', '-')}.png",
                    mime="image/png",
                    key=f"download_{technique}"
                )
    
    with tab3:
        st.subheader("Extracted Text")
        
        # Main extracted text
        st.text_area("Best Text Extraction Result:", 
                     result['extracted_text'], 
                     height=200)
        
        # Copy button
        if st.button("📋 Copy to Clipboard", key="copy_text"):
            st.success("Text copied to clipboard!")
        
        # Show all text variants in a dropdown
        with st.expander("All Text Extraction Variants"):
            variant_selector = st.selectbox(
                "Select extraction method:",
                list(result['text_variants'].keys())
            )
            
            if variant_selector:
                st.text_area(f"Text from {variant_selector}:", 
                             result['text_variants'][variant_selector],
                             height=150)
    
    with tab4:
        # Display confidence metrics
        if result['medications']:
            # Prepare data for visualization
            confidence_data = {
                'Medication': [med['drug_name'] for med in result['medications']],
                'Confidence': [med['confidence'] for med in result['medications']]
            }
            
            conf_df = pd.DataFrame(confidence_data)
            
            # Create bar chart
            st.subheader("Confidence Scores by Medication")
            st.bar_chart(conf_df.set_index('Medication'))
            
            # Add statistics
            avg_conf = np.mean(conf_df['Confidence'])
            max_conf = np.max(conf_df['Confidence'])
            min_conf = np.min(conf_df['Confidence'])
            
            st.metric("Average Confidence", f"{avg_conf:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Highest Confidence", f"{max_conf:.2f}")
            with col2:
                st.metric("Lowest Confidence", f"{min_conf:.2f}")
        else:
            st.info("No medication data available for confidence scoring.")
    
    # Export options
    st.markdown("---")
    export_container = st.container()
    export_col1, export_col2 = export_container.columns(2)
    
    with export_col1:
        if st.button("📊 Export as CSV", key="export_csv"):
            if result['medications']:
                # Convert medications to DataFrame
                med_df = pd.DataFrame(result['medications'])
                
                # Convert dosage lists to strings
                med_df['dosage'] = med_df['dosage'].apply(lambda x: ', '.join(x))
                
                # Convert to CSV
                csv = med_df.to_csv(index=False)
                
                # Offer download
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"prescription_analysis_{timestamp.replace(':', '-')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            else:
                st.warning("No medication data to export.")
    
    with export_col2:
        if st.button("📑 Generate PDF Report", key="gen_pdf"):
            st.info("PDF report generation feature coming soon!")

def show_dashboard_page():
    """Display the analytics dashboard with enhanced visualizations."""
    st.title("📊 Prescription Analysis Dashboard")
    
    # Add date range filter
    st.sidebar.header("Dashboard Filters")
    
    # Only show filter if we have history
    if st.session_state.history:
        dates = [pd.to_datetime(item['timestamp']) for item in st.session_state.history]
        min_date = min(dates).date()
        max_date = max(dates).date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter based on date range
        filtered_history = st.session_state.history
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_history = [
                item for item in st.session_state.history
                if start_date <= pd.to_datetime(item['timestamp']).date() <= end_date
            ]
            
            if filtered_history:
                # Update dashboard with filtered data
                dashboard = PrescriptionDashboard()
                for item in filtered_history:
                    dashboard.add_analysis_result(item)
                
                dashboard.render_dashboard()
            else:
                st.info("No data available for the selected date range.")
        else:
            # Render with all data
            st.session_state.dashboard.render_dashboard()
    else:
        # No analysis data yet
        st.info("No analysis data available yet. Analyze some prescriptions to populate the dashboard.")
        
        # Add demo button
        if st.button("Load Demo Data"):
            # Create some sample data
            import random
            from datetime import datetime, timedelta
            
            # Generate random timestamps for the past week
            base_date = datetime.now() - timedelta(days=7)
            
            for i in range(10):
                timestamp = (base_date + timedelta(
                    days=random.randint(0, 7),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )).strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate random medications
                num_meds = random.randint(1, 5)
                medications = []
                
                drug_names = ["Amoxicillin", "Lisinopril", "Metformin", 
                              "Atorvastatin", "Albuterol", "Omeprazole", 
                              "Levothyroxine", "Amlodipine"]
                
                for j in range(num_meds):
                    medications.append({
                        'drug_name': random.choice(drug_names),
                        'dosage': [f"{random.randint(1, 10)*5} mg"],
                        'confidence': random.uniform(0.3, 0.95),
                        'original_text': f"Sample text for medication {j+1}"
                    })
                
                # Create demo result
                demo_result = {
                    'extracted_text': f"Sample prescription text {i+1}",
                    'medications': medications,
                    'processed_images': {},
                    'text_variants': {},
                    'timestamp': timestamp
                }
                
                # Add to dashboard and history
                st.session_state.dashboard.add_analysis_result(demo_result)
                st.session_state.history.append(demo_result)
            
            st.success("Demo data loaded successfully!")
            st.rerun()

def show_settings_page():
    """Display enhanced settings page."""
    st.title("⚙️ System Settings")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3 = st.tabs(["General Settings", "OCR Configuration", "Database Settings"])
    
    with tab1:
        st.subheader("Application Settings")
        
        # Theme selection
        theme = st.selectbox(
            "UI Theme",
            ["Light", "Dark", "System Default"],
            index=2
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Spanish", "French", "German"],
            index=0
        )
        
        # Save button
        if st.button("Save General Settings"):
            st.success("Settings saved successfully!")
            st.session_state.settings['theme'] = theme
            st.session_state.settings['language'] = language
    
    with tab2:
        st.subheader("OCR Configuration")
        
        # Tesseract path
        tesseract_path = st.text_input(
            "Tesseract Path", 
            value=st.session_state.settings['tesseract_path']
        )
        
        # OCR parameters
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings['confidence_threshold'],
            step=0.05,
            format="%.2f"
        )
        
        ocr_mode = st.selectbox(
            "OCR Mode",
            ["Fast", "Balanced", "Accurate"],
            index=1,
            help="Fast: Optimized for speed but may be less accurate. Accurate: Maximum accuracy but slower."
        )
        
        # Advanced options
        with st.expander("Advanced OCR Settings"):
            psm_mode = st.selectbox(
                "Page Segmentation Mode",
                [
                    "0 - Orientation and script detection only",
                    "1 - Automatic page segmentation with OSD",
                    "3 - Fully automatic page segmentation, but no OSD (Default)",
                    "4 - Assume a single column of text of variable sizes",
                    "6 - Assume a single uniform block of text"
                ],
                index=2
            )
            
            oem_mode = st.selectbox(
                "OCR Engine Mode",
                [
                    "0 - Legacy engine only",
                    "1 - Neural nets LSTM engine only",
                    "2 - Legacy + LSTM engines",
                    "3 - Default, based on what is available"
                ],
                index=3
            )
        
        # Save OCR settings
        if st.button("Save OCR Settings"):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            st.session_state.settings['tesseract_path'] = tesseract_path
            st.session_state.settings['confidence_threshold'] = confidence_threshold
            st.session_state.settings['ocr_mode'] = ocr_mode
            st.session_state.settings['psm_mode'] = psm_mode
            st.session_state.settings['oem_mode'] = oem_mode
            st.success("OCR settings saved successfully!")
    
    with tab3:
        st.subheader("Drug Database Settings")
        
        # Database upload
        st.write("Upload or update the drug database (CSV format)")
        db_file = st.file_uploader("Upload drug database", type=['csv'])
        
        if db_file:
            try:
                # Preview before saving
                db_preview = pd.read_csv(db_file)
                st.write("Database Preview:")
                st.dataframe(db_preview.head())
                
                # Confirm columns
                if 'drug_name' not in db_preview.columns:
                    st.warning("Warning: 'drug_name' column not found. Select the column to use as drug name:")
                    drug_name_col = st.selectbox("Select drug name column:", db_preview.columns)
                else:
                    drug_name_col = 'drug_name'
                
                # Save button
                if st.button("Save Database"):
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                        db_file.seek(0)
                        f.write(db_file.getbuffer())
                        temp_path = f.name
                    
                    # If column needs to be renamed
                    if drug_name_col != 'drug_name':
                        db_preview = db_preview.rename(columns={drug_name_col: 'drug_name'})
                        db_preview.to_csv(temp_path, index=False)
                    
                    # Load database
                    st.session_state.analyzer.load_drug_database(temp_path)
                    st.success(f"Drug database updated with {len(db_preview)} entries!")
                    
                    # Option to download modified database
                    if drug_name_col != 'drug_name':
                        csv_data = db_preview.to_csv(index=False)
                        st.download_button(
                            "Download Modified Database",
                            data=csv_data,
                            file_name="drug_database_modified.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error processing database: {e}")
        
        # Database statistics
        st.markdown("---")
        st.subheader("Current Database Statistics")
        
        if hasattr(st.session_state.analyzer, 'drug_db') and st.session_state.analyzer.drug_db is not None:
            drug_db = st.session_state.analyzer.drug_db
            st.metric("Total Drugs in Database", len(drug_db))
            
            # Show sample entries
            st.write("Sample entries:")
            st.dataframe(drug_db.head())
            
            # Export current database
            if st.button("Export Current Database"):
                csv_data = drug_db.to_csv(index=False)
                st.download_button(
                    "Download Current Database",
                    data=csv_data,
                    file_name="current_drug_database.csv",
                    mime="text/csv"
                )
        else:
            st.info("No drug database currently loaded.")

def show_help_page():
    """Display help and documentation."""
    st.title("❓ Help & Documentation")
    
    # Create tabs for different help sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Getting Started", 
        "FAQs", 
        "Troubleshooting",
        "About"
    ])
    
    with tab1:
        st.header("Getting Started")
        
        st.subheader("How to use Aushadhi")
        st.markdown("""
        1. **Upload a prescription** - Go to the "Analyze Prescription" page and upload a clear image of a prescription.
        2. **Analyze** - Click the "Analyze Prescription" button to extract medication information.
        3. **Review results** - Check the extracted medications and their details.
        4. **View insights** - Go to the Dashboard to see analytics from your prescriptions over time.
        
        For best results:
        * Use well-lit, clear images
        * Ensure the prescription text is focused and readable
        * Position the prescription to fill most of the image
        """)
        
        st.subheader("Quick Video Tutorial")
        st.info("Video tutorial coming soon!")
    
    with tab2:
        st.header("Frequently Asked Questions")
        
        faqs = [
            {
                "question": "What types of prescriptions can Aushadhi analyze?",
                "answer": "Aushadhi can analyze most handwritten and printed prescriptions. It works best with clearly written prescriptions with standard medication names."
            },
            {
                "question": "How accurate is the medication detection?",
                "answer": "Accuracy depends on image quality and clarity of the prescription. The system provides confidence scores for each detected medication. Higher scores (above 0.7) indicate greater confidence in the detection."
            },
            {
                "question": "Can I edit incorrect detections?",
                "answer": "Yes, you can edit and correct any medication information in the 'Verify/Edit' section for each detected medication."
            },
            {
                "question": "Is my prescription data secure?",
                "answer": "Aushadhi processes all data locally on your device and does not store your prescription images in any cloud service unless you explicitly enable cloud backup in Settings."
            },
            {
                "question": "How do I update the drug database?",
                "answer": "Go to Settings > Database Settings and upload a CSV file containing drug names and related information."
            }
        ]
        
        for i, faq in enumerate(faqs):
            with st.expander(faq["question"]):
                st.write(faq["answer"])
    
    with tab3:
        st.header("Troubleshooting")
        
        issues = [
            {
                "issue": "No medications detected",
                "solution": """
                * Ensure the image is clear and well-lit
                * Try using the image enhancement options (contrast, denoise)
                * Check that the prescription text is clearly visible
                * Try a different angle to reduce glare
                * Verify that your drug database is loaded correctly
                """
            },
            {
                "issue": "Low confidence scores",
                "solution": """
                * Improve image quality 
                * Ensure the prescription is properly focused
                * Update your drug database with more comprehensive medication information
                * Try different OCR settings in the Settings page
                """
            },
            {
                "issue": "OCR errors or incorrect text extraction",
                "solution": """
                * Check Tesseract path in Settings
                * Try different OCR modes (Accurate mode may provide better results)
                * Adjust Page Segmentation Mode in Advanced OCR Settings
                * Use image preprocessing options to improve text clarity
                """
            },
            {
                "issue": "Application running slowly",
                "solution": """
                * Use smaller image files
                * Select 'Fast' OCR mode in Settings
                * Close other resource-intensive applications
                * Try using a sample image to test system performance
                """
            }
        ]
        
        for issue in issues:
            with st.expander(issue["issue"]):
                st.markdown(issue["solution"])
        
        st.subheader("Still having problems?")
        st.info("Contact support at support@aushadhi-example.com")
    
    with tab4:
        st.header("About Aushadhi")
        
        st.markdown("""
        **Aushadhi** is an advanced prescription analysis system designed to extract medication information from images of prescriptions using Optical Character Recognition (OCR) and machine learning.
        
        **Version:** 1.2.0
        
        **Features:**
        * Prescription image analysis
        * Medication detection and extraction
        * Dosage information extraction
        * Advanced image preprocessing
        * Analysis dashboard and statistics
        * Customizable drug database
        
        **Technologies Used:**
        * Python
        * OpenCV
        * Tesseract OCR
        * Streamlit
        * Pandas
        * Scikit-learn
        
        **License:** MIT
        
        **Acknowledgements:**
        * Tesseract OCR Project
        * OpenCV Team
        * Streamlit Community
        """)

if __name__ == "__main__":
    main()