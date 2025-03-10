import streamlit as st
import os
import pandas as pd
from datetime import datetime
import json
from ocr_module import extract_text_from_image, parse_prescription
from advanced_ocr import extract_text_with_multiple_engines
from analysis_module import analyze_prescription
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Aushadhi",
    page_icon="💊",
    layout="wide"
)

# Initialize session state for patient dashboard
if 'patients' not in st.session_state:
    st.session_state.patients = []

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the uploads directory."""
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def main():
    st.title("Aushadhi")
    st.markdown("### Prescription Analysis & Medication Management System")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Analyze Prescription", "Patient Dashboard", "About"])
    
    with tab1:
        st.header("Upload Prescription")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose a prescription image", type=["jpg", "jpeg", "png"])
            
            # Option to use a test image from the dataset
            use_test_image = st.checkbox("Use test image from dataset")
            if use_test_image:
                if os.path.exists("test_data") and any(os.listdir("test_data")):
                    test_images = [f for f in os.listdir("test_data") 
                                if os.path.isfile(os.path.join("test_data", f)) and 
                                any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
                    if test_images:
                        selected_test_image = st.selectbox("Select test image", test_images)
                        test_image_path = os.path.join("test_data", selected_test_image)
                        st.image(test_image_path, caption="Selected Test Image", width=400)
                    else:
                        st.warning("No test images found. Please download the dataset first.")
                else:
                    st.warning("Test data directory not found or empty. Please run download_dataset.py first.")
            
            if not use_test_image and uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Prescription", width=400)
            
            ocr_method = st.radio(
                "OCR Method",
                ["Basic OCR", "Advanced Multi-Engine OCR"],
                index=0
            )
        
        with col2:
            if (uploaded_file is not None or (use_test_image and 'selected_test_image' in locals())):
                file_path = None
                if uploaded_file is not None:
                    # Save the uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                elif use_test_image and 'selected_test_image' in locals():
                    file_path = os.path.join("test_data", selected_test_image)
                
                if file_path and st.button("Analyze Prescription"):
                    with st.spinner("Processing with OCR..."):
                        if ocr_method == "Basic OCR":
                            # Extract text from the image
                            extracted_text = extract_text_from_image(file_path)
                            ocr_results = None
                        else:
                            # Use advanced multi-engine OCR
                            ocr_results = extract_text_with_multiple_engines(file_path)
                            # Use the best result as the primary text
                            extracted_text = ocr_results.get("easyocr_sharpened", "") or ocr_results.get("tesseract_adaptive", "")
                        
                        # Display extracted text
                        st.subheader("Extracted Text")
                        st.text_area("OCR Result", extracted_text, height=200)
                        
                        # If using advanced OCR, show toggle for all results
                        if ocr_method == "Advanced Multi-Engine OCR" and ocr_results:
                            with st.expander("View all OCR results"):
                                for engine, text in ocr_results.items():
                                    st.markdown(f"**{engine}:**")
                                    st.text(text)
                                    st.markdown("---")
                        
                        # Analyze with local system and Gemini
                        with st.spinner("Analyzing prescription..."):
                            analysis_result = analyze_prescription(extracted_text, ocr_results)
                            
                            if "error" in analysis_result and "llm_analysis" not in analysis_result:
                                st.error(f"Error with Gemini API: {analysis_result['error']}")
                                st.info("Showing local analysis only")
                            
                            # Display local analysis results
                            st.subheader("Medication Analysis")
                            
                            local_analysis = analysis_result.get("local_analysis", {})
                            
                            # Medications identified
                            meds = local_analysis.get("medications", [])
                            if meds:
                                st.markdown("#### Medications Identified")
                                for med in meds:
                                    st.markdown(f"**{med['name'].title()}**")
                            else:
                                st.info("No medications identified in the prescription text")
                            
                            # Dosage information
                            dosages = local_analysis.get("dosages", [])
                            if dosages:
                                st.markdown("#### Dosage Information")
                                for dose in dosages:
                                    st.markdown(f"**{dose['medication'].title()}**")
                                    st.markdown(f"- Found dosage: {dose['found_dosage']}")
                                    st.markdown(f"- Standard dosage: {dose['standard_dosage']}")
                                    st.markdown(f"- Maximum daily: {dose['max_daily']}")
                            
                            # Interactions
                            interactions = local_analysis.get("interactions", [])
                            if interactions:
                                st.markdown("#### Potential Drug Interactions")
                                for interaction in interactions:
                                    st.markdown(f"**{', '.join(interaction['medications']).title()}**")
                                    st.markdown(f"- Severity: {interaction['severity']}")
                                    st.markdown(f"- {interaction['description']}")
                            
                            # Gemini analysis
                            if "llm_analysis" in analysis_result:
                                st.subheader("Comprehensive Analysis (Gemini)")
                                st.markdown(analysis_result["llm_analysis"])
                            
                            # Add to patient dashboard
                            st.subheader("Add to Patient Records")
                            patient_name = st.text_input("Patient Name")
                            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
                            patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])
                            
                            if st.button("Save to Patient Records"):
                                if patient_name:
                                    # Create a record with all the data
                                    patient_record = {
                                        "name": patient_name,
                                        "age": patient_age,
                                        "gender": patient_gender,
                                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        "prescription_text": extracted_text,
                                        "medications": [med["name"] for med in meds],
                                        "dosages": dosages,
                                        "interactions": interactions,
                                        "analysis": analysis_result.get("llm_analysis", "No Gemini analysis available")
                                    }
                                    
                                    # Add to session state
                                    st.session_state.patients.append(patient_record)
                                    
                                    # Save to a JSON file as well for persistence
                                    try:
                                        if not os.path.exists("data"):
                                            os.makedirs("data")
                                        
                                        file_path = os.path.join("data", "patients.json")
                                        
                                        # Load existing data if available
                                        if os.path.exists(file_path):
                                            with open(file_path, "r") as f:
                                                patients_data = json.load(f)
                                        else:
                                            patients_data = []
                                        
                                        # Add new patient
                                        patients_data.append(patient_record)
                                        
                                        # Save updated data
                                        with open(file_path, "w") as f:
                                            json.dump(patients_data, f, indent=2)
                                        
                                        st.success(f"Added {patient_name} to patient records")
                                    except Exception as e:
                                        st.warning(f"Note: Could not save to permanent storage: {str(e)}")
                                        st.success(f"Added {patient_name} to session (temporary storage)")
                                else:
                                    st.error("Please enter a patient name")
    
    with tab2:
        st.header("Patient Dashboard")
        
        # Load patients from file if session is empty
        if not st.session_state.patients and os.path.exists(os.path.join("data", "patients.json")):
            try:
                with open(os.path.join("data", "patients.json"), "r") as f:
                    st.session_state.patients = json.load(f)
            except Exception as e:
                st.error(f"Error loading patient data: {str(e)}")
        
        if not st.session_state.patients:
            st.info("No patients in the dashboard yet. Analyze prescriptions to add patients.")
        else:
            # Create a simple dataframe for the dashboard
            patient_summary = []
            for p in st.session_state.patients:
                med_count = len(p.get("medications", []))
                interaction_count = len(p.get("interactions", []))
                
                patient_summary.append({
                    "Name": p["name"],
                    "Age": p.get("age", "N/A"),
                    "Gender": p.get("gender", "N/A"),
                    "Date": p["date"],
                    "Medications": med_count,
                    "Interactions": interaction_count
                })
            
            patients_df = pd.DataFrame(patient_summary)
            
            # Display search and filter
            st.subheader("Patient Search")
            search_term = st.text_input("Search by name", "")
            
            if search_term:
                filtered_df = patients_df[patients_df["Name"].str.contains(search_term, case=False)]
            else:
                filtered_df = patients_df
            
            # Display the dashboard
            st.subheader("Patient List")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Patient details
            if st.session_state.patients:
                st.subheader("Patient Details")
                selected_patient = st.selectbox(
                    "Select Patient", 
                    options=[p["name"] for p in st.session_state.patients]
                )
                
                patient_data = next((p for p in st.session_state.patients if p["name"] == selected_patient), None)
                
                if patient_data:
                    # Display patient info in columns
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Name:** {patient_data['name']}")
                    with col2:
                        st.markdown(f"**Age:** {patient_data.get('age', 'N/A')}")
                    with col3:
                        st.markdown(f"**Gender:** {patient_data.get('gender', 'N/A')}")
                    
                    st.markdown(f"**Date:** {patient_data['date']}")
                    
                    # Medications tab
                    with st.expander("Current Medications", expanded=True):
                        if "dosages" in patient_data and patient_data["dosages"]:
                            for dose in patient_data["dosages"]:
                                st.markdown(f"**{dose['medication'].title()}**")
                                st.markdown(f"- Dosage: {dose['found_dosage']}")
                                st.markdown(f"- Standard: {dose['standard_dosage']}")
                        elif "medications" in patient_data and patient_data["medications"]:
                            for med in patient_data["medications"]:
                                st.markdown(f"- {med.title()}")
                        else:
                            st.info("No medication information available")
                    
                    # Interactions
                    if "interactions" in patient_data and patient_data["interactions"]:
                        with st.expander("Drug Interactions"):
                            for interaction in patient_data["interactions"]:
                                st.markdown(f"**{', '.join(interaction['medications']).title()}**")
                                st.markdown(f"- Severity: {interaction['severity']}")
                                st.markdown(f"- {interaction['description']}")
                    
                    # Full analysis
                    with st.expander("Full Analysis"):
                        st.markdown(patient_data.get("analysis", "No analysis available"))
                    
                    # Original prescription
                    with st.expander("Original Prescription Text"):
                        st.text(patient_data.get("prescription_text", "Not available"))
    
    with tab3:
        st.header("About Aushadhi")
        st.markdown("""
        ### Prescription Analysis & Medication Management System
        
        Aushadhi is a powerful tool designed to help healthcare professionals and patients manage medications safely and effectively.
        
        **Key Features:**
        - Prescription OCR: Extract text from prescription images
        - Multi-engine analysis: Utilize multiple OCR techniques for accuracy
        - Medication identification and dosage checking
        - Drug interaction detection
        - Patient medication tracking dashboard
        - AI-powered comprehensive analysis
        
        **Technologies Used:**
        - Computer Vision & OCR (Tesseract, EasyOCR)
        - Google Gemini AI for comprehensive analysis
        - Streamlit for the user interface
        - Python for backend processing
        
        **Dataset:**
        The system can be tested with the Illegible Medical Prescription Images Dataset from Kaggle, which contains various prescription images for testing OCR capabilities.
        
        **Note:** This system is intended for educational and demonstration purposes. Always consult with healthcare professionals regarding medication decisions.
        """)

if __name__ == "__main__":
    main()