import os
import google.generativeai as genai
from medication_checker import identify_medications, extract_dosage_info, check_interactions

def analyze_prescription(prescription_text, ocr_results=None):
    """
    Analyze prescription using medication dictionary and Gemini Pro.
    """
    # First, use our medication dictionary
    identified_meds = identify_medications(prescription_text)
    
    # Extract dosage information
    dosage_info = extract_dosage_info(prescription_text, identified_meds)
    
    # Check for drug interactions
    interactions = check_interactions(identified_meds)
    
    # Now use Gemini Pro for enhanced analysis
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return {
            "error": "GOOGLE_API_KEY not found in environment variables",
            "local_analysis": {
                "medications": identified_meds,
                "dosages": dosage_info,
                "interactions": interactions
            }
        }
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Different OCR engines might have different results, so include all
    all_ocr_text = prescription_text
    if ocr_results:
        all_ocr_text += "\n\nAdditional OCR Results:\n"
        for engine, text in ocr_results.items():
            all_ocr_text += f"\n--- {engine} ---\n{text}\n"
    
    # Create a prompt for Gemini
    prompt = f"""
    You are Aushadhi, a prescription analysis assistant. Analyze this prescription text extracted using OCR:
    
    {all_ocr_text}
    
    Based on your medical knowledge:
    1. Identify all medications mentioned in the prescription
    2. For each medication, determine the prescribed dosage
    3. Verify if the dosages appear to be within normal ranges
    4. Note any potential drug interactions
    5. List common side effects to watch for
    6. Provide clear patient instructions for taking these medications
    
    If the OCR text seems unclear or incomplete, make your best assessment but note the uncertainty.
    """
    
   # Call Gemini API
    try:
        # First try with the standard model name
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            gemini_analysis = response.text
        except Exception as e:
            # If that fails, try to list available models and use the first suitable one
            print(f"Error with gemini-pro model: {str(e)}")
            print("Trying to list available models...")
            
            models = genai.list_models()
            gemini_models = [m.name for m in models if 'gemini' in m.name.lower()]
            
            if gemini_models:
                print(f"Found alternative models: {gemini_models}")
                model = genai.GenerativeModel(gemini_models[0])
                response = model.generate_content(prompt)
                gemini_analysis = response.text
            else:
                # If no Gemini models are available, fall back to basic analysis
                raise Exception("No Gemini models available")
        
        return {
            "llm_analysis": gemini_analysis,
            "local_analysis": {
                "medications": identified_meds,
                "dosages": dosage_info,
                "interactions": interactions
            }
        }
    except Exception as e:
        print(f"Error using Gemini API: {str(e)}")
        print("Falling back to local analysis only")
        return {
            "error": str(e),
            "local_analysis": {
                "medications": identified_meds,
                "dosages": dosage_info,
                "interactions": interactions
            }
        }