# A simple medication dictionary with common medications and dosage ranges
# In a production system, this would be a proper database
MEDICATION_DICT = {
    "paracetamol": {
        "aliases": ["acetaminophen", "tylenol", "crocin", "dolo"],
        "normal_dose": "500-1000mg, 3-4 times per day",
        "max_daily": "4000mg",
        "forms": ["tablet", "syrup", "injection"]
    },
    "amoxicillin": {
        "aliases": ["amox", "amoxil"],
        "normal_dose": "250-500mg, 3 times per day",
        "max_daily": "1500mg",
        "forms": ["capsule", "tablet", "syrup"]
    },
    "metformin": {
        "aliases": ["glucophage", "glycomet"],
        "normal_dose": "500-1000mg, 2-3 times per day",
        "max_daily": "2550mg",
        "forms": ["tablet", "extended-release"]
    },
    "atorvastatin": {
        "aliases": ["lipitor", "atorva"],
        "normal_dose": "10-80mg once daily",
        "max_daily": "80mg",
        "forms": ["tablet"]
    },
    "ibuprofen": {
        "aliases": ["advil", "motrin", "brufen"],
        "normal_dose": "200-400mg, 3-4 times per day",
        "max_daily": "3200mg",
        "forms": ["tablet", "capsule", "liquid"]
    },
    "aspirin": {
        "aliases": ["ecosprin", "disprin"],
        "normal_dose": "75-325mg daily",
        "max_daily": "4000mg",
        "forms": ["tablet", "enteric-coated"]
    },
    "omeprazole": {
        "aliases": ["prilosec", "omez"],
        "normal_dose": "20-40mg once daily",
        "max_daily": "120mg",
        "forms": ["capsule", "tablet"]
    },
    "levothyroxine": {
        "aliases": ["synthroid", "eltroxin"],
        "normal_dose": "50-200mcg once daily",
        "max_daily": "200mcg",
        "forms": ["tablet"]
    }
}

# Common drug interactions
DRUG_INTERACTIONS = [
    {
        "drugs": ["metformin", "atorvastatin"],
        "severity": "moderate",
        "description": "May increase risk of muscle pain, weakness or breakdown (myopathy)"
    },
    {
        "drugs": ["paracetamol", "warfarin"],
        "severity": "moderate",
        "description": "May increase bleeding risk when used regularly"
    },
    {
        "drugs": ["ibuprofen", "aspirin"],
        "severity": "moderate",
        "description": "Increased risk of gastrointestinal bleeding"
    },
    {
        "drugs": ["omeprazole", "clopidogrel"],
        "severity": "severe",
        "description": "May reduce the effectiveness of clopidogrel"
    },
    {
        "drugs": ["levothyroxine", "calcium"],
        "severity": "moderate",
        "description": "Calcium may reduce absorption of levothyroxine"
    }
]

def identify_medications(text):
    """Identify possible medications from text"""
    text_lower = text.lower()
    identified_meds = []
    
    for med, info in MEDICATION_DICT.items():
        # Check for the medication name
        if med in text_lower:
            identified_meds.append({"name": med, "info": info})
        
        # Check for aliases
        for alias in info["aliases"]:
            if alias in text_lower and med not in [x["name"] for x in identified_meds]:
                identified_meds.append({"name": med, "info": info})
                break
    
    return identified_meds

def extract_dosage_info(text, medications):
    """Extract dosage information for identified medications"""
    results = []
    
    for med in medications:
        med_name = med["name"]
        med_info = med["info"]
        
        # Simple regex-free approach using string operations
        text_lower = text.lower()
        
        # Find medication name position
        med_pos = text_lower.find(med_name)
        if med_pos == -1:
            # Try aliases
            for alias in med_info["aliases"]:
                med_pos = text_lower.find(alias)
                if med_pos != -1:
                    break
        
        if med_pos != -1:
            # Look for dosage information after the medication name
            # This is a simplified approach - a real system would use more robust NLP
            after_med = text_lower[med_pos:med_pos+100]  # Look at next 100 chars
            
            # Look for mg or ml
            dosage = "Unknown"
            for unit in ["mg", "ml", "mcg", "g"]:
                unit_pos = after_med.find(unit)
                if unit_pos != -1:
                    # Look for numbers before the unit
                    start = unit_pos
                    while start > 0 and (after_med[start-1].isdigit() or after_med[start-1] == '.'):
                        start -= 1
                    
                    if start < unit_pos:
                        dosage = after_med[start:unit_pos+len(unit)]
                        break
            
            results.append({
                "medication": med_name,
                "found_dosage": dosage,
                "standard_dosage": med_info["normal_dose"],
                "max_daily": med_info["max_daily"]
            })
    
    return results

def check_interactions(medications):
    """Check for potential drug interactions"""
    interactions = []
    med_names = [med["name"] for med in medications]
    
    for interaction in DRUG_INTERACTIONS:
        # Check if at least 2 drugs from the interaction list are present
        matching_drugs = [drug for drug in interaction["drugs"] if drug in med_names]
        if len(matching_drugs) >= 2:
            interactions.append({
                "medications": matching_drugs,
                "severity": interaction["severity"],
                "description": interaction["description"]
            })
    
    return interactions