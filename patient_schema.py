from datetime import datetime

class Patient:
    def __init__(self, name, age=None, gender=None):
        self.name = name
        self.age = age
        self.gender = gender
        self.prescriptions = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_prescription(self, prescription_data):
        """Add a prescription to the patient's record"""
        prescription = {
            "date": datetime.now(),
            "medications": prescription_data.get("medications", []),
            "dosages": prescription_data.get("dosages", []),
            "interactions": prescription_data.get("interactions", []),
            "analysis": prescription_data.get("analysis", ""),
            "prescription_text": prescription_data.get("prescription_text", ""),
        }
        
        self.prescriptions.append(prescription)
        self.updated_at = datetime.now()
    
    def to_dict(self):
        """Convert patient object to dictionary for JSON storage"""
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "prescriptions": self.prescriptions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a Patient object from dictionary data"""
        patient = cls(data["name"], data.get("age"), data.get("gender"))
        patient.prescriptions = data.get("prescriptions", [])
        
        # Convert string dates to datetime objects
        if "created_at" in data:
            patient.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            patient.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return patient