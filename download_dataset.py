import os
import kaggle
from dotenv import load_dotenv
import zipfile
import shutil

# Load environment variables for Kaggle API credentials
load_dotenv()

def download_kaggle_dataset():
    """Download the Illegible Medical Prescription Images Dataset from Kaggle"""
    print("Downloading dataset from Kaggle...")
    
    # Create test_data directory if it doesn't exist
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
    
    # Download the dataset using the Kaggle API
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mehaksingal/illegible-medical-prescription-images-dataset",
            path="test_data",
            unzip=True
        )
        print("Dataset downloaded and extracted successfully!")
        
        # Move files from subdirectory to test_data if needed
        dataset_dir = os.path.join("test_data", "illegible-medical-prescription-images")
        if os.path.exists(dataset_dir):
            print("Moving files to test_data directory...")
            for filename in os.listdir(dataset_dir):
                shutil.move(
                    os.path.join(dataset_dir, filename),
                    os.path.join("test_data", filename)
                )
            os.rmdir(dataset_dir)
            
        print(f"Dataset ready in test_data directory! Found {len(os.listdir('test_data'))} files.")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nManual download instructions:")
        print("1. Go to https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset")
        print("2. Click 'Download'")
        print("3. Extract the ZIP file")
        print("4. Place the images in the test_data directory of this project")

if __name__ == "__main__":
    download_kaggle_dataset()