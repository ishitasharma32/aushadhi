import os
import argparse
from advanced_ocr import extract_text_with_multiple_engines
from ocr_module import extract_text_from_image
from medication_checker import identify_medications, extract_dosage_info, check_interactions

def test_single_image(image_path):
    """Test OCR on a single image"""
    print(f"Testing OCR on image: {image_path}")
    
    # Test basic OCR
    print("\n=== Basic OCR ===")
    basic_result = extract_text_from_image(image_path)
    print(basic_result)
    
    # Test advanced OCR
    print("\n=== Advanced OCR ===")
    advanced_results = extract_text_with_multiple_engines(image_path)
    
    for engine, text in advanced_results.items():
        print(f"\n--- {engine} ---")
        print(text)
    
    # Test medication identification
    best_result = advanced_results.get("easyocr_sharpened", "") or advanced_results.get("tesseract_adaptive", "") or basic_result
    
    print("\n=== Medication Identification ===")
    identified_meds = identify_medications(best_result)
    
    if identified_meds:
        for med in identified_meds:
            print(f"Found: {med['name']}")
        
        print("\n=== Dosage Information ===")
        dosage_info = extract_dosage_info(best_result, identified_meds)
        
        for dose in dosage_info:
            print(f"{dose['medication']}:")
            print(f"  Found dosage: {dose['found_dosage']}")
            print(f"  Standard dosage: {dose['standard_dosage']}")
            print(f"  Maximum daily: {dose['max_daily']}")
        
        print("\n=== Interaction Check ===")
        interactions = check_interactions(identified_meds)
        
        if interactions:
            for interaction in interactions:
                print(f"Interaction between: {', '.join(interaction['medications'])}")
                print(f"  Severity: {interaction['severity']}")
                print(f"  Description: {interaction['description']}")
        else:
            print("No interactions found")
    else:
        print("No medications identified")

def test_dataset(dataset_dir):
    """Test OCR on a directory of images"""
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(dataset_dir) 
                  if os.path.isfile(os.path.join(dataset_dir, f)) and 
                  any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"No image files found in {dataset_dir}")
        return
    
    print(f"Found {len(image_files)} image files in dataset")
    
    for i, image_file in enumerate(image_files):
        print(f"\n\n{'='*50}")
        print(f"Testing image {i+1}/{len(image_files)}: {image_file}")
        print(f"{'='*50}\n")
        
        image_path = os.path.join(dataset_dir, image_file)
        test_single_image(image_path)

def main():
    parser = argparse.ArgumentParser(description='Test OCR capabilities')
    parser.add_argument('--image', help='Path to a single prescription image')
    parser.add_argument('--dataset', help='Path to a directory containing prescription images')
    args = parser.parse_args()
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            return
        test_single_image(args.image)
    elif args.dataset:
        test_dataset(args.dataset)
    else:
        print("Please provide either --image or --dataset argument")
        print("Example: python test_ocr.py --image path/to/image.jpg")
        print("Example: python test_ocr.py --dataset test_data")

if __name__ == "__main__":
    main()