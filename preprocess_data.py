#!/usr/bin/env python3
"""
Script to preprocess X-ray dataset for self-supervised learning
"""

import os
import zipfile
import shutil
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def extract_zip(zip_path, extract_dir):
    """Extract zip file"""
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed!")

def convert_to_rgb(image_path):
    """Convert image to RGB format"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(image_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def resize_image(image_path, target_size=(224, 224)):
    """Resize image to target size"""
    try:
        with Image.open(image_path) as img:
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                img.save(image_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return False

def preprocess_images(data_dir, target_size=(224, 224)):
    """Preprocess all images in the directory"""
    print(f"Preprocessing images in {data_dir}...")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Process images
    processed_count = 0
    failed_count = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Convert to RGB
        if convert_to_rgb(image_path):
            # Resize image
            if resize_image(image_path, target_size):
                processed_count += 1
            else:
                failed_count += 1
        else:
            failed_count += 1
    
    print(f"Processing completed!")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")

def organize_data(data_dir, output_dir):
    """Organize data into a flat structure"""
    print(f"Organizing data from {data_dir} to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    # Copy files to output directory with new names
    for i, image_path in enumerate(tqdm(image_files, desc="Organizing images")):
        # Get file extension
        _, ext = os.path.splitext(image_path)
        
        # Create new filename
        new_filename = f"xray_{i:06d}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        
        # Copy file
        shutil.copy2(image_path, new_path)
    
    print(f"Data organization completed! {len(image_files)} files copied to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess X-ray dataset for SSL")
    parser.add_argument("--zip_path", type=str, default="clarity_dataset_v6.zip",
                       help="Path to zip file containing X-ray images")
    parser.add_argument("--extract_dir", type=str, default="./extracted_data",
                       help="Directory to extract zip contents")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Final organized data directory")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Target image size")
    parser.add_argument("--skip_extraction", action="store_true",
                       help="Skip zip extraction if already done")
    
    args = parser.parse_args()
    
    # Extract zip file
    if not args.skip_extraction:
        extract_zip(args.zip_path, args.extract_dir)
    
    # Preprocess images
    preprocess_images(args.extract_dir, (args.image_size, args.image_size))
    
    # Organize data
    organize_data(args.extract_dir, args.output_dir)
    
    print(f"\nDataset preprocessing completed!")
    print(f"Final data directory: {args.output_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"You can now use this directory for SSL training with --data_dir {args.output_dir}")

if __name__ == "__main__":
    main()


