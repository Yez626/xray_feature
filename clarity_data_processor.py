"""
Clarity Data Processor - Extract labels and barcode information from folder structure
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

@dataclass
class ImageInfo:
    """Image information data class"""
    path: str
    filename: str
    product_name: str
    model: str
    condition: str
    barcode: Optional[str] = None
    anomaly_type: str = "Unknown"
    is_anomaly: bool = False

class ClarityDataProcessor:
    """Clarity Data Processor - Extract labels from folder structure"""
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.image_info_list: List[ImageInfo] = []
        
        # Anomaly type mapping
        self.anomaly_mapping = {
            "Good": "Grade A",
            "GradeA": "Grade A", 
            "Mint": "Grade A",
            "Counterfeit": "Counterfeit",
            # Anomaly types that need manual annotation
            "Incomplete": "Incomplete",
            "Defective": "Defective"
        }
    
    def extract_barcode_from_filename(self, filename: str) -> Optional[str]:
        """Extract barcode from filename"""
        # Match numeric barcode at the beginning of filename
        match = re.match(r'^(\d{8,20})_', filename)
        if match:
            return match.group(1)
        return None
    
    def parse_folder_name(self, folder_name: str) -> Tuple[str, str, str]:
        """Parse folder name"""
        # Format: ProductName_Model_Condition
        parts = folder_name.split('_')
        
        if len(parts) >= 3:
            # Handle multi-part product names
            if len(parts) > 3:
                product_name = '_'.join(parts[:-2])
                model = parts[-2]
                condition = parts[-1]
            else:
                product_name = parts[0]
                model = parts[1]
                condition = parts[2]
        else:
            # Handle simple format
            product_name = parts[0] if len(parts) > 0 else "Unknown"
            model = parts[1] if len(parts) > 1 else "Unknown"
            condition = parts[2] if len(parts) > 2 else "Unknown"
        
        return product_name, model, condition
    
    def process_directory(self, directory: str = None) -> List[ImageInfo]:
        """Process all images in directory"""
        if directory is None:
            directory = self.data_root
        
        image_info_list = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    # Get relative path
                    rel_path = os.path.relpath(os.path.join(root, file), self.data_root)
                    
                    # Parse path information
                    path_parts = rel_path.split(os.sep)
                    folder_name = path_parts[-2] if len(path_parts) > 1 else "Unknown"
                    
                    # Parse folder name
                    product_name, model, condition = self.parse_folder_name(folder_name)
                    
                    # Extract barcode
                    barcode = self.extract_barcode_from_filename(file)
                    
                    # Determine anomaly type
                    anomaly_type = self.anomaly_mapping.get(condition, "Unknown")
                    is_anomaly = anomaly_type not in ["Grade A", "Unknown"]
                    
                    # Create image information
                    image_info = ImageInfo(
                        path=os.path.join(root, file),
                        filename=file,
                        product_name=product_name,
                        model=model,
                        condition=condition,
                        barcode=barcode,
                        anomaly_type=anomaly_type,
                        is_anomaly=is_anomaly
                    )
                    
                    image_info_list.append(image_info)
        
        self.image_info_list = image_info_list
        return image_info_list
    
    def get_statistics(self) -> Dict:
        """Get data statistics"""
        if not self.image_info_list:
            self.process_directory()
        
        stats = {
            "total_images": len(self.image_info_list),
            "anomaly_types": {},
            "products": {},
            "barcodes": {},
            "conditions": {}
        }
        
        for info in self.image_info_list:
            # Anomaly type statistics
            stats["anomaly_types"][info.anomaly_type] = stats["anomaly_types"].get(info.anomaly_type, 0) + 1
            
            # Product statistics
            product_key = f"{info.product_name}_{info.model}"
            stats["products"][product_key] = stats["products"].get(product_key, 0) + 1
            
            # Barcode statistics
            if info.barcode:
                stats["barcodes"][info.barcode] = stats["barcodes"].get(info.barcode, 0) + 1
            
            # Condition statistics
            stats["conditions"][info.condition] = stats["conditions"].get(info.condition, 0) + 1
        
        return stats
    
    def create_dataframe(self) -> pd.DataFrame:
        """Create dataframe"""
        if not self.image_info_list:
            self.process_directory()
        
        data = []
        for info in self.image_info_list:
            data.append({
                "path": info.path,
                "filename": info.filename,
                "product_name": info.product_name,
                "model": info.model,
                "condition": info.condition,
                "barcode": info.barcode,
                "anomaly_type": info.anomaly_type,
                "is_anomaly": info.is_anomaly
            })
        
        return pd.DataFrame(data)
    
    def get_images_by_barcode(self, barcode: str) -> List[ImageInfo]:
        """Get images by barcode"""
        return [info for info in self.image_info_list if info.barcode == barcode]
    
    def get_images_by_anomaly_type(self, anomaly_type: str) -> List[ImageInfo]:
        """Get images by anomaly type"""
        return [info for info in self.image_info_list if info.anomaly_type == anomaly_type]
    
    def get_gallery_images(self, barcode: str) -> List[ImageInfo]:
        """Get gallery images (normal state images)"""
        return [info for info in self.image_info_list 
                if info.barcode == barcode and info.anomaly_type == "Grade A"]
    
    def get_query_images(self, barcode: str) -> List[ImageInfo]:
        """Get query images (all state images)"""
        return [info for info in self.image_info_list if info.barcode == barcode]
    
    def save_labels(self, output_path: str):
        """Save labels to file"""
        df = self.create_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Labels saved to: {output_path}")
    
    def print_summary(self):
        """Print data summary"""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("CLARITY Data Summary")
        print("=" * 60)
        print(f"Total images: {stats['total_images']}")
        
        print("\nAnomaly type distribution:")
        for anomaly_type, count in stats["anomaly_types"].items():
            print(f"  {anomaly_type}: {count}")
        
        print(f"\nNumber of product types: {len(stats['products'])}")
        print(f"Unique barcodes: {len(stats['barcodes'])}")
        
        print("\nCondition distribution:")
        for condition, count in stats["conditions"].items():
            print(f"  {condition}: {count}")
        
        print("\nTop 10 products:")
        sorted_products = sorted(stats["products"].items(), key=lambda x: x[1], reverse=True)
        for product, count in sorted_products[:10]:
            print(f"  {product}: {count}")

def main():
    """Main function - Process data and display summary"""
    # Set data path
    data_root = "./extracted_data/clarity_dataset_v6"
    
    # Create processor
    processor = ClarityDataProcessor(data_root)
    
    # Process data
    print("Processing data...")
    image_info_list = processor.process_directory()
    
    # Display summary
    processor.print_summary()
    
    # Save labels
    processor.save_labels("./clarity_labels.csv")
    
    # Show some examples
    print("\nExample data:")
    df = processor.create_dataframe()
    print(df.head(10))
    
    # Group by barcode example
    print("\nGroup by barcode example:")
    barcode_groups = df.groupby('barcode').size().head(5)
    for barcode, count in barcode_groups.items():
        print(f"Barcode {barcode}: {count} images")

if __name__ == "__main__":
    main()
