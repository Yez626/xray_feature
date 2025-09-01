import argparse
import os
import torch
import numpy as np
from trainer import SSLTrainer
from utils import compute_similarity_matrix, find_similar_images, visualize_features


def main():
    parser = argparse.ArgumentParser(description="Extract features from trained SSL model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing X-ray images")
    parser.add_argument("--output_dir", type=str, default="./features",
                       help="Directory to save extracted features")
    parser.add_argument("--visualize", action="store_true",
                       help="Whether to visualize features")
    
    args = parser.parse_args()
    
    print("🔍 Starting feature extraction...")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print("✅ Output directory created")
    
    try:
        # Load configuration from checkpoint
        print("🔄 Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        config.data_dir = args.data_dir
        
        print(f"✅ Checkpoint loaded from {args.checkpoint_path}")
        print(f"✅ SSL method: {config.ssl_method}")
        print(f"✅ Model: {config.model_name}")
        
        # Create trainer and load checkpoint
        print("🔄 Creating trainer...")
        trainer = SSLTrainer(config)
        print("✅ Trainer created")
        
        print("🔄 Loading checkpoint into trainer...")
        trainer.load_checkpoint(args.checkpoint_path)
        print("✅ Checkpoint loaded into trainer")
        
        # Extract features
        print("🔄 Extracting features...")
        features, image_paths = trainer.extract_features()
        print(f"✅ Features extracted: {features.shape}")
        print(f"✅ Image paths: {len(image_paths)}")
        
        # Save features
        features_path = os.path.join(args.output_dir, f"{config.ssl_method}_{config.model_name}_features.npy")
        np.save(features_path, features)
        print(f"✅ Features saved to {features_path}")
        
        # Save image paths
        paths_path = os.path.join(args.output_dir, f"{config.ssl_method}_{config.model_name}_paths.txt")
        with open(paths_path, 'w') as f:
            for path in image_paths:
                f.write(f"{path}\n")
        print(f"✅ Image paths saved to {paths_path}")
        
        # Compute similarity matrix
        print("🔄 Computing similarity matrix...")
        similarity_matrix = compute_similarity_matrix(features)
        similarity_path = os.path.join(args.output_dir, f"{config.ssl_method}_{config.model_name}_similarity.npy")
        np.save(similarity_path, similarity_matrix)
        print(f"✅ Similarity matrix saved to {similarity_path}")
        
        # Visualize features if requested
        if args.visualize:
            print("🔄 Generating visualization...")
            viz_path = os.path.join(args.output_dir, f"{config.ssl_method}_{config.model_name}_tsne.png")
            visualize_features(features, viz_path)
        
        # Show some example similarities
        print("\n📊 Example image similarities:")
        for i in range(min(3, len(image_paths))):
            similar = find_similar_images(similarity_matrix, i, top_k=3)
            print(f"\nQuery image {i}: {os.path.basename(image_paths[i])}")
            for idx, sim in similar:
                print(f"  Similar: {os.path.basename(image_paths[idx])} (similarity: {sim:.3f})")
        
        print("\n🎉 Feature extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

