#!/usr/bin/env python3
"""
Main script for training self-supervised learning models on X-ray images
"""

import os
import argparse
from config import SSLConfig
from trainer import SSLTrainer

def main():
    parser = argparse.ArgumentParser(description="Train SSL model on X-ray images")
    parser.add_argument("--ssl_method", type=str, default="simclr", 
                       choices=["simclr", "mae", "dino", "byol"],
                       help="SSL method to use")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                       help="Backbone model name")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing X-ray images")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--feature_dim", type=int, default=128,
                       help="Feature dimension for downstream tasks")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SSLConfig(
        ssl_method=args.ssl_method,
        model_name=args.model_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        feature_dim=args.feature_dim
    )
    
    print(f"Training {args.ssl_method} with {args.model_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Feature dimension: {args.feature_dim}")
    
    # Create trainer and start training
    trainer = SSLTrainer(config)
    trainer.train()
    
    print("Training completed!")

if __name__ == "__main__":
    main()


