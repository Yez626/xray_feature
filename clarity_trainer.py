"""
Clarity Trainer - Train model using extracted labels
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

from clarity_models import ClarityFeatureExtractor, SimilarityMatcher
from clarity_config import ClarityConfig
from clarity_data_processor import ClarityDataProcessor

class ClarityDataset(Dataset):
    """Clarity Dataset - Using extracted labels"""
    
    def __init__(self, dataframe: pd.DataFrame, image_size: int = 224, is_training: bool = True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.is_training = is_training
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.dataframe['label_encoded'] = self.label_encoder.fit_transform(self.dataframe['anomaly_type'])
        
        # Data augmentation
        if is_training:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent=0.1,
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, 
                        contrast_limit=0.2
                    ),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ], p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load image
        image_path = row['path']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transformations
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Labels
        label = row['label_encoded']
        is_anomaly = row['is_anomaly']
        barcode = row['barcode']
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'is_anomaly': torch.tensor(is_anomaly, dtype=torch.bool),
            'barcode': barcode,
            'anomaly_type': row['anomaly_type'],
            'path': image_path
        }

class ClarityTrainer:
    """Clarity Trainer"""
    
    def __init__(self, config: ClarityConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        self.model = ClarityFeatureExtractor(
            backbone_name=config.backbone_name,
            feature_dim=config.feature_dim,
            num_anomaly_classes=config.num_anomaly_classes,
            pretrained=config.pretrained
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def prepare_data(self, labels_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data"""
        # Split data
        train_df, temp_df = train_test_split(
            labels_df, 
            test_size=0.3, 
            random_state=42,
            stratify=labels_df['anomaly_type']
        )
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            random_state=42,
            stratify=temp_df['anomaly_type']
        )
        
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        # Create datasets
        train_dataset = ClarityDataset(train_df, self.config.image_size, is_training=True)
        val_dataset = ClarityDataset(val_df, self.config.image_size, is_training=False)
        test_dataset = ClarityDataset(test_df, self.config.image_size, is_training=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images, labels)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def train(self, labels_df: pd.DataFrame):
        """Main training loop"""
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(labels_df)
        
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint("best_model.pth")
                print(f"New best model saved! Validation accuracy: {best_val_acc:.4f}")
            
            # Periodically save checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        # Final testing
        test_metrics = self.validate(test_loader)
        print(f"\nFinal test results:")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        return test_metrics
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")

def main():
    """Main function"""
    # Load configuration
    config = ClarityConfig()
    
    # Load label data
    if os.path.exists("clarity_labels.csv"):
        labels_df = pd.read_csv("clarity_labels.csv")
        print(f"Loaded {len(labels_df)} labels")
    else:
        print("Label file not found, please run clarity_data_processor.py first")
        return
    
    # Create trainer
    trainer = ClarityTrainer(config)
    
    # Start training
    print("Starting training...")
    test_metrics = trainer.train(labels_df)
    
    print("Training completed!")
    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
