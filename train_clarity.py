"""
Clarity Model Training Script - Optimized Version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ClarityDataset(Dataset):
    """Clarity Dataset"""
    
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
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create blank image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transformations
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Labels
        label = row['label_encoded']
        is_anomaly = row['is_anomaly']
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'is_anomaly': torch.tensor(is_anomaly, dtype=torch.bool),
            'anomaly_type': row['anomaly_type'],
            'path': image_path
        }

class ClarityModel(nn.Module):
    """Clarity Model - Simplified Version"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        # Use pre-trained ViT model
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Feature dimension
        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-Base
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{correct/total:.4f}'
        })
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total:.4f}'
            })
    
    return total_loss / len(val_loader), correct / total, all_predictions, all_labels

def main():
    """Main training function"""
    print("=" * 60)
    print("🚀 Starting Clarity Model Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading data...")
    labels_df = pd.read_csv("clarity_labels.csv")
    print(f"Total data: {len(labels_df)} images")
    print(f"Anomaly type distribution:")
    print(labels_df['anomaly_type'].value_counts())
    
    # Split data
    print("🔄 Splitting data...")
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
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    # Create datasets
    print("🔄 Creating datasets...")
    train_dataset = ClarityDataset(train_df, is_training=True)
    val_dataset = ClarityDataset(val_df, is_training=False)
    test_dataset = ClarityDataset(test_df, is_training=False)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("🏗️ Creating model...")
    model = ClarityModel(num_classes=2, pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training loop
    print("🎯 Starting training...")
    num_epochs = 20
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Training - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_clarity_model.pth')
            print(f"🎉 New best model! Validation accuracy: {best_val_acc:.4f}")
        
        # Periodically save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'clarity_model_epoch_{epoch+1}.pth')
    
    # Final testing
    print("\n🧪 Final testing...")
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Generate classification report
    print("\n📊 Classification report:")
    print(classification_report(test_labels, test_preds, 
                              target_names=['Grade A', 'Counterfeit']))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Grade A', 'Counterfeit'],
                yticklabels=['Grade A', 'Counterfeit'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print("Model saved as: best_clarity_model.pth")

if __name__ == "__main__":
    main()
