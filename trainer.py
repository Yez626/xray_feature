"""
Trainer for self-supervised learning on X-ray images
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import SSLConfig
from models import create_ssl_model
from data_loader import SSLDataLoader

class SSLTrainer:
    """Trainer for self-supervised learning"""
    
    def __init__(self, config: SSLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        self.model = create_ssl_model(config).to(self.device)
        
        # Create data loader
        self.data_loader = SSLDataLoader(config)
        
        # Create optimizer
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
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (x1, x2, _) in enumerate(pbar):
            x1, x2 = x1.to(self.device), x2.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.ssl_method == "simclr":
                z1, z2 = self.model(x1, x2)
                loss = self.model.contrastive_loss(z1, z2, self.config.temperature)
            
            elif self.config.ssl_method == "mae":
                decoded, ids_restore = self.model(x1)
                loss = F.mse_loss(decoded, torch.randn_like(decoded))
            
            elif self.config.ssl_method == "dino":
                student_out, teacher_out = self.model(x1, x2)
                loss = self.model.dino_loss(student_out, teacher_out)
                self.model.update_teacher()
            
            elif self.config.ssl_method == "byol":
                online_out, target_out = self.model(x1, x2)
                loss = self.model.byol_loss(online_out, target_out)
                self.model.update_target()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(train_loader)
    
    def train(self):
        """Main training loop"""
        train_loader = self.data_loader.get_train_loader()
        val_loader = self.data_loader.get_val_loader()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x1, x2, _ in tqdm(val_loader, desc="Validation"):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                if self.config.ssl_method == "simclr":
                    z1, z2 = self.model(x1, x2)
                    loss = self.model.contrastive_loss(z1, z2, self.config.temperature)
                elif self.config.ssl_method == "mae":
                    decoded, ids_restore = self.model(x1)
                    loss = F.mse_loss(decoded, torch.randn_like(decoded))
                elif self.config.ssl_method == "dino":
                    student_out, teacher_out = self.model(x1, x2)
                    loss = self.model.dino_loss(student_out, teacher_out)
                elif self.config.ssl_method == "byol":
                    online_out, target_out = self.model(x1, x2)
                    loss = self.model.byol_loss(online_out, target_out)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config
        }
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        # Handle both full paths and filenames
        if os.path.isabs(checkpoint_path) or checkpoint_path.startswith('./') or checkpoint_path.startswith('../'):
            # It's already a relative or absolute path, use as is
            path = checkpoint_path
        else:
            # It's just a filename, join with checkpoint directory
            path = os.path.join(self.config.checkpoint_dir, checkpoint_path)
        
        print(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Checkpoint loaded from {path}")
    
    def extract_features(self, save_path: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Extract features from the trained model"""
        self.model.eval()
        feature_loader = self.data_loader.get_feature_loader()
        
        print(f"Feature loader has {len(feature_loader)} batches")
        print(f"Batch size: {feature_loader.batch_size}")
        
        features = []
        image_paths = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(feature_loader, desc="Extracting features")):
                print(f"Processing batch {batch_idx}: {type(batch)}, length: {len(batch)}")
                
                # Handle different batch formats
                if len(batch) == 2:
                    x, paths = batch
                    print(f"  Batch format: (x, paths) - x shape: {x.shape}, paths: {len(paths)}")
                elif len(batch) == 3:
                    x, _, paths = batch  # Skip the second augmentation
                    print(f"  Batch format: (x1, x2, paths) - x shape: {x.shape}, paths: {len(paths)}")
                else:
                    print(f"  Unexpected batch format: {batch}")
                    continue
                
                x = x.to(self.device)
                print(f"  Input tensor shape: {x.shape}")
                
                try:
                    # Try to extract features from the model
                    if hasattr(self.model, 'backbone'):
                        print("  Using model.backbone")
                        if hasattr(self.model.backbone, 'forward'):
                            # For models that support return_features
                            try:
                                feat = self.model.backbone(x, return_features=True)
                            except TypeError:
                                # Fallback to regular forward
                                feat = self.model.backbone(x)
                        else:
                            feat = self.model.backbone(x)
                    else:
                        print("  Using model directly")
                        # Try to get features from the main model
                        if hasattr(self.model, 'get_features'):
                            feat = self.model.get_features(x)
                        else:
                            # Use the last layer before projection head
                            feat = self.model(x, return_features=True)
                    
                    print(f"  Feature shape: {feat.shape}")
                    
                    # Ensure features are 2D (batch_size, feature_dim)
                    if feat.dim() > 2:
                        feat = feat.view(feat.size(0), -1)
                        print(f"  Flattened feature shape: {feat.shape}")
                    
                    features.append(feat.cpu().numpy())
                    image_paths.extend(paths)
                    print(f"  Successfully processed batch {batch_idx}")
                    
                except Exception as e:
                    print(f"  Error processing batch {batch_idx}: {e}")
                    continue
        
        print(f"Total features collected: {len(features)}")
        print(f"Total image paths: {len(image_paths)}")
        
        if not features:
            raise ValueError("No features were extracted. Check the model and data loader.")
        
        features = np.concatenate(features, axis=0)
        print(f"Final features shape: {features.shape}")
        
        if save_path:
            np.save(save_path, features)
            with open(save_path.replace('.npy', '_paths.txt'), 'w') as f:
                for path in image_paths:
                    f.write(f"{path}\n")
            print(f"Features saved to {save_path}")
        
        return features, image_paths

