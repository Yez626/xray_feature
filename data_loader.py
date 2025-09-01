"""
Data loader for X-ray images with SSL augmentations
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional
import random

class XRayDataset(Dataset):
    """X-ray dataset for self-supervised learning"""
    
    def __init__(self, data_dir: str, image_size: int = 224, is_training: bool = True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_training = is_training
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.image_files.extend([
                os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if f.lower().endswith(ext)
            ])
        
        # SSL augmentations
        if is_training:
            self.augmentations = self._get_ssl_augmentations()
        else:
            self.augmentations = self._get_test_augmentations()
    
    def _get_ssl_augmentations(self):
        """Get augmentations for self-supervised learning"""
        return A.Compose([
            A.RandomResizedCrop(
                size=(self.image_size, self.image_size), 
                scale=(0.2, 1.0)
            ),
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
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2
                ),
                A.RandomGamma(gamma_limit=(80, 120)),
                A.CLAHE(clip_limit=2.0),
            ], p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def _get_test_augmentations(self):
        """Get augmentations for testing/evaluation"""
        return A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply augmentations
        if self.is_training:
            # For SSL, we need two augmented views
            aug1 = self.augmentations(image=image)['image']
            aug2 = self.augmentations(image=image)['image']
            return aug1, aug2, image_path
        else:
            # For testing, single augmentation
            aug = self.augmentations(image=image)['image']
            return aug, aug, image_path 

class SSLDataLoader:
    """Data loader wrapper for self-supervised learning"""
    
    def __init__(self, config):
        self.config = config
        
        # Create datasets
        self.train_dataset = XRayDataset(
            data_dir=config.data_dir,
            image_size=config.image_size,
            is_training=True
        )
        
        self.val_dataset = XRayDataset(
            data_dir=config.data_dir,
            image_size=config.image_size,
            is_training=False
        )
    
    def get_train_loader(self):
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
    
    def get_val_loader(self):
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
    
    def get_feature_loader(self):
        """Get data loader for feature extraction"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )


