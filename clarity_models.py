"""
Enhanced models for Clarity project - DINOv3 + ArcFace integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import timm
from transformers import ViTModel, ViTConfig

class ArcFaceHead(nn.Module):
    """
    ArcFace head for similarity learning with margin-based classification
    """
    
    def __init__(self, feature_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight matrix for ArcFace
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass for ArcFace
        
        Args:
            features: Input features [batch_size, feature_dim]
            labels: Ground truth labels [batch_size] (optional)
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Normalize features and weights
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(features, weight)
        
        if labels is not None and self.training:
            # Apply ArcFace margin during training
            cosine = cosine * self.scale
            
            # Create one-hot encoding
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            # Apply margin
            cosine_margin = cosine - one_hot * self.margin
            
            return cosine_margin
        else:
            # During inference, return cosine similarity
            return cosine * self.scale

class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone with enhanced feature extraction for X-ray images
    """
    
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        
        if model_name.startswith("vit"):
            # Use DINOv3 ViT model
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )
            self.feature_dim = self.backbone.embed_dim
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through DINOv3 backbone
        
        Args:
            x: Input images [batch_size, 3, height, width]
            return_features: Whether to return raw features
        
        Returns:
            features: Extracted features [batch_size, feature_dim]
        """
        # Extract features using DINOv3
        features = self.backbone.forward_features(x)
        
        # Global average pooling if needed
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        if return_features:
            return features
        
        # Apply L2 normalization
        features = F.normalize(features, dim=1)
        return features

class ClarityFeatureExtractor(nn.Module):
    """
    Complete feature extraction pipeline for Clarity project
    Combines DINOv3 backbone with ArcFace head for similarity learning
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        feature_dim: int = 128,
        num_classes: int = 4,  # Grade A, Incomplete, Counterfeit, Defective
        pretrained: bool = True
    ):
        super().__init__()
        
        # DINOv3 backbone
        self.backbone = DINOv3Backbone(backbone_name, pretrained)
        
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # ArcFace head for similarity learning
        self.arcface_head = ArcFaceHead(feature_dim, num_classes)
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the complete pipeline
        
        Args:
            x: Input images [batch_size, 3, height, width]
            labels: Ground truth labels [batch_size] (optional)
            return_features: Whether to return raw features
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Extract features using DINOv3
        features = self.backbone(x, return_features=True)
        
        # Project to feature space
        projected_features = self.feature_proj(features)
        
        if return_features:
            return projected_features
        
        # Apply ArcFace head
        logits = self.arcface_head(projected_features, labels)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized features for similarity matching"""
        with torch.no_grad():
            features = self.backbone(x, return_features=True)
            projected_features = self.feature_proj(features)
            return F.normalize(projected_features, dim=1)

class SimilarityMatcher:
    """
    Similarity matching system for gallery-based retrieval
    """
    
    def __init__(self, feature_extractor: ClarityFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.gallery_features = None
        self.gallery_paths = None
        self.gallery_barcodes = None
    
    def build_gallery(self, gallery_loader, barcode_mapping: dict):
        """
        Build gallery from data loader with barcode mapping
        
        Args:
            gallery_loader: DataLoader containing gallery images
            barcode_mapping: Dictionary mapping image paths to barcodes
        """
        self.feature_extractor.eval()
        features_list = []
        paths_list = []
        barcodes_list = []
        
        with torch.no_grad():
            for batch in gallery_loader:
                if len(batch) == 3:
                    images, _, paths = batch
                else:
                    images, paths = batch
                
                # Extract features
                features = self.feature_extractor.extract_features(images)
                
                features_list.append(features.cpu())
                paths_list.extend(paths)
                
                # Get barcodes for this batch
                batch_barcodes = [barcode_mapping.get(path, None) for path in paths]
                barcodes_list.extend(batch_barcodes)
        
        self.gallery_features = torch.cat(features_list, dim=0)
        self.gallery_paths = paths_list
        self.gallery_barcodes = barcodes_list
        
        print(f"Gallery built with {len(self.gallery_paths)} items")
    
    def find_similar(
        self, 
        query_features: torch.Tensor, 
        barcode: str, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Find similar items in gallery for given barcode
        
        Args:
            query_features: Query image features [1, feature_dim]
            barcode: Product barcode for filtering
            top_k: Number of top similar items to return
        
        Returns:
            similarities: Similarity scores [top_k]
            indices: Indices of similar items [top_k]
            paths: Paths of similar items [top_k]
        """
        if self.gallery_features is None:
            raise ValueError("Gallery not built. Call build_gallery() first.")
        
        # Filter gallery by barcode
        barcode_mask = torch.tensor([
            b == barcode for b in self.gallery_barcodes
        ])
        
        if not barcode_mask.any():
            # No items found for this barcode
            return torch.tensor([]), torch.tensor([]), []
        
        # Get relevant gallery features
        relevant_features = self.gallery_features[barcode_mask]
        relevant_indices = torch.where(barcode_mask)[0]
        relevant_paths = [self.gallery_paths[i] for i in relevant_indices]
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_features.unsqueeze(0), 
            relevant_features, 
            dim=1
        )
        
        # Get top-k similar items
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        top_paths = [relevant_paths[i] for i in top_indices]
        
        return top_similarities, top_indices, top_paths
    
    def compute_recall_at_1(self, query_loader, ground_truth_mapping: dict) -> float:
        """
        Compute Recall@1 metric for evaluation
        
        Args:
            query_loader: DataLoader with query images
            ground_truth_mapping: Dictionary mapping query paths to ground truth gallery paths
        
        Returns:
            recall_at_1: Recall@1 score
        """
        self.feature_extractor.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in query_loader:
                if len(batch) == 3:
                    images, _, paths = batch
                else:
                    images, paths = batch
                
                for i, path in enumerate(paths):
                    # Extract query features
                    query_features = self.feature_extractor.extract_features(images[i:i+1])
                    
                    # Get ground truth gallery path
                    gt_path = ground_truth_mapping.get(path)
                    if gt_path is None:
                        continue
                    
                    # Find most similar item
                    similarities, indices, _ = self.find_similar(
                        query_features, 
                        barcode="",  # Use all gallery items
                        top_k=1
                    )
                    
                    if len(similarities) > 0:
                        # Check if the most similar item matches ground truth
                        most_similar_path = self.gallery_paths[indices[0]]
                        if most_similar_path == gt_path:
                            correct += 1
                    
                    total += 1
        
        return correct / total if total > 0 else 0.0

def create_clarity_model(
    backbone_name: str = "vit_base_patch16_224",
    feature_dim: int = 128,
    num_classes: int = 4,
    pretrained: bool = True
) -> ClarityFeatureExtractor:
    """
    Create Clarity feature extractor model
    
    Args:
        backbone_name: Name of backbone model
        feature_dim: Dimension of extracted features
        num_classes: Number of anomaly classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        ClarityFeatureExtractor: Complete model for feature extraction
    """
    return ClarityFeatureExtractor(
        backbone_name=backbone_name,
        feature_dim=feature_dim,
        num_classes=num_classes,
        pretrained=pretrained
    )
