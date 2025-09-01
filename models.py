"""
Model definitions for X-ray self-supervised learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import ViTModel, ViTConfig
from typing import Tuple, Optional
import einops

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class BackboneModel(nn.Module):
    """Backbone model (ViT or ResNet)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.model_name.startswith("vit"):
            self.backbone = self._create_vit()
        else:
            self.backbone = self._create_resnet()
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def _create_vit(self):
        """Create Vision Transformer backbone"""
        if self.config.model_name == "vit_base_patch16_224":
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                is_encoder_decoder=False,
                image_encoder_type="ViTModel",
                backbone_output_dim=768,
            )
            return ViTModel(config)
        else:
            # Use timm for other ViT variants
            return timm.create_model(self.config.model_name, pretrained=False, num_classes=0)
    
    def _create_resnet(self):
        """Create ResNet backbone"""
        return timm.create_model(self.config.model_name, pretrained=False, num_classes=0)
    
    def forward(self, x, return_features=False):
        if self.config.model_name.startswith("vit"):
            if isinstance(self.backbone, ViTModel):
                outputs = self.backbone(pixel_values=x)
                features = outputs.last_hidden_state[:, 0]  # [CLS] token
            else:
                features = self.backbone.forward_features(x)
                if hasattr(features, 'shape') and len(features.shape) > 2:
                    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            features = self.backbone.forward_features(x)
            if hasattr(features, 'shape') and len(features.shape) > 2:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        if return_features:
            return features
        
        projected_features = self.feature_proj(features)
        return projected_features

class SimCLR(nn.Module):
    """SimCLR implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = BackboneModel(config)
        
        # Projection head
        self.projection = ProjectionHead(
            config.feature_dim, 
            config.feature_dim, 
            config.feature_dim
        )
    
    def forward(self, x1, x2):
        # Get features from backbone
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        
        # Project to representation space
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def contrastive_loss(self, z1, z2, temperature=0.07):
        """Compute contrastive loss"""
        batch_size = z1.shape[0]
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2
        )
        
        # Define positive pairs
        positives = torch.cat([
            torch.arange(batch_size, batch_size * 2),
            torch.arange(batch_size)
        ])
        
        # Compute loss
        logits = similarity_matrix / temperature
        labels = positives.to(logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

class MAE(nn.Module):
    """Masked Autoencoder implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = BackboneModel(config)
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.decoder_dim),
            nn.GELU(),
            nn.Linear(config.decoder_dim, config.decoder_dim),
            nn.GELU(),
            nn.Linear(config.decoder_dim, 3 * 16 * 16)  # Reconstruct patches
        )
        
        self.mask_ratio = config.mask_ratio
        self.patch_size = 16
    
    def random_masking(self, x, mask_ratio):
        """Random masking of patches"""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(-1, -1, D))
        
        return x_masked, ids_restore
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x, return_features=True)
        
        # Apply masking
        x_masked, ids_restore = self.random_masking(features, self.mask_ratio)
        
        # Decode
        decoded = self.decoder(x_masked)
        
        return decoded, ids_restore
    
    def reconstruction_loss(self, pred, target, mask):
        """Compute reconstruction loss"""
        loss = F.mse_loss(pred, target, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        return loss

class DINO(nn.Module):
    """DINO implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = BackboneModel(config)
        
        # Student and teacher projection heads
        self.student_head = ProjectionHead(
            config.feature_dim, 
            config.feature_dim, 
            config.feature_dim
        )
        
        self.teacher_head = ProjectionHead(
            config.feature_dim, 
            config.feature_dim, 
            config.feature_dim
        )
        
        # Initialize teacher with student weights
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        
        self.center = torch.zeros(config.feature_dim)
    
    def forward(self, x1, x2):
        # Student forward pass
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        
        z1_student = self.student_head(h1)
        z2_student = self.student_head(h2)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            z1_teacher = self.teacher_head(h1)
            z2_teacher = self.teacher_head(h2)
        
        return (z1_student, z2_student), (z1_teacher, z2_teacher)
    
    def update_teacher(self, momentum=0.996):
        """Update teacher network"""
        for param_student, param_teacher in zip(
            self.student_head.parameters(), 
            self.teacher_head.parameters()
        ):
            param_teacher.data = (
                momentum * param_teacher.data + 
                (1 - momentum) * param_student.data
            )
    
    def dino_loss(self, student_output, teacher_output, temperature=0.04):
        """Compute DINO loss"""
        student_out = student_output / temperature
        teacher_out = F.softmax(teacher_output / 0.04, dim=-1)
        
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        return loss.mean()

class BYOL(nn.Module):
    """BYOL implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = BackboneModel(config)
        
        # Online network
        self.online_predictor = ProjectionHead(
            config.feature_dim, 
            config.feature_dim, 
            config.feature_dim
        )
        
        # Target network
        self.target_predictor = ProjectionHead(
            config.feature_dim, 
            config.feature_dim, 
            config.feature_dim
        )
        
        # Initialize target with online weights
        for param in self.target_predictor.parameters():
            param.requires_grad = False
    
    def forward(self, x1, x2):
        # Online network
        h1_online = self.backbone(x1)
        h2_online = self.backbone(x2)
        
        q1 = self.online_predictor(h1_online)
        q2 = self.online_predictor(h2_online)
        
        # Target network (no gradients)
        with torch.no_grad():
            h1_target = self.backbone(x1)
            h2_target = self.backbone(x2)
            
            z1 = self.target_predictor(h1_target)
            z2 = self.target_predictor(h2_target)
        
        return (q1, q2), (z1, z2)
    
    def update_target(self, momentum=0.99):
        """Update target network"""
        for param_online, param_target in zip(
            self.backbone.parameters(), 
            self.target_predictor.parameters()
        ):
            param_target.data = (
                momentum * param_target.data + 
                (1 - momentum) * param_online.data
            )
    
    def byol_loss(self, online_output, target_output):
        """Compute BYOL loss"""
        q1, q2 = online_output
        z1, z2 = target_output
        
        # Normalize
        q1 = F.normalize(q1, dim=1)
        q2 = F.normalize(q2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Loss
        loss1 = 2 - 2 * (q1 * z2).sum(dim=1)
        loss2 = 2 - 2 * (q2 * z1).sum(dim=1)
        
        return (loss1 + loss2).mean()

def create_ssl_model(config):
    """Create SSL model based on config"""
    if config.ssl_method == "simclr":
        return SimCLR(config)
    elif config.ssl_method == "mae":
        return MAE(config)
    elif config.ssl_method == "dino":
        return DINO(config)
    elif config.ssl_method == "byol":
        return BYOL(config)
    else:
        raise ValueError(f"Unknown SSL method: {config.ssl_method}")


