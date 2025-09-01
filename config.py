"""
Configuration file for X-ray self-supervised learning
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

@dataclass
class SSLConfig:
    """Self-supervised learning configuration"""
    
    # Model settings
    model_name: str = "vit_base_patch16_224"  # or "resnet50"
    pretrained: bool = False
    embedding_dim: int = 768  # for ViT, 2048 for ResNet
    
    # Training settings
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    
    # Data settings
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation settings
    min_scale: float = 0.2
    max_scale: float = 1.0
    color_jitter: float = 0.4
    gaussian_blur: float = 0.5
    
    # SSL specific settings
    ssl_method: str = "simclr"  # "simclr", "mae", "dino", "byol"
    temperature: float = 0.07  # for contrastive learning
    queue_size: int = 65536  # for MoCo-style methods
    
    # MAE specific
    mask_ratio: float = 0.75
    decoder_dim: int = 512
    decoder_depth: int = 2
    
    # DINO specific
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    center_momentum: float = 0.9
    
    # BYOL specific
    moving_average_decay: float = 0.99
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Feature extraction
    feature_dim: int = 128  # final feature dimension for downstream tasks
    use_pooling: bool = True  # whether to use global pooling
    
    def __post_init__(self):
        if self.model_name.startswith("vit"):
            self.embedding_dim = 768 if "base" in self.model_name else 1024
        elif self.model_name.startswith("resnet"):
            self.embedding_dim = 2048


