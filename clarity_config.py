"""
Configuration for Clarity project
Enhanced configuration for X-ray validation with barcode integration
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch

@dataclass
class ClarityConfig:
    """Configuration for Clarity X-ray validation system"""
    
    # Model settings
    backbone_name: str = "vit_base_patch16_224"  # DINOv3 backbone
    feature_dim: int = 128  # Final feature dimension
    num_anomaly_classes: int = 4  # Grade A, Incomplete, Counterfeit, Defective
    pretrained: bool = True
    
    # Training settings
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    
    # ArcFace settings
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    
    # Data settings
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation settings for aligned images
    min_scale: float = 0.8
    max_scale: float = 1.0
    color_jitter: float = 0.2
    gaussian_blur: float = 0.3
    
    # Similarity matching settings
    similarity_threshold: float = 0.7  # Threshold for anomaly detection
    top_k_retrieval: int = 5  # Number of similar items to retrieve
    
    # Evaluation settings
    recall_at_1: bool = True
    classification_metrics: bool = True
    similarity_analysis: bool = True
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = None
    
    # Paths
    data_dir: str = "./data"
    gallery_dir: str = "./gallery"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"
    
    # Barcode settings
    barcode_length_min: int = 8
    barcode_length_max: int = 20
    barcode_patterns: List[str] = None  # Regex patterns for barcode validation
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        if self.barcode_patterns is None:
            self.barcode_patterns = [
                r'^\d{8,20}$',  # Numeric barcodes
                r'^[A-Z0-9]{8,20}$',  # Alphanumeric barcodes
            ]
        
        # Create directories
        import os
        for directory in [self.checkpoint_dir, self.log_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class AnomalyType:
    """Anomaly type configuration"""
    name: str
    description: str
    similarity_threshold: float
    color: str = "#FF6B6B"  # Default color for visualization

# Predefined anomaly types
ANOMALY_TYPES = {
    "Grade A": AnomalyType(
        name="Grade A",
        description="Mint condition, no anomalies",
        similarity_threshold=0.8,
        color="#4ECDC4"
    ),
    "Incomplete": AnomalyType(
        name="Incomplete",
        description="Missing components or parts",
        similarity_threshold=0.6,
        color="#FFE66D"
    ),
    "Counterfeit": AnomalyType(
        name="Counterfeit",
        description="Fake or counterfeit item",
        similarity_threshold=0.4,
        color="#FF6B6B"
    ),
    "Defective": AnomalyType(
        name="Defective",
        description="Damaged or defective item",
        similarity_threshold=0.5,
        color="#A8E6CF"
    )
}

@dataclass
class GalleryConfig:
    """Configuration for gallery management"""
    max_gallery_size: int = 10000
    gallery_update_frequency: int = 24  # hours
    similarity_cache_size: int = 1000
    enable_gallery_caching: bool = True
    
    # Gallery filtering
    min_similarity_for_gallery: float = 0.3
    max_items_per_barcode: int = 100

@dataclass
class APIConfig:
    """Configuration for API endpoints"""
    enable_cors: bool = True
    cors_origins: List[str] = None
    rate_limit_per_minute: int = 100
    request_timeout: int = 30  # seconds
    
    # Authentication (if needed)
    enable_auth: bool = False
    auth_token: Optional[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

def create_default_config() -> ClarityConfig:
    """Create default configuration for Clarity project"""
    return ClarityConfig()

def create_production_config() -> ClarityConfig:
    """Create production configuration with optimized settings"""
    config = ClarityConfig()
    
    # Production optimizations
    config.batch_size = 64
    config.num_workers = 8
    config.enable_gallery_caching = True
    config.similarity_threshold = 0.75
    
    # API settings for production
    config.rate_limit_per_minute = 1000
    config.request_timeout = 60
    
    return config

def create_development_config() -> ClarityConfig:
    """Create development configuration with debugging settings"""
    config = ClarityConfig()
    
    # Development settings
    config.batch_size = 16
    config.num_epochs = 10
    config.num_workers = 2
    
    # Enable all evaluation metrics
    config.recall_at_1 = True
    config.classification_metrics = True
    config.similarity_analysis = True
    
    return config

# Configuration presets
CONFIG_PRESETS = {
    "default": create_default_config,
    "production": create_production_config,
    "development": create_development_config
}

def load_config(preset: str = "default") -> ClarityConfig:
    """
    Load configuration from preset
    
    Args:
        preset: Configuration preset name
    
    Returns:
        ClarityConfig: Configuration object
    """
    if preset not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(CONFIG_PRESETS.keys())}")
    
    return CONFIG_PRESETS[preset]()
