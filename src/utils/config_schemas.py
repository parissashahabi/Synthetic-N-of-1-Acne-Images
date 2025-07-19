"""
Configuration schemas for the acne diffusion project.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os


@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    
    # Data settings
    img_size: int = 128
    batch_size: int = 16
    train_split: float = 0.8
    num_workers: int = 4
    pin_memory: bool = True
    
    # Device settings
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 10
    val_interval: int = 10
    
    # Paths
    data_dir: str = "/data/acne_dataset"
    experiment_dir: str = "./experiments"
    
    # Wandb settings
    use_wandb: bool = False
    wandb_project: str = "acne-diffusion"
    wandb_entity: Optional[str] = None  # Your wandb username/team
    wandb_tags: Optional[List[str]] = None


@dataclass
class DataConfig:
    """Data-specific configuration."""
    dataset_path: str = "./data/acne_dataset"
    severity_levels: Optional[List[int]] = None  # None means all levels
    apply_augmentation: bool = True
    
    # Data loading
    drop_last: bool = True
    shuffle_train: bool = True
    
    # Class mapping
    num_classes: int = 4  # 0, 1, 2, 3 severity levels


@dataclass
class DiffusionModelConfig:
    """Configuration for the diffusion U-Net model."""
    
    # Model architecture
    spatial_dims: int = 2
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 256
    channels_multiple: Tuple[float, ...] = (1, 1, 2, 3, 4, 4)
    attention_levels: Tuple[bool, ...] = (False, False, False, False, True, False)
    num_res_blocks: int = 2 # Depth = 2
    num_head_channels: int = 64
    with_conditioning: bool = False
    resblock_updown: bool = True
    dropout: float = 0.0


@dataclass
class DiffusionTrainingConfig(BaseConfig):
    """Configuration for diffusion model training."""
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    num_train_timesteps: int = 1000
    n_epochs: int = 10
    
    # Inference settings
    num_inference_steps: int = 1000
    intermediate_steps: int = 100
    save_intermediates: bool = True
    
    # Generation settings
    num_samples: int = 1
    sample_interval: int = 10
    process_interval: int = 10


@dataclass
class ClassifierModelConfig:
    """Configuration for the classifier model."""
    
    # Model architecture
    spatial_dims: int = 2
    in_channels: int = 3
    out_channels: int = 4  # Number of severity levels
    base_channels: int = 128
    channels_multiple: Tuple[float, ...] = (1, 1, 2, 3, 4)
    attention_levels: Tuple[bool, ...] = (False, False, False, False, True)
    num_res_blocks: Tuple[int, ...] = (2, 2, 2, 2, 2)
    num_head_channels: int = 64
    with_conditioning: bool = False


@dataclass
class ClassifierTrainingConfig(BaseConfig):
    """Configuration for classifier training."""
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    n_epochs: int = 200
    
    # Noise augmentation
    noise_timesteps_train: int = 1000
    noise_timesteps_val: int = 1