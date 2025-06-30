"""
Base configuration settings for the acne diffusion project.
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
    
    # def __post_init__(self):
    #     """Create necessary directories."""
    #     os.makedirs(self.experiment_dir, exist_ok=True)
    #     os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
    #     os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
    #     os.makedirs(os.path.join(self.experiment_dir, "results"), exist_ok=True)


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