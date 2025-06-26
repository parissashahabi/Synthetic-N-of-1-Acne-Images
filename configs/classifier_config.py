"""
Classifier model configuration.
"""
from dataclasses import dataclass
from typing import Tuple
from .base_config import BaseConfig


@dataclass
class ClassifierModelConfig:
    """Configuration for the classifier model."""
    
    # Model architecture
    spatial_dims: int = 2
    in_channels: int = 3
    out_channels: int = 4  # Number of severity levels
    channels: Tuple[int, ...] = (128, 128, 256, 256, 512)
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
    
    def __post_init__(self):
        super().__post_init__()
        # Create classifier-specific directories
        import os
        os.makedirs(os.path.join(self.experiment_dir, "classifier"), exist_ok=True)