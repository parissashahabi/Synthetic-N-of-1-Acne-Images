"""
Diffusion model configuration.
"""
from dataclasses import dataclass
from typing import Tuple
from .base_config import BaseConfig


@dataclass
class DiffusionModelConfig:
    """Configuration for the diffusion U-Net model."""
    
    # Model architecture
    spatial_dims: int = 2
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 256
    channels_multiple: Tuple[float, ...] = (1, 1, 2, 3, 4, 4)
    # num_channels: Tuple[int, ...] = (128, 128, 256, 256, 512, 512)
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
    
    # def __post_init__(self):
    #     super().__post_init__()
    #     # Create diffusion-specific directories
    #     import os
    #     os.makedirs(os.path.join(self.experiment_dir, "diffusion"), exist_ok=True)
    #     os.makedirs(os.path.join(self.experiment_dir, "samples"), exist_ok=True)