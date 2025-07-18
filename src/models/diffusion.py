"""
Diffusion model implementation.
"""
import torch
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.inferers import DiffusionInferer

from configs.diffusion_config import DiffusionModelConfig, DiffusionTrainingConfig


class DiffusionModel:
    """Wrapper class for diffusion model components."""
    
    def __init__(self, config: DiffusionModelConfig, training_config: DiffusionTrainingConfig):
        self.config = config
        self.training_config = training_config
        
        # Create model components
        self.model = self._create_model()
        self.scheduler = self._create_scheduler()
        self.inferer = self._create_inferer()
        print(f"config: {self.config}")
        
    def _create_model(self) -> DiffusionModelUNet:
        """Create the U-Net model."""

        num_channels = tuple(
            int(self.config.base_channels * mult) for mult in self.config.channels_multiple
        )

        return DiffusionModelUNet(
            spatial_dims=self.config.spatial_dims,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            num_channels=num_channels,
            attention_levels=self.config.attention_levels,
            num_res_blocks=self.config.num_res_blocks,
            num_head_channels=self.config.num_head_channels,
            with_conditioning=self.config.with_conditioning,
            resblock_updown=self.config.resblock_updown,
        )
    
    def _create_scheduler(self) -> DDPMScheduler:
        """Create the DDPM scheduler."""
        return DDPMScheduler(
            num_train_timesteps=self.training_config.num_train_timesteps
        )
    
    def _create_inferer(self) -> DiffusionInferer:
        """Create the diffusion inferer."""
        return DiffusionInferer(self.scheduler)
    
    def to(self, device: torch.device):
        """Move model to device."""
        self.model.to(device)
        return self
    
    def get_model_summary(self) -> dict:
        """Get model summary statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
        }