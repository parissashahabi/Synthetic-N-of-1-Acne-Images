"""
Classifier model implementation with fixed encoder.
"""
import torch
import torch.nn as nn
from typing import Union, Optional
from monai.networks.nets.diffusion_model_unet import DiffusionModelEncoder

from configs.classifier_config import ClassifierModelConfig


class FixedDiffusionModelEncoder(DiffusionModelEncoder):
    """
    Fixed version of DiffusionModelEncoder that handles dynamic feature sizes.
    Uses Global Average Pooling to avoid hardcoded dimensions.
    """
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_res_blocks=(2, 2, 2, 2),
        channels=(32, 64, 64, 64),
        attention_levels=(False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels=8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim=None,
        num_class_embeds=None,
        upcast_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ):
        # Initialize parent class
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            channels=channels,
            attention_levels=attention_levels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            with_conditioning=with_conditioning,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        
        # Replace the hardcoded output layer with our flexible version
        self._setup_fixed_output_layer(spatial_dims, channels, out_channels)
    
    def _setup_fixed_output_layer(self, spatial_dims: int, channels, out_channels: int):
        """Replace the hardcoded output layer with a flexible one."""
        # Add global average pooling
        if spatial_dims == 2:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif spatial_dims == 3:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        else:
            raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")
        
        # Replace the hardcoded output layer
        self.out = nn.Sequential(
            nn.Linear(channels[-1], 512), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(512, out_channels)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with fixed feature size handling."""
        # 1. time embedding
        from monai.networks.nets.diffusion_model_unet import get_timestep_embedding
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class embedding
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)

        # 4. down blocks
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        
        for downsample_block in self.down_blocks:
            h, _ = downsample_block(hidden_states=h, temb=emb, context=context)

        # 5. Fixed output processing
        # Apply global average pooling instead of flatten
        h = self.global_pool(h)
        h = h.reshape(h.shape[0], -1)  # (batch_size, channels[-1])
        
        output: torch.Tensor = self.out(h)
        return output


class ClassifierModel:
    """Wrapper class for classifier model."""
    
    def __init__(self, config: ClassifierModelConfig):
        self.config = config
        self.model = self._create_model()
        print(f"config: {self.config}")
    
    def _create_model(self) -> FixedDiffusionModelEncoder:
        """Create the classifier model."""

        num_channels = tuple(
            int(self.config.base_channels * mult) for mult in self.config.channels_multiple
        )

        return FixedDiffusionModelEncoder(
            spatial_dims=self.config.spatial_dims,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            channels=num_channels,
            attention_levels=self.config.attention_levels,
            num_res_blocks=self.config.num_res_blocks,
            num_head_channels=self.config.num_head_channels,
            with_conditioning=self.config.with_conditioning,
        )
    
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