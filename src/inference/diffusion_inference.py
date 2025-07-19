"""
Diffusion model inference utilities.
"""
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
from torch.cuda.amp import autocast
from pathlib import Path

from generative.networks.schedulers.ddim import DDIMScheduler

from models.diffusion import DiffusionModel
from utils.config_schemas import DiffusionModelConfig, DiffusionTrainingConfig
from utils.checkpoints import CheckpointManager


class DiffusionInference:
    """Inference engine for trained diffusion models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        model_config: Optional[DiffusionModelConfig] = None,
        training_config: Optional[DiffusionTrainingConfig] = None
    ):
        """
        Initialize diffusion inference engine.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
            model_config: Model configuration (if None, uses default)
            training_config: Training configuration (if None, uses default)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Setup configurations
        self.model_config = model_config or DiffusionModelConfig()
        self.training_config = training_config or DiffusionTrainingConfig()
        
        # Initialize model
        self.model = DiffusionModel(self.model_config, self.training_config)
        self.model.to(self.device)
        
        # Load checkpoint
        self._load_checkpoint()
        
        print(f"🎯 Diffusion inference ready on {self.device}")
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.dirname(self.checkpoint_path),
            model_name="diffusion"
        )
        
        checkpoint = checkpoint_manager.load_checkpoint(
            self.checkpoint_path,
            self.model.model,
            device=self.device
        )
        
        if checkpoint is None:
            raise ValueError(f"Failed to load checkpoint: {self.checkpoint_path}")
        
        self.model.model.eval()
        print(f"✅ Loaded checkpoint: {os.path.basename(self.checkpoint_path)}")
    
    def generate(
        self,
        num_samples: int = 1,
        img_size: Optional[int] = None,
        num_inference_steps: int = 1000,
        save_intermediates: bool = False,
        intermediate_steps: int = 100,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Generate images using the diffusion model.
        
        Args:
            num_samples: Number of images to generate
            img_size: Image size (if None, uses config default)
            num_inference_steps: Number of denoising steps
            save_intermediates: Whether to save intermediate denoising steps
            intermediate_steps: Number of intermediate steps to save
            guidance_scale: Classifier-free guidance scale (if supported)
            seed: Random seed for reproducible generation
        
        Returns:
            Tuple of (generated_images, intermediate_images)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        img_size = img_size or self.training_config.img_size
        
        print(f"🎨 Generating {num_samples} image(s)...")
        print(f"   Image size: {img_size}x{img_size}")
        print(f"   Inference steps: {num_inference_steps}")
        print(f"   Device: {self.device}")
        
        # Create random noise
        noise = torch.randn(
            (num_samples, self.model_config.in_channels, img_size, img_size)
        ).to(self.device)
        
        # Set scheduler timesteps
        self.model.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.model.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        
        # Generate images
        start_time = time.time()
        
        with torch.no_grad():
            with autocast(enabled=True):
                if save_intermediates:
                    images, intermediates = self.model.inferer.sample(
                        input_noise=noise,
                        diffusion_model=self.model.model,
                        scheduler=self.model.scheduler,
                        save_intermediates=True,
                        intermediate_steps=intermediate_steps
                    )
                else:
                    images = self.model.inferer.sample(
                        input_noise=noise,
                        diffusion_model=self.model.model,
                        scheduler=self.model.scheduler
                    )
                    intermediates = None
        
        generation_time = time.time() - start_time
        print(f"✅ Generation completed in {generation_time:.2f}s")
        print(f"   Time per image: {generation_time/num_samples:.2f}s")
        
        return images, intermediates
    
    def generate_and_save(
        self,
        output_dir: str,
        num_samples: int = 1,
        img_size: Optional[int] = None,
        num_inference_steps: int = 1000,
        save_intermediates: bool = True,
        save_process: bool = True,
        filename_prefix: str = "generated",
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Generate images and save them to disk.
        
        Args:
            output_dir: Directory to save generated images
            num_samples: Number of images to generate
            img_size: Image size
            num_inference_steps: Number of denoising steps
            save_intermediates: Whether to save intermediate steps
            save_process: Whether to save denoising process visualization
            filename_prefix: Prefix for saved filenames
            seed: Random seed
        
        Returns:
            List of saved file paths
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate images
        images, intermediates = self.generate(
            num_samples=num_samples,
            img_size=img_size,
            num_inference_steps=num_inference_steps,
            save_intermediates=save_intermediates,
            seed=seed
        )
        
        saved_files = []
        
        # Save individual images
        for i in range(num_samples):
            # Save final image
            image_path = output_path / f"{filename_prefix}_sample_{i+1}.png"
            self._save_image(images[i], image_path)
            saved_files.append(str(image_path))
            
            # Save denoising process if requested
            if save_process and intermediates:
                process_path = output_path / f"{filename_prefix}_process_{i+1}.png"
                self._save_process(intermediates, process_path, sample_idx=i)
                saved_files.append(str(process_path))
        
        # Save grid of all samples if multiple
        if num_samples > 1:
            grid_path = output_path / f"{filename_prefix}_grid.png"
            self._save_image_grid(images, grid_path)
            saved_files.append(str(grid_path))
        
        print(f"💾 Saved {len(saved_files)} files to: {output_dir}")
        return saved_files
    
    def _save_image(self, image: torch.Tensor, save_path: Path):
        """Save a single image."""
        plt.figure(figsize=(8, 8))
        
        # Process image for display
        if image.shape[0] == 3:  # RGB
            img_display = image.permute(1, 2, 0).cpu()
        else:  # Grayscale
            img_display = image[0].cpu()
        
        img_display = torch.clamp(img_display, 0, 1)
        
        plt.imshow(img_display, cmap="gray" if image.shape[0] == 1 else None)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
    
    def _save_image_grid(self, images: torch.Tensor, save_path: Path, max_cols: int = 4):
        """Save a grid of images."""
        num_images = images.shape[0]
        cols = min(num_images, max_cols)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        
        if num_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for i in range(num_images):
            # Process image for display
            if images[i].shape[0] == 3:  # RGB
                img_display = images[i].permute(1, 2, 0).cpu()
            else:  # Grayscale
                img_display = images[i][0].cpu()
            
            img_display = torch.clamp(img_display, 0, 1)
            
            axes[i].imshow(img_display, cmap="gray" if images[i].shape[0] == 1 else None)
            axes[i].axis("off")
            axes[i].set_title(f"Sample {i+1}")
        
        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def _save_process(self, intermediates: List[torch.Tensor], save_path: Path, sample_idx: int = 0):
        """Save denoising process visualization."""
        if not intermediates:
            return
        
        # Concatenate intermediate images
        chain = torch.cat(intermediates, dim=-1)
        
        plt.figure(figsize=(20, 4))
        
        if chain[sample_idx].shape[0] == 3:  # RGB
            chain_display = chain[sample_idx].permute(1, 2, 0).cpu()
        else:  # Grayscale
            chain_display = chain[sample_idx][0].cpu()
        
        chain_display = torch.clamp(chain_display, 0, 1)
        
        plt.imshow(chain_display, cmap="gray" if chain[sample_idx].shape[0] == 1 else None)
        plt.axis("off")
        plt.title(f"Denoising Process - Sample {sample_idx + 1}")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
    
    def interpolate(
        self,
        num_steps: int = 10,
        img_size: Optional[int] = None,
        num_inference_steps: int = 1000,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate interpolation between two random noise vectors.
        
        Args:
            num_steps: Number of interpolation steps
            img_size: Image size
            num_inference_steps: Number of denoising steps
            seed: Random seed
        
        Returns:
            Tensor of interpolated images
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        img_size = img_size or self.training_config.img_size
        
        # Generate two random noise vectors
        noise1 = torch.randn(
            (1, self.model_config.in_channels, img_size, img_size)
        ).to(self.device)
        noise2 = torch.randn(
            (1, self.model_config.in_channels, img_size, img_size)
        ).to(self.device)
        
        # Create interpolation steps
        interpolated_images = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interpolated_noise = (1 - alpha) * noise1 + alpha * noise2
            
            # Generate image
            with torch.no_grad():
                with autocast(enabled=True):
                    image = self.model.inferer.sample(
                        input_noise=interpolated_noise,
                        diffusion_model=self.model.model,
                        scheduler=self.model.scheduler
                    )
            
            interpolated_images.append(image)
        
        return torch.cat(interpolated_images, dim=0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        
        return {
            'checkpoint_path': self.checkpoint_path,
            'device': str(self.device),
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


def load_diffusion_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> DiffusionInference:
    """
    Convenience function to load a diffusion model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        DiffusionInference instance
    """
    return DiffusionInference(checkpoint_path, device)


def batch_generate(
    checkpoint_path: str,
    output_dir: str,
    num_batches: int = 10,
    batch_size: int = 4,
    **generation_kwargs
) -> List[str]:
    """
    Generate images in batches to avoid memory issues.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory
        num_batches: Number of batches
        batch_size: Images per batch
        **generation_kwargs: Additional arguments for generation
    
    Returns:
        List of all saved file paths
    """
    inference_engine = DiffusionInference(checkpoint_path)
    all_saved_files = []
    
    for batch_idx in range(num_batches):
        print(f"🎨 Generating batch {batch_idx + 1}/{num_batches}")
        
        batch_output_dir = Path(output_dir) / f"batch_{batch_idx + 1}"
        
        saved_files = inference_engine.generate_and_save(
            output_dir=str(batch_output_dir),
            num_samples=batch_size,
            filename_prefix=f"batch_{batch_idx + 1}",
            **generation_kwargs
        )
        
        all_saved_files.extend(saved_files)
    
    print(f"✅ Generated {num_batches * batch_size} images in {len(all_saved_files)} files")
    return all_saved_files