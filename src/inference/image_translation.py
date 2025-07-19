"""
Image-to-image translation for acne severity transformation using diffusion models.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List

from generative.networks.schedulers.ddim import DDIMScheduler

from models.diffusion import DiffusionModel
from models.classifier import ClassifierModel
from utils.checkpoints import CheckpointManager
from utils.config_reader import ConfigReader


class AcneSeverityTranslator:
    """Image-to-image translation for acne severity transformation."""
    
    def __init__(
        self,
        diffusion_checkpoint: str,
        classifier_checkpoint: str,
        config_path: str = "config.yaml",
        device: Optional[torch.device] = None
    ):
        """
        Initialize the acne severity translator.
        
        Args:
            diffusion_checkpoint: Path to trained diffusion model
            classifier_checkpoint: Path to trained classifier model
            config_path: Path to configuration file
            device: Device to run on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.config_reader = ConfigReader(config_path)
        
        # Initialize models
        self._load_models(diffusion_checkpoint, classifier_checkpoint)
        
        # Class names for visualization
        self.class_names = [f"Severity {i}" for i in range(4)]
        
        print(f"🎯 Acne Severity Translator initialized on {self.device}")
    
    def _load_models(self, diffusion_checkpoint: str, classifier_checkpoint: str):
        """Load diffusion and classifier models."""
        # Load diffusion model
        diffusion_model_config = self.config_reader.get_diffusion_model_config()
        diffusion_training_config = self.config_reader.get_diffusion_training_config()
        
        self.diffusion_model = DiffusionModel(diffusion_model_config, diffusion_training_config)
        self.diffusion_model.to(self.device)
        
        # Load diffusion checkpoint
        diffusion_manager = CheckpointManager(
            checkpoint_dir=os.path.dirname(diffusion_checkpoint),
            model_name="diffusion"
        )
        diffusion_manager.load_checkpoint(
            diffusion_checkpoint, self.diffusion_model.model, device=self.device
        )
        self.diffusion_model.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.diffusion_model.model.eval()
        
        # Load classifier model
        classifier_model_config = self.config_reader.get_classifier_model_config()
        self.classifier_model = ClassifierModel(classifier_model_config)
        self.classifier_model.to(self.device)
        
        # Load classifier checkpoint
        classifier_manager = CheckpointManager(
            checkpoint_dir=os.path.dirname(classifier_checkpoint),
            model_name="classifier"
        )
        classifier_manager.load_checkpoint(
            classifier_checkpoint, self.classifier_model.model, device=self.device
        )
        self.classifier_model.model.eval()
        
        print(f"✅ Models loaded successfully")
    
    def encode_image_to_noise(
        self, 
        image: torch.Tensor, 
        num_steps: int = 250
    ) -> torch.Tensor:
        """
        Encode input image to noise using reversed DDIM sampling.
        
        Args:
            image: Input image tensor [1, C, H, W]
            num_steps: Number of encoding steps
        
        Returns:
            Encoded noisy image
        """
        print(f"🔄 Encoding image to noise ({num_steps} steps)...")
        
        current_img = image.to(self.device)
        self.diffusion_model.scheduler.set_timesteps(num_inference_steps=1000)
        
        progress_bar = tqdm(range(num_steps), desc="Encoding")
        
        for t in progress_bar:
            with autocast(enabled=False):
                with torch.no_grad():
                    model_output = self.diffusion_model.model(
                        current_img, 
                        timesteps=torch.tensor([t]).to(self.device)
                    )
            
            # Use reversed step for encoding
            current_img, _ = self.diffusion_model.scheduler.reversed_step(
                model_output, t, current_img
            )
        
        return current_img
    
    def translate_severity(
        self,
        image: torch.Tensor,
        source_severity: int,
        target_severity: int,
        num_steps: int = 250,
        guidance_scale: float = 10.0,
        save_gradients: bool = False,
        gradient_save_interval: int = 35
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        """
        Translate image to target acne severity using gradient guidance.
        
        Args:
            image: Input image tensor [1, C, H, W]
            target_severity: Target severity level (0-3)
            num_steps: Number of translation steps
            guidance_scale: Gradient guidance scale
            save_gradients: Whether to save gradient visualizations
            gradient_save_interval: Interval for saving gradients
        
        Returns:
            Tuple of (translated_image, gradient_images, alpha_values)
        """
        print(f"🎨 Translating to severity level {target_severity} ({num_steps} steps)...")
        
        # First encode the image to noise
        current_img = self.encode_image_to_noise(image, num_steps)
        
        # Prepare for guided denoising
        target_label = torch.tensor([source_severity]).to(self.device)
        gradient_images = []
        alpha_values = []
        
        progress_bar = tqdm(range(num_steps), desc=f"Translating to severity {target_severity}")
        
        for i in progress_bar:
            t = num_steps - i
            
            with autocast(enabled=True):
                # Get diffusion model prediction
                with torch.no_grad():
                    model_output = self.diffusion_model.model(
                        current_img, 
                        timesteps=torch.tensor([t]).to(self.device)
                    ).detach()
                
                # Compute gradients for guidance
                with torch.enable_grad():
                    x_in = current_img.detach().requires_grad_(True)
                    
                    # Get classifier prediction
                    logits = self.classifier_model.model(
                        x_in, 
                        timesteps=torch.tensor([t]).to(self.device)
                    )
                    
                    # Compute log probabilities
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), target_label.view(-1)]
                    
                    # Compute gradients
                    gradients = torch.autograd.grad(selected.sum(), x_in)[0]
                    
                    # Get alpha values for proper scaling
                    alpha_prod_t = self.diffusion_model.scheduler.alphas_cumprod[t]
                    alpha_values.append((1 - alpha_prod_t).sqrt())
                    
                    # Save gradients for visualization
                    if save_gradients and i % gradient_save_interval == 0:
                        gradient_img = gradients[0, 0].cpu().detach().numpy()
                        gradient_images.append(gradient_img)
                    
                    # Update predicted noise with classifier guidance
                    updated_noise = (
                        model_output - (1 - alpha_prod_t).sqrt() * guidance_scale * gradients
                    )
            
            # Perform denoising step
            current_img, _ = self.diffusion_model.scheduler.step(
                updated_noise, t, current_img
            )
            
            torch.cuda.empty_cache()
        
        return current_img, gradient_images, alpha_values
    
    def translate_and_visualize(
        self,
        input_image: torch.Tensor,
        source_severity: int,
        target_severity: int,
        output_dir: str,
        prefix: str = "translation",
        num_steps: int = 250,
        guidance_scale: float = 10.0,
        save_process: bool = True
    ) -> dict:
        """
        Complete translation pipeline with visualization.
        
        Args:
            input_image: Input image tensor [1, C, H, W]
            source_severity: Source severity level
            target_severity: Target severity level
            output_dir: Directory to save results
            prefix: Filename prefix
            num_steps: Number of steps
            guidance_scale: Guidance scale
            save_process: Whether to save intermediate visualizations
        
        Returns:
            Dictionary with results and paths
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting translation: Severity {source_severity} → {target_severity}")
        
        # Perform translation
        translated_image, gradient_images, alpha_values = self.translate_severity(
            input_image, source_severity, target_severity, num_steps, guidance_scale, 
            save_gradients=save_process
        )
        
        # Save results
        results = {
            'input_image': input_image,
            'translated_image': translated_image,
            'source_severity': source_severity,
            'target_severity': target_severity,
            'gradient_images': gradient_images,
            'alpha_values': alpha_values
        }
        
        # Visualize and save results
        saved_files = self._save_results(results, output_path, prefix, save_process)
        
        return {
            'results': results,
            'saved_files': saved_files,
            'output_dir': str(output_path)
        }
    
    def _save_results(
        self, 
        results: dict, 
        output_path: Path, 
        prefix: str, 
        save_process: bool
    ) -> List[str]:
        """Save visualization results."""
        saved_files = []
        
        # Convert tensors to numpy for visualization
        input_img = results['input_image'][0].permute(1, 2, 0).cpu().numpy()
        translated_img = results['translated_image'][0].permute(1, 2, 0).cpu().numpy()
        
        # Ensure images are in [0, 1] range
        input_img = np.clip(input_img, 0, 1)
        translated_img = np.clip(translated_img, 0, 1)
        
        # Handle grayscale if needed
        if input_img.shape[2] == 1:
            input_img = input_img[:, :, 0]
            translated_img = translated_img[:, :, 0]
            cmap = 'gray'
        else:
            cmap = None
        
        # Save individual images
        plt.figure(figsize=(8, 6))
        plt.imshow(input_img, cmap=cmap)
        plt.title(f"Original Image (Severity {results['source_severity']})")
        plt.axis('off')
        input_path = output_path / f"{prefix}_input_severity_{results['source_severity']}.png"
        plt.savefig(input_path, bbox_inches='tight', dpi=300)
        plt.close()
        saved_files.append(str(input_path))
        
        plt.figure(figsize=(8, 6))
        plt.imshow(translated_img, cmap=cmap)
        plt.title(f"Translated Image (Severity {results['target_severity']})")
        plt.axis('off')
        output_path_img = output_path / f"{prefix}_output_severity_{results['target_severity']}.png"
        plt.savefig(output_path_img, bbox_inches='tight', dpi=300)
        plt.close()
        saved_files.append(str(output_path_img))
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].imshow(input_img, cmap=cmap)
        axes[0].set_title(f"Input: {self.class_names[results['source_severity']]}", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(translated_img, cmap=cmap)
        axes[1].set_title(f"Output: {self.class_names[results['target_severity']]}", fontsize=14)
        axes[1].axis('off')
        
        plt.suptitle(f"Acne Severity Translation", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        comparison_path = output_path / f"{prefix}_comparison.png"
        plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
        plt.close()
        saved_files.append(str(comparison_path))
        
        # Save gradient visualizations if available
        if save_process and results['gradient_images']:
            self._save_gradient_grid(
                results['gradient_images'], 
                output_path / f"{prefix}_gradients.png"
            )
            saved_files.append(str(output_path / f"{prefix}_gradients.png"))
        
        # Compute and save difference map
        diff = np.abs(input_img - translated_img)
        plt.figure(figsize=(8, 6))
        plt.imshow(diff, cmap='inferno')
        plt.title('Difference Map')
        plt.colorbar()
        plt.axis('off')
        diff_path = output_path / f"{prefix}_difference.png"
        plt.savefig(diff_path, bbox_inches='tight', dpi=300)
        plt.close()
        saved_files.append(str(diff_path))
        
        print(f"💾 Saved {len(saved_files)} visualization files to {output_path}")
        
        return saved_files
    
    def _save_gradient_grid(self, gradient_images: List[np.ndarray], save_path: Path):
        """Save gradient visualizations in a grid."""
        if not gradient_images:
            return
        
        ncols = min(4, len(gradient_images))
        nrows = (len(gradient_images) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        
        if nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, grad_img in enumerate(gradient_images):
            axes[i].imshow(grad_img, cmap='inferno')
            axes[i].set_title(f'Gradient Step {i*35}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(gradient_images), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Gradient Evolution During Translation', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def batch_translate(
        self,
        images: List[torch.Tensor],
        source_severities: List[int],
        target_severities: List[int],
        output_dir: str,
        **kwargs
    ) -> List[dict]:
        """Translate multiple images in batch."""
        results = []
        
        for i, (img, src_sev, tgt_sev) in enumerate(
            zip(images, source_severities, target_severities)
        ):
            print(f"\n--- Processing image {i+1}/{len(images)} ---")
            
            result = self.translate_and_visualize(
                img, src_sev, tgt_sev, 
                output_dir=f"{output_dir}/image_{i+1}",
                prefix=f"translation_{i+1}",
                **kwargs
            )
            results.append(result)
        
        return results


def create_translator(
    diffusion_checkpoint: str,
    classifier_checkpoint: str,
    config_path: str = "config.yaml"
) -> AcneSeverityTranslator:
    """Convenience function to create translator."""
    return AcneSeverityTranslator(
        diffusion_checkpoint, 
        classifier_checkpoint, 
        config_path
    )