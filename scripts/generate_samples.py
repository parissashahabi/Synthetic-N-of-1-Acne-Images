#!/usr/bin/env python3
"""
Generate samples using trained diffusion model.
"""
import os
import sys
import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.diffusion import DiffusionModel
from configs.diffusion_config import DiffusionModelConfig, DiffusionTrainingConfig
from utils.checkpoints import CheckpointManager
from utils.config_parser import add_config_args, update_config_from_args


def display_results(image, intermediates, config, save_dir, title_prefix, generation_time):
    """Display and save inference results."""
    
    # Display final generated image
    plt.figure(figsize=(8, 8))
    
    # Process image for display
    if image.shape[1] == 3:  # RGB
        img_display = image[0].permute(1, 2, 0).cpu()  # CHW -> HWC
    else:  # Grayscale
        img_display = image[0, 0].cpu()
    
    img_display = torch.clamp(img_display, 0, 1)
    
    # Display image
    plt.imshow(img_display, cmap="gray" if image.shape[1] == 1 else None)
    plt.axis("off")
    plt.title(f"{title_prefix} Image\nGeneration time: {generation_time:.2f}s", 
              fontsize=12)
    plt.tight_layout()
    
    # Save final image
    if save_dir:
        final_image_path = os.path.join(save_dir, f'{title_prefix.lower()}_final_image.png')
        plt.savefig(final_image_path, bbox_inches='tight', dpi=300)
        print(f"💾 Final image saved: {final_image_path}")
    
    plt.show()
    
    # Display generation process
    if intermediates and config.save_intermediates:
        # Create denoising process visualization
        chain = torch.cat(intermediates, dim=-1)
        
        plt.style.use("default")
        plt.figure(figsize=(20, 4))
        
        if chain.shape[1] == 3:  # RGB
            rgb_chain = chain[0].permute(1, 2, 0).cpu()  # CHW -> HWC
        else:  # Grayscale
            rgb_chain = chain[0, 0].cpu()
        
        rgb_chain = torch.clamp(rgb_chain, 0, 1)
        
        plt.imshow(rgb_chain, cmap="gray" if chain.shape[1] == 1 else None)
        plt.axis("off")
        plt.title(f"{title_prefix} - Denoising Process ({config.intermediate_steps} steps)", 
                  fontsize=12)
        plt.tight_layout()
        
        # Save process visualization
        if save_dir:
            process_path = os.path.join(save_dir, f'{title_prefix.lower()}_denoising_process.png')
            plt.savefig(process_path, bbox_inches='tight', pad_inches=0, dpi=300)
            print(f"🔄 Denoising process saved: {process_path}")
        
        plt.show()


def run_inference(model, config, device, save_dir=None, title_prefix="Generated"):
    """Run inference to generate images."""
    print(f"🚀 Starting inference...")
    
    # Set model to evaluation mode
    model.model.eval()
    
    # Create random noise
    noise = torch.randn((
        config.num_samples, 
        3, 
        config.img_size, 
        config.img_size
    )).to(device)
    
    print(f"   Noise shape: {noise.shape}")
    print(f"   Inference steps: {config.num_inference_steps}")
    
    # Set scheduler timesteps
    model.scheduler.set_timesteps(num_inference_steps=config.num_inference_steps)
    
    # Generate images
    start_time = time.time()
    
    with torch.no_grad():
        with autocast(enabled=True):
            if config.save_intermediates:
                image, intermediates = model.inferer.sample(
                    input_noise=noise, 
                    diffusion_model=model.model, 
                    scheduler=model.scheduler, 
                    save_intermediates=True, 
                    intermediate_steps=config.intermediate_steps
                )
            else:
                image = model.inferer.sample(
                    input_noise=noise, 
                    diffusion_model=model.model, 
                    scheduler=model.scheduler
                )
                intermediates = None
    
    generation_time = time.time() - start_time
    print(f"   Generation completed in {generation_time:.2f}s")
    
    # Display and save results
    display_results(image, intermediates, config, save_dir, title_prefix, generation_time)
    
    return image, intermediates


def main():
    parser = argparse.ArgumentParser(description='Generate samples with diffusion model')
    
    # Add all config arguments automatically
    add_config_args(parser, DiffusionModelConfig, prefix="model-")
    add_config_args(parser, DiffusionTrainingConfig, prefix="train-")
    
    # Keep existing manual arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./generated_samples',
                       help='Output directory for generated samples')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup configurations
    model_config = DiffusionModelConfig()
    training_config = DiffusionTrainingConfig()
    
    # Override with command line arguments
    model_config = update_config_from_args(model_config, args, prefix="model_")
    training_config = update_config_from_args(training_config, args, prefix="train_")
    
    # Create model
    print("🏗️ Creating diffusion model...")
    model = DiffusionModel(model_config, training_config)
    model.to(device)
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.dirname(args.checkpoint),
        model_name="diffusion"
    )
    
    checkpoint = checkpoint_manager.load_checkpoint(
        args.checkpoint, model.model, device=device
    )
    
    if checkpoint is None:
        print("❌ Failed to load checkpoint")
        return
    
    # Run inference
    print(f"🎨 Generating {args.num_samples} sample(s)...")
    
    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        
        # Create sample-specific save directory
        sample_dir = os.path.join(args.output_dir, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Generate sample
        image, intermediates = run_inference(
            model=model,
            config=training_config,
            device=device,
            save_dir=sample_dir,
            title_prefix=f"Sample_{i+1}"
        )
    
    print(f"\n✅ Generation completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()