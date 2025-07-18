#!/usr/bin/env python3
"""
Generate samples using trained diffusion model - reads configuration from config.yaml only.
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
from utils.checkpoints import CheckpointManager
from utils.config_reader import ConfigReader


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
    if intermediates and config.get('save_intermediates', False):
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
        plt.title(f"{title_prefix} - Denoising Process ({config.get('intermediate_steps', 100)} steps)", 
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
        config['num_samples'], 
        3, 
        config['img_size'], 
        config['img_size']
    )).to(device)
    
    print(f"   Noise shape: {noise.shape}")
    print(f"   Inference steps: {config['num_inference_steps']}")
    
    # Set scheduler timesteps
    model.scheduler.set_timesteps(num_inference_steps=config['num_inference_steps'])
    
    # Generate images
    start_time = time.time()
    
    with torch.no_grad():
        with autocast(enabled=True):
            if config.get('save_intermediates', False):
                image, intermediates = model.inferer.sample(
                    input_noise=noise, 
                    diffusion_model=model.model, 
                    scheduler=model.scheduler, 
                    save_intermediates=True, 
                    intermediate_steps=config.get('intermediate_steps', 100)
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
    
    # Essential arguments only
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./generated_samples',
                       help='Output directory for generated samples')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible generation')
    
    args = parser.parse_args()
    
    # Load configuration from YAML
    print(f"📖 Loading configuration from: {args.config}")
    try:
        config_reader = ConfigReader(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    # Get configurations
    model_config = config_reader.get_diffusion_model_config()
    training_config = config_reader.get_diffusion_training_config()
    generation_config = config_reader.get_generation_config()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    
    # Set random seed if provided
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration summary
    print(f"\n📋 Generation Configuration:")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Number of samples: {generation_config['num_samples']}")
    print(f"   Image size: {generation_config['img_size']}")
    print(f"   Inference steps: {generation_config['num_inference_steps']}")
    print(f"   Save intermediates: {generation_config['save_intermediates']}")
    
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
        return 1
    
    # Run inference
    print(f"🎨 Generating {generation_config['num_samples']} sample(s)...")
    
    for i in range(generation_config['num_samples']):
        print(f"\n--- Sample {i+1}/{generation_config['num_samples']} ---")
        
        # Create sample-specific save directory
        sample_dir = os.path.join(args.output_dir, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Generate sample
        image, intermediates = run_inference(
            model=model,
            config=generation_config,
            device=device,
            save_dir=sample_dir,
            title_prefix=f"Sample_{i+1}"
        )
    
    print(f"\n✅ Generation completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())