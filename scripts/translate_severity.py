#!/usr/bin/env python3
"""
Acne severity translation script - Convert images between different acne severity levels.
"""
import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.image_translation import AcneSeverityTranslator
from data.transforms import create_transforms
from utils.config_reader import ConfigReader


def load_image_from_path(image_path: str, img_size: int = 128) -> torch.Tensor:
    """Load and preprocess image from path."""
    # Load image
    image = Image.open(image_path)
    
    # Create preprocessing transforms (no augmentation)
    _, transforms = create_transforms(img_size=img_size, apply_augmentation=True)
    
    # Apply transforms
    sample = {"image": image_path, "label": 0}  # Dummy label
    transformed = transforms(sample)
    
    # Add batch dimension
    return transformed["image"].unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='Translate acne severity levels')
    
    # Required arguments
    parser.add_argument('--diffusion-checkpoint', type=str, required=True,
                       help='Path to trained diffusion model checkpoint')
    parser.add_argument('--classifier-checkpoint', type=str, required=True,
                       help='Path to trained classifier model checkpoint')
    parser.add_argument('--input-image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--target-severity', type=int, required=True, choices=[0, 1, 2, 3],
                       help='Target severity level (0=clear, 1=mild, 2=moderate, 3=severe)')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--source-severity', type=int, choices=[0, 1, 2, 3],
                       help='Source severity level (if known, for visualization)')
    parser.add_argument('--output-dir', type=str, default='./translation_results',
                       help='Output directory for results')
    parser.add_argument('--num-steps', type=int, default=250,
                       help='Number of translation steps')
    parser.add_argument('--guidance-scale', type=float, default=30.0,
                       help='Gradient guidance scale')
    parser.add_argument('--save-process', action='store_true',
                       help='Save intermediate gradient visualizations')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    
    # Validate input file
    if not os.path.exists(args.input_image):
        print(f"❌ Input image not found: {args.input_image}")
        return 1
    
    # Validate checkpoints
    for checkpoint_path in [args.diffusion_checkpoint, args.classifier_checkpoint]:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return 1
    
    print(f"📋 Translation Configuration:")
    print(f"   Input image: {args.input_image}")
    print(f"   Target severity: {args.target_severity}")
    print(f"   Source severity: {args.source_severity or 'unknown'}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Translation steps: {args.num_steps}")
    print(f"   Guidance scale: {args.guidance_scale}")
    print(f"   Save process: {args.save_process}")
    
    # Load configuration
    try:
        config_reader = ConfigReader(args.config)
        img_size = config_reader.get('diffusion.training.img_size', 128)
    except Exception as e:
        print(f"⚠️ Could not load config: {e}")
        img_size = 128
    
    # Load and preprocess input image
    print(f"📸 Loading input image...")
    try:
        input_image = load_image_from_path(args.input_image, img_size)
        print(f"   Image shape: {input_image.shape}")
        print(f"   Image range: [{input_image.min():.3f}, {input_image.max():.3f}]")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return 1
    
    # Initialize translator
    print(f"🏗️ Initializing translator...")
    try:
        translator = AcneSeverityTranslator(
            diffusion_checkpoint=args.diffusion_checkpoint,
            classifier_checkpoint=args.classifier_checkpoint,
            config_path=args.config,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to initialize translator: {e}")
        return 1
    
    # Perform translation
    print(f"🎨 Starting translation...")
    try:
        result = translator.translate_and_visualize(
            input_image=input_image,
            source_severity=args.source_severity or -1,  # Use -1 if unknown
            target_severity=args.target_severity,
            output_dir=args.output_dir,
            prefix=f"severity_{args.target_severity}",
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            save_process=args.save_process
        )
        
        print(f"✅ Translation completed successfully!")
        print(f"📁 Results saved to: {result['output_dir']}")
        print(f"📄 Generated files:")
        for file_path in result['saved_files']:
            print(f"   - {file_path}")
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())