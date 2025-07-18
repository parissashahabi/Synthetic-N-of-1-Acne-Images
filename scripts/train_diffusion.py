#!/usr/bin/env python3
"""
Training script for diffusion model - reads configuration from config.yaml only.
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import AcneDataset
from data.transforms import create_transforms
from models.diffusion import DiffusionModel
from training.diffusion_trainer import DiffusionTrainer
from utils.visualization import show_batch
from utils.config_reader import ConfigReader


def setup_data(training_config, data_config):
    """Setup data loaders."""
    print("📁 Setting up data loaders...")
    
    # Create transforms
    train_transforms, val_transforms = create_transforms(
        img_size=training_config.img_size,
        apply_augmentation=data_config.apply_augmentation
    )
    
    # Create dataset
    full_dataset = AcneDataset(
        data_dir=data_config.dataset_path,
        transform=train_transforms,
        severity_levels=data_config.severity_levels
    )
    
    # Print dataset statistics
    stats = full_dataset.get_statistics()
    total_images = len(full_dataset)
    
    print(f"📊 Dataset loaded successfully!")
    print(f"   Total images: {total_images}")
    print("   Images by severity level:")
    for severity, count in sorted(stats.items()):
        percentage = (count / total_images) * 100
        print(f"      Level {severity}: {count} images ({percentage:.1f}%)")
    
    # Split dataset
    train_size = int(training_config.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transforms
    val_ds.dataset.transform = val_transforms
    
    print(f"📈 Dataset split:")
    print(f"   Training: {len(train_ds)} images")
    print(f"   Validation: {len(val_ds)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=training_config.batch_size, 
        shuffle=True, 
        num_workers=training_config.num_workers, 
        drop_last=True, 
        pin_memory=training_config.pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=training_config.batch_size, 
        shuffle=False, 
        num_workers=training_config.num_workers, 
        drop_last=True,
        pin_memory=training_config.pin_memory
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model')
    
    # Only essential arguments that don't change configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with reduced epochs')
    parser.add_argument('--enable-wandb', action='store_true',
                       help='Enable wandb logging (overrides config)')
    parser.add_argument('--wandb-name', type=str, 
                       help='Wandb run name (overrides experiment name)')
    
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
    data_config = config_reader.get_data_config()
    
    # Apply quick test settings if requested
    if args.quick_test:
        quick_config = config_reader.get('diffusion.quick_test', {})
        training_config.n_epochs = quick_config.get('n_epochs', 5)
        training_config.batch_size = quick_config.get('batch_size', 8)
        print(f"⚡ Quick test mode: {training_config.n_epochs} epochs, batch size {training_config.batch_size}")
    
    # Apply wandb settings if enabled
    if args.enable_wandb:
        wandb_config = config_reader.get_wandb_config()
        training_config.use_wandb = True
        training_config.wandb_project = wandb_config.get('project', 'acne-diffusion')
        training_config.wandb_entity = wandb_config.get('entity')
        training_config.wandb_tags = wandb_config.get('tags', ['diffusion'])
        print("🌐 Wandb logging enabled")
    
    # Set custom wandb name if provided
    if args.wandb_name:
        import time
        timestamp = int(time.time())
        training_config.experiment_dir = f"{config_reader.get('project.experiments_dir')}/{args.wandb_name}_{timestamp}"
        print(f"📝 Custom experiment name: {args.wandb_name}")
    
    # Setup device
    device_config = training_config.device
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    print(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Print configuration summary
    print(f"\n📋 Configuration Summary:")
    print(f"   Data directory: {data_config.dataset_path}")
    print(f"   Experiment directory: {training_config.experiment_dir}")
    print(f"   Epochs: {training_config.n_epochs}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Image size: {training_config.img_size}")
    print(f"   Model base channels: {model_config.base_channels}")
    print(f"   Mixed precision: {training_config.mixed_precision}")
    
    # Setup data
    train_loader, val_loader = setup_data(training_config, data_config)
    
    # Show sample batch
    sample_dir = os.path.join(training_config.experiment_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, 'training_samples.png')
    show_batch(train_loader, num_samples=5, title="Training Samples", save_path=sample_path)
    
    # Create model
    print("🏗️ Creating diffusion model...")
    model = DiffusionModel(model_config, training_config)
    model.to(device)
    
    # Print model summary
    summary = model.get_model_summary()
    print(f"📊 Model Summary:")
    print(f"   Total parameters: {summary['total_parameters']:,}")
    print(f"   Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"   Model size: {summary['model_size_mb']:.1f} MB")
    
    # Create trainer
    trainer = DiffusionTrainer(model, training_config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        if not trainer.resume_from_checkpoint(args.resume):
            print("❌ Failed to resume from checkpoint")
            return 1
    
    # Start training
    print("🚀 Starting training...")
    final_model_path = trainer.train(train_loader, val_loader)
    
    print("✅ Training completed!")
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"📁 Experiment directory: {training_config.experiment_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())