#!/usr/bin/env python3
"""
Training script for classifier model.
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
from models.classifier import ClassifierModel
from training.classifier_trainer import ClassifierTrainer
from configs.classifier_config import ClassifierModelConfig, ClassifierTrainingConfig
from configs.base_config import DataConfig
from utils.visualization import show_batch, plot_learning_curves


def setup_data(config: ClassifierTrainingConfig, data_config: DataConfig):
    """Setup data loaders."""
    print("📁 Setting up data loaders...")
    
    # Create transforms
    train_transforms, val_transforms = create_transforms(
        img_size=config.img_size,
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
    train_size = int(config.train_split * len(full_dataset))
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
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        drop_last=True, 
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        drop_last=True,
        pin_memory=config.pin_memory
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train classifier model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, help='Path to dataset')
    parser.add_argument('--experiment-dir', type=str, default='./experiments', 
                       help='Experiment directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    # Wandb arguments
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='acne-diffusion',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, help='Wandb entity/username')
    parser.add_argument('--wandb-tags', type=str, nargs='+', help='Wandb tags')
    parser.add_argument('--wandb-name', type=str, help='Wandb run name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Setup configurations
    model_config = ClassifierModelConfig()
    training_config = ClassifierTrainingConfig()
    data_config = DataConfig()
    
    # Override config with command line arguments
    if args.data_dir:
        data_config.dataset_path = args.data_dir
    if args.experiment_dir:
        training_config.experiment_dir = args.experiment_dir
    if args.epochs:
        training_config.n_epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.lr:
        training_config.learning_rate = args.lr
    
    # Wandb configuration
    if args.wandb:
        training_config.use_wandb = True
    if args.wandb_project:
        training_config.wandb_project = args.wandb_project
    if args.wandb_entity:
        training_config.wandb_entity = args.wandb_entity
    if args.wandb_tags:
        training_config.wandb_tags = args.wandb_tags
    
    # Create experiment name with timestamp if wandb name provided
    if args.wandb_name:
        import time
        timestamp = int(time.time())
        training_config.experiment_dir = f"{training_config.experiment_dir}/{args.wandb_name}_{timestamp}"
    
    # Setup data
    train_loader, val_loader = setup_data(training_config, data_config)
    
    # Show sample batch
    sample_dir = os.path.join(training_config.experiment_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, 'training_samples.png')
    show_batch(train_loader, num_samples=5, title="Training Samples", save_path=sample_path)
    
    # Create model
    print("🏗️ Creating classifier model...")
    model = ClassifierModel(model_config)
    model.to(device)
    
    # Print model summary
    summary = model.get_model_summary()
    print(f"📊 Model Summary:")
    print(f"   Total parameters: {summary['total_parameters']:,}")
    print(f"   Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"   Model size: {summary['model_size_mb']:.1f} MB")
    print(f"   Classes: {model_config.out_channels} (acne severity levels)")
    
    # Create trainer
    trainer = ClassifierTrainer(model, training_config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        if not trainer.resume_from_checkpoint(args.resume):
            print("❌ Failed to resume from checkpoint")
            return
    
    # Start training
    print("🚀 Starting training...")
    final_model_path = trainer.train(train_loader, val_loader)
    
    # Plot learning curves
    results_dir = os.path.join(training_config.experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    curves_path = os.path.join(results_dir, 'learning_curves_classifier.png')
    
    plot_learning_curves(
        train_losses=trainer.epoch_loss_list,
        val_losses=trainer.val_epoch_loss_list,
        val_accuracies=trainer.val_accuracy_list,
        save_path=curves_path
    )
    
    print("✅ Training completed!")
    print(f"🌟 Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"🎯 Best validation accuracy: {trainer.best_val_accuracy:.2f}%")
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"📁 Experiment directory: {training_config.experiment_dir}")


if __name__ == "__main__":
    main()