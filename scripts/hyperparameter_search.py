#!/usr/bin/env python3
"""
Fixed config-driven hyperparameter search with grid search and proper wandb handling.
"""
import os
import sys
import wandb
import yaml
import argparse
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Print class distribution for training and validation sets
def print_class_distribution(dataset, set_name):
    """Print class distribution for a dataset."""
    class_counts = {}
    for idx in range(len(dataset)):
        label = dataset[idx]["label"]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = len(dataset)
    print(f"   {set_name} set class distribution:")
    for severity in sorted(class_counts.keys()):
        count = class_counts[severity]
        percentage = (count / total_samples) * 100
        print(f"      Level {severity}: {count} images ({percentage:.1f}%)")

def load_hypersearch_config(config_path="config.yaml"):
    """Load hyperparameter search configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'hypersearch' not in config or 'classifier' not in config['hypersearch']:
        raise ValueError("hypersearch.classifier section missing from config.yaml")
    
    return config['hypersearch']['classifier']


def calculate_model_parameters(base_channels, channels_multiple, num_res_blocks, attention_levels, img_size=128):
    """Calculate estimated parameters."""
    channels = [int(base_channels * mult) for mult in channels_multiple]
    
    params = 0
    
    # Initial conv (3->first_channel)
    params += 3 * channels[0] * 3 * 3
    
    # Down blocks
    for i, (ch, num_res, has_attention) in enumerate(zip(channels, num_res_blocks, attention_levels)):
        # ResBlocks
        params += num_res * (ch * ch * 3 * 3 * 2)
        
        # Attention
        if has_attention:
            current_spatial_size = img_size // (2 ** i)
            spatial_elements = current_spatial_size * current_spatial_size
            params += ch * ch * 3 + spatial_elements * ch
        
        # Downsampling conv
        if i < len(channels) - 1:
            params += ch * channels[i+1] * 3 * 3
    
    # Classifier head
    params += channels[-1] * 256 + 256 * 4
    
    return params


def validate_architecture(base_channels, channels_multiple, attention_levels, num_res_blocks, num_head_channels):
    """Validate architecture parameters."""
    channels = [int(base_channels * mult) for mult in channels_multiple]
    
    # Check array lengths match
    if len(channels_multiple) != len(attention_levels) or len(channels_multiple) != len(num_res_blocks):
        return False, f"Array length mismatch"
    
    # Check normalization group constraints
    for ch in channels:
        if ch % 32 != 0:
            return False, f"Channel {ch} not divisible by 32"
    
    # Check attention head constraints
    for ch, has_attention in zip(channels, attention_levels):
        if has_attention and ch % num_head_channels != 0:
            return False, f"Channel {ch} not divisible by num_head_channels {num_head_channels}"
    
    return True, "Valid"


def train_with_sweep():
    """Training function with proper wandb handling."""
    # Initialize wandb FIRST - this is critical!
    run = wandb.init()
    
    try:
        config = wandb.config
        print(f"🔧 Starting run with config: {dict(config)}")
        
        # Load hypersearch config from YAML
        hypersearch_config = load_hypersearch_config("config.yaml")
        
        # Get the selected architecture from config
        arch_name = config.architecture_type
        if arch_name not in hypersearch_config['architectures']:
            print(f"❌ Unknown architecture: {arch_name}")
            wandb.log({"status": "failed", "error": f"Unknown architecture: {arch_name}"})
            return
        
        # Get architecture definition from config.yaml
        arch_config = hypersearch_config['architectures'][arch_name]
        
        # Extract parameters from config.yaml
        channels_multiple = arch_config['channels_multiple']
        attention_levels = arch_config['attention_levels']
        num_res_blocks = arch_config['num_res_blocks']
        
        # Get other parameters from sweep config
        base_channels = config.base_channels
        img_size = config.img_size
        num_head_channels = config.num_head_channels
        
        # Validate architecture
        valid, msg = validate_architecture(
            base_channels, channels_multiple, attention_levels,
            num_res_blocks, num_head_channels
        )
        
        if not valid:
            print(f"❌ Invalid architecture: {msg}")
            wandb.log({"status": "failed", "error": f"Invalid architecture: {msg}"})
            return
        
        # Calculate estimated parameters
        estimated_params = calculate_model_parameters(
            base_channels, channels_multiple, num_res_blocks, attention_levels, img_size
        )
        
        print(f"🔢 Estimated parameters: {estimated_params:,}")
        
        # Check parameter limit from config
        max_params = hypersearch_config.get('max_parameters', 2_000_000)
        if estimated_params > max_params:
            print(f"❌ Model too large: {estimated_params:,} parameters (limit: {max_params:,})")
            wandb.log({"estimated_parameters": estimated_params, "status": "skipped_too_large"})
            return
        
        wandb.log({"estimated_parameters": estimated_params})
        
        # Create temporary config by modifying the base config
        base_config_path = "config.yaml"
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Update only the specific values for this run
        base_config['classifier']['model'].update({
            'spatial_dims': 2,
            'in_channels': 3,
            'out_channels': 4,
            'base_channels': base_channels,
            'channels_multiple': channels_multiple,
            'attention_levels': attention_levels,
            'num_res_blocks': num_res_blocks,
            'num_head_channels': num_head_channels,
            'with_conditioning': False
        })
        
        base_config['classifier']['training'].update({
            'img_size': img_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'batch_size': config.batch_size,
            'n_epochs': config.n_epochs,
            'noise_timesteps_train': config.noise_timesteps_train,
            'use_wandb': False,  # DISABLE wandb in trainer to avoid conflicts!
            'wandb_project': hypersearch_config.get('wandb_project', 'acne-classifier-sweep')
        })
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(base_config, f)
            temp_config_path = f.name
        
        try:
            # Import and run training
            print("🚀 Starting training...")
            
            os.environ['PYTHONPATH'] = f"{os.getcwd()}:{os.getcwd()}/src:{os.environ.get('PYTHONPATH', '')}"
            sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
            
            from data.dataset import AcneDataset
            from data.transforms import create_transforms
            from models.classifier import ClassifierModel
            from training.classifier_trainer import ClassifierTrainer
            from torch.utils.data import DataLoader, random_split
            import torch
            from utils.config_reader import ConfigReader
            
            # Load configuration
            config_reader = ConfigReader(temp_config_path)
            model_config = config_reader.get_classifier_model_config()
            training_config = config_reader.get_classifier_training_config()
            data_config = config_reader.get_data_config()
            
            # IMPORTANT: Disable wandb in trainer to avoid conflicts
            training_config.use_wandb = False
            
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🖥️ Using device: {device}")
            
            # Setup data
            train_transforms, val_transforms = create_transforms(
                img_size=training_config.img_size,
                apply_augmentation=data_config.apply_augmentation
            )
            
            full_dataset = AcneDataset(
                data_dir=data_config.dataset_path,
                transform=train_transforms,
                severity_levels=data_config.severity_levels
            )
            
            stats = full_dataset.get_statistics()
            print(f"📊 Dataset statistics: {stats}")
            
            # Split dataset
            train_size = int(training_config.train_split * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_ds, val_ds = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            val_ds.dataset.transform = val_transforms
            
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
            
            print(f"📊 Dataset: {len(train_ds)} train, {len(val_ds)} val")

            print_class_distribution(train_ds, "Training")
            print_class_distribution(val_ds, "Validation")
            
            # Create model
            print("🏗️ Creating model...")
            model = ClassifierModel(model_config)
            model.to(device)
            
            # Get actual parameter count
            actual_params = sum(p.numel() for p in model.model.parameters())
            print(f"🔢 Actual parameters: {actual_params:,}")
            wandb.log({"actual_parameters": actual_params})
            
            # Create trainer (with wandb disabled in trainer)
            trainer = ClassifierTrainer(model, training_config, device)
            
            # Train model and manually log to wandb from here
            print("🎓 Starting training...")
            
            # Train and manually track progress
            for epoch in range(training_config.n_epochs):
                train_loss, train_accuracy = trainer.train_epoch(train_loader, epoch)
                
                # Log to wandb manually
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                }, step=epoch)
                
                # Validation every 10 epochs
                if (epoch + 1) % 10 == 0:
                    val_loss, val_accuracy, class_accuracies = trainer.validate(val_loader)
                    
                    # Log validation metrics
                    val_metrics = {
                        "val/loss": val_loss,
                        "val/accuracy": val_accuracy,
                    }
                    
                    # Log per-class accuracies
                    for i, acc in enumerate(class_accuracies):
                        val_metrics[f"val/class_{i}_accuracy"] = acc
                    
                    wandb.log(val_metrics, step=epoch)
                    
                    # Update best metrics
                    if val_loss < trainer.best_val_loss:
                        trainer.best_val_loss = val_loss
                        trainer.best_val_accuracy = val_accuracy
                    
                    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_accuracy:.2f}%, "
                          f"Val Loss {val_loss:.4f}, Val Acc {val_accuracy:.2f}%")
            
            # Log final metrics
            final_metrics = {
                "final_val_accuracy": trainer.best_val_accuracy,
                "final_val_loss": trainer.best_val_loss,
                "total_epochs": training_config.n_epochs,
                "architecture_type": arch_name,
                "base_channels": base_channels,
                "img_size": img_size,
                "status": "completed"
            }
            
            wandb.log(final_metrics)
            
            print(f"✅ Training completed! Best val accuracy: {trainer.best_val_accuracy:.2f}%")
            
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        wandb.log({"status": "failed", "error": str(e)})
        raise e
    
    finally:
        # Make sure wandb run is finished
        wandb.finish()


def create_sweep_config(config_path="config.yaml"):
    """Create sweep configuration from config.yaml values - GRID SEARCH for all combinations."""
    hypersearch_config = load_hypersearch_config(config_path)
    
    # Get architecture names from config
    architecture_names = list(hypersearch_config['architectures'].keys())
    
    sweep_config = {
        'method': 'grid',  # CHANGED TO GRID to try all combinations!
        'metric': {
            'name': hypersearch_config.get('metric_name', 'final_val_accuracy'),
            'goal': hypersearch_config.get('metric_goal', 'maximize')
        },
        'parameters': {
            # Architecture selection from config
            'architecture_type': {
                'values': architecture_names
            },
            
            # Base channels from config
            'base_channels': {
                'values': hypersearch_config.get('base_channels_options', [32, 64])
            },
            
            # Image sizes from config
            'img_size': {
                'values': hypersearch_config.get('image_sizes', [128])
            },
            
            # Attention head channels from config
            'num_head_channels': {
                'values': hypersearch_config.get('num_head_channels_options', [32, 64])
            },
            
            # Training parameters from config - ALL VALUES, NOT RANGES
            'learning_rate': {
                'values': hypersearch_config.get('learning_rates', [1e-4])
            },
            'weight_decay': {
                'values': hypersearch_config.get('weight_decays', [0.05])
            },
            'batch_size': {
                'values': hypersearch_config.get('batch_sizes', [32])
            },
            'n_epochs': {
                'value': hypersearch_config.get('n_epochs', 100)
            },
            'noise_timesteps_train': {
                'values': hypersearch_config.get('noise_timesteps', [1000])
            }
        }
    }
    
    return sweep_config


def main():
    parser = argparse.ArgumentParser(description='Config-driven hyperparameter search')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to config file')
    parser.add_argument('--project', type=str, help='Wandb project name (overrides config)')
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--create-sweep', action='store_true')
    parser.add_argument('--sweep-id', type=str)
    parser.add_argument('--test-config', action='store_true')
    
    args = parser.parse_args()
    
    if args.test_config:
        # Test configuration loading
        try:
            hypersearch_config = load_hypersearch_config(args.config)
            print("🧪 Testing configuration from config.yaml:")
            print(f"📋 Found {len(hypersearch_config['architectures'])} architectures:")
            
            # Calculate total combinations
            total_combinations = 1
            total_combinations *= len(hypersearch_config['architectures'])
            total_combinations *= len(hypersearch_config.get('base_channels_options', [32]))
            total_combinations *= len(hypersearch_config.get('image_sizes', [128]))
            total_combinations *= len(hypersearch_config.get('num_head_channels_options', [64]))
            total_combinations *= len(hypersearch_config.get('learning_rates', [1e-4]))
            total_combinations *= len(hypersearch_config.get('weight_decays', [0.05]))
            total_combinations *= len(hypersearch_config.get('batch_sizes', [32]))
            total_combinations *= len(hypersearch_config.get('noise_timesteps', [1000]))
            
            print(f"🎯 Total combinations to try: {total_combinations}")
            
            for arch_name, arch_config in hypersearch_config['architectures'].items():
                for base_channels in hypersearch_config.get('base_channels_options', [32]):
                    for img_size in hypersearch_config.get('image_sizes', [128]):
                        params = calculate_model_parameters(
                            base_channels,
                            arch_config['channels_multiple'],
                            arch_config['num_res_blocks'],
                            arch_config['attention_levels'],
                            img_size
                        )
                        channels = [int(base_channels * mult) for mult in arch_config['channels_multiple']]
                        print(f"✅ {arch_name:20} | base={base_channels:2d} | img={img_size:3d} | "
                              f"channels={channels} | params={params:>7,}")
            
            print(f"\n📊 Search space summary:")
            print(f"   Architectures: {list(hypersearch_config['architectures'].keys())}")
            print(f"   Base channels: {hypersearch_config.get('base_channels_options', 'not set')}")
            print(f"   Image sizes: {hypersearch_config.get('image_sizes', 'not set')}")
            print(f"   Learning rates: {hypersearch_config.get('learning_rates', 'not set')}")
            print(f"   Batch sizes: {hypersearch_config.get('batch_sizes', 'not set')}")
            print(f"   Weight decays: {hypersearch_config.get('weight_decays', 'not set')}")
            print(f"   Noise timesteps: {hypersearch_config.get('noise_timesteps', 'not set')}")
            print(f"   Max parameters: {hypersearch_config.get('max_parameters', 'not set'):,}")
            
        except Exception as e:
            print(f"❌ Config test failed: {e}")
            return 1
        
        return 0
    
    if args.create_sweep:
        try:
            sweep_config = create_sweep_config(args.config)
            hypersearch_config = load_hypersearch_config(args.config)
            
            project_name = args.project or hypersearch_config.get('wandb_project', 'acne-classifier-sweep')
            
            print("🔧 Grid search configuration from config.yaml:")
            print(yaml.dump(sweep_config, default_flow_style=False))
            
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            
            print(f"🚀 Created GRID sweep: {sweep_id}")
            print(f"📋 Project: {project_name}")
            print(f"🌐 URL: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
            print(f"🏃 To run: export WANDB_SWEEP_ID={sweep_id} && python scripts/hyperparameter_search.py --sweep-id {sweep_id}")
            
            return sweep_id
            
        except Exception as e:
            print(f"❌ Failed to create sweep: {e}")
            return 1
    
    elif args.sweep_id:
        hypersearch_config = load_hypersearch_config(args.config)
        project_name = args.project or hypersearch_config.get('wandb_project', 'acne-classifier-sweep')
        
        print(f"🏃 Starting sweep agent for: {args.sweep_id}")
        wandb.agent(args.sweep_id, train_with_sweep, count=args.count, project=project_name)
        
    else:
        print("❌ Either --create-sweep or --sweep-id must be specified")
        print("💡 Use --test-config to validate your config.yaml and see total combinations")
        return 1


if __name__ == "__main__":
    main()