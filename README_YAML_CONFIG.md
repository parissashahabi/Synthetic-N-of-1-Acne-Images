# Acne Diffusion Model

A comprehensive deep learning framework for acne image generation and classification using diffusion models and the ACNE04 dataset.

## 🎛️ Configuration-Driven Approach

This project uses a **single configuration file** (`config.yaml`) for all settings. No command-line arguments are needed to change training parameters - simply edit the YAML file and run the scripts.

## Features

- **Diffusion Model**: High-quality acne image generation using DDPM
- **Classifier**: Acne severity level classification (0-3 levels)
- **Centralized Configuration**: All settings in `config.yaml`
- **GPU Cluster Ready**: Optimized for distributed training
- **Comprehensive Logging**: Training monitoring and visualization

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd acne_diffusion

# Create conda environment
conda create -n diffusion-env python=3.9 -y
conda activate diffusion-env

# Install dependencies
make install

# Verify installation
make verify-install
```

### 2. Configure Your Project

Edit `config.yaml` to set your preferences:

```yaml
# Example configurations
project:
  data_dir: "./data/acne_dataset"  # Path to your dataset
  
diffusion:
  training:
    n_epochs: 100                  # Number of training epochs
    batch_size: 16                 # Batch size
    learning_rate: 1e-4            # Learning rate

classifier:
  training:
    n_epochs: 200                  # Number of training epochs
    batch_size: 32                 # Batch size
    learning_rate: 3e-4            # Learning rate
```

### 3. Prepare Data

```bash
# Setup data directory structure
mkdir -p data/acne_dataset

# Your ACNE04 dataset should be organized as:
# data/acne_dataset/
# ├── acne0_1024/  # Clear skin (Level 0)
# ├── acne1_1024/  # Mild acne (Level 1)
# ├── acne2_1024/  # Moderate acne (Level 2)
# └── acne3_1024/  # Severe acne (Level 3)
```

### 4. Local Training

```bash
# Quick tests (uses settings from config.yaml)
make train-diffusion-quick        # 5 epochs test
make train-classifier-quick       # 50 epochs test

# Full training
make train-diffusion              # Full diffusion training
make train-classifier             # Full classifier training

# With Weights & Biases logging
make train-diffusion-wandb
make train-classifier-wandb
```

### 5. Cluster Usage

```bash
# Generate SLURM job scripts
make generate-slurm-scripts

# Submit jobs
sbatch job_test.sh                # Test setup (30 min)
sbatch job_diffusion.sh           # Train diffusion (24h)
sbatch job_classifier.sh          # Train classifier (24h)

# Resume training
export CHECKPOINT=path/to/checkpoint.pth
sbatch job_resume_diffusion.sh

# Generate samples
export CHECKPOINT=path/to/diffusion_checkpoint.pth
sbatch job_generate.sh

# Check jobs
make check-jobs
```

## 📋 Configuration Structure

All settings are controlled through `config.yaml`:

```yaml
# Project Settings
project:
  name: "acne-diffusion"
  data_dir: "./data/acne_dataset"
  experiments_dir: "./experiments"

# Diffusion Model Configuration
diffusion:
  model:
    base_channels: 256
    channels_multiple: [1, 1, 2, 3, 4, 4]
    attention_levels: [false, false, false, false, true, false]
  training:
    n_epochs: 100
    batch_size: 16
    learning_rate: 1e-4
    img_size: 128

# Classifier Configuration
classifier:
  model:
    base_channels: 128
    out_channels: 4  # 4 severity levels
  training:
    n_epochs: 200
    batch_size: 32
    learning_rate: 3e-4
    weight_decay: 0.05

# Generation Settings
generation:
  num_samples: 10
  num_inference_steps: 1000
  save_intermediates: true

# Cluster Settings
cluster:
  account: "your-account"
  partition: "gpu"

# Wandb Settings
wandb:
  project: "acne-diffusion"
```

## 🚀 Usage Examples

### Training Scripts

All scripts read from `config.yaml` automatically:

```bash
# Basic training
python scripts/train_diffusion.py --config config.yaml
python scripts/train_classifier.py --config config.yaml

# Quick tests
python scripts/train_diffusion.py --config config.yaml --quick-test
python scripts/train_classifier.py --config config.yaml --quick-test

# With Wandb logging
python scripts/train_diffusion.py --config config.yaml --enable-wandb
python scripts/train_classifier.py --config config.yaml --enable-wandb

# Resume training
python scripts/train_diffusion.py --config config.yaml --resume path/to/checkpoint.pth
```

### Generation and Evaluation

```bash
# Generate samples (reads generation config from YAML)
python scripts/generate_samples.py --config config.yaml --checkpoint path/to/diffusion_checkpoint.pth

# Evaluate classifier (reads evaluation config from YAML)
python scripts/evaluate_model.py --config config.yaml --checkpoint path/to/classifier_checkpoint.pth
```

### Configuration Management

```bash
# View current configuration
make show-config

# Edit configuration
make edit-config               # Opens config.yaml in editor

# Validate configuration
make validate-config
```

## 🔧 Changing Settings

To modify any training parameter:

1. **Edit `config.yaml`** - Change the desired values
2. **Run your command** - Scripts automatically read the new settings

**Example: Change diffusion epochs**
```yaml
# In config.yaml
diffusion:
  training:
    n_epochs: 500        # Changed from 100 to 500
```

Then simply run:
```bash
make train-diffusion     # Uses the new 500 epochs setting
```

## 📊 Available Commands

### Training
- `make train-diffusion-quick` - Quick diffusion test
- `make train-diffusion` - Full diffusion training  
- `make train-classifier-quick` - Quick classifier test
- `make train-classifier` - Full classifier training
- `make train-*-wandb` - Training with Wandb logging

### Generation & Evaluation
- `make generate CHECKPOINT=path` - Generate samples
- `make evaluate CHECKPOINT=path` - Evaluate classifier

### Cluster
- `make generate-slurm-scripts` - Create SLURM job files
- `make submit-test` - Submit test job
- `make submit-diffusion` - Submit diffusion training
- `make submit-classifier` - Submit classifier training

### Configuration
- `make show-config` - View current settings
- `make edit-config` - Edit config.yaml
- `make validate-config` - Check config validity

## 🎛️ Advanced Configuration

### Model Architecture
```yaml
diffusion:
  model:
    base_channels: 512              # Larger model
    channels_multiple: [1, 2, 3, 4, 5, 6]
    attention_levels: [false, false, true, true, true, true]
    num_res_blocks: 3               # More depth
```

### Training Settings
```yaml
diffusion:
  training:
    n_epochs: 2000                  # Long training
    batch_size: 8                   # For limited GPU memory
    learning_rate: 5e-5             # Lower learning rate
    mixed_precision: true           # Memory optimization
```

### Hyperparameter Search
```yaml
hypersearch:
  diffusion:
    learning_rates: [1e-4, 5e-5, 1e-5]
    batch_sizes: [16, 32]
    base_channels: [128, 256, 512]
    epochs: 50
```

## 📁 Project Structure

```
acne_diffusion/
├── config.yaml                 # ← All configuration here
├── scripts/
│   ├── train_diffusion.py      # Reads from config.yaml
│   ├── train_classifier.py     # Reads from config.yaml  
│   ├── generate_samples.py     # Reads from config.yaml
│   ├── evaluate_model.py       # Reads from config.yaml
│   └── generate_slurm_jobs.py  # Creates SLURM scripts
├── src/
│   └── utils/
│       └── config_reader.py    # Configuration reader
├── Makefile                    # Uses config.yaml
└── job_*.sh                    # Auto-generated SLURM scripts
```

## 🛠️ Troubleshooting

### Common Issues

1. **Config file not found**
   ```bash
   make validate-config          # Check if config.yaml exists and is valid
   ```

2. **Invalid configuration**
   ```bash
   python -c "from src.utils.config_reader import ConfigReader; ConfigReader('config.yaml')"
   ```

3. **GPU memory issues**
   ```yaml
   # In config.yaml - reduce batch size
   diffusion:
     training:
       batch_size: 8             # Reduced from 16
   ```

### Getting Help

```bash
make help                       # Show all available commands
make status                     # Show project status
make show-config               # Show current configuration
```

## ✅ Benefits of Configuration-Driven Approach

1. **Centralized Control**: All settings in one place
2. **Reproducible**: Easy to share exact configurations
3. **Version Control**: Track configuration changes in git
4. **No CLI Complexity**: Simple, consistent command interface
5. **Cluster Friendly**: SLURM scripts automatically use config settings
6. **Error Prevention**: No risk of CLI argument mistakes

## 🎯 Quick Reference

**To change ANY setting:** Edit `config.yaml`  
**To run training:** Use make commands or scripts with `--config config.yaml`  
**To use on cluster:** Generate SLURM scripts with `make generate-slurm-scripts`  
**For help:** Run `make help`

Happy training! 🚀