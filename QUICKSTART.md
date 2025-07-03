# Quick Start Guide - Updated

Get up and running with the Acne Diffusion framework in under 10 minutes using conda with full configuration control!

## 🚀 Quick Setup

### 1. Clone and Setup Conda Environment
```bash
git clone <repository-url>
cd acne_diffusion

# Create conda environment
make env-create

# Activate environment
conda activate diffusion-env

# Install dependencies (PyTorch + CUDA + all packages)
make install

# Setup project structure
make setup-cluster

# Generate SLURM scripts
make generate-slurm-scripts
```

### 2. Check Setup
```bash
# Verify installation
make verify-install

# Check GPU availability
make check-gpu

# Check dataset
make check-data

# Complete status check
make status
```

### 3. For Cluster Users
```bash
# Request interactive GPU session
make interactive-gpu
# This runs: srun --partition=gpu-interactive --account=sci-lippert --cpus-per-task=4 --gpus=1 --mem=16G --time=02:00:00 --pty bash

# Or for larger jobs
make interactive-gpu-large

# Then activate your environment
conda activate diffusion-env
```

### 4. Prepare Data
```bash
# Create data directory
make prepare-data

# Copy your ACNE04 dataset to: data/acne_dataset/
# Check dataset structure
make check-data
```

Expected structure:
```
data/acne_dataset/
├── acne0_1024/  # Clear skin (Level 0)
├── acne1_1024/  # Mild acne (Level 1)
├── acne2_1024/  # Moderate acne (Level 2)
└── acne3_1024/  # Severe acne (Level 3)
```

### 5. Start Training (New Argument Format)
```bash
# Ensure environment is active
conda activate diffusion-env

# Quick test (5 epochs, ~10 minutes)
make train-quick-diffusion

# Full training with new arguments
make train-diffusion
make train-classifier

# Training with wandb
make train-diffusion-wandb
make train-classifier-wandb
```

## 🎛️ New Configuration System

All configuration parameters can now be specified via command line arguments using prefixes:

### Prefixes:
- `--model-*` for model architecture parameters
- `--train-*` for training parameters  
- `--data-*` for dataset parameters

### Example Training Commands

#### Basic Diffusion Training
```bash
python scripts/train_diffusion.py \
    --data-dataset-path ./data/acne_dataset \
    --train-experiment-dir ./experiments/my_diffusion \
    --train-n-epochs 100 \
    --train-batch-size 16 \
    --train-learning-rate 1e-4
```

#### Custom Model Architecture
```bash
python scripts/train_diffusion.py \
    --data-dataset-path ./data/acne_dataset \
    --train-experiment-dir ./experiments/custom_diffusion \
    --model-base-channels 512 \
    --model-channels-multiple 1 2 3 4 5 \
    --model-attention-levels False False True True True \
    --model-num-res-blocks 3 \
    --train-num-train-timesteps 2000
```

#### Classifier with Custom Settings
```bash
python scripts/train_classifier.py \
    --data-dataset-path ./data/acne_dataset \
    --train-experiment-dir ./experiments/custom_classifier \
    --model-base-channels 256 \
    --model-out-channels 4 \
    --train-learning-rate 3e-4 \
    --train-weight-decay 0.1 \
    --train-noise-timesteps-train 1500
```

## 📋 Common Commands (Updated)

### 🎓 Training with Makefile
```bash
# Basic training
make train-diffusion
make train-classifier

# Quick tests
make train-quick-diffusion
make train-quick-classifier

# Custom training with parameters
make train-diffusion-custom EPOCHS=200 BATCH_SIZE=32 CHANNELS=512
make train-classifier-custom EPOCHS=300 LR=1e-4 WD=0.05

# Training with wandb logging
make train-diffusion-wandb
make train-classifier-wandb

# Hyperparameter search
make hypersearch-diffusion
make hypersearch-classifier
```

### 🔄 Resume Training
```bash
# Using Makefile
make resume-diffusion CHECKPOINT=path/to/checkpoint.pth
make resume-classifier CHECKPOINT=path/to/checkpoint.pth

# Auto-resume from latest checkpoint
make auto-resume-diffusion
make auto-resume-classifier

# Custom resume with parameters
make resume-diffusion-custom CHECKPOINT=path/to/checkpoint.pth EPOCHS=50 LR=5e-5
```

### 🎨 Generation
```bash
# Basic generation
make generate CHECKPOINT=path/to/diffusion_checkpoint.pth

# Batch generation
make generate-batch CHECKPOINT=path/to/checkpoint.pth NUM_SAMPLES=50

# High-quality generation
make generate-hq CHECKPOINT=path/to/checkpoint.pth

# Direct command
python scripts/generate_samples.py \
    --checkpoint path/to/checkpoint.pth \
    --output-dir ./my_samples \
    --train-num-samples 20 \
    --train-num-inference-steps 2000 \
    --train-save-intermediates
```

### 📊 Evaluation
```bash
# Using Makefile
make evaluate CHECKPOINT=path/to/classifier_checkpoint.pth

# Direct command
python scripts/evaluate_model.py \
    --checkpoint path/to/checkpoint.pth \
    --data-dataset-path ./data/acne_dataset \
    --output-dir ./evaluation_results \
    --batch-size 64
```

## 🖥️ Cluster Usage (Updated SLURM Scripts)

### Generate All SLURM Scripts
```bash
make generate-slurm-scripts
```

This creates comprehensive job scripts with the new argument format:

#### Basic Jobs
- `job_test_diffusion.sh` - Quick test (30 min)
- `job_test_classifier.sh` - Quick test (30 min)
- `job_diffusion.sh` - Standard training
- `job_classifier.sh` - Standard training
- `job_diffusion_wandb.sh` - Training with wandb
- `job_classifier_wandb.sh` - Training with wandb

#### Advanced Jobs
- `job_diffusion_hq.sh` - High-quality training
- `job_classifier_robust.sh` - Robust training with high noise
- `job_diffusion_long.sh` - Long-term training (72h)
- `job_diffusion_multi_gpu.sh` - Multi-GPU training

#### Research Jobs
- `job_diffusion_hyperparam.sh` - Hyperparameter search
- `job_classifier_hyperparam.sh` - Hyperparameter search
- `job_ablation_attention.sh` - Ablation studies
- `job_ensemble_diffusion.sh` - Ensemble training

### Submit Jobs
```bash
# Test setup first
make submit-test

# Submit training jobs
make submit-diffusion
make submit-classifier

# Submit all jobs
make submit-all

# Monitor jobs
make check-jobs
make tail-logs
```

### Resume Training on Cluster
```bash
# Set checkpoint path
export CHECKPOINT_TO_RESUME=experiments/diffusion_*/checkpoints/checkpoint_epoch_50.pth
sbatch job_diffusion_resume.sh

export CHECKPOINT_TO_RESUME=experiments/classifier_*/checkpoints/checkpoint_epoch_100.pth
sbatch job_classifier_resume.sh
```

### Generation on Cluster
```bash
# Set checkpoint path
export CHECKPOINT_TO_USE=experiments/diffusion_*/checkpoints/best_diffusion.pth
sbatch job_generate_samples.sh

# High-quality generation
sbatch job_generate_hq.sh
```

## 🎛️ Full Configuration Examples

### All Available Parameters

#### Diffusion Model Parameters
```bash
--model-spatial-dims 2                          # 2D or 3D
--model-in-channels 3                           # Input channels (RGB=3)
--model-out-channels 3                          # Output channels
--model-base-channels 256                       # Base channel count
--model-channels-multiple 1 1 2 3 4 4          # Channel multipliers per level
--model-attention-levels False False False False True False  # Attention at each level
--model-num-res-blocks 2                        # ResNet blocks per level
--model-num-head-channels 64                    # Attention head channels
--model-with-conditioning                        # Enable conditioning
--model-resblock-updown                         # Use ResBlocks for up/downsampling
--model-dropout 0.0                             # Dropout rate
```

#### Training Parameters
```bash
--train-img-size 128                            # Image resolution
--train-batch-size 16                           # Batch size
--train-train-split 0.8                         # Train/val split
--train-num-workers 4                           # Data loader workers
--train-pin-memory                              # Pin memory for GPU
--train-device cuda                             # Device (cuda/cpu)
--train-mixed-precision                         # Enable AMP
--train-log-interval 10                         # Log every N batches
--train-checkpoint-interval 10                  # Save checkpoint every N epochs
--train-val-interval 10                         # Validate every N epochs
--train-experiment-dir ./experiments/my_run     # Experiment directory
--train-learning-rate 1e-4                      # Learning rate
--train-n-epochs 100                            # Number of epochs
--train-num-train-timesteps 1000                # Diffusion timesteps
--train-num-inference-steps 1000                # Inference steps
--train-num-samples 1                           # Samples to generate
--train-save-intermediates                      # Save intermediate steps
```

#### Data Parameters
```bash
--data-dataset-path ./data/acne_dataset         # Dataset path
--data-severity-levels 0 1 2 3                  # Severity levels to include
--data-apply-augmentation                       # Enable data augmentation
--data-drop-last                                # Drop last batch
--data-shuffle-train                            # Shuffle training data
--data-num-classes 4                            # Number of classes
```

### Complex Training Examples

#### Large High-Quality Model
```bash
python scripts/train_diffusion.py \
    --data-dataset-path ./data/acne_dataset \
    --train-experiment-dir ./experiments/hq_diffusion \
    --train-n-epochs 2000 \
    --train-batch-size 8 \
    --train-learning-rate 5e-5 \
    --model-base-channels 512 \
    --model-channels-multiple 1 2 3 4 5 6 \
    --model-attention-levels False False True True True True \
    --model-num-res-blocks 3 \
    --train-num-train-timesteps 2000 \
    --train-checkpoint-interval 50 \
    --wandb \
    --wandb-name hq_diffusion_run
```

#### Robust Classifier
```bash
python scripts/train_classifier.py \
    --data-dataset-path ./data/acne_dataset \
    --train-experiment-dir ./experiments/robust_classifier \
    --train-n-epochs 3000 \
    --train-batch-size 32 \
    --train-learning-rate 1e-4 \
    --train-weight-decay 0.1 \
    --model-base-channels 256 \
    --model-channels-multiple 1 2 3 4 5 \
    --train-noise-timesteps-train 1500 \
    --train-checkpoint-interval 100 \
    --wandb \
    --wandb-name robust_classifier
```

## 📊 Results and Outputs (Updated)

### Training Results Structure
```
experiments/diffusion_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_diffusion.pth           # Best model
│   ├── final_diffusion.pth          # Final model  
│   └── diffusion_checkpoint_*.pth   # Regular checkpoints
├── samples/
│   ├── generated_image_*.png        # Generated samples during training
│   └── generation_process_*.png     # Denoising process visualization
├── results/
│   └── learning_curves.png          # Training curves
└── logs/
    └── training.log                 # Training logs
```

### Check Results
```bash
# Show recent results
make show-results
make show-checkpoints
make show-experiments

# Check project status
make status

# View specific experiment
ls experiments/diffusion_*/
ls experiments/classifier_*/
```

## ⚡ Performance and Optimization

### Benchmarking
```bash
# Run performance benchmark
make benchmark

# Analyze model architecture
make analyze-model

# Monitor system resources
make monitor-gpu
```

### Memory Optimization
```bash
# For limited GPU memory (< 8GB)
python scripts/train_diffusion.py \
    --train-batch-size 4 \
    --model-base-channels 128 \
    --train-mixed-precision

# For high-end GPUs (> 16GB)
python scripts/train_diffusion.py \
    --train-batch-size 64 \
    --model-base-channels 512 \
    --model-num-res-blocks 3
```

## 🛠️ Maintenance and Cleanup

### Cleanup Commands
```bash
# Clean old experiments
make clean-experiments

# Clean logs
make clean-logs

# Clean Python cache
make clean-cache

# Complete cleanup
make clean-all
```

### Backup
```bash
# Backup experiments
make backup-experiments

# Backup only checkpoints
make backup-checkpoints
```

## 🔧 Configuration Examples Reference

### View All Available Configurations
```bash
# Get help for any script
python scripts/train_diffusion.py --help
python scripts/train_classifier.py --help
python scripts/generate_samples.py --help

# See configuration examples
make config-examples
```

### Makefile Help
```bash
# See all available make commands
make help

# Get quickstart guide
make quickstart
```

## 🆘 Troubleshooting (Updated)

### Common Issues

1. **Import Errors**: Run `make verify-install`
2. **GPU Issues**: Run `make check-gpu`
3. **Dataset Issues**: Run `make check-data`
4. **Environment Issues**: Run `make conda-info`

### Debug Training
```bash
# Quick test to verify setup
make train-quick-diffusion

# Check logs
make tail-logs

# Monitor resources
make monitor-gpu
```

### Cluster Issues
```bash
# Check job status
make check-jobs

# View recent logs
make show-job-logs

# Cancel all jobs if needed
make cancel-jobs
```

## 🎯 Success Indicators

You'll know everything is working when:

✅ `make verify-install` passes all checks  
✅ `make check-gpu` shows CUDA available  
✅ `make check-data` finds your dataset  
✅ `make train-quick-diffusion` completes successfully  
✅ Generated samples appear in experiments/*/samples/  
✅ Learning curves show decreasing loss  
✅ SLURM jobs submit and run successfully  

## 🚀 Advanced Workflows

### Research Workflow
```bash
# 1. Quick test
make train-quick-diffusion

# 2. Hyperparameter search
make hypersearch-diffusion

# 3. Ablation studies
sbatch job_ablation_attention.sh

# 4. Ensemble training
sbatch job_ensemble_diffusion.sh

# 5. Long-term training
sbatch job_diffusion_long.sh
```

### Production Workflow
```bash
# 1. Test setup
make submit-test

# 2. Train models
make submit-diffusion
make submit-classifier

# 3. Generate samples
export CHECKPOINT_TO_USE=best_model.pth
sbatch job_generate_samples.sh

# 4. Evaluate performance
export CHECKPOINT_TO_EVALUATE=best_classifier.pth
sbatch job_evaluate_classifier.sh
```

Happy training with full configuration control! 🚀🎛️