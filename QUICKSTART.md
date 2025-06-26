# Quick Start Guide

Get up and running with the Acne Diffusion framework in under 10 minutes using conda!

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
```

### 2. Check GPU Availability
```bash
# Check if CUDA/GPU is available
make check-gpu

# Monitor GPU usage (optional)
make monitor-gpu
```

### 3. For Cluster Users (using your srun command)
```bash
# Request interactive GPU session
make interactive-gpu
# This runs: srun --partition=gpu-interactive --account=sci-lippert --cpus-per-task=4 --gpus=1 --mem=16G --time=02:00:00 --pty bash

# Or for larger jobs
make interactive-gpu-large
# This runs: srun --partition=gpu-interactive --account=sci-lippert --cpus-per-task=8 --gpus=1 --mem=32G --time=04:00:00 --pty bash

# Then activate your environment
conda activate diffusion-env
```

### 4. Prepare Data
```bash
# Create data directory
mkdir -p data

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

### 5. Start Training
```bash
# Ensure environment is active
conda activate diffusion-env

# Train diffusion model (4-6 hours on single GPU)
make train-diffusion

# Train classifier (2-3 hours on single GPU)
make train-classifier

# Quick test (5 epochs, ~10 minutes)
make train-quick
```

## 📋 Common Commands

### 🐍 Conda Environment Management
```bash
# Create environment
make env-create

# Activate environment
conda activate diffusion-env

# Export current environment
make env-export

# Create environment from exported file
make env-from-file

# Check conda info
make conda-info

# Remove environment (if needed)
make env-remove
```

### 🎓 Training
```bash
# Custom diffusion training
python scripts/train_diffusion.py \
    --data-dir ./data/acne_dataset \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --experiment-dir ./experiments/my_diffusion_run

# Custom classifier training
python scripts/train_classifier.py \
    --data-dir ./data/acne_dataset \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --experiment-dir ./experiments/my_classifier_run

# Resume training from checkpoint
python scripts/train_diffusion.py \
    --resume ./experiments/diffusion_*/checkpoints/diffusion_checkpoint_epoch_50.pth \
    --epochs 50
```

### 🎨 Generation
```bash
# Find your trained models
make show-checkpoints

# Generate samples using make command
make generate CHECKPOINT=experiments/diffusion_*/checkpoints/best_diffusion.pth

# Or generate directly with custom settings
python scripts/generate_samples.py \
    --checkpoint path/to/best_diffusion.pth \
    --output-dir ./my_samples \
    --num-samples 20 \
    --num-steps 1000 \
    --save-process
```

### 📊 Evaluation
```bash
# Evaluate classifier performance
make evaluate CHECKPOINT=experiments/classifier_*/checkpoints/best_classifier.pth

# Or evaluate with custom settings
python scripts/evaluate_model.py \
    --checkpoint path/to/best_classifier.pth \
    --data-dir ./data/acne_dataset \
    --output-dir ./my_evaluation \
    --batch-size 64
```

### 🖥️ GPU Cluster Usage (Your Setup)

#### Interactive Sessions
```bash
# Quick 2-hour session (4 CPUs, 16GB RAM)
make interactive-gpu

# Longer 4-hour session (8 CPUs, 32GB RAM)  
make interactive-gpu-large

# Custom session (modify as needed)
srun --partition=gpu-interactive --account=sci-lippert \
     --cpus-per-task=8 --gpus=1 --mem=32G --time=06:00:00 --pty bash
```

#### Generate SLURM Job Scripts
```bash
# Create all job script templates (updates with your account)
./slurm_job_examples.sh
```

#### Submit Training Jobs
```bash
# Single GPU diffusion training
sbatch job_diffusion_single.sh

# Single GPU classifier training  
sbatch job_classifier_single.sh

# Multi-GPU training (if available)
sbatch job_diffusion_multi.sh

# Hyperparameter search
sbatch job_hyperparam_search.sh
```

#### Monitor Cluster Jobs
```bash
# Check job status
make check-jobs
# or directly: squeue -u $USER

# Cancel jobs if needed
make cancel-jobs
# or directly: scancel -u $USER

# Check logs
tail -f logs/diffusion_*.out
tail -f logs/classifier_*.out
```

## 📊 Results and Outputs

### Training Results
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

# Check project status
make status

# View learning curves and samples
ls experiments/*/samples/
ls experiments/*/results/
```

## ⚡ Cluster-Specific Tips

### 🖥️ Your Workflow
```bash
# 1. Request interactive session
make interactive-gpu

# 2. Once in the session, activate conda
conda activate diffusion-env

# 3. Run training
make train-diffusion

# 4. Monitor with another terminal
squeue -u $USER
watch -n 30 'squeue -u $USER'
```

### 📦 Environment Management on Cluster
```bash
# Export environment for reproducibility
make env-export

# Share environment.yml with colleagues
git add environment.yml
git commit -m "Add conda environment"

# Recreate exact environment later
make env-from-file
```

### 💡 Optimization Tips

**For Your GPU Setup (4 CPUs, 16GB RAM):**
```bash
# Optimal batch sizes
--batch-size 16  # For diffusion
--batch-size 32  # For classifier
--num-workers 4  # Match your CPU count
```

**For Larger Sessions (8 CPUs, 32GB RAM):**
```bash
# Increase throughput
--batch-size 32  # For diffusion
--batch-size 64  # For classifier  
--num-workers 8  # Match your CPU count
```

**For CPU-Only Testing:**
```bash
# Install CPU-only PyTorch
make install-cpu

# Train on CPU (much slower, for testing only)
--device cpu --batch-size 4
```

### 🔧 Common Issues

**Environment Not Found:**
```bash
# Check available environments
conda env list

# Recreate if missing
make env-create
conda activate diffusion-env
make install
```

**CUDA Version Mismatch:**
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**Job Queue Issues:**
```bash
# Check queue status
sinfo -p gpu-interactive

# Check your account
sacctmgr show user $USER

# Check job limits
scontrol show qos
```

**Dataset Not Found:**
```bash
# Check data structure
make check-data

# Fix paths in scripts if needed
--data-dir /path/to/your/acne_dataset
```

### 🎯 Quick Validation

**Test Installation:**
```bash
# Check environment
conda activate diffusion-env
make status

# Quick training test
make train-quick

# Check if models load
python -c "
from src.models.diffusion import DiffusionModel
from src.models.classifier import ClassifierModel
print('✅ Models load successfully')
"
```

**Test GPU:**
```bash
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

## 🚀 Complete Cluster Workflow Example

Here's a complete example using your cluster setup:

### Step 1: Setup (One-time)
```bash
# On login node
git clone <repository-url>
cd acne_diffusion
make env-create
conda activate diffusion-env
make install
make setup-cluster

# Upload your dataset to data/acne_dataset/
```

### Step 2: Interactive Development
```bash
# Request interactive session
make interactive-gpu

# In the GPU session:
conda activate diffusion-env
make check-gpu
make check-data
make train-quick  # Test with 5 epochs
```

### Step 3: Production Training
```bash
# Create SLURM job scripts
./slurm_job_examples.sh

# Edit job scripts to use your account (they'll use sci-lippert)
# Submit longer training jobs
sbatch job_diffusion_single.sh
sbatch job_classifier_single.sh

# Monitor
make check-jobs
tail -f logs/diffusion_*.out
```

### Step 4: Generate and Evaluate
```bash
# After training completes
make show-checkpoints

# Generate samples
make generate CHECKPOINT=experiments/diffusion_*/checkpoints/best_diffusion.pth

# Evaluate classifier
make evaluate CHECKPOINT=experiments/classifier_*/checkpoints/best_classifier.pth
```

## 🛠️ SLURM Job Script Templates

The generated job scripts will automatically use your account. Here's what they'll look like:

```bash
#!/bin/bash
#SBATCH --job-name=acne_diffusion
#SBATCH --account=sci-lippert          # Your account
#SBATCH --partition=gpu                # Adjust for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Training
python scripts/train_diffusion.py \
    --data-dir ./data/acne_dataset \
    --experiment-dir ./experiments/diffusion_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch-size 16 \
    --device cuda
```

## 📈 Expected Performance on Your Setup

### Interactive Sessions (4 CPUs, 16GB, 1 GPU)
- **Diffusion training:** ~6-8 hours (100 epochs)
- **Classifier training:** ~3-4 hours (200 epochs)
- **Quick test:** ~10-15 minutes (5 epochs)
- **Inference:** 30-60 seconds per image

### Batch Jobs (8 CPUs, 32GB, 1 GPU)
- **Diffusion training:** ~4-6 hours (100 epochs)
- **Classifier training:** ~2-3 hours (200 epochs)
- **Batch generation:** ~5-10 minutes for 50 images

## 🎯 Next Steps

### 1. Experiment with Hyperparameters
```bash
# Try different learning rates in interactive session
make interactive-gpu
conda activate diffusion-env

python scripts/train_classifier.py --lr 1e-3 --epochs 20
python scripts/train_classifier.py --lr 1e-5 --epochs 20
```

### 2. Advanced Features
```python
# Use in Python scripts
from src.inference.diffusion_inference import DiffusionInference
from src.inference.classifier_inference import ClassifierInference

# Load models
diffusion = DiffusionInference("checkpoints/best_diffusion.pth")
classifier = ClassifierInference("checkpoints/best_classifier.pth")

# Generate samples
images, process = diffusion.generate(num_samples=10)

# Classify images
result = classifier.predict("image.jpg")
print(f"Predicted severity: {result['predicted_class_name']}")
```

### 3. Batch Processing
```python
# Batch generation for large datasets
from src.inference.diffusion_inference import batch_generate

batch_generate(
    checkpoint_path="best_diffusion.pth",
    output_dir="./large_generation",
    num_batches=50,
    batch_size=10
)

# Batch classification
from src.inference.classifier_inference import batch_predict_from_folder

batch_predict_from_folder(
    checkpoint_path="best_classifier.pth",
    image_folder="./test_images",
    output_csv="classification_results.csv"
)
```

### 4. Environment Sharing
```bash
# Export your working environment
make env-export

# Share with teammates
git add environment.yml
git commit -m "Add working conda environment"

# Others can recreate with:
make env-from-file
```

## 🆘 Getting Help

### Check Everything
```bash
# Comprehensive status check
make status

# Conda environment info
make conda-info

# Check if you're in correct environment
echo $CONDA_DEFAULT_ENV  # Should show: diffusion-env
```

### Debug Issues
```bash
# Check installation
conda list | grep torch
conda list | grep monai

# Test imports
python -c "import torch, monai; print('✅ Core imports OK')"

# Check file permissions
ls -la scripts/
chmod +x scripts/*.py  # Fix if needed
```

### Cluster-Specific Debugging
```bash
# Check account limits
sacctmgr show user $USER

# Check available partitions
sinfo

# Check GPU availability
sinfo -N -o "%N %G %C %m %f"
```

## 🎉 Success Indicators

You'll know everything is working when:

✅ `conda activate diffusion-env` works without errors  
✅ `make check-gpu` shows CUDA available  
✅ `make check-data` finds your dataset  
✅ `make train-quick` completes in interactive session  
✅ `make interactive-gpu` gets you a GPU node  
✅ You see generated samples in `experiments/*/samples/`  
✅ Learning curves show decreasing loss  
✅ SLURM jobs submit successfully with `sbatch`  

## 🔄 Daily Workflow

### Quick Development
```bash
make interactive-gpu
conda activate diffusion-env
# Code, test, experiment...
```

### Production Training
```bash
sbatch job_diffusion_single.sh
make check-jobs
# Wait for completion, check results
```

### Generate Results
```bash
make show-checkpoints
make generate CHECKPOINT=best_model.pth
make evaluate CHECKPOINT=best_classifier.pth
```

Happy training on your cluster! 🚀🖥️# Quick Start Guide

Get up and running with the Acne Diffusion framework in under 10 minutes!

## 🚀 Quick Setup

### 1. Clone and Setup Environment
```bash
git clone <repository-url>
cd acne_diffusion

# Create virtual environment
make env-create

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
make install

# Setup project structure
make setup-cluster
```

### 2. Check GPU Availability
```bash
# Check if CUDA/GPU is available
make check-gpu

# Monitor GPU usage (optional)
make monitor-gpu
```

### 3. Prepare Data
```bash
# Create data directory
mkdir -p data

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

### 4. Start Training
```bash
# Train diffusion model (4-6 hours on single GPU)
make train-diffusion

# Train classifier (2-3 hours on single GPU)
make train-classifier

# Quick test (5 epochs, ~10 minutes)
make train-quick
```

## 📋 Common Commands

### 🎓 Training
```bash
# Custom diffusion training
python scripts/train_diffusion.py \
    --data-dir ./data/acne_dataset \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --experiment-dir ./experiments/my_diffusion_run

# Custom classifier training
python scripts/train_classifier.py \
    --data-dir ./data/acne_dataset \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --experiment-dir ./experiments/my_classifier_run

# Resume training from checkpoint
python scripts/train_diffusion.py \
    --resume ./experiments/diffusion_*/checkpoints/diffusion_checkpoint_epoch_50.pth \
    --epochs 50
```

### 🎨 Generation
```bash
# Find your trained models
make show-checkpoints

# Generate samples using make command
make generate CHECKPOINT=experiments/diffusion_*/checkpoints/best_diffusion.pth

# Or generate directly with custom settings
python scripts/generate_samples.py \
    --checkpoint path/to/best_diffusion.pth \
    --output-dir ./my_samples \
    --num-samples 20 \
    --num-steps 1000 \
    --save-process
```

### 📊 Evaluation
```bash
# Evaluate classifier performance
make evaluate CHECKPOINT=experiments/classifier_*/checkpoints/best_classifier.pth

# Or evaluate with custom settings
python scripts/evaluate_model.py \
    --checkpoint path/to/best_classifier.pth \
    --data-dir ./data/acne_dataset \
    --output-dir ./my_evaluation \
    --batch-size 64
```

### 🖥️ GPU Cluster Usage

#### Generate SLURM Job Scripts
```bash
# Create all job script templates
./slurm_job_examples.sh
```

#### Submit Training Jobs
```bash
# Single GPU diffusion training
sbatch job_diffusion_single.sh

# Single GPU classifier training  
sbatch job_classifier_single.sh

# Multi-GPU training (if available)
sbatch job_diffusion_multi.sh

# Hyperparameter search
sbatch job_hyperparam_search.sh
```

#### Monitor Cluster Jobs
```bash
# Check job status
make check-jobs

# Or use SLURM commands directly
squeue -u $USER
scancel <job_id>  # Cancel job if needed

# Check logs
tail -f logs/diffusion_*.out
tail -f logs/classifier_*.out
```

## 📊 Results and Outputs

### Training Results
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

# Check project status
make status

# View learning curves and samples
ls experiments/*/samples/
ls experiments/*/results/
```

## ⚡ Quick Tips & Troubleshooting

### 💡 Optimization Tips

**For Limited GPU Memory (< 8GB):**
```bash
# Reduce batch size
--batch-size 8

# Enable mixed precision
# (automatically enabled in configs)
```

**For High-End GPUs (> 16GB):**
```bash
# Increase batch size for faster training
--batch-size 64

# Higher image resolution (if desired)
# Modify img_size in configs/
```

**For CPU-Only Training:**
```bash
# Install CPU-only PyTorch
make install-torch-cpu

# Train on CPU (slower)
--device cpu --batch-size 4
```

### 🔧 Common Issues

**CUDA Out of Memory:**
```bash
# Check GPU usage
make monitor-gpu

# Reduce batch size
--batch-size 8

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Dataset Not Found:**
```bash
# Check data structure
make check-data

# Verify path
ls -la ./data/acne_dataset/
```

**Slow Training:**
```bash
# Check data loading
--num-workers 8

# Monitor system resources
htop  # CPU usage
nvidia-smi  # GPU usage
```

**Import Errors:**
```bash
# Reinstall package
pip install -e .

# Check installation
python -c "import src.models.diffusion; print('✅ Installation OK')"
```

### 🎯 Quick Validation

**Test Installation:**
```bash
# Quick training test
make train-quick

# Check if models load
python -c "
from src.models.diffusion import DiffusionModel
from src.models.classifier import ClassifierModel
print('✅ Models load successfully')
"
```

**Test GPU:**
```bash
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
"
```

## 🚀 Next Steps

### 1. Experiment with Hyperparameters
```bash
# Try different learning rates
python scripts/train_classifier.py --lr 1e-3
python scripts/train_classifier.py --lr 1e-5

# Different batch sizes
python scripts/train_diffusion.py --batch-size 32
```

### 2. Advanced Features
```python
# Use in Python scripts
from src.inference.diffusion_inference import DiffusionInference
from src.inference.classifier_inference import ClassifierInference

# Load models
diffusion = DiffusionInference("checkpoints/best_diffusion.pth")
classifier = ClassifierInference("checkpoints/best_classifier.pth")

# Generate samples
images, process = diffusion.generate(num_samples=10)

# Classify images
result = classifier.predict("image.jpg")
print(f"Predicted severity: {result['predicted_class_name']}")
```

### 3. Customize Configurations
```python
# Edit configs for your needs
# configs/diffusion_config.py - Model architecture
# configs/classifier_config.py - Classification settings
# configs/base_config.py - General settings
```

### 4. Extend for Your Dataset
```python
# Modify src/data/dataset.py for custom datasets
# Add new transforms in src/data/transforms.py
# Create custom training loops in src/training/
```

## 📈 Expected Performance

### Diffusion Model
- **Training time:** 4-6 hours (100 epochs, RTX 3080)
- **Memory usage:** 6-8 GB GPU memory
- **Final loss:** ~0.01-0.05  
- **Generation time:** 30-60 seconds per image

### Classifier  
- **Training time:** 2-3 hours (200 epochs, RTX 3080)
- **Memory usage:** 4-6 GB GPU memory
- **Expected accuracy:** 85-95%
- **Inference time:** <1 second per image

## 🆘 Getting Help

1. **Check logs:** `tail -f experiments/*/logs/training.log`
2. **Verify setup:** `make status`
3. **Test components:** `make train-quick` 
4. **Monitor resources:** `make monitor-gpu`
5. **Check documentation:** See README.md for detailed info

## 🎉 Success Indicators

You'll know everything is working when:

✅ `make check-gpu` shows CUDA available  
✅ `make check-data` finds your dataset  
✅ `make train-quick` completes without errors  
✅ You see generated samples in `experiments/*/samples/`  
✅ Learning curves show decreasing loss  

Happy training! 🚀🎨