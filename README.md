# Acne Diffusion Model

A comprehensive deep learning framework for acne image generation and classification using diffusion models and the ACNE04 dataset.

## Features

- **Diffusion Model**: High-quality acne image generation using DDPM
- **Classifier**: Acne severity level classification (0-3 levels)
- **Flexible Architecture**: Works with various image sizes
- **GPU Cluster Ready**: Optimized for distributed training
- **Easy Configuration**: Modular config system
- **Comprehensive Logging**: Training monitoring and visualization

## Project Structure

```
acne_diffusion/
├── configs/                 # Configuration files
│   ├── base_config.py      # Base configuration
│   ├── diffusion_config.py # Diffusion model config
│   └── classifier_config.py# Classifier config
├── src/
│   ├── data/               # Data handling
│   │   ├── dataset.py      # ACNE04 dataset class
│   │   └── transforms.py   # Data transforms
│   ├── models/             # Model implementations
│   │   ├── diffusion.py    # Diffusion model
│   │   └── classifier.py   # Classifier model
│   ├── training/           # Training logic
│   │   ├── diffusion_trainer.py
│   │   └── classifier_trainer.py
│   ├── inference/          # Inference utilities
│   └── utils/              # Utilities
│       ├── checkpoints.py  # Checkpoint management
│       ├── visualization.py# Plotting and visualization
│       └── logging.py      # Logging utilities
├── scripts/                # Training scripts
│   ├── train_diffusion.py
│   ├── train_classifier.py
│   ├── generate_samples.py
│   └── evaluate_model.py
└── experiments/            # Experiment outputs
    ├── logs/
    ├── checkpoints/
    └── results/
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd acne_diffusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### 1. Training Diffusion Model

```bash
# Basic training
python scripts/train_diffusion.py \
    --data-dir /path/to/acne_dataset \
    --experiment-dir ./experiments/diffusion_run1 \
    --epochs 100 \
    --batch-size 16

# Resume from checkpoint
python scripts/train_diffusion.py \
    --resume ./experiments/diffusion_run1/checkpoints/best_diffusion.pth \
    --epochs 50
```

### 2. Training Classifier

```bash
# Train classifier
python scripts/train_classifier.py \
    --data-dir /path/to/acne_dataset \
    --experiment-dir ./experiments/classifier_run1 \
    --epochs 200 \
    --batch-size 32 \
    --lr 3e-4
```

### 3. Generate Samples

```bash
# Generate images
python scripts/generate_samples.py \
    --checkpoint ./experiments/diffusion_run1/checkpoints/best_diffusion.pth \
    --output-dir ./generated_samples \
    --num-samples 10 \
    --save-process
```

## Configuration

### Diffusion Model Configuration

```python
from configs.diffusion_config import DiffusionModelConfig, DiffusionTrainingConfig

# Model architecture
model_config = DiffusionModelConfig(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_channels=(128, 128, 256, 256, 512, 512),
    attention_levels=(False, False, False, False, True, False),
    num_res_blocks=2,
    num_head_channels=64
)

# Training parameters
training_config = DiffusionTrainingConfig(
    learning_rate=1e-4,
    n_epochs=100,
    batch_size=16,
    num_train_timesteps=1000,
    img_size=128
)
```

### Classifier Configuration

```python
from configs.classifier_config import ClassifierModelConfig, ClassifierTrainingConfig

# Model architecture
model_config = ClassifierModelConfig(
    spatial_dims=2,
    in_channels=3,
    out_channels=4,  # 4 severity levels
    channels=(128, 128, 256, 256, 512),
    attention_levels=(False, False, False, False, True)
)

# Training parameters
training_config = ClassifierTrainingConfig(
    learning_rate=3e-4,
    weight_decay=0.05,
    n_epochs=200,
    batch_size=32
)
```

## GPU Cluster Usage

### SLURM Job Script Example

```bash
#!/bin/bash
#SBATCH --job-name=acne_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err

# Load modules
module load python/3.9
module load cuda/11.8

# Activate environment
source venv/bin/activate

# Run training
python scripts/train_diffusion.py \
    --data-dir $SCRATCH/acne_dataset \
    --experiment-dir $SCRATCH/experiments/diffusion_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch-size 16 \
    --device cuda
```

### Multi-GPU Training

```bash
# Using DataParallel
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/train_diffusion.py \
    --batch-size 64 \
    --device cuda
```

## Dataset

The project uses the ACNE04 dataset with the following structure:
```
acne_dataset/
├── acne0_1024/  # Severity level 0 (clear skin)
├── acne1_1024/  # Severity level 1 (mild)
├── acne2_1024/  # Severity level 2 (moderate)
└── acne3_1024/  # Severity level 3 (severe)
```

## Model Architecture

### Diffusion Model
- **Base**: U-Net with attention mechanisms
- **Scheduler**: DDPM with 1000 timesteps
- **Resolution**: 128x128 RGB images
- **Training**: MSE loss with mixed precision

### Classifier
- **Base**: Fixed DiffusionModelEncoder
- **Features**: Global Average Pooling for flexible input sizes
- **Output**: 4 classes (severity levels 0-3)
- **Training**: Cross-entropy loss with noise augmentation

## Key Features

### 1. Flexible Input Sizes
The classifier uses Global Average Pooling, allowing it to work with any input image size without hardcoded dimensions.

### 2. Noise Augmentation
The classifier is trained with noise augmentation (following the original paper's approach) to improve robustness.

### 3. Comprehensive Checkpointing
- Automatic checkpoint saving and loading
- Best model preservation
- Resume training capability
- Model state dict export for inference

### 4. Visualization
- Training progress plots
- Sample generation during training
- Denoising process visualization
- Dataset statistics and sample batches

## Monitoring and Logging

### TensorBoard (Optional)
```python
# Add to training config
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./experiments/logs')
# Log metrics during training
```

### Weights & Biases (Optional)
```python
import wandb

wandb.init(project="acne-diffusion")
# Log metrics and artifacts
```

## Advanced Usage

### Custom Data Pipeline
```python
from src.data.dataset import AcneDataset
from src.data.transforms import create_transforms

# Custom severity levels
dataset = AcneDataset(
    data_dir="./data",
    severity_levels=[0, 1, 2],  # Only mild to moderate
    transform=transforms
)
```

### Custom Model Configuration
```python
# Custom diffusion model
config = DiffusionModelConfig(
    num_channels=(64, 128, 256, 512),
    attention_levels=(False, True, True, True),
    num_res_blocks=3
)
```

### Inference API
```python
from src.models.diffusion import DiffusionModel
from src.utils.checkpoints import CheckpointManager

# Load trained model
model = DiffusionModel(model_config, training_config)
checkpoint_manager = CheckpointManager("./checkpoints", "diffusion")
checkpoint_manager.load_checkpoint("best_diffusion.pth", model.model)

# Generate samples
samples = model.generate(num_samples=5, num_steps=1000)
```

## Performance Optimization

### Memory Optimization
- Mixed precision training (AMP)
- Gradient checkpointing
- Optimized data loading with multiple workers
- Pin memory for GPU transfers

### Speed Optimization
- Compiled models (PyTorch 2.0+)
- Efficient attention mechanisms
- Optimized batch sizes for your hardware

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model

2. **Slow Training**
   - Increase number of workers
   - Use pin_memory=True
   - Check data loading bottlenecks

3. **Poor Generation Quality**
   - Train for more epochs
   - Adjust learning rate
   - Check data preprocessing

### Debug Mode
```bash
python scripts/train_diffusion.py --debug
# Enables detailed logging and smaller batches
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@misc{acne-diffusion,
  title={Acne Diffusion Model: Deep Learning Framework for Acne Image Generation and Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/acne-diffusion}
}
```

## Acknowledgments

- MONAI team for the diffusion model implementation
- ACNE04 dataset creators
- PyTorch and CUDA teams for the deep learning framework