#!/bin/bash
#SBATCH --job-name=acne_diffusion_resume
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/acne_diffusion_resume_%j.out
#SBATCH --error=logs/acne_diffusion_resume_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@utah.edu

# Description: Resume diffusion training from checkpoint

echo "🚀 Starting job: acne_diffusion_resume"
echo "📅 Job started at: $(date)"
echo "🖥️ Running on node: $SLURMD_NODENAME"
echo "💼 Job ID: $SLURM_JOB_ID"
echo "📊 Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "💾 Allocated memory: $SLURM_MEM_PER_NODE MB"

# Load modules (adjust as needed for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate conda environment
echo "🐍 Activating conda environment..."
source $HOME/.bashrc
conda activate diffusion-env

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'diffusion-env'"
    echo "💡 Make sure you've created the environment with: make env-create"
    exit 1
fi

# Verify CUDA availability
echo "🖥️ Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No GPUs available')"

if [ $? -ne 0 ]; then
    echo "❌ Python/PyTorch check failed"
    exit 1
fi

# Check if dataset exists
if [ ! -d "data/acne_dataset" ]; then
    echo "❌ Dataset not found at data/acne_dataset"
    echo "💡 Please make sure your dataset is in the correct location"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

# Create experiment directory with timestamp
EXPERIMENT_DIR="./experiments/acne_diffusion_resume_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

echo "📁 Experiment directory: $EXPERIMENT_DIR"

# Run the training command
echo "🏃 Starting training..."
# Set checkpoint path before submitting
CHECKPOINT_PATH="experiments/diffusion_*/checkpoints/diffusion_checkpoint_epoch_*.pth"
if [ -z "$CHECKPOINT_TO_RESUME" ]; then
    echo "❌ Please set CHECKPOINT_TO_RESUME environment variable"
    echo "💡 Example: export CHECKPOINT_TO_RESUME=experiments/diffusion_20241201/checkpoints/diffusion_checkpoint_epoch_50.pth"
    exit 1
fi

python scripts/train_diffusion.py \
    --resume "$CHECKPOINT_TO_RESUME" \
    --data-dir ./data/acne_dataset \
    --experiment-dir "$EXPERIMENT_DIR" \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --device cuda

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Results saved in: $EXPERIMENT_DIR"
    echo "📊 Check logs at: logs/acne_diffusion_resume_${SLURM_JOB_ID}.out"
else
    echo "❌ Training failed with exit code $?"
    exit 1
fi

echo "🎉 Job completed at: $(date)"
