#!/bin/bash
#SBATCH --job-name=config_classifier_sweep
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/config_hypersearch_%j.out
#SBATCH --error=logs/config_hypersearch_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@utah.edu

# Config-driven hyperparameter search for classifier

echo "🚀 Starting config-driven hyperparameter search"
echo "📅 Started at: $(date)"
echo "🖥️ Node: $SLURMD_NODENAME"
echo "💼 Job ID: $SLURM_JOB_ID"

# Check if WANDB_SWEEP_ID is set
if [ -z "$WANDB_SWEEP_ID" ]; then
    echo "❌ WANDB_SWEEP_ID environment variable not set"
    echo "💡 Create a sweep first with:"
    echo "   python scripts/hyperparameter_search.py --create-sweep"
    echo "   Then set: export WANDB_SWEEP_ID=your_sweep_id"
    exit 1
fi

echo "🔍 Using sweep ID: $WANDB_SWEEP_ID"

# Activate environment
source $HOME/.bashrc
conda activate diffusion-env

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'diffusion-env'"
    exit 1
fi

# Verify CUDA
echo "🖥️ Checking CUDA availability..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

# Check dataset
if [ ! -d "data/acne_dataset" ]; then
    echo "❌ Dataset not found at data/acne_dataset"
    exit 1
fi

# Check config file
if [ ! -f "config.yaml" ]; then
    echo "❌ config.yaml not found"
    exit 1
fi

# Test config first
echo "🧪 Testing config.yaml..."
python scripts/hyperparameter_search.py --test-config

if [ $? -ne 0 ]; then
    echo "❌ Config validation failed"
    exit 1
fi

# Set Python path
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

# Check wandb login
echo "🌐 Checking wandb configuration..."
python -c "import wandb; print(f'Wandb logged in: {wandb.api.api_key is not None}')" 2>/dev/null || echo "⚠️ Wandb may not be configured"

# Set the number of runs for this agent
NUM_RUNS=${WANDB_AGENT_COUNT:-24200}

echo "🏃 Starting wandb agent with $NUM_RUNS runs..."
echo "📊 All parameters read from config.yaml"
echo "🌐 Sweep URL: https://wandb.ai/$(python -c 'import wandb; print(wandb.api.default_entity if wandb.api.default_entity else "your-entity")')/acne-classifier-sweep/sweeps/$WANDB_SWEEP_ID"

# Run the config-driven hyperparameter search
python scripts/hyperparameter_search.py \
    --sweep-id "$WANDB_SWEEP_ID" \
    --count $NUM_RUNS \
    --config config.yaml

if [ $? -eq 0 ]; then
    echo "✅ Hyperparameter search completed successfully!"
else
    echo "❌ Hyperparameter search failed"
    exit 1
fi

echo "🎉 Job completed at: $(date)"