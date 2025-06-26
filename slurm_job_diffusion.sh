#!/bin/bash
#SBATCH --job-name=acne_diffusion_long
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=72:00:00                    # 72 hours (3 days)
#SBATCH --output=logs/diffusion_long_%j.out
#SBATCH --error=logs/diffusion_long_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=prisashahbi@gmail.com  # Replace with your email
#SBATCH --requeue                          # Allow job to be requeued if node fails

# =============================================================================
# LONG DURATION DIFFUSION TRAINING SCRIPT
# Designed for 500-2000 epochs over 48-72 hours
# =============================================================================

# Set strict error handling
set -e
set -u
set -o pipefail

# Environment setup
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "========================"

# Load environment
source ~/.bashrc
conda activate diffusion-env

# Verify environment
if [ "$CONDA_DEFAULT_ENV" != "diffusion-env" ]; then
    echo "ERROR: Failed to activate diffusion-env"
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p experiments
mkdir -p checkpoints_backup

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Check system status
echo "=== System Check ==="
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "Free disk space: $(df -h . | tail -1 | awk '{print $4}')"
echo "===================="

# Training configuration
DATA_DIR="./data/acne_dataset"
EXPERIMENT_NAME="diffusion_long_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="./experiments/${EXPERIMENT_NAME}"
EPOCHS=1000                    # Large number of epochs
BATCH_SIZE=16                  # Adjust based on your GPU memory
LEARNING_RATE=1e-4
CHECKPOINT_INTERVAL=50         # Save checkpoint every 50 epochs
SAMPLE_INTERVAL=100            # Generate samples every 100 epochs

# Verify data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Dataset not found at $DATA_DIR"
    echo "Please ensure your ACNE04 dataset is properly placed"
    exit 1
fi

echo "=== Training Configuration ==="
echo "Data directory: $DATA_DIR"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "=============================="

# Function to backup important checkpoints
backup_checkpoint() {
    local checkpoint_path=$1
    local backup_name=$2
    if [ -f "$checkpoint_path" ]; then
        cp "$checkpoint_path" "./checkpoints_backup/${backup_name}_$(date +%Y%m%d_%H%M%S).pth"
        echo "Backed up checkpoint: $backup_name"
    fi
}

# Function to monitor training progress
monitor_progress() {
    if [ -f "$EXPERIMENT_DIR/logs/training.log" ]; then
        echo "=== Latest Training Progress ==="
        tail -n 10 "$EXPERIMENT_DIR/logs/training.log"
        echo "==============================="
    fi
}

# Set up signal handling for graceful shutdown
trap 'echo "Received interrupt signal. Backing up current state..."; backup_checkpoint "$EXPERIMENT_DIR/checkpoints/latest_diffusion.pth" "interrupted"; exit 130' INT TERM

# Start training with automatic checkpoint resumption
echo "=== Starting Long Duration Training ==="
echo "Training will run for up to $EPOCHS epochs"
echo "Checkpoints saved every $CHECKPOINT_INTERVAL epochs"
echo "Samples generated every $SAMPLE_INTERVAL epochs"
echo "========================================"

# Check if there's a checkpoint to resume from (useful for requeued jobs)
RESUME_ARG=""
if [ -f "$EXPERIMENT_DIR/checkpoints/latest_diffusion.pth" ]; then
    echo "Found existing checkpoint, resuming training..."
    RESUME_ARG="--resume $EXPERIMENT_DIR/checkpoints/latest_diffusion.pth"
fi

# Run the training
python scripts/train_diffusion.py \
    --data-dir "$DATA_DIR" \
    --experiment-dir "$EXPERIMENT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --sample-interval $SAMPLE_INTERVAL \
    --device cuda \
    --enable-mixed-precision \
    --gradient-clip-val 1.0 \
    --num-workers 4 \
    --pin-memory \
    --persistent-workers \
    $RESUME_ARG \
    2>&1 | tee -a logs/training_full_${SLURM_JOB_ID}.log

TRAINING_EXIT_CODE=$?

# Post-training actions
echo "=== Training Completed ==="
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Finished at: $(date)"

# Final system check
echo "=== Final System Status ==="
nvidia-smi
df -h .
echo "============================"

# Backup final models
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    backup_checkpoint "$EXPERIMENT_DIR/checkpoints/best_diffusion.pth" "final_best"
    backup_checkpoint "$EXPERIMENT_DIR/checkpoints/final_diffusion.pth" "final_last"
    
    # Generate final sample batch
    echo "Generating final sample batch..."
    python scripts/generate_samples.py \
        --checkpoint "$EXPERIMENT_DIR/checkpoints/best_diffusion.pth" \
        --output-dir "$EXPERIMENT_DIR/final_samples" \
        --num-samples 20 \
        --save-process \
        --device cuda || echo "Sample generation failed, but training was successful"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
    backup_checkpoint "$EXPERIMENT_DIR/checkpoints/latest_diffusion.pth" "failed_latest"
fi

# Create summary report
cat > "$EXPERIMENT_DIR/training_summary.txt" << EOF
=== Training Summary ===
Job ID: $SLURM_JOB_ID
Started: $(head -1 logs/training_full_${SLURM_JOB_ID}.log | grep "Job started" || echo "Unknown")
Finished: $(date)
Exit Code: $TRAINING_EXIT_CODE
Node: $(hostname)
Target Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Learning Rate: $LEARNING_RATE

Final Model Location: $EXPERIMENT_DIR/checkpoints/best_diffusion.pth
Backup Location: ./checkpoints_backup/
Logs: logs/training_full_${SLURM_JOB_ID}.log

GPU Used: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
Peak GPU Memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MB
========================
EOF

echo "Training job completed. Summary saved to: $EXPERIMENT_DIR/training_summary.txt"

exit $TRAINING_EXIT_CODE