#!/bin/bash

# =============================================================================
# SLURM Job Examples for GPU Cluster Training with Conda
# Updated for sci-lippert account
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Single GPU Diffusion Training
# -----------------------------------------------------------------------------
cat > job_diffusion_single.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_diffusion
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
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

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Check environment
echo "Starting job at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

# Run training
python scripts/train_diffusion.py \
    --data-dir ./data/acne_dataset \
    --experiment-dir ./experiments/diffusion_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 2. Single GPU Classifier Training
# -----------------------------------------------------------------------------
cat > job_classifier_single.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_classifier
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/classifier_%j.out
#SBATCH --error=logs/classifier_%j.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Check environment
echo "Starting job at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

# Run training
python scripts/train_classifier.py \
    --data-dir ./data/acne_dataset \
    --experiment-dir ./experiments/classifier_$(date +%Y%m%d_%H%M%S) \
    --epochs 200 \
    --batch-size 32 \
    --lr 3e-4 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 3. Multi-GPU Training (DataParallel)
# -----------------------------------------------------------------------------
cat > job_diffusion_multi.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_diffusion_multi
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/diffusion_multi_%j.out
#SBATCH --error=logs/diffusion_multi_%j.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Create log directory
mkdir -p logs

# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16

# Check environment
echo "Starting job at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check GPUs
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Run training with larger batch size for multi-GPU
python scripts/train_diffusion.py \
    --data-dir ./data/acne_dataset \
    --experiment-dir ./experiments/diffusion_multi_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 4. High Memory Training (Large Batch)
# -----------------------------------------------------------------------------
cat > job_diffusion_highmem.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_diffusion_highmem
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu-highmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/diffusion_highmem_%j.out
#SBATCH --error=logs/diffusion_highmem_%j.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Check environment
echo "Starting job at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check resources
nvidia-smi
free -h

# Run training with large batch size
python scripts/train_diffusion.py \
    --data-dir ./data/acne_dataset \
    --experiment-dir ./experiments/diffusion_highmem_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 5. Resume Training Job
# -----------------------------------------------------------------------------
cat > job_resume_training.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_resume
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Check environment
echo "Starting job at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Resume from checkpoint (update path as needed)
CHECKPOINT_PATH="./experiments/diffusion_20241201_120000/checkpoints/best_diffusion.pth"

echo "Resuming from: $CHECKPOINT_PATH"

python scripts/train_diffusion.py \
    --resume $CHECKPOINT_PATH \
    --epochs 50 \
    --batch-size 16 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 6. Inference/Generation Job
# -----------------------------------------------------------------------------
cat > job_generate.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_generate
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Check environment
echo "Starting job at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Generate samples
CHECKPOINT_PATH="./experiments/diffusion_20241201_120000/checkpoints/best_diffusion.pth"
OUTPUT_DIR="./generated_samples_$(date +%Y%m%d_%H%M%S)"

echo "Generating from: $CHECKPOINT_PATH"
echo "Output to: $OUTPUT_DIR"

python scripts/generate_samples.py \
    --checkpoint $CHECKPOINT_PATH \
    --output-dir $OUTPUT_DIR \
    --num-samples 50 \
    --num-steps 1000 \
    --save-process \
    --device cuda

echo "Generation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
EOF

# -----------------------------------------------------------------------------
# 7. Evaluation Job
# -----------------------------------------------------------------------------
cat > job_evaluate.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_evaluate
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Check environment
echo "Starting job at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Evaluate classifier
CHECKPOINT_PATH="./experiments/classifier_20241201_120000/checkpoints/best_classifier.pth"
OUTPUT_DIR="./evaluation_results_$(date +%Y%m%d_%H%M%S)"

echo "Evaluating: $CHECKPOINT_PATH"
echo "Output to: $OUTPUT_DIR"

python scripts/evaluate_model.py \
    --checkpoint $CHECKPOINT_PATH \
    --data-dir ./data/acne_dataset \
    --output-dir $OUTPUT_DIR \
    --batch-size 64 \
    --device cuda

echo "Evaluation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
EOF

# -----------------------------------------------------------------------------
# 8. Array Job for Hyperparameter Search
# -----------------------------------------------------------------------------
cat > job_hyperparam_search.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_hypersearch
#SBATCH --account=sci-lippert
#SBATCH --array=1-12
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/hypersearch_%A_%a.out
#SBATCH --error=logs/hypersearch_%A_%a.err

# Environment setup
source ~/.bashrc
conda activate diffusion-env

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Check environment
echo "Starting array job $SLURM_ARRAY_TASK_ID at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Define hyperparameter combinations
declare -a learning_rates=("1e-4" "3e-4" "1e-3")
declare -a batch_sizes=("16" "32" "64" "128")

# Calculate parameters for this array job
lr_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / 4 ))
bs_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % 4 ))

LR=${learning_rates[$lr_idx]}
BS=${batch_sizes[$bs_idx]}

echo "Array task $SLURM_ARRAY_TASK_ID: LR=$LR, BS=$BS"

# Run training with specific hyperparameters
python scripts/train_classifier.py \
    --data-dir ./data/acne_dataset \
    --experiment-dir ./experiments/hypersearch_lr${LR}_bs${BS}_$(date +%Y%m%d_%H%M%S) \
    --epochs 50 \
    --batch-size $BS \
    --lr $LR \
    --device cuda

echo "Hyperparameter search job $SLURM_ARRAY_TASK_ID completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 9. Interactive Job (Alternative to make interactive-gpu)
# -----------------------------------------------------------------------------
cat > job_interactive.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_interactive
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

# This creates an interactive session
# Usage: sbatch job_interactive.sh
# Then: squeue -u $USER to get job ID
# Then: srun --jobid=<job_id> --pty bash

echo "Interactive GPU session ready"
echo "Use: srun --jobid=$SLURM_JOB_ID --pty bash"
echo "Then: conda activate diffusion-env"

# Keep job alive
sleep 7200  # 2 hours
EOF

# -----------------------------------------------------------------------------
# Usage Instructions
# -----------------------------------------------------------------------------
echo "SLURM Job Scripts Created for sci-lippert account!"
echo ""
echo "📁 Files created:"
echo "  job_diffusion_single.sh     - Single GPU diffusion training"
echo "  job_classifier_single.sh    - Single GPU classifier training"
echo "  job_diffusion_multi.sh      - Multi-GPU training"
echo "  job_diffusion_highmem.sh    - High memory training"
echo "  job_resume_training.sh      - Resume from checkpoint"
echo "  job_generate.sh             - Generate samples"
echo "  job_evaluate.sh             - Evaluate model"
echo "  job_hyperparam_search.sh    - Hyperparameter search (array job)"
echo "  job_interactive.sh          - Interactive session"
echo ""
echo "🚀 Usage:"
echo "1. Single GPU diffusion:       sbatch job_diffusion_single.sh"
echo "2. Single GPU classifier:      sbatch job_classifier_single.sh"
echo "3. Multi-GPU training:         sbatch job_diffusion_multi.sh"
echo "4. High memory training:       sbatch job_diffusion_highmem.sh"
echo "5. Resume training:            sbatch job_resume_training.sh"
echo "6. Generate samples:           sbatch job_generate.sh"
echo "7. Evaluate model:             sbatch job_evaluate.sh"
echo "8. Hyperparameter search:      sbatch job_hyperparam_search.sh"
echo "9. Interactive session:        sbatch job_interactive.sh"
echo ""
echo "📊 Monitor jobs:"
echo "  squeue -u $USER              - Check job status"
echo "  scancel <job_id>             - Cancel job"
echo "  scancel -u $USER             - Cancel all your jobs"
echo "  tail -f logs/*.out           - Watch logs"
echo ""
echo "🔧 Important notes:"
echo "  - All scripts use account: sci-lippert"
echo "  - Conda environment: diffusion-env (must be created first)"
echo "  - Update checkpoint paths in resume/generate/evaluate jobs"
echo "  - Adjust partition names if your cluster uses different ones"
echo "  - Modify resource requirements based on your cluster limits"
echo ""
echo "🐍 Before submitting jobs, ensure:"
echo "  conda activate diffusion-env"
echo "  make install"
echo "  make setup-cluster"
echo ""
echo "💡 Quick start:"
echo "  1. make interactive-gpu                    # Get interactive session"
echo "  2. conda activate diffusion-env           # Activate environment"  
echo "  3. make train-quick                       # Test with 5 epochs"
echo "  4. sbatch job_diffusion_single.sh         # Submit full training"diffusion_multi_%j.err

# Environment setup
module load python/3.9
module load cuda/11.8
source venv/bin/activate

# Create log directory
mkdir -p logs

# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16

# Run training with larger batch size for multi-GPU
python scripts/train_diffusion.py \
    --data-dir $SCRATCH/acne_dataset \
    --experiment-dir $SCRATCH/experiments/diffusion_multi_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 4. High Memory Training (Large Batch)
# -----------------------------------------------------------------------------
cat > job_diffusion_highmem.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_diffusion_highmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-highmem
#SBATCH --output=logs/diffusion_highmem_%j.out
#SBATCH --error=logs/diffusion_highmem_%j.err

# Environment setup
module load python/3.9
module load cuda/11.8
source venv/bin/activate

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Run training with large batch size
python scripts/train_diffusion.py \
    --data-dir $SCRATCH/acne_dataset \
    --experiment-dir $SCRATCH/experiments/diffusion_highmem_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 5. Resume Training Job
# -----------------------------------------------------------------------------
cat > job_resume_training.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

# Environment setup
module load python/3.9
module load cuda/11.8
source venv/bin/activate

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Resume from checkpoint (update path as needed)
CHECKPOINT_PATH="$SCRATCH/experiments/diffusion_20241201_120000/checkpoints/best_diffusion.pth"

python scripts/train_diffusion.py \
    --resume $CHECKPOINT_PATH \
    --epochs 50 \
    --batch-size 16 \
    --device cuda

echo "Job completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# 6. Inference/Generation Job
# -----------------------------------------------------------------------------
cat > job_generate.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

# Environment setup
module load python/3.9
module load cuda/11.8
source venv/bin/activate

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Generate samples
CHECKPOINT_PATH="$SCRATCH/experiments/diffusion_20241201_120000/checkpoints/best_diffusion.pth"
OUTPUT_DIR="$SCRATCH/generated_samples_$(date +%Y%m%d_%H%M%S)"

python scripts/generate_samples.py \
    --checkpoint $CHECKPOINT_PATH \
    --output-dir $OUTPUT_DIR \
    --num-samples 50 \
    --num-steps 1000 \
    --save-process \
    --device cuda

echo "Generation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
EOF

# -----------------------------------------------------------------------------
# 7. Evaluation Job
# -----------------------------------------------------------------------------
cat > job_evaluate.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

# Environment setup
module load python/3.9
module load cuda/11.8
source venv/bin/activate

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Evaluate classifier
CHECKPOINT_PATH="$SCRATCH/experiments/classifier_20241201_120000/checkpoints/best_classifier.pth"
OUTPUT_DIR="$SCRATCH/evaluation_results_$(date +%Y%m%d_%H%M%S)"

python scripts/evaluate_model.py \
    --checkpoint $CHECKPOINT_PATH \
    --data-dir $SCRATCH/acne_dataset \
    --output-dir $OUTPUT_DIR \
    --batch-size 64 \
    --device cuda

echo "Evaluation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
EOF

# -----------------------------------------------------------------------------
# 8. Array Job for Hyperparameter Search
# -----------------------------------------------------------------------------
cat > job_hyperparam_search.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=acne_hypersearch
#SBATCH --array=1-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/hypersearch_%A_%a.out
#SBATCH --error=logs/hypersearch_%A_%a.err

# Environment setup
module load python/3.9
module load cuda/11.8
source venv/bin/activate

# Create log directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Define hyperparameter combinations
declare -a learning_rates=("1e-4" "3e-4" "1e-3")
declare -a batch_sizes=("16" "32" "64" "128")

# Calculate parameters for this array job
lr_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / 4 ))
bs_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % 4 ))

LR=${learning_rates[$lr_idx]}
BS=${batch_sizes[$bs_idx]}

echo "Array task $SLURM_ARRAY_TASK_ID: LR=$LR, BS=$BS"

# Run training with specific hyperparameters
python scripts/train_classifier.py \
    --data-dir $SCRATCH/acne_dataset \
    --experiment-dir $SCRATCH/experiments/hypersearch_lr${LR}_bs${BS}_$(date +%Y%m%d_%H%M%S) \
    --epochs 50 \
    --batch-size $BS \
    --lr $LR \
    --device cuda

echo "Hyperparameter search job $SLURM_ARRAY_TASK_ID completed at $(date)"
EOF

# -----------------------------------------------------------------------------
# Usage Instructions
# -----------------------------------------------------------------------------
echo "SLURM Job Scripts Created!"
echo ""
echo "Usage:"
echo "1. Single GPU diffusion training:  sbatch job_diffusion_single.sh"
echo "2. Single GPU classifier training: sbatch job_classifier_single.sh"
echo "3. Multi-GPU training:             sbatch job_diffusion_multi.sh"
echo "4. High memory training:           sbatch job_diffusion_highmem.sh"
echo "5. Resume training:                sbatch job_resume_training.sh"
echo "6. Generate samples:               sbatch job_generate.sh"
echo "7. Evaluate model:                 sbatch job_evaluate.sh"
echo "8. Hyperparameter search:          sbatch job_hyperparam_search.sh"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Cancel job with: scancel <job_id>"
echo ""
echo "Remember to:"
echo "- Update paths in the scripts as needed"
echo "- Adjust partition names for your cluster"
echo "- Modify resource requirements based on your needs"
echo "- Set up your virtual environment first"