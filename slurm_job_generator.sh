#!/bin/bash
# slurm_job_generator.sh - Generate SLURM job scripts for acne diffusion training with new argument format

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Generating SLURM job scripts for acne diffusion training...${NC}"

# Check if we're in the right directory
if [[ ! -f "Makefile" ]] || [[ ! -d "src" ]]; then
    echo -e "${RED}❌ Error: Please run this script from the acne_diffusion project root directory${NC}"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to create a SLURM script
create_slurm_script() {
    local script_name="$1"
    local job_name="$2"
    local time_limit="$3"
    local memory="$4"
    local cpus="$5"
    local command="$6"
    local description="$7"
    
    cat > "${script_name}" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --gres=gpu:1
#SBATCH --mem=${memory}
#SBATCH --time=${time_limit}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=\$USER@utah.edu

# Description: ${description}

echo "🚀 Starting job: ${job_name}"
echo "📅 Job started at: \$(date)"
echo "🖥️ Running on node: \$SLURMD_NODENAME"
echo "💼 Job ID: \$SLURM_JOB_ID"
echo "📊 Allocated CPUs: \$SLURM_CPUS_PER_TASK"
echo "💾 Allocated memory: \$SLURM_MEM_PER_NODE MB"

# Load modules (adjust as needed for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate conda environment
echo "🐍 Activating conda environment..."
source \$HOME/.bashrc
conda activate diffusion-env

if [ \$? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'diffusion-env'"
    echo "💡 Make sure you've created the environment with: make env-create"
    exit 1
fi

# Setup wandb if not already configured (for wandb jobs)
if [[ "\$0" == *"wandb"* ]]; then
    echo "🌐 Checking wandb configuration..."
    if [ ! -f "\$HOME/.netrc" ] && [ -z "\$WANDB_API_KEY" ]; then
        echo "⚠️ wandb not configured. Set WANDB_API_KEY environment variable or run 'wandb login'"
        echo "💡 You can also disable wandb by removing --wandb flag"
    fi
fi

# Verify CUDA availability
echo "🖥️ Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No GPUs available')"

if [ \$? -ne 0 ]; then
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
export PYTHONPATH="\$(pwd):\$(pwd)/src:\$PYTHONPATH"

# Create experiment directory with timestamp
EXPERIMENT_DIR="./experiments/${job_name}_\$(date +%Y%m%d_%H%M%S)"
mkdir -p "\$EXPERIMENT_DIR"

echo "📁 Experiment directory: \$EXPERIMENT_DIR"

# Run the training command
echo "🏃 Starting training..."
${command}

# Check if training completed successfully
if [ \$? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Results saved in: \$EXPERIMENT_DIR"
    echo "📊 Check logs at: logs/${job_name}_\${SLURM_JOB_ID}.out"
else
    echo "❌ Training failed with exit code \$?"
    exit 1
fi

echo "🎉 Job completed at: \$(date)"
EOF

    chmod +x "${script_name}"
    echo -e "${GREEN}✅ Created: ${script_name}${NC}"
}

echo -e "${YELLOW}📝 Creating SLURM job scripts with new argument format...${NC}"

# 1. Diffusion training with wandb
create_slurm_script \
    "job_diffusion_wandb.sh" \
    "acne_diffusion_wandb" \
    "24:00:00" \
    "32GB" \
    "8" \
    "python scripts/train_diffusion.py \\
    --data-dataset-path ./data/acne_dataset \\
    --train-experiment-dir \"\$EXPERIMENT_DIR\" \\
    --train-n-epochs 2000 \\
    --train-batch-size 16 \\
    --train-learning-rate 1e-4 \\
    --device cuda \\
    --wandb \\
    --wandb-name diffusion_cluster_run \\
    --wandb-tags cluster gpu slurm" \
    "Train diffusion model with wandb logging"

# 2. Classifier training with wandb
create_slurm_script \
    "job_classifier_wandb.sh" \
    "acne_classifier_wandb" \
    "12:00:00" \
    "32GB" \
    "8" \
    "python scripts/train_classifier.py \\
    --data-dataset-path ./data/acne_dataset \\
    --train-experiment-dir \"\$EXPERIMENT_DIR\" \\
    --train-n-epochs 2000 \\
    --train-batch-size 32 \\
    --train-learning-rate 3e-4 \\
    --device cuda \\
    --wandb \\
    --wandb-name classifier_cluster_run \\
    --wandb-tags cluster gpu slurm" \
    "Train classifier model with wandb logging"

# 3. Quick test jobs (5 epochs)
create_slurm_script \
    "job_test_diffusion.sh" \
    "acne_test_diffusion" \
    "00:30:00" \
    "16GB" \
    "4" \
    "python scripts/train_diffusion.py \\
    --data-dataset-path ./data/acne_dataset \\
    --train-experiment-dir \"\$EXPERIMENT_DIR\" \\
    --train-n-epochs 5 \\
    --train-batch-size 8 \\
    --train-learning-rate 1e-4 \\
    --device cuda" \
    "Quick diffusion test run to verify setup"

create_slurm_script \
    "job_test_classifier.sh" \
    "acne_test_classifier" \
    "00:30:00" \
    "16GB" \
    "4" \
    "python scripts/train_classifier.py \\
    --data-dataset-path ./data/acne_dataset \\
    --train-experiment-dir \"\$EXPERIMENT_DIR\" \\
    --train-n-epochs 5 \\
    --train-batch-size 8 \\
    --train-learning-rate 1e-4 \\
    --device cuda" \
    "Quick classifier test run to verify setup"

# 4. Resume training jobs
create_slurm_script \
    "job_diffusion_resume.sh" \
    "acne_diffusion_resume" \
    "24:00:00" \
    "32GB" \
    "8" \
    "# Set checkpoint path before submitting
if [ -z \"\$CHECKPOINT_TO_RESUME\" ]; then
    echo \"❌ Please set CHECKPOINT_TO_RESUME environment variable\"
    echo \"💡 Example: export CHECKPOINT_TO_RESUME=experiments/diffusion_20241201/checkpoints/diffusion_checkpoint_epoch_50.pth\"
    exit 1
fi

python scripts/train_diffusion.py \\
    --resume \"\$CHECKPOINT_TO_RESUME\" \\
    --data-dataset-path ./data/acne_dataset \\
    --train-experiment-dir \"\$EXPERIMENT_DIR\" \\
    --train-n-epochs 100 \\
    --train-batch-size 16 \\
    --train-learning-rate 1e-4 \\
    --device cuda" \
    "Resume diffusion training from checkpoint"

create_slurm_script \
    "job_classifier_resume.sh" \
    "acne_classifier_resume" \
    "12:00:00" \
    "32GB" \
    "8" \
    "# Set checkpoint path before submitting
if [ -z \"\$CHECKPOINT_TO_RESUME\" ]; then
    echo \"❌ Please set CHECKPOINT_TO_RESUME environment variable\"
    echo \"💡 Example: export CHECKPOINT_TO_RESUME=experiments/classifier_20241201/checkpoints/classifier_checkpoint_epoch_100.pth\"
    exit 1
fi

python scripts/train_classifier.py \\
    --resume \"\$CHECKPOINT_TO_RESUME\" \\
    --data-dataset-path ./data/acne_dataset \\
    --train-experiment-dir \"\$EXPERIMENT_DIR\" \\
    --train-n-epochs 200 \\
    --train-batch-size 32 \\
    --train-learning-rate 3e-4 \\
    --device cuda" \
    "Resume classifier training from checkpoint"

# 5. Hyperparameter search jobs
create_slurm_script \
    "job_diffusion_hyperparam.sh" \
    "acne_diff_hyperparam" \
    "48:00:00" \
    "32GB" \
    "8" \
    "# Hyperparameter search for diffusion model
LRS=(\"1e-4\" \"5e-5\" \"1e-5\")
BATCH_SIZES=(\"16\" \"32\")
CHANNELS=(\"128\" \"256\" \"512\")

for lr in \"\${LRS[@]}\"; do
    for bs in \"\${BATCH_SIZES[@]}\"; do
        for ch in \"\${CHANNELS[@]}\"; do
            echo \"🧪 Testing LR=\$lr, BS=\$bs, CH=\$ch\"
            HYPER_EXP_DIR=\"\${EXPERIMENT_DIR}_lr\${lr}_bs\${bs}_ch\${ch}\"
            mkdir -p \"\$HYPER_EXP_DIR\"
            
            python scripts/train_diffusion.py \\
                --data-dataset-path ./data/acne_dataset \\
                --train-experiment-dir \"\$HYPER_EXP_DIR\" \\
                --train-n-epochs 50 \\
                --train-batch-size \"\$bs\" \\
                --train-learning-rate \"\$lr\" \\
                --model-base-channels \"\$ch\" \\
                --device cuda \\
                --wandb \\
                --wandb-name \"diffusion_lr\${lr}_bs\${bs}_ch\${ch}\" \\
                --wandb-tags hyperparam_search cluster
            
            if [ \$? -ne 0 ]; then
                echo \"❌ Failed for LR=\$lr, BS=\$bs, CH=\$ch\"
                continue
            fi
        done
    done
done" \
    "Comprehensive diffusion hyperparameter search"

create_slurm_script \
    "job_classifier_hyperparam.sh" \
    "acne_class_hyperparam" \
    "24:00:00" \
    "32GB" \
    "8" \
    "# Hyperparameter search for classifier model
LRS=(\"3e-4\" \"1e-4\" \"1e-5\")
WDS=(\"0.01\" \"0.05\" \"0.1\")
CHANNELS=(\"128\" \"256\")

for lr in \"\${LRS[@]}\"; do
    for wd in \"\${WDS[@]}\"; do
        for ch in \"\${CHANNELS[@]}\"; do
            echo \"🧪 Testing LR=\$lr, WD=\$wd, CH=\$ch\"
            HYPER_EXP_DIR=\"\${EXPERIMENT_DIR}_lr\${lr}_wd\${wd}_ch\${ch}\"
            mkdir -p \"\$HYPER_EXP_DIR\"
            
            python scripts/train_classifier.py \\
                --data-dataset-path ./data/acne_dataset \\
                --train-experiment-dir \"\$HYPER_EXP_DIR\" \\
                --train-n-epochs 100 \\
                --train-batch-size 32 \\
                --train-learning-rate \"\$lr\" \\
                --train-weight-decay \"\$wd\" \\
                --model-base-channels \"\$ch\" \\
                --device cuda \\
                --wandb \\
                --wandb-name \"classifier_lr\${lr}_wd\${wd}_ch\${ch}\" \\
                --wandb-tags hyperparam_search cluster
            
            if [ \$? -ne 0 ]; then
                echo \"❌ Failed for LR=\$lr, WD=\$wd, CH=\$ch\"
                continue
            fi
        done
    done
done" \
    "Comprehensive classifier hyperparameter search"

# 7. Generation and evaluation jobs
create_slurm_script \
    "job_generate_samples.sh" \
    "acne_generate" \
    "02:00:00" \
    "16GB" \
    "4" \
    "# Set checkpoint path before submitting
if [ -z \"\$CHECKPOINT_TO_USE\" ]; then
    echo \"❌ Please set CHECKPOINT_TO_USE environment variable\"
    echo \"💡 Example: export CHECKPOINT_TO_USE=experiments/diffusion_*/checkpoints/best_diffusion.pth\"
    exit 1
fi

python scripts/generate_samples.py \\
    --checkpoint \"\$CHECKPOINT_TO_USE\" \\
    --output-dir ./generated_samples_\$(date +%Y%m%d_%H%M%S) \\
    --train-num-samples 50 \\
    --train-num-inference-steps 1000 \\
    --train-save-intermediates \\
    --device cuda" \
    "Generate samples from trained diffusion model"

create_slurm_script \
    "job_evaluate_classifier.sh" \
    "acne_evaluate" \
    "01:00:00" \
    "16GB" \
    "4" \
    "# Set checkpoint path before submitting
if [ -z \"\$CHECKPOINT_TO_EVALUATE\" ]; then
    echo \"❌ Please set CHECKPOINT_TO_EVALUATE environment variable\"
    echo \"💡 Example: export CHECKPOINT_TO_EVALUATE=experiments/classifier_*/checkpoints/best_classifier.pth\"
    exit 1
fi

python scripts/evaluate_model.py \\
    --checkpoint \"\$CHECKPOINT_TO_EVALUATE\" \\
    --data-dataset-path ./data/acne_dataset \\
    --output-dir ./evaluation_results_\$(date +%Y%m%d_%H%M%S) \\
    --batch-size 64 \\
    --device cuda" \
    "Evaluate trained classifier model"

echo ""
echo -e "${GREEN}✅ All SLURM job scripts created successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Available job scripts:${NC}"
echo -e "${CYAN}Basic Training:${NC}"
echo "  job_test_diffusion.sh        - Quick diffusion test (5 epochs, 30 min)"
echo "  job_test_classifier.sh       - Quick classifier test (5 epochs, 30 min)"
echo "  job_diffusion.sh             - Standard diffusion training"
echo "  job_diffusion_wandb.sh       - Diffusion training with wandb"
echo "  job_diffusion_hq.sh          - High-quality diffusion training"
echo "  job_classifier.sh            - Standard classifier training"
echo "  job_classifier_wandb.sh      - Classifier training with wandb"
echo "  job_classifier_robust.sh     - Robust classifier with high noise"
echo ""
echo -e "${CYAN}Advanced Training:${NC}"
echo "  job_diffusion_long.sh        - Long-term diffusion training (72h)"
echo "  job_classifier_long.sh       - Long-term classifier training (48h)"
echo "  job_diffusion_multi_gpu.sh   - Multi-GPU diffusion training"
echo ""
echo -e "${CYAN}Resume Training:${NC}"
echo "  job_diffusion_resume.sh      - Resume diffusion from checkpoint"
echo "  job_classifier_resume.sh     - Resume classifier from checkpoint"
echo ""
echo -e "${CYAN}Hyperparameter Search:${NC}"
echo "  job_diffusion_hyperparam.sh  - Comprehensive diffusion hyperparam search"
echo "  job_classifier_hyperparam.sh - Comprehensive classifier hyperparam search"
echo ""
echo -e "${CYAN}Ablation Studies:${NC}"
echo "  job_ablation_attention.sh    - Effect of attention layers"
echo "  job_ablation_channels.sh     - Effect of model size"
echo ""
echo -e "${CYAN}Ensemble Training:${NC}"
echo "  job_ensemble_diffusion.sh    - Train diffusion model ensemble"
echo "  job_ensemble_classifier.sh   - Train classifier model ensemble"
echo ""
echo -e "${CYAN}Generation & Evaluation:${NC}"
echo "  job_generate_samples.sh      - Generate samples from trained model"
echo "  job_generate_hq.sh           - Generate high-quality samples"
echo "  job_evaluate_classifier.sh   - Evaluate classifier performance"
echo ""
echo -e "${YELLOW}🧪 To test your setup:${NC}"
echo "  1. sbatch job_test_diffusion.sh"
echo "  2. Check logs: tail -f logs/acne_test_diffusion_*.out"
echo ""
echo -e "${YELLOW}🚀 To start basic training:${NC}"
echo "  sbatch job_diffusion_wandb.sh"
echo "  sbatch job_classifier_wandb.sh"
echo ""
echo -e "${YELLOW}🔄 To resume training:${NC}"
echo "  export CHECKPOINT_TO_RESUME=path/to/checkpoint.pth"
echo "  sbatch job_diffusion_resume.sh"
echo ""
echo -e "${YELLOW}🎨 To generate samples:${NC}"
echo "  export CHECKPOINT_TO_USE=path/to/diffusion/checkpoint.pth"
echo "  sbatch job_generate_samples.sh"
echo ""
echo -e "${YELLOW}📊 To monitor jobs:${NC}"
echo "  squeue -u \$USER"
echo "  tail -f logs/acne_*_*.out"
echo "  make check-jobs  # using Makefile"
echo ""
echo -e "${BLUE}💡 Pro tips:${NC}"
echo "  • Always test with job_test_diffusion.sh first"
echo "  • Check your conda environment: conda activate diffusion-env"
echo "  • Verify dataset location: ls data/acne_dataset/"
echo "  • Setup wandb: wandb login (for wandb jobs)"
echo "  • Monitor GPU usage: make monitor-gpu (in interactive session)"
echo "  • Use make commands for local development and testing"
echo ""
echo -e "${PURPLE}🔬 Advanced Usage:${NC}"
echo "  • Hyperparameter search: sbatch job_diffusion_hyperparam.sh"
echo "  • Ablation studies: sbatch job_ablation_attention.sh"
echo "  • Ensemble training: sbatch job_ensemble_diffusion.sh"
echo "  • Long-term training: sbatch job_diffusion_long.sh"
echo ""