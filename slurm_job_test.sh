#!/bin/bash
#SBATCH --job-name=acne_test
#SBATCH --account=sci-lippert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=00:30:00                    # 30 minutes for testing
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# =============================================================================
# SLURM TESTING SCRIPT
# Tests all components before long training
# =============================================================================

set -e

echo "=== SLURM Environment Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "=============================="

# Test 1: Environment activation
echo "TEST 1: Environment Activation"
source ~/.bashrc
conda activate diffusion-env
if [ "$CONDA_DEFAULT_ENV" != "diffusion-env" ]; then
    echo "❌ FAILED: Could not activate diffusion-env"
    exit 1
fi
echo "✅ PASSED: Environment activated successfully"
echo ""

# Test 2: Python imports
echo "TEST 2: Python Package Imports"
python -c "
import torch
import monai
import numpy as np
import matplotlib.pyplot as plt
print('✅ PASSED: All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'MONAI version: {monai.__version__}')
"
echo ""

# Test 3: CUDA availability
echo "TEST 3: CUDA/GPU Check"
python -c "
import torch
if not torch.cuda.is_available():
    print('❌ FAILED: CUDA not available')
    exit(1)
print(f'✅ PASSED: CUDA available')
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo ""

# Test 4: GPU memory allocation
echo "TEST 4: GPU Memory Allocation"
python -c "
import torch
try:
    # Test basic GPU operations
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f'✅ PASSED: GPU computation successful')
    print(f'Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')
    torch.cuda.empty_cache()
except Exception as e:
    print(f'❌ FAILED: GPU computation failed: {e}')
    exit(1)
"
echo ""

# Test 5: Data directory
echo "TEST 5: Dataset Verification"
if [ ! -d "./data/acne_dataset" ]; then
    echo "❌ FAILED: Dataset directory not found at ./data/acne_dataset"
    echo "Please create the directory and add your ACNE04 dataset"
    exit 1
fi

# Count images in each severity level
for level in acne0_1024 acne1_1024 acne2_1024 acne3_1024; do
    if [ -d "./data/acne_dataset/$level" ]; then
        count=$(find "./data/acne_dataset/$level" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
        echo "  $level: $count images"
        if [ $count -eq 0 ]; then
            echo "⚠️  WARNING: No images found in $level"
        fi
    else
        echo "❌ FAILED: Directory ./data/acne_dataset/$level not found"
        exit 1
    fi
done
echo "✅ PASSED: Dataset structure verified"
echo ""

# Test 6: Model imports
echo "TEST 6: Model Loading Test"
python -c "
import sys
sys.path.append('./src')
try:
    from models.diffusion import DiffusionModel
    from models.classifier import ClassifierModel
    print('✅ PASSED: Model classes imported successfully')
except ImportError as e:
    print(f'❌ FAILED: Model import failed: {e}')
    exit(1)
"
echo ""

# Test 7: Quick training test (3 epochs)
echo "TEST 7: Quick Training Test (3 epochs)"
mkdir -p logs
mkdir -p experiments/test_run

python scripts/train_diffusion.py \
    --data-dir ./data/acne_dataset \
    --experiment-dir ./experiments/test_run \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda \
    --checkpoint-interval 1 \
    2>&1 | tee logs/quick_test_${SLURM_JOB_ID}.log

if [ $? -eq 0 ]; then
    echo "✅ PASSED: Quick training completed successfully"
else
    echo "❌ FAILED: Quick training failed"
    echo "Check logs/quick_test_${SLURM_JOB_ID}.log for details"
    exit 1
fi
echo ""

# Test 8: Checkpoint loading
echo "TEST 8: Checkpoint Loading Test"
if [ -f "./experiments/test_run/checkpoints/diffusion_checkpoint_epoch_3.pth" ]; then
    python -c "
import torch
checkpoint = torch.load('./experiments/test_run/checkpoints/diffusion_checkpoint_epoch_3.pth', map_location='cuda')
print('✅ PASSED: Checkpoint loaded successfully')
print(f'Checkpoint epoch: {checkpoint.get(\"epoch\", \"unknown\")}')
print(f'Keys in checkpoint: {list(checkpoint.keys())}')
"
else
    echo "❌ FAILED: Checkpoint not found"
    exit 1
fi
echo ""

# Test 9: Sample generation
echo "TEST 9: Sample Generation Test"
python scripts/generate_samples.py \
    --checkpoint ./experiments/test_run/checkpoints/diffusion_checkpoint_epoch_3.pth \
    --output-dir ./experiments/test_run/test_samples \
    --num-samples 2 \
    --num-steps 100 \
    --device cuda

if [ $? -eq 0 ]; then
    echo "✅ PASSED: Sample generation completed"
    echo "Generated samples:"
    ls -la ./experiments/test_run/test_samples/
else
    echo "❌ FAILED: Sample generation failed"
    exit 1
fi
echo ""

# Test 10: Disk space check
echo "TEST 10: Disk Space Check"
available_space=$(df . | tail -1 | awk '{print $4}')
available_gb=$((available_space / 1024 / 1024))
echo "Available disk space: ${available_gb} GB"

if [ $available_gb -lt 10 ]; then
    echo "⚠️  WARNING: Low disk space (< 10 GB). Long training may fail."
    echo "Consider cleaning up or requesting more storage."
else
    echo "✅ PASSED: Sufficient disk space available"
fi
echo ""

# Test 11: Email notification test (if configured)
echo "TEST 11: Email Configuration"
if [ -n "${SBATCH_MAIL_USER:-}" ]; then
    echo "✅ Email notifications configured for: $SBATCH_MAIL_USER"
else
    echo "⚠️  WARNING: No email notifications configured"
    echo "Add --mail-user=your_email@domain.com to get job status updates"
fi
echo ""

# Final summary
echo "================================="
echo "🎉 ALL TESTS COMPLETED SUCCESSFULLY!"
echo "================================="
echo ""
echo "Your system is ready for long duration training!"
echo ""
echo "Next steps:"
echo "1. Update the email in job_diffusion_long.sh"
echo "2. Adjust epochs, batch_size, and time limit as needed"
echo "3. Submit the long training job: sbatch job_diffusion_long.sh"
echo ""
echo "Monitoring commands:"
echo "  squeue -u \$USER                    # Check job status"
echo "  tail -f logs/diffusion_long_*.out   # Watch training progress"
echo "  scancel <job_id>                    # Cancel job if needed"
echo ""
echo "Test completed at: $(date)"

# Cleanup test files (optional)
echo ""
echo "Cleaning up test files..."
rm -rf ./experiments/test_run
echo "Test cleanup completed."