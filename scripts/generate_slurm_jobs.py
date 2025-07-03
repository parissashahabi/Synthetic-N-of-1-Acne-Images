#!/usr/bin/env python3
"""
Generate essential SLURM job scripts from config.yaml
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config_reader import ConfigReader


def create_slurm_script(filename, job_name, resource_type, command, description, config):
    """Create a SLURM job script."""
    
    resource = config.get(f'resources.{resource_type}')
    
    script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={config.get('cluster.account')}
#SBATCH --partition={config.get('cluster.partition')}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={resource.get('cpus', 4)}
#SBATCH --gres=gpu:1
#SBATCH --mem={resource.get('memory')}
#SBATCH --time={resource.get('time')}
#SBATCH --output={config.get('project.logs_dir')}/{job_name}_%j.out
#SBATCH --error={config.get('project.logs_dir')}/{job_name}_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@{config.get('cluster.email_domain')}

# {description}

echo "🚀 Starting job: {job_name}"
echo "📅 Started at: $(date)"
echo "🖥️ Node: $SLURMD_NODENAME"
echo "💼 Job ID: $SLURM_JOB_ID"

# Activate environment
source $HOME/.bashrc
conda activate {config.get('project.conda_env')}

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

# Verify CUDA
python -c "import torch; print(f'CUDA: {{torch.cuda.is_available()}}')"

# Check dataset
if [ ! -d "{config.get('project.data_dir')}" ]; then
    echo "❌ Dataset not found"
    exit 1
fi

# Set Python path
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

# Create experiment directory
EXPERIMENT_DIR="{config.get('project.experiments_dir')}/{job_name}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_DIR"

echo "📁 Experiment dir: $EXPERIMENT_DIR"
echo "🏃 Starting training..."

# Run command
{command}

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed"
    exit 1
fi

echo "🎉 Job completed at: $(date)"
'''
    
    with open(filename, 'w') as f:
        f.write(script_content)
    
    os.chmod(filename, 0o755)
    print(f"✅ Created: {filename}")


def main():
    # Load config
    try:
        config = ConfigReader("config.yaml")
    except FileNotFoundError:
        print("❌ config.yaml not found")
        return 1
    
    print("📝 Generating SLURM job scripts...")
    
    # Create logs directory
    os.makedirs(config.get('project.logs_dir'), exist_ok=True)
    
    scripts = []
    
    # 1. Test job
    create_slurm_script(
        "job_test.sh",
        "acne_test",
        "test",
        f'''python scripts/train_diffusion.py \\
    --data-dataset-path {config.get('project.data_dir')} \\
    --train-experiment-dir "$EXPERIMENT_DIR" \\
    --train-n-epochs {config.get('diffusion.quick_epochs')} \\
    --train-batch-size 8''',
        "Quick test run",
        config
    )
    scripts.append("job_test.sh")
    
    # 2. Diffusion training
    create_slurm_script(
        "job_diffusion.sh",
        "acne_diffusion",
        "train",
        f'''python scripts/train_diffusion.py \\
    --data-dataset-path {config.get('project.data_dir')} \\
    --train-experiment-dir "$EXPERIMENT_DIR" \\
    --train-n-epochs {config.get('diffusion.epochs')} \\
    --train-batch-size {config.get('diffusion.batch_size')} \\
    --train-learning-rate {config.get('diffusion.learning_rate')} \\
    --wandb \\
    --wandb-project {config.get('wandb.project')} \\
    --wandb-name diffusion_cluster \\
    --wandb-tags cluster''',
        "Train diffusion model",
        config
    )
    scripts.append("job_diffusion.sh")
    
    # 3. Classifier training
    create_slurm_script(
        "job_classifier.sh",
        "acne_classifier",
        "train",
        f'''python scripts/train_classifier.py \\
    --data-dataset-path {config.get('project.data_dir')} \\
    --train-experiment-dir "$EXPERIMENT_DIR" \\
    --train-n-epochs {config.get('classifier.epochs')} \\
    --train-batch-size {config.get('classifier.batch_size')} \\
    --train-learning-rate {config.get('classifier.learning_rate')} \\
    --wandb \\
    --wandb-project {config.get('wandb.project')} \\
    --wandb-name classifier_cluster \\
    --wandb-tags cluster''',
        "Train classifier model",
        config
    )
    scripts.append("job_classifier.sh")
    
    # 4. Resume diffusion
    create_slurm_script(
        "job_resume_diffusion.sh",
        "acne_resume_diffusion",
        "train",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_resume_diffusion.sh"
    exit 1
fi

python scripts/train_diffusion.py \\
    --resume "$CHECKPOINT" \\
    --data-dataset-path {config.get('project.data_dir')} \\
    --train-experiment-dir "$EXPERIMENT_DIR" \\
    --train-n-epochs {config.get('diffusion.epochs')} \\
    --train-batch-size {config.get('diffusion.batch_size')} \\
    --wandb \\
    --wandb-project {config.get('wandb.project')} \\
    --wandb-name diffusion_resumed \\
    --wandb-tags cluster resumed''',
        "Resume diffusion training",
        config
    )
    scripts.append("job_resume_diffusion.sh")
    
    # 5. Resume classifier
    create_slurm_script(
        "job_resume_classifier.sh",
        "acne_resume_classifier",
        "train",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_resume_classifier.sh"
    exit 1
fi

python scripts/train_classifier.py \\
    --resume "$CHECKPOINT" \\
    --data-dataset-path {config.get('project.data_dir')} \\
    --train-experiment-dir "$EXPERIMENT_DIR" \\
    --train-n-epochs {config.get('classifier.epochs')} \\
    --train-batch-size {config.get('classifier.batch_size')} \\
    --wandb \\
    --wandb-project {config.get('wandb.project')} \\
    --wandb-name classifier_resumed \\
    --wandb-tags cluster resumed''',
        "Resume classifier training",
        config
    )
    scripts.append("job_resume_classifier.sh")
    
    # 6. Generate samples
    create_slurm_script(
        "job_generate.sh",
        "acne_generate",
        "generate",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_generate.sh"
    exit 1
fi

python scripts/generate_samples.py \\
    --checkpoint "$CHECKPOINT" \\
    --output-dir ./generated_$(date +%Y%m%d_%H%M%S) \\
    --train-num-samples {config.get('generation.num_samples')} \\
    --train-num-inference-steps {config.get('generation.num_inference_steps')} \\
    --train-save-intermediates''',
        "Generate samples",
        config
    )
    scripts.append("job_generate.sh")
    
    # 7. Evaluate classifier
    create_slurm_script(
        "job_evaluate.sh",
        "acne_evaluate",
        "generate",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_evaluate.sh"
    exit 1
fi

python scripts/evaluate_model.py \\
    --checkpoint "$CHECKPOINT" \\
    --data-dataset-path {config.get('project.data_dir')} \\
    --output-dir ./evaluation_$(date +%Y%m%d_%H%M%S)''',
        "Evaluate classifier",
        config
    )
    scripts.append("job_evaluate.sh")
    
    # 8. Hyperparameter search
    create_slurm_script(
        "job_hypersearch.sh",
        "acne_hypersearch",
        "hypersearch",
        f'''# Hyperparameter search for both models
echo "🧪 Starting hyperparameter search..."

# Diffusion hyperparameter search
echo "🔥 Diffusion hyperparameter search..."
for lr in {' '.join([str(lr) for lr in config.get('hypersearch.diffusion.learning_rates')])}; do
    for bs in {' '.join([str(bs) for bs in config.get('hypersearch.diffusion.batch_sizes')])}; do
        echo "Testing diffusion LR=$lr, BS=$bs"
        EXP_DIR="{config.get('project.experiments_dir')}/hypersearch_diffusion_lr${{lr}}_bs${{bs}}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$EXP_DIR"
        
        python scripts/train_diffusion.py \\
            --data-dataset-path {config.get('project.data_dir')} \\
            --train-experiment-dir "$EXP_DIR" \\
            --train-n-epochs {config.get('hypersearch.diffusion.epochs')} \\
            --train-batch-size $bs \\
            --train-learning-rate $lr \\
            --wandb \\
            --wandb-project {config.get('wandb.project')} \\
            --wandb-name "diffusion_lr${{lr}}_bs${{bs}}" \\
            --wandb-tags cluster hypersearch diffusion
    done
done

# Classifier hyperparameter search
echo "🎓 Classifier hyperparameter search..."
for lr in {' '.join([str(lr) for lr in config.get('hypersearch.classifier.learning_rates')])}; do
    for wd in {' '.join([str(wd) for wd in config.get('hypersearch.classifier.weight_decays')])}; do
        echo "Testing classifier LR=$lr, WD=$wd"
        EXP_DIR="{config.get('project.experiments_dir')}/hypersearch_classifier_lr${{lr}}_wd${{wd}}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$EXP_DIR"
        
        python scripts/train_classifier.py \\
            --data-dataset-path {config.get('project.data_dir')} \\
            --train-experiment-dir "$EXP_DIR" \\
            --train-n-epochs {config.get('hypersearch.classifier.epochs')} \\
            --train-batch-size {config.get('classifier.batch_size')} \\
            --train-learning-rate $lr \\
            --train-weight-decay $wd \\
            --wandb \\
            --wandb-project {config.get('wandb.project')} \\
            --wandb-name "classifier_lr${{lr}}_wd${{wd}}" \\
            --wandb-tags cluster hypersearch classifier
    done
done''',
        "Hyperparameter search",
        config
    )
    scripts.append("job_hypersearch.sh")
    
    print(f"\n✅ Generated {len(scripts)} SLURM job scripts!")
    print("\n📋 Available scripts:")
    for script in scripts:
        print(f"  {script}")
    
    print(f"\n🧪 Usage:")
    print("  Test setup:        sbatch job_test.sh")
    print("  Train models:      sbatch job_diffusion.sh")
    print("                     sbatch job_classifier.sh") 
    print("  Resume training:   export CHECKPOINT=path && sbatch job_resume_diffusion.sh")
    print("  Generate samples:  export CHECKPOINT=path && sbatch job_generate.sh")
    print("  Evaluate:          export CHECKPOINT=path && sbatch job_evaluate.sh")
    print("  Hypersearch:       sbatch job_hypersearch.sh")
    print("  Check jobs:        squeue -u $USER")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())