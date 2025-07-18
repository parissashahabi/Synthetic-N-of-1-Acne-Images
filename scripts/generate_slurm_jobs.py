#!/usr/bin/env python3
"""
Generate essential SLURM job scripts from config.yaml - scripts read config.yaml directly
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
    
    print("📝 Generating SLURM job scripts that read from config.yaml...")
    
    # Create logs directory
    os.makedirs(config.get('project.logs_dir'), exist_ok=True)
    
    scripts = []
    
    # 1. Test job
    create_slurm_script(
        "job_test.sh",
        "acne_test",
        "test",
        f'''python scripts/train_diffusion.py --config config.yaml --quick-test''',
        "Quick test run",
        config
    )
    scripts.append("job_test.sh")
    
    # 2. Diffusion training
    create_slurm_script(
        "job_diffusion.sh",
        "acne_diffusion",
        "train",
        f'''python scripts/train_diffusion.py --config config.yaml --enable-wandb --wandb-name diffusion_cluster''',
        "Train diffusion model",
        config
    )
    scripts.append("job_diffusion.sh")
    
    # 3. Classifier training
    create_slurm_script(
        "job_classifier.sh",
        "acne_classifier",
        "train",
        f'''python scripts/train_classifier.py --config config.yaml --enable-wandb --wandb-name classifier_cluster''',
        "Train classifier model",
        config
    )
    scripts.append("job_classifier.sh")
    
    # 4. Diffusion quick test
    create_slurm_script(
        "job_diffusion_quick.sh",
        "acne_diffusion_quick",
        "test",
        f'''python scripts/train_diffusion.py --config config.yaml --quick-test''',
        "Quick diffusion test",
        config
    )
    scripts.append("job_diffusion_quick.sh")
    
    # 5. Classifier quick test
    create_slurm_script(
        "job_classifier_quick.sh",
        "acne_classifier_quick",
        "test",
        f'''python scripts/train_classifier.py --config config.yaml --quick-test''',
        "Quick classifier test",
        config
    )
    scripts.append("job_classifier_quick.sh")
    
    # 6. Resume diffusion
    create_slurm_script(
        "job_resume_diffusion.sh",
        "acne_resume_diffusion",
        "train",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_resume_diffusion.sh"
    exit 1
fi

python scripts/train_diffusion.py --config config.yaml --resume "$CHECKPOINT" --enable-wandb --wandb-name diffusion_resumed''',
        "Resume diffusion training",
        config
    )
    scripts.append("job_resume_diffusion.sh")
    
    # 7. Resume classifier
    create_slurm_script(
        "job_resume_classifier.sh",
        "acne_resume_classifier",
        "train",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_resume_classifier.sh"
    exit 1
fi

python scripts/train_classifier.py --config config.yaml --resume "$CHECKPOINT" --enable-wandb --wandb-name classifier_resumed''',
        "Resume classifier training",
        config
    )
    scripts.append("job_resume_classifier.sh")
    
    # 8. Generate samples
    create_slurm_script(
        "job_generate.sh",
        "acne_generate",
        "generate",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_generate.sh"
    exit 1
fi

python scripts/generate_samples.py --config config.yaml --checkpoint "$CHECKPOINT" --output-dir ./generated_$(date +%Y%m%d_%H%M%S)''',
        "Generate samples",
        config
    )
    scripts.append("job_generate.sh")
    
    # 9. Evaluate classifier
    create_slurm_script(
        "job_evaluate.sh",
        "acne_evaluate",
        "generate",
        f'''if [ -z "$CHECKPOINT" ]; then
    echo "❌ Set CHECKPOINT environment variable"
    echo "Usage: export CHECKPOINT=path/to/checkpoint.pth && sbatch job_evaluate.sh"
    exit 1
fi

python scripts/evaluate_model.py --config config.yaml --checkpoint "$CHECKPOINT" --output-dir ./evaluation_$(date +%Y%m%d_%H%M%S)''',
        "Evaluate classifier",
        config
    )
    scripts.append("job_evaluate.sh")
    
    # 10. Hyperparameter search
    create_slurm_script(
        "job_hypersearch.sh",
        "acne_hypersearch",
        "hypersearch",
        f'''# Hyperparameter search for both models
echo "🧪 Starting hyperparameter search..."

# Note: This script will modify config.yaml temporarily for each hyperparameter combination
# Create backup of original config
cp config.yaml config_backup.yaml

# Diffusion hyperparameter search
echo "🔥 Diffusion hyperparameter search..."
for lr in {' '.join([str(lr) for lr in config.get('hypersearch.diffusion.learning_rates')])}; do
    for bs in {' '.join([str(bs) for bs in config.get('hypersearch.diffusion.batch_sizes')])}; do
        for ch in {' '.join([str(ch) for ch in config.get('hypersearch.diffusion.base_channels', [256])])}; do
            echo "Testing diffusion LR=$lr, BS=$bs, CH=$ch"
            
            # Create temporary config with modified parameters
            python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['diffusion']['training']['learning_rate'] = $lr
config['diffusion']['training']['batch_size'] = $bs
config['diffusion']['training']['n_epochs'] = {config.get('hypersearch.diffusion.epochs', 50)}
config['diffusion']['model']['base_channels'] = $ch
with open('config_temp.yaml', 'w') as f:
    yaml.dump(config, f)
"
            
            python scripts/train_diffusion.py --config config_temp.yaml --enable-wandb --wandb-name "diffusion_lr${{lr}}_bs${{bs}}_ch${{ch}}"
            
            if [ $? -ne 0 ]; then
                echo "❌ Failed for LR=$lr, BS=$bs, CH=$ch"
                continue
            fi
        done
    done
done

# Classifier hyperparameter search
echo "🎓 Classifier hyperparameter search..."
for lr in {' '.join([str(lr) for lr in config.get('hypersearch.classifier.learning_rates')])}; do
    for wd in {' '.join([str(wd) for wd in config.get('hypersearch.classifier.weight_decays')])}; do
        for ch in {' '.join([str(ch) for ch in config.get('hypersearch.classifier.base_channels', [128])])}; do
            echo "Testing classifier LR=$lr, WD=$wd, CH=$ch"
            
            # Create temporary config with modified parameters
            python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['classifier']['training']['learning_rate'] = $lr
config['classifier']['training']['weight_decay'] = $wd
config['classifier']['training']['n_epochs'] = {config.get('hypersearch.classifier.epochs', 100)}
config['classifier']['model']['base_channels'] = $ch
with open('config_temp.yaml', 'w') as f:
    yaml.dump(config, f)
"
            
            python scripts/train_classifier.py --config config_temp.yaml --enable-wandb --wandb-name "classifier_lr${{lr}}_wd${{wd}}_ch${{ch}}"
            
            if [ $? -ne 0 ]; then
                echo "❌ Failed for LR=$lr, WD=$wd, CH=$ch"
                continue
            fi
        done
    done
done

# Restore original config
mv config_backup.yaml config.yaml
rm -f config_temp.yaml

echo "✅ Hyperparameter search completed!"''',
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
    print("  Quick tests:       sbatch job_diffusion_quick.sh")
    print("                     sbatch job_classifier_quick.sh")
    print("  Train models:      sbatch job_diffusion.sh")
    print("                     sbatch job_classifier.sh") 
    print("  Resume training:   export CHECKPOINT=path && sbatch job_resume_diffusion.sh")
    print("  Generate samples:  export CHECKPOINT=path && sbatch job_generate.sh")
    print("  Evaluate:          export CHECKPOINT=path && sbatch job_evaluate.sh")
    print("  Hypersearch:       sbatch job_hypersearch.sh")
    print("  Check jobs:        squeue -u $USER")
    
    print(f"\n📝 Note: All scripts now read configuration from config.yaml")
    print("         To change settings, edit config.yaml directly")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())