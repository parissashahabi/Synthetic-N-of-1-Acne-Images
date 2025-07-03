# Simple YAML Configuration System

A clean, simple configuration system that puts all settings in one YAML file.

## 📁 Files Overview

```
config.yaml                     # ← All configuration here
src/utils/config_reader.py      # ← Reads YAML config  
scripts/generate_slurm_jobs.py  # ← Generates SLURM jobs
Makefile                        # ← Uses config.yaml
config.mk                       # ← Auto-generated from YAML
job_*.sh                        # ← Auto-generated SLURM scripts
```

## 🚀 Quick Setup

1. **Install PyYAML**: `pip install PyYAML`
2. **Copy the files** from the artifacts above
3. **Run setup**: `./setup_simple_config.sh`
4. **Test**: `make train-diffusion-quick`

## ⚙️ Configuration

Edit `config.yaml` to change any setting:

```yaml
# Training parameters
diffusion:
  epochs: 100        # ← Change this
  batch_size: 16
  learning_rate: 1e-4

classifier:
  epochs: 200
  batch_size: 32
  learning_rate: 3e-4

# Cluster settings  
cluster:
  account: "sci-lippert"    # ← Change this
  partition: "gpu"

# Wandb (used for all jobs)
wandb:
  project: "acne-diffusion"  # ← Change this
```

## 🎯 Usage

### Local Training
```bash
make train-diffusion-quick      # Quick test (5 epochs)
make train-diffusion            # Full training (100 epochs)
make train-classifier           # Full training (200 epochs)

# Resume training
make resume-diffusion CHECKPOINT=path/to/checkpoint.pth
make resume-classifier CHECKPOINT=path/to/checkpoint.pth

# Generate samples
make generate CHECKPOINT=path/to/diffusion_checkpoint.pth

# Evaluate
make evaluate CHECKPOINT=path/to/classifier_checkpoint.pth
```

### Cluster Usage
```bash
# Generate SLURM scripts (do this once)
make generate-slurm-scripts

# Submit jobs
make submit-test                # Test job (30 min)
make submit-diffusion           # Train diffusion (24h)
make submit-classifier          # Train classifier (24h)

# Resume training
export CHECKPOINT=path/to/checkpoint.pth
sbatch job_resume_diffusion.sh

# Generate samples  
export CHECKPOINT=path/to/diffusion_checkpoint.pth
sbatch job_generate.sh

# Check jobs
make check-jobs
```

## 📋 Available SLURM Jobs

| Script | Purpose | Time | Resources |
|--------|---------|------|-----------|
| `job_test.sh` | Quick test | 30min | 4 CPUs, 16GB |
| `job_diffusion.sh` | Train diffusion | 24h | 8 CPUs, 32GB |
| `job_classifier.sh` | Train classifier | 24h | 8 CPUs, 32GB |
| `job_resume_diffusion.sh` | Resume diffusion | 24h | 8 CPUs, 32GB |
| `job_resume_classifier.sh` | Resume classifier | 24h | 8 CPUs, 32GB |
| `job_generate.sh` | Generate samples | 2h | 4 CPUs, 16GB |
| `job_evaluate.sh` | Evaluate classifier | 2h | 4 CPUs, 16GB |
| `job_hypersearch.sh` | Hyperparameter search | 48h | 8 CPUs, 32GB |

## 🔧 How It Works

1. **`config.yaml`** contains all settings
2. **`config_reader.py`** reads the YAML and creates Makefile variables
3. **`Makefile`** uses those variables for training commands
4. **`generate_slurm_jobs.py`** creates SLURM scripts from the YAML
5. **All jobs use wandb** automatically

## 📊 Hyperparameter Search

The `job_hypersearch.sh` script will test all combinations defined in `config.yaml`:

```yaml
hypersearch:
  diffusion:
    learning_rates: [1e-4, 5e-5, 1e-5]
    batch_sizes: [16, 32]
    epochs: 50
  
  classifier:
    learning_rates: [3e-4, 1e-4, 1e-5] 
    weight_decays: [0.01, 0.05, 0.1]
    epochs: 100
```

This will run 6 diffusion experiments and 9 classifier experiments, all logged to wandb.

## 🎛️ Customization

### Change Training Parameters
Edit `config.yaml`:
```yaml
diffusion:
  epochs: 500           # Instead of 100
  batch_size: 8         # Instead of 16
  learning_rate: 5e-5   # Instead of 1e-4
```

### Change Cluster Settings
Edit `config.yaml`:
```yaml
cluster:
  account: "your-account"
  partition: "your-partition"

resources:
  train:
    time: "48:00:00"    # Instead of 24:00:00
    memory: "64GB"      # Instead of 32GB
```

### Change Wandb Settings
Edit `config.yaml`:
```yaml
wandb:
  project: "my-project"
  entity: "my-team"
  tags: ["experiment", "v2"]
```

## 🛠️ Commands Reference

### Setup Commands
- `make install` - Install all dependencies
- `make env-create` - Create conda environment
- `make verify-install` - Check installation
- `make interactive-gpu` - Get interactive GPU session

### Training Commands  
- `make train-diffusion-quick` - Quick diffusion test
- `make train-diffusion` - Full diffusion training
- `make train-classifier-quick` - Quick classifier test
- `make train-classifier` - Full classifier training

### Generation Commands
- `make generate CHECKPOINT=path` - Generate samples
- `make evaluate CHECKPOINT=path` - Evaluate classifier

### Cluster Commands
- `make generate-slurm-scripts` - Generate job scripts
- `make submit-test` - Submit test job
- `make submit-diffusion` - Submit diffusion training
- `make submit-classifier` - Submit classifier training
- `make check-jobs` - Check job status

### Utility Commands
- `make status` - Show project status
- `make show-config` - Show current configuration
- `make help` - Show all commands

## ✅ Benefits

1. **Simple**: One YAML file for everything
2. **Consistent**: Same config used everywhere  
3. **Easy**: Change YAML, everything updates
4. **Clean**: No complex presets or templates
5. **Standard**: All jobs use wandb automatically
6. **Effective**: Focus on the essential jobs you actually use

That's it! Simple, clean, and effective. 🚀