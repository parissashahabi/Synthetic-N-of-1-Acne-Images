# Makefile for Acne Diffusion Project

.PHONY: help install install-dev test clean format lint setup-cluster env-create env-activate

# Default target
help:
	@echo "Available targets:"
	@echo "  install           Install package and dependencies"
	@echo "  install-dev       Install package with development dependencies"
	@echo "  env-create        Create conda environment"
	@echo "  env-activate      Show command to activate environment"
	@echo "  env-export        Export environment to YAML"
	@echo "  env-from-file     Create environment from YAML file"
	@echo "  test              Run tests"
	@echo "  clean             Clean build artifacts"
	@echo "  format            Format code with black and isort"
	@echo "  lint              Run code linting"
	@echo "  setup-cluster     Setup for cluster usage"
	@echo "  train-diffusion   Train diffusion model"
	@echo "  train-classifier  Train classifier model"
	@echo "  generate          Generate samples"
	@echo "  evaluate          Evaluate model"
	@echo "  check-gpu         Check GPU availability"
	@echo "  monitor-gpu       Monitor GPU usage"
	@echo "  interactive-gpu   Request interactive GPU session"
	@echo "  setup             Complete project setup"

# Environment setup with conda
env-create:
	@echo "🐍 Creating conda environment: diffusion-env"
	conda create -n diffusion-env python=3.9 -y
	@echo "✅ Environment created! Activate with:"
	@echo "  conda activate diffusion-env"

env-activate:
	@echo "To activate the conda environment:"
	@echo "  conda activate diffusion-env"

env-export:
	@echo "📦 Exporting conda environment..."
	conda env export -n diffusion-env > environment.yml
	@echo "✅ Environment exported to environment.yml"

env-from-file:
	@echo "📥 Creating environment from environment.yml..."
	conda env create -f environment.yml
	@echo "✅ Environment created from file!"

env-remove:
	@echo "🗑️ Removing conda environment..."
	conda env remove -n diffusion-env -y
	@echo "✅ Environment removed!"

# Installation
install:
	@echo "📦 Installing dependencies..."
	conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
	pip install monai matplotlib tqdm
	pip install "monai-weekly[pillow, tqdm]"
	pip install monai-generative
	pip install "diffusers[torch]"
	pip install scikit-learn tensorboard wandb
	pip install hydra-core omegaconf einops
	pip install -e .
	@echo "✅ Installation complete!"

install-dev:
	make install
	pip install pytest black flake8 isort mypy
	@echo "✅ Development environment ready!"

install-cpu:
	@echo "📦 Installing CPU-only PyTorch..."
	conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
	pip install monai matplotlib tqdm
	pip install "monai-weekly[pillow, tqdm]"
	pip install monai-generative
	pip install "diffusers[torch]"
	pip install scikit-learn tensorboard wandb
	pip install hydra-core omegaconf
	pip install -e .

# Testing
test:
	pytest tests/ -v

# Code quality
format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/
	mypy src/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/
	rm -rf experiments/*/logs/*.out experiments/*/logs/*.err
	rm -rf venv  # Remove any old venv directories

# GPU utilities
check-gpu:
	@echo "🖥️ Checking GPU availability..."
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

monitor-gpu:
	@echo "📊 Monitoring GPU usage (press Ctrl+C to stop)..."
	@if command -v nvidia-smi > /dev/null; then \
		watch -n 1 nvidia-smi; \
	elif command -v gpustat > /dev/null; then \
		pip install gpustat && watch -n 1 gpustat; \
	else \
		echo "nvidia-smi not available"; \
	fi

# Interactive GPU session for cluster
interactive-gpu:
	@echo "🖥️ Requesting interactive GPU session..."
	srun --partition=gpu-interactive --account=sci-lippert --cpus-per-task=4 --gpus=1 --mem=16G --time=02:00:00 --pty bash

interactive-gpu-large:
	@echo "🖥️ Requesting large interactive GPU session..."
	srun --partition=gpu-interactive --account=sci-lippert --cpus-per-task=8 --gpus=1 --mem=32G --time=04:00:00 --pty bash

# Cluster setup
setup-cluster:
	mkdir -p logs experiments
	chmod +x scripts/*.py
	chmod +x slurm_job_examples.sh
	@echo "📁 Created directory structure for cluster usage"

# Training shortcuts
train-diffusion:
	@echo "🚀 Starting diffusion model training..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_diffusion.py \
		--data-dir ./data/acne_dataset \
		--experiment-dir ./experiments/diffusion_$(shell date +%Y%m%d_%H%M%S) \
		--epochs 100 \
		--batch-size 16

train-classifier:
	@echo "🎓 Starting classifier training..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_classifier.py \
		--data-dir ./data/acne_dataset \
		--experiment-dir ./experiments/classifier_$(shell date +%Y%m%d_%H%M%S) \
		--epochs 200 \
		--batch-size 32

# Quick training for testing
train-quick-diffusion:
	@echo "⚡ Quick training test (5 epochs)..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_diffusion.py \
		--data-dir ./data/acne_dataset \
		--experiment-dir ./experiments/quick_test_diffusion_$(shell date +%Y%m%d_%H%M%S) \
		--epochs 5 \
		--batch-size 8

train-quick-classifier:
	@echo "⚡ Quick training test (5 epochs)..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_classifier.py \
		--data-dir ./data/acne_dataset \
		--experiment-dir ./experiments/quick_test_classifier_$(shell date +%Y%m%d_%H%M%S) \
		--epochs 50 \
		--batch-size 8

# Resume training shortcuts
resume-diffusion:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Usage: make resume-diffusion CHECKPOINT=path/to/checkpoint.pth"; \
		echo "💡 Example: make resume-diffusion CHECKPOINT=experiments/diffusion_*/checkpoints/diffusion_checkpoint_epoch_50.pth"; \
		echo "📁 Available checkpoints:"; \
		find experiments/ -name "*diffusion*checkpoint*.pth" -type f -exec ls -la {} \; 2>/dev/null | head -5 || echo "No diffusion checkpoints found"; \
		exit 1; \
	fi
	@echo "🔄 Resuming diffusion model training from checkpoint..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_diffusion.py \
		--resume $(CHECKPOINT) \
		--data-dir ./data/acne_dataset \
		--epochs 1000 \
		--batch-size 16

resume-classifier:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Usage: make resume-classifier CHECKPOINT=path/to/checkpoint.pth"; \
		echo "💡 Example: make resume-classifier CHECKPOINT=experiments/classifier_*/checkpoints/classifier_checkpoint_epoch_100.pth"; \
		echo "📁 Available checkpoints:"; \
		find experiments/ -name "*classifier*checkpoint*.pth" -type f -exec ls -la {} \; 2>/dev/null | head -5 || echo "No classifier checkpoints found"; \
		exit 1; \
	fi
	@echo "🔄 Resuming classifier training from checkpoint..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_classifier.py \
		--resume $(CHECKPOINT) \
		--data-dir ./data/acne_dataset \
		--epochs 200 \
		--batch-size 32

# Resume with custom parameters
resume-diffusion-custom:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Usage: make resume-diffusion-custom CHECKPOINT=path/to/checkpoint.pth EPOCHS=50 BATCH_SIZE=16 LR=1e-4"; \
		echo "💡 Example: make resume-diffusion-custom CHECKPOINT=experiments/diffusion_*/checkpoints/diffusion_checkpoint_epoch_50.pth EPOCHS=50"; \
		exit 1; \
	fi
	@echo "🔄 Resuming diffusion training with custom parameters..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_diffusion.py \
		--resume $(CHECKPOINT) \
		--data-dir ./data/acne_dataset \
		--epochs $(or $(EPOCHS),100) \
		--batch-size $(or $(BATCH_SIZE),16) \
		--lr $(or $(LR),1e-4) \
		--experiment-dir ./experiments/resumed_diffusion_$(shell date +%Y%m%d_%H%M%S)

resume-classifier-custom:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Usage: make resume-classifier-custom CHECKPOINT=path/to/checkpoint.pth EPOCHS=100 BATCH_SIZE=32 LR=3e-4"; \
		echo "💡 Example: make resume-classifier-custom CHECKPOINT=experiments/classifier_*/checkpoints/classifier_checkpoint_epoch_100.pth EPOCHS=100"; \
		exit 1; \
	fi
	@echo "🔄 Resuming classifier training with custom parameters..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_classifier.py \
		--resume $(CHECKPOINT) \
		--data-dir ./data/acne_dataset \
		--epochs $(or $(EPOCHS),200) \
		--batch-size $(or $(BATCH_SIZE),32) \
		--lr $(or $(LR),3e-4) \
		--experiment-dir ./experiments/resumed_classifier_$(shell date +%Y%m%d_%H%M%S)

# Auto-resume from latest checkpoint
auto-resume-diffusion:
	@echo "🔍 Finding latest diffusion checkpoint..."
	@LATEST_CHECKPOINT=$$(find experiments/ -name "*diffusion*checkpoint*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2); \
	if [ -z "$$LATEST_CHECKPOINT" ]; then \
		echo "❌ No diffusion checkpoints found. Start fresh training with: make train-diffusion"; \
		exit 1; \
	fi; \
	echo "📁 Latest checkpoint: $$LATEST_CHECKPOINT"; \
	echo "🔄 Resuming training..."; \
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_diffusion.py \
		--resume "$$LATEST_CHECKPOINT" \
		--data-dir ./data/acne_dataset \
		--epochs 100 \
		--batch-size 16

auto-resume-classifier:
	@echo "🔍 Finding latest classifier checkpoint..."
	@LATEST_CHECKPOINT=$$(find experiments/ -name "*classifier*checkpoint*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2); \
	if [ -z "$$LATEST_CHECKPOINT" ]; then \
		echo "❌ No classifier checkpoints found. Start fresh training with: make train-classifier"; \
		exit 1; \
	fi; \
	echo "📁 Latest checkpoint: $$LATEST_CHECKPOINT"; \
	echo "🔄 Resuming training..."; \
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/train_classifier.py \
		--resume "$$LATEST_CHECKPOINT" \
		--data-dir ./data/acne_dataset \
		--epochs 200 \
		--batch-size 32

# Generation and evaluation
generate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Usage: make generate CHECKPOINT=path/to/checkpoint.pth"; \
		echo "💡 Example: make generate CHECKPOINT=experiments/diffusion_*/checkpoints/best_diffusion.pth"; \
		exit 1; \
	fi
	@echo "🎨 Generating samples..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/generate_samples.py \
		--checkpoint $(CHECKPOINT) \
		--output-dir ./generated_samples_$(shell date +%Y%m%d_%H%M%S) \
		--num-samples 10 \
		--save-process

evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Usage: make evaluate CHECKPOINT=path/to/checkpoint.pth"; \
		echo "💡 Example: make evaluate CHECKPOINT=experiments/classifier_*/checkpoints/best_classifier.pth"; \
		exit 1; \
	fi
	@echo "📊 Evaluating model..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$PYTHONPATH" python scripts/evaluate_model.py \
		--checkpoint $(CHECKPOINT) \
		--data-dir ./data/acne_dataset \
		--output-dir ./evaluation_results_$(shell date +%Y%m%d_%H%M%S)

# Find and show latest checkpoints
show-checkpoints:
	@echo "📁 Latest checkpoints:"
	@find experiments/ -name "*.pth" -type f -exec ls -la {} \; 2>/dev/null | head -10 || echo "No checkpoints found"

show-results:
	@echo "📊 Recent experiment results:"
	@find experiments/ -name "*.png" -o -name "*.txt" -o -name "*.json" | head -10 || echo "No results found"

# Data utilities
check-data:
	@echo "📊 Checking dataset..."
	@if [ -d "./data/acne_dataset" ]; then \
		echo "✅ Dataset directory found"; \
		find ./data/acne_dataset -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l | sed 's/^/   Total images: /'; \
		find ./data/acne_dataset -type d -name "acne*" | while read dir; do \
			count=$$(find "$$dir" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l); \
			echo "   $$(basename $$dir): $$count images"; \
		done; \
	else \
		echo "❌ Dataset not found at ./data/acne_dataset"; \
		echo "💡 Please create the directory and add your ACNE04 dataset"; \
	fi

# Complete setup for new environment
setup:
	@echo "🚀 Setting up Acne Diffusion project with conda..."
	@echo ""
	@echo "📝 Setup steps:"
	@echo "1. Create conda environment:"
	@echo "   make env-create"
	@echo ""
	@echo "2. Activate environment:"
	@echo "   conda activate diffusion-env"
	@echo ""
	@echo "3. Install dependencies:"
	@echo "   make install"
	@echo ""
	@echo "4. Setup for cluster:"
	@echo "   make setup-cluster"
	@echo ""
	@echo "5. Check GPU availability:"
	@echo "   make check-gpu"
	@echo ""
	@echo "6. Place your dataset in ./data/acne_dataset/"
	@echo ""
	@echo "7. Start training:"
	@echo "   make train-diffusion"

# Development helpers
dev-setup:
	make env-create
	@echo "Now run: conda activate diffusion-env && make install-dev && make setup-cluster"

# Cluster job shortcuts
submit-diffusion:
	@if [ ! -f "job_diffusion_single.sh" ]; then \
		echo "❌ SLURM job script not found. Run: ./slurm_job_examples.sh"; \
		exit 1; \
	fi
	sbatch job_diffusion_single.sh

submit-classifier:
	@if [ ! -f "job_classifier_single.sh" ]; then \
		echo "❌ SLURM job script not found. Run: ./slurm_job_examples.sh"; \
		exit 1; \
	fi
	sbatch job_classifier_single.sh

check-jobs:
	@echo "📋 Current SLURM jobs:"
	@squeue -u $$USER 2>/dev/null || echo "Not on a SLURM cluster or squeue not available"

cancel-jobs:
	@echo "🛑 Cancelling all your jobs..."
	@scancel -u $$USER 2>/dev/null || echo "No jobs to cancel"

# Comprehensive status check
status:
	@echo "📊 Project Status Check"
	@echo "======================="
	@echo ""
	@echo "🐍 Conda environment:"
	@conda info --envs | grep diffusion-env || echo "   diffusion-env not found"
	@echo ""
	make check-gpu
	@echo ""
	make check-data
	@echo ""
	make show-checkpoints
	@echo ""
	@echo "📁 Disk usage:"
	@du -h experiments/ 2>/dev/null | tail -1 || echo "No experiments directory"

# Environment management helpers
conda-info:
	@echo "🐍 Conda Environment Info:"
	@echo "Current environment: $${CONDA_DEFAULT_ENV:-none}"
	@echo "Available environments:"
	@conda env list

env-update:
	@echo "📦 Updating environment packages..."
	conda update --all -y
	pip install --upgrade pip