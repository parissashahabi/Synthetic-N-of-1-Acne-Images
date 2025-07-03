# Simple Makefile for Acne Diffusion Project

# Include config variables
-include config.mk

# Generate config.mk from config.yaml
config.mk: config.yaml src/utils/config_reader.py
	@python src/utils/config_reader.py --config config.yaml --export-makefile

.PHONY: help install check-gpu verify-install interactive-gpu status

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install                     Install dependencies"
	@echo "  verify-install              Verify installation"
	@echo "  check-gpu                   Check GPU availability"
	@echo "  interactive-gpu             Request interactive GPU session"
	@echo ""
	@echo "Training:"
	@echo "  train-diffusion-quick       Quick diffusion test"
	@echo "  train-diffusion             Train diffusion model"
	@echo "  train-classifier-quick      Quick classifier test"
	@echo "  train-classifier            Train classifier model"
	@echo ""
	@echo "Generation & Evaluation:"
	@echo "  generate CHECKPOINT=path    Generate samples"
	@echo "  evaluate CHECKPOINT=path    Evaluate classifier"
	@echo ""
	@echo "Cluster:"
	@echo "  generate-slurm-scripts      Generate SLURM job scripts"
	@echo "  submit-test                 Submit test job"
	@echo "  submit-diffusion            Submit diffusion training"
	@echo "  submit-classifier           Submit classifier training"
	@echo "  check-jobs                  Check job status"

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install PyYAML
	conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
	pip install monai matplotlib tqdm "monai-weekly[pillow, tqdm]" monai-generative
	pip install "diffusers[torch]" scikit-learn tensorboard wandb hydra-core omegaconf einops
	pip install -e .
	@echo "✅ Installation complete!"

# Environment
env-create:
	conda create -n $(CONDA_ENV) python=3.9 -y

# GPU utilities
check-gpu:
	@python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

verify-install:
	@echo "🔍 Verifying installation..."
	@python -c "import torch; print('✅ PyTorch')" || echo "❌ PyTorch"
	@python -c "import monai; print('✅ MONAI')" || echo "❌ MONAI"
	@python -c "import generative; print('✅ MONAI Generative')" || echo "❌ MONAI Generative"
	@python -c "from src.utils.config_reader import ConfigReader; ConfigReader(); print('✅ Config system')" || echo "❌ Config system"

# Interactive session
interactive-gpu: config.mk
	srun --partition=$(CLUSTER_PARTITION_INTERACTIVE) --account=$(CLUSTER_ACCOUNT) \
		--cpus-per-task=4 --gpus=1 --mem=16G --time=02:00:00 --pty bash

# Training commands
train-diffusion-quick: config.mk
	@echo "⚡ Quick diffusion test ($(DIFFUSION_QUICK_EPOCHS) epochs)..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_diffusion.py \
		--data-dataset-path $(DATA_DIR) \
		--train-experiment-dir $(EXPERIMENTS_DIR)/quick_diffusion_$(shell date +%Y%m%d_%H%M%S) \
		--train-n-epochs $(DIFFUSION_QUICK_EPOCHS) \
		--train-batch-size 16

train-diffusion: config.mk
	@echo "🚀 Training diffusion model ($(DIFFUSION_EPOCHS) epochs)..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_diffusion.py \
		--data-dataset-path $(DATA_DIR) \
		--train-experiment-dir $(EXPERIMENTS_DIR)/diffusion_$(shell date +%Y%m%d_%H%M%S) \
		--train-n-epochs $(DIFFUSION_EPOCHS) \
		--train-batch-size $(DIFFUSION_BATCH_SIZE) \
		--train-learning-rate $(DIFFUSION_LR) \
		--wandb --wandb-project $(WANDB_PROJECT) --wandb-name diffusion_local

train-classifier-quick: config.mk
	@echo "⚡ Quick classifier test ($(CLASSIFIER_QUICK_EPOCHS) epochs)..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_classifier.py \
		--data-dataset-path $(DATA_DIR) \
		--train-experiment-dir $(EXPERIMENTS_DIR)/quick_classifier_$(shell date +%Y%m%d_%H%M%S) \
		--train-n-epochs $(CLASSIFIER_QUICK_EPOCHS) \
		--train-batch-size 16

train-classifier: config.mk
	@echo "🎓 Training classifier model ($(CLASSIFIER_EPOCHS) epochs)..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_classifier.py \
		--data-dataset-path $(DATA_DIR) \
		--train-experiment-dir $(EXPERIMENTS_DIR)/classifier_$(shell date +%Y%m%d_%H%M%S) \
		--train-n-epochs $(CLASSIFIER_EPOCHS) \
		--train-batch-size $(CLASSIFIER_BATCH_SIZE) \
		--train-learning-rate $(CLASSIFIER_LR) \
		--wandb --wandb-project $(WANDB_PROJECT) --wandb-name classifier_local

# Resume training
resume-diffusion: config.mk
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make resume-diffusion CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_diffusion.py \
		--resume $(CHECKPOINT) \
		--data-dataset-path $(DATA_DIR) \
		--train-experiment-dir $(EXPERIMENTS_DIR)/resumed_diffusion_$(shell date +%Y%m%d_%H%M%S) \
		--wandb --wandb-project $(WANDB_PROJECT) --wandb-name diffusion_resumed

resume-classifier: config.mk
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make resume-classifier CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_classifier.py \
		--resume $(CHECKPOINT) \
		--data-dataset-path $(DATA_DIR) \
		--train-experiment-dir $(EXPERIMENTS_DIR)/resumed_classifier_$(shell date +%Y%m%d_%H%M%S) \
		--wandb --wandb-project $(WANDB_PROJECT) --wandb-name classifier_resumed

# Generation and evaluation
generate: config.mk
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make generate CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/generate_samples.py \
		--checkpoint $(CHECKPOINT) \
		--output-dir ./generated_$(shell date +%Y%m%d_%H%M%S) \
		--train-num-samples 10 \
		--train-save-intermediates

evaluate: config.mk
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make evaluate CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/evaluate_model.py \
		--checkpoint $(CHECKPOINT) \
		--data-dataset-path $(DATA_DIR) \
		--output-dir ./evaluation_$(shell date +%Y%m%d_%H%M%S)

# SLURM job management
generate-slurm-scripts: config.mk
	@echo "📝 Generating SLURM job scripts..."
	python scripts/generate_slurm_jobs.py

submit-test:
	sbatch job_test.sh

submit-diffusion:
	sbatch job_diffusion.sh

submit-classifier:
	sbatch job_classifier.sh

check-jobs:
	@squeue -u $$USER 2>/dev/null || echo "No SLURM jobs or not on cluster"

# Status and utilities
status: config.mk
	@echo "📊 Project Status"
	@echo "================="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Data: $(DATA_DIR)"
	@echo "Conda env: $(CONDA_ENV)"
	@echo ""
	make check-gpu
	@echo ""
	@if [ -d "$(DATA_DIR)" ]; then \
		echo "✅ Dataset found"; \
		find $(DATA_DIR) -name "*.jpg" -o -name "*.png" | wc -l | sed 's/^/Images: /'; \
	else \
		echo "❌ Dataset not found at $(DATA_DIR)"; \
	fi

show-config: config.mk
	@echo "📋 Configuration"
	@echo "==============="
	@echo "Diffusion: $(DIFFUSION_EPOCHS) epochs, batch $(DIFFUSION_BATCH_SIZE), lr $(DIFFUSION_LR)"
	@echo "Classifier: $(CLASSIFIER_EPOCHS) epochs, batch $(CLASSIFIER_BATCH_SIZE), lr $(CLASSIFIER_LR)"
	@echo "Wandb project: $(WANDB_PROJECT)"
	@echo "Cluster account: $(CLUSTER_ACCOUNT)"

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f config.mk