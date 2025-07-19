# Simple Makefile for Acne Diffusion Project - Uses config.yaml for all settings

# Include auto-generated config variables
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
	@echo "Training (reads from config.yaml):"
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
	@echo ""
	@echo "Configuration:"
	@echo "  show-config                 Show current config settings"
	@echo "  edit-config                 Edit config.yaml"

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

# Training commands (all read from config.yaml)
train-diffusion-quick:
	@echo "⚡ Quick diffusion test..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_diffusion.py --config config.yaml --quick-test

train-diffusion:
	@echo "🚀 Training diffusion model..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_diffusion.py --config config.yaml

train-diffusion-wandb:
	@echo "🚀 Training diffusion model with wandb..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_diffusion.py --config config.yaml --enable-wandb

train-classifier-quick:
	@echo "⚡ Quick classifier test..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_classifier.py --config config.yaml --quick-test

train-classifier:
	@echo "🎓 Training classifier model..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_classifier.py --config config.yaml

train-classifier-wandb:
	@echo "🎓 Training classifier model with wandb..."
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_classifier.py --config config.yaml --enable-wandb

# Resume training
resume-diffusion:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make resume-diffusion CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_diffusion.py \
		--config config.yaml --resume $(CHECKPOINT) --enable-wandb --wandb-name diffusion_resumed

resume-classifier:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make resume-classifier CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/train_classifier.py \
		--config config.yaml --resume $(CHECKPOINT) --enable-wandb --wandb-name classifier_resumed

# Generation and evaluation
generate:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make generate CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/generate_samples.py \
		--config config.yaml --checkpoint $(CHECKPOINT) --output-dir ./generated_$(shell date +%Y%m%d_%H%M%S)

evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make evaluate CHECKPOINT=path"; exit 1; fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/evaluate_model.py \
		--config config.yaml --checkpoint $(CHECKPOINT) --output-dir ./evaluation_$(shell date +%Y%m%d_%H%M%S)

# Translation commands
translate-severity:
	@if [ -z "$(DIFFUSION_CHECKPOINT)" ] || [ -z "$(CLASSIFIER_CHECKPOINT)" ] || [ -z "$(INPUT_IMAGE)" ] || [ -z "$(TARGET_SEVERITY)" ]; then \
		echo "❌ Missing required arguments"; \
		echo "Usage: make translate-severity DIFFUSION_CHECKPOINT=path CLASSIFIER_CHECKPOINT=path INPUT_IMAGE=path TARGET_SEVERITY=0-3"; \
		echo "Optional: SOURCE_SEVERITY=0-3 OUTPUT_DIR=path GUIDANCE_SCALE=30.0 NUM_STEPS=250"; \
		exit 1; \
	fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/translate_severity.py \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT) \
		--classifier-checkpoint $(CLASSIFIER_CHECKPOINT) \
		--input-image $(INPUT_IMAGE) \
		--target-severity $(TARGET_SEVERITY) \
		--config config.yaml \
		$(if $(SOURCE_SEVERITY),--source-severity $(SOURCE_SEVERITY)) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),--output-dir ./translation_results) \
		$(if $(GUIDANCE_SCALE),--guidance-scale $(GUIDANCE_SCALE)) \
		$(if $(NUM_STEPS),--num-steps $(NUM_STEPS)) \
		$(if $(SAVE_PROCESS),--save-process)

translate-with-process:
	@if [ -z "$(DIFFUSION_CHECKPOINT)" ] || [ -z "$(CLASSIFIER_CHECKPOINT)" ] || [ -z "$(INPUT_IMAGE)" ] || [ -z "$(TARGET_SEVERITY)" ]; then \
		echo "❌ Missing required arguments"; \
		exit 1; \
	fi
	PYTHONPATH="$(shell pwd):$(shell pwd)/src:$$PYTHONPATH" python scripts/translate_severity.py \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT) \
		--classifier-checkpoint $(CLASSIFIER_CHECKPOINT) \
		--input-image $(INPUT_IMAGE) \
		--target-severity $(TARGET_SEVERITY) \
		--config config.yaml \
		--save-process \
		$(if $(SOURCE_SEVERITY),--source-severity $(SOURCE_SEVERITY)) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR),--output-dir ./translation_results) \
		$(if $(GUIDANCE_SCALE),--guidance-scale $(GUIDANCE_SCALE)) \
		$(if $(NUM_STEPS),--num-steps $(NUM_STEPS))

# SLURM job management
generate-slurm-scripts:
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

# Configuration management
show-config: config.mk
	@echo "📋 Current Configuration (from config.yaml)"
	@echo "==========================================="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Data: $(DATA_DIR)"
	@echo "Conda env: $(CONDA_ENV)"
	@echo ""
	@echo "Diffusion:"
	@echo "  Epochs: $(DIFFUSION_EPOCHS)"
	@echo "  Batch size: $(DIFFUSION_BATCH_SIZE)"
	@echo "  Learning rate: $(DIFFUSION_LR)"
	@echo "  Quick test epochs: $(DIFFUSION_QUICK_EPOCHS)"
	@echo ""
	@echo "Classifier:"
	@echo "  Epochs: $(CLASSIFIER_EPOCHS)"
	@echo "  Batch size: $(CLASSIFIER_BATCH_SIZE)"
	@echo "  Learning rate: $(CLASSIFIER_LR)"
	@echo "  Quick test epochs: $(CLASSIFIER_QUICK_EPOCHS)"
	@echo ""
	@echo "Wandb project: $(WANDB_PROJECT)"
	@echo "Cluster account: $(CLUSTER_ACCOUNT)"
	@echo ""
	@echo "📝 To change settings, edit config.yaml"

edit-config:
	@echo "📝 Opening config.yaml for editing..."
	@if command -v code >/dev/null 2>&1; then \
		code config.yaml; \
	elif command -v nano >/dev/null 2>&1; then \
		nano config.yaml; \
	elif command -v vim >/dev/null 2>&1; then \
		vim config.yaml; \
	else \
		echo "❌ No editor found. Please edit config.yaml manually"; \
	fi

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
	@echo ""
	@echo "📝 Configuration: All settings read from config.yaml"

validate-config:
	@echo "🔍 Validating config.yaml..."
	@python -c "from src.utils.config_reader import ConfigReader; c = ConfigReader('config.yaml'); print('✅ config.yaml is valid')" || echo "❌ config.yaml has errors"

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f config.mk