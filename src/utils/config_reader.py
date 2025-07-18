"""
Enhanced configuration reader for YAML config with dataclass integration.
"""
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Any, Union


class ConfigReader:
    """Enhanced configuration reader with dataclass integration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'diffusion.training.epochs')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_diffusion_model_config(self):
        """Get diffusion model configuration as dataclass."""
        from configs.diffusion_config import DiffusionModelConfig
        
        model_config = DiffusionModelConfig()
        model_data = self.get('diffusion.model', {})
        
        for field in fields(model_config):
            if field.name in model_data:
                value = model_data[field.name]
                # Convert lists to tuples if needed
                if field.name in ['channels_multiple', 'attention_levels'] and isinstance(value, list):
                    value = tuple(value)
                setattr(model_config, field.name, value)
        
        return model_config
    
    def get_diffusion_training_config(self):
        """Get diffusion training configuration as dataclass."""
        from configs.diffusion_config import DiffusionTrainingConfig
        
        training_config = DiffusionTrainingConfig()
        training_data = self.get('diffusion.training', {})
        
        # Set experiment directory with timestamp
        import time
        timestamp = int(time.time())
        experiment_name = f"diffusion_{timestamp}"
        training_data['experiment_dir'] = f"{self.get('project.experiments_dir')}/{experiment_name}"
        
        for field in fields(training_config):
            if field.name in training_data:
                setattr(training_config, field.name, training_data[field.name])
        
        return training_config
    
    def get_classifier_model_config(self):
        """Get classifier model configuration as dataclass."""
        from configs.classifier_config import ClassifierModelConfig
        
        model_config = ClassifierModelConfig()
        model_data = self.get('classifier.model', {})
        
        for field in fields(model_config):
            if field.name in model_data:
                value = model_data[field.name]
                # Convert lists to tuples if needed
                if field.name in ['channels_multiple', 'attention_levels', 'num_res_blocks'] and isinstance(value, list):
                    value = tuple(value)
                setattr(model_config, field.name, value)
        
        return model_config
    
    def get_classifier_training_config(self):
        """Get classifier training configuration as dataclass."""
        from configs.classifier_config import ClassifierTrainingConfig
        
        training_config = ClassifierTrainingConfig()
        training_data = self.get('classifier.training', {})
        
        # Set experiment directory with timestamp
        import time
        timestamp = int(time.time())
        experiment_name = f"classifier_{timestamp}"
        training_data['experiment_dir'] = f"{self.get('project.experiments_dir')}/{experiment_name}"
        
        for field in fields(training_config):
            if field.name in training_data:
                setattr(training_config, field.name, training_data[field.name])
        
        return training_config
    
    def get_data_config(self):
        """Get data configuration as dataclass."""
        from configs.base_config import DataConfig
        
        data_config = DataConfig()
        data_data = self.get('data', {})
        
        for field in fields(data_config):
            if field.name in data_data:
                setattr(data_config, field.name, data_data[field.name])
        
        return data_config
    
    def get_generation_config(self):
        """Get generation configuration."""
        return self.get('generation', {})
    
    def get_evaluation_config(self):
        """Get evaluation configuration."""
        return self.get('evaluation', {})
    
    def get_wandb_config(self):
        """Get wandb configuration."""
        return self.get('wandb', {})
    
    def export_for_makefile(self, output_file: str = "config.mk"):
        """Export config as Makefile variables."""
        lines = [
            "# Auto-generated from config.yaml",
            "",
            f"PROJECT_NAME := {self.get('project.name')}",
            f"DATA_DIR := {self.get('project.data_dir')}",
            f"EXPERIMENTS_DIR := {self.get('project.experiments_dir')}",
            f"LOGS_DIR := {self.get('project.logs_dir')}",
            f"CONDA_ENV := {self.get('project.conda_env')}",
            "",
            f"CLUSTER_ACCOUNT := {self.get('cluster.account')}",
            f"CLUSTER_PARTITION := {self.get('cluster.partition')}",
            f"CLUSTER_PARTITION_INTERACTIVE := {self.get('cluster.partition_interactive')}",
            "",
            f"WANDB_PROJECT := {self.get('wandb.project')}",
            "",
            f"DIFFUSION_EPOCHS := {self.get('diffusion.training.n_epochs')}",
            f"DIFFUSION_BATCH_SIZE := {self.get('diffusion.training.batch_size')}",
            f"DIFFUSION_LR := {self.get('diffusion.training.learning_rate')}",
            f"DIFFUSION_QUICK_EPOCHS := {self.get('diffusion.quick_test.n_epochs')}",
            "",
            f"CLASSIFIER_EPOCHS := {self.get('classifier.training.n_epochs')}",
            f"CLASSIFIER_BATCH_SIZE := {self.get('classifier.training.batch_size')}",
            f"CLASSIFIER_LR := {self.get('classifier.training.learning_rate')}",
            f"CLASSIFIER_QUICK_EPOCHS := {self.get('classifier.quick_test.n_epochs')}",
        ]
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--export-makefile", action="store_true")
    parser.add_argument("--get", help="Get config value")
    
    args = parser.parse_args()
    
    config = ConfigReader(args.config)
    
    if args.export_makefile:
        output_file = config.export_for_makefile()
        print(f"✅ Exported to {output_file}")
    
    if args.get:
        value = config.get(args.get)
        print(f"{args.get}: {value}")