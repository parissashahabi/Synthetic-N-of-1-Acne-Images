"""
Simple configuration reader for YAML config.
"""
import yaml
import os
from pathlib import Path


class ConfigReader:
    """Simple configuration reader."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'diffusion.epochs')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
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
            f"DIFFUSION_EPOCHS := {self.get('diffusion.epochs')}",
            f"DIFFUSION_BATCH_SIZE := {self.get('diffusion.batch_size')}",
            f"DIFFUSION_LR := {self.get('diffusion.learning_rate')}",
            f"DIFFUSION_QUICK_EPOCHS := {self.get('diffusion.quick_epochs')}",
            "",
            f"CLASSIFIER_EPOCHS := {self.get('classifier.epochs')}",
            f"CLASSIFIER_BATCH_SIZE := {self.get('classifier.batch_size')}",
            f"CLASSIFIER_LR := {self.get('classifier.learning_rate')}",
            f"CLASSIFIER_QUICK_EPOCHS := {self.get('classifier.quick_epochs')}",
        ]
        
        # Add resource configs
        for resource_name, resource_config in self.get('resources', {}).items():
            prefix = f"RESOURCE_{resource_name.upper()}"
            lines.extend([
                f"{prefix}_TIME := {resource_config.get('time')}",
                f"{prefix}_MEMORY := {resource_config.get('memory')}",
                f"{prefix}_CPUS := {resource_config.get('cpus')}",
            ])
        
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