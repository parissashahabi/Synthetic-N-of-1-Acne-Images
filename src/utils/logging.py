"""
Logging utilities for training and inference.
"""
import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


class TrainingLogger:
    """Comprehensive logging system for training experiments."""
    
    def __init__(
        self, 
        experiment_dir: str, 
        name: str = "training",
        level: int = logging.INFO,
        use_colors: bool = True,
        log_to_file: bool = True,
        log_to_console: bool = True
    ):
        """
        Initialize training logger.
        
        Args:
            experiment_dir: Directory to save logs
            name: Logger name
            level: Logging level
            use_colors: Use colored output for console
            log_to_file: Save logs to file
            log_to_console: Print logs to console
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if use_colors:
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
        
        # File handler
        if log_to_file:
            log_file = self.experiment_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = {}
        self.start_time = time.time()
        
        self.info(f"🚀 Logger initialized: {name}")
        self.info(f"📁 Experiment directory: {experiment_dir}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        epoch_time: Optional[float] = None
    ):
        """Log epoch results."""
        message = f"Epoch {epoch:3d}"
        
        if train_loss is not None:
            message += f" | Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            message += f" | Val Loss: {val_loss:.4f}"
        
        if train_acc is not None:
            message += f" | Train Acc: {train_acc:.2f}%"
        
        if val_acc is not None:
            message += f" | Val Acc: {val_acc:.2f}%"
        
        if lr is not None:
            message += f" | LR: {lr:.2e}"
        
        if epoch_time is not None:
            message += f" | Time: {epoch_time:.1f}s"
        
        self.info(message)
        
        # Store metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': lr,
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        self.metrics[epoch].update(epoch_metrics)
    
    def log_model_info(self, model: torch.nn.Module, model_name: str = "Model"):
        """Log model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        self.info(f"📊 {model_name} Information:")
        self.info(f"   Total parameters: {total_params:,}")
        self.info(f"   Trainable parameters: {trainable_params:,}")
        self.info(f"   Model size: {model_size_mb:.1f} MB")
        self.info(f"   Trainable ratio: {trainable_params/total_params:.2%}")
    
    def log_config(self, config: Dict[str, Any], config_name: str = "Configuration"):
        """Log configuration."""
        self.info(f"⚙️ {config_name}:")
        for key, value in config.items():
            self.info(f"   {key}: {value}")
    
    def log_device_info(self, device: torch.device):
        """Log device information."""
        self.info(f"🖥️ Device Information:")
        self.info(f"   Device: {device}")
        
        if device.type == 'cuda':
            self.info(f"   GPU Name: {torch.cuda.get_device_name(device)}")
            self.info(f"   GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
            self.info(f"   CUDA Version: {torch.version.cuda}")
        
        self.info(f"   PyTorch Version: {torch.__version__}")
    
    def log_dataset_info(self, train_size: int, val_size: int, batch_size: int):
        """Log dataset information."""
        self.info(f"📊 Dataset Information:")
        self.info(f"   Training samples: {train_size:,}")
        self.info(f"   Validation samples: {val_size:,}")
        self.info(f"   Total samples: {train_size + val_size:,}")
        self.info(f"   Batch size: {batch_size}")
        self.info(f"   Training batches: {train_size // batch_size}")
        self.info(f"   Validation batches: {val_size // batch_size}")
    
    def log_memory_usage(self, device: torch.device):
        """Log memory usage."""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            
            self.info(f"💾 GPU Memory Usage:")
            self.info(f"   Allocated: {allocated:.2f} GB")
            self.info(f"   Reserved: {reserved:.2f} GB")
            self.info(f"   Total: {total:.2f} GB")
            self.info(f"   Utilization: {(reserved/total)*100:.1f}%")
    
    def log_checkpoint(self, checkpoint_path: str, epoch: int, is_best: bool = False):
        """Log checkpoint saving."""
        checkpoint_type = "best" if is_best else "regular"
        self.info(f"💾 Saved {checkpoint_type} checkpoint: {checkpoint_path} (epoch {epoch})")
    
    def log_training_start(self, epochs: int):
        """Log training start."""
        self.info("🚀 " + "="*50)
        self.info(f"🚀 STARTING TRAINING - {epochs} EPOCHS")
        self.info("🚀 " + "="*50)
        self.start_time = time.time()
    
    def log_training_end(self, epochs_completed: int):
        """Log training completion."""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.info("🎉 " + "="*50)
        self.info(f"🎉 TRAINING COMPLETED - {epochs_completed} EPOCHS")
        self.info(f"🎉 Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.info("🎉 " + "="*50)
    
    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save metrics to JSON file."""
        metrics_file = self.experiment_dir / filename
        
        # Convert any numpy types to native Python types
        def convert_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        converted_metrics = convert_types(self.metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(converted_metrics, f, indent=2, default=str)
        
        self.info(f"📊 Metrics saved to: {metrics_file}")
        return metrics_file
    
    def load_metrics(self, filename: str = "training_metrics.json") -> Dict[str, Any]:
        """Load metrics from JSON file."""
        metrics_file = self.experiment_dir / filename
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.metrics = json.load(f)
            self.info(f"📊 Metrics loaded from: {metrics_file}")
        else:
            self.warning(f"📊 Metrics file not found: {metrics_file}")
        
        return self.metrics


class WandbLogger:
    """Weights & Biases integration for experiment tracking."""
    
    def __init__(
        self, 
        project: str, 
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """
        Initialize Weights & Biases logger.
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
            tags: List of tags for the run
        """
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
            
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags
            )
            
            print(f"🌐 W&B logging enabled: {wandb.run.url}")
            
        except ImportError:
            print("⚠️ W&B not installed. Install with: pip install wandb")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, name: str = "model"):
        """Log model artifact to W&B."""
        if self.enabled:
            artifact = self.wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            self.wandb.log_artifact(artifact)
    
    def log_image(self, image, caption: str = "", step: Optional[int] = None):
        """Log image to W&B."""
        if self.enabled:
            self.wandb.log({caption: self.wandb.Image(image)}, step=step)
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            self.wandb.finish()


class TensorBoardLogger:
    """TensorBoard integration for experiment tracking."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            print(f"📊 TensorBoard logging enabled: {log_dir}")
            print(f"📊 Start TensorBoard with: tensorboard --logdir {log_dir}")
            
        except ImportError:
            print("⚠️ TensorBoard not installed. Install with: pip install tensorboard")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ TensorBoard initialization failed: {e}")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image."""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_graph(self, model, input_to_model):
        """Log model graph."""
        if self.enabled:
            self.writer.add_graph(model, input_to_model)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()


class ExperimentLogger:
    """Combined logger that integrates multiple logging backends."""
    
    def __init__(
        self,
        experiment_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment logger with multiple backends.
        
        Args:
            experiment_dir: Directory for experiment outputs
            experiment_name: Name of the experiment
            use_wandb: Enable W&B logging
            use_tensorboard: Enable TensorBoard logging
            wandb_project: W&B project name
            config: Configuration dictionary
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training logger
        self.logger = TrainingLogger(
            experiment_dir=experiment_dir,
            name=experiment_name
        )
        
        # Initialize W&B logger
        self.wandb_logger = None
        if use_wandb and wandb_project:
            self.wandb_logger = WandbLogger(
                project=wandb_project,
                name=experiment_name,
                config=config
            )
        
        # Initialize TensorBoard logger
        self.tb_logger = None
        if use_tensorboard:
            tb_log_dir = self.experiment_dir / "tensorboard"
            self.tb_logger = TensorBoardLogger(str(tb_log_dir))
        
        self.step_count = 0
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """Log metrics to all enabled backends."""
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.tb_logger:
            for tag, value in metrics.items():
                self.tb_logger.log_scalar(tag, value, step)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        epoch_time: Optional[float] = None
    ):
        """Log epoch results to all backends."""
        # Log to training logger
        self.logger.log_epoch(
            epoch=epoch,
            train_loss=train_metrics.get('loss'),
            val_loss=val_metrics.get('loss') if val_metrics else None,
            train_acc=train_metrics.get('accuracy'),
            val_acc=val_metrics.get('accuracy') if val_metrics else None,
            lr=lr,
            epoch_time=epoch_time
        )
        
        # Prepare metrics for other loggers
        all_metrics = {}
        
        # Add training metrics
        for key, value in train_metrics.items():
            all_metrics[f"train/{key}"] = value
        
        # Add validation metrics
        if val_metrics:
            for key, value in val_metrics.items():
                all_metrics[f"val/{key}"] = value
        
        # Add learning rate
        if lr is not None:
            all_metrics["learning_rate"] = lr
        
        # Add epoch time
        if epoch_time is not None:
            all_metrics["epoch_time"] = epoch_time
        
        # Log to all backends
        self.log_metrics(all_metrics, step=epoch)
    
    def log_model_weights(self, model: torch.nn.Module, step: int):
        """Log model weights histograms."""
        if self.tb_logger:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.tb_logger.log_histogram(f"weights/{name}", param.data, step)
                    if param.grad is not None:
                        self.tb_logger.log_histogram(f"gradients/{name}", param.grad, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image to all backends."""
        if self.wandb_logger:
            self.wandb_logger.log_image(image, caption=tag, step=step)
        
        if self.tb_logger:
            self.tb_logger.log_image(tag, image, step)
    
    def save_checkpoint_info(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        """Log checkpoint information."""
        self.logger.log_checkpoint(checkpoint_path, epoch, is_best=metrics.get('is_best', False))
        
        if self.wandb_logger:
            self.wandb_logger.log_model(checkpoint_path, f"checkpoint_epoch_{epoch}")
    
    def finish(self):
        """Finish logging and cleanup."""
        # Save training logger metrics
        self.logger.save_metrics()
        
        # Close W&B
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        # Close TensorBoard
        if self.tb_logger:
            self.tb_logger.close()
        
        self.logger.info("📊 Experiment logging finished")


def setup_logging(
    experiment_dir: str,
    experiment_name: str,
    level: int = logging.INFO,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    wandb_project: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> ExperimentLogger:
    """
    Setup comprehensive logging for an experiment.
    
    Args:
        experiment_dir: Directory for experiment outputs
        experiment_name: Name of the experiment
        level: Logging level
        use_wandb: Enable W&B logging
        use_tensorboard: Enable TensorBoard logging
        wandb_project: W&B project name
        config: Configuration dictionary
    
    Returns:
        Configured ExperimentLogger instance
    """
    return ExperimentLogger(
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        wandb_project=wandb_project,
        config=config
    )