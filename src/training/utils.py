"""
Training utilities and helper functions.
"""
import os
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
            
        return False


class LearningRateScheduler:
    """Custom learning rate scheduler with multiple strategies."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, strategy: str = "cosine", **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            strategy: Scheduling strategy ('cosine', 'step', 'exponential', 'plateau')
            **kwargs: Additional arguments for specific strategies
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.kwargs = kwargs
        self.step_count = 0
        self.best_loss = float('inf')
        
        if strategy == "step":
            self.step_size = kwargs.get('step_size', 30)
            self.gamma = kwargs.get('gamma', 0.1)
        elif strategy == "exponential":
            self.gamma = kwargs.get('gamma', 0.95)
        elif strategy == "cosine":
            self.max_epochs = kwargs.get('max_epochs', 100)
        elif strategy == "plateau":
            self.factor = kwargs.get('factor', 0.5)
            self.patience = kwargs.get('patience', 10)
            self.wait = 0
    
    def step(self, epoch: Optional[int] = None, val_loss: Optional[float] = None):
        """Update learning rate based on strategy."""
        self.step_count += 1
        
        if self.strategy == "step":
            if epoch and epoch % self.step_size == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.gamma
                    
        elif self.strategy == "exponential":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
                
        elif self.strategy == "cosine":
            if epoch:
                lr = self.kwargs.get('initial_lr', 1e-4) * 0.5 * (1 + np.cos(np.pi * epoch / self.max_epochs))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
        elif self.strategy == "plateau":
            if val_loss is not None:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= self.factor
                        self.wait = 0
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.start_time = None
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.start_time = time.time()
    
    def end_epoch(self, loss: float, accuracy: Optional[float] = None, lr: Optional[float] = None):
        """Mark the end of an epoch and record metrics."""
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
        
        self.losses.append(loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def get_average_epoch_time(self) -> float:
        """Get average epoch time."""
        return np.mean(self.epoch_times) if self.epoch_times else 0.0
    
    def get_best_loss(self) -> Tuple[float, int]:
        """Get best loss and epoch."""
        if not self.losses:
            return float('inf'), -1
        best_idx = np.argmin(self.losses)
        return self.losses[best_idx], best_idx
    
    def get_best_accuracy(self) -> Tuple[float, int]:
        """Get best accuracy and epoch."""
        if not self.accuracies:
            return 0.0, -1
        best_idx = np.argmax(self.accuracies)
        return self.accuracies[best_idx], best_idx
    
    def plot_metrics(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training metrics."""
        n_plots = 1 + (len(self.accuracies) > 0) + (len(self.learning_rates) > 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot losses
        epochs = range(1, len(self.losses) + 1)
        axes[plot_idx].plot(epochs, self.losses, 'b-', label='Loss', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Training Loss')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend()
        plot_idx += 1
        
        # Plot accuracies if available
        if self.accuracies:
            axes[plot_idx].plot(epochs, self.accuracies, 'g-', label='Accuracy', linewidth=2)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Accuracy (%)')
            axes[plot_idx].set_title('Training Accuracy')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            plot_idx += 1
        
        # Plot learning rates if available
        if self.learning_rates:
            axes[plot_idx].plot(epochs, self.learning_rates, 'r-', label='Learning Rate', linewidth=2)
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Learning Rate')
            axes[plot_idx].set_title('Learning Rate Schedule')
            axes[plot_idx].set_yscale('log')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Metrics plot saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


class GradientClipping:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms = []
    
    def clip_gradients(self, model: torch.nn.Module) -> float:
        """Clip gradients and return the gradient norm."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=self.max_norm, 
            norm_type=self.norm_type
        )
        self.grad_norms.append(grad_norm.item())
        return grad_norm.item()
    
    def get_average_grad_norm(self) -> float:
        """Get average gradient norm."""
        return np.mean(self.grad_norms) if self.grad_norms else 0.0


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model size and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
    }


def count_dataset_classes(dataloader: DataLoader) -> Dict[int, int]:
    """Count the number of samples per class in a dataset."""
    class_counts = {}
    
    for batch in dataloader:
        labels = batch["label"].numpy()
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
    
    return class_counts


def estimate_training_time(dataloader: DataLoader, seconds_per_batch: float, epochs: int) -> Dict[str, float]:
    """Estimate total training time."""
    total_batches = len(dataloader) * epochs
    total_seconds = total_batches * seconds_per_batch
    
    return {
        'total_batches': total_batches,
        'total_seconds': total_seconds,
        'total_minutes': total_seconds / 60,
        'total_hours': total_seconds / 3600,
        'batches_per_epoch': len(dataloader)
    }


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_memory_usage(device: torch.device) -> Dict[str, float]:
    """Check GPU memory usage."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        max_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': max_memory,
            'free_gb': max_memory - reserved,
            'utilization_percent': (reserved / max_memory) * 100
        }
    else:
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'total_gb': 0.0,
            'free_gb': 0.0,
            'utilization_percent': 0.0
        }


def get_device_info() -> Dict[str, str]:
    """Get device information."""
    device_info = {
        'device_type': 'cpu',
        'device_count': 0,
        'device_name': 'CPU',
        'cuda_version': 'N/A'
    }
    
    if torch.cuda.is_available():
        device_info.update({
            'device_type': 'cuda',
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda
        })
    
    return device_info


class ModelEMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}