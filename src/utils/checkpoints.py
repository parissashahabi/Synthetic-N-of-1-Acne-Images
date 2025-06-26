"""
Checkpoint management utilities.
"""
import os
import glob
import torch
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manages model checkpoints and saving/loading."""
    
    def __init__(self, checkpoint_dir: str, model_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_losses: list,
        val_losses: list,
        config: Any,
        is_best: bool = False,
        extra_data: Optional[Dict] = None
    ) -> str:
        """Save a training checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': config.__dict__ if hasattr(config, '__dict__') else config,
        }
        
        # Add extra data if provided
        if extra_data:
            checkpoint.update(extra_data)
        
        if is_best:
            filename = f'best_{self.model_name}.pth'
            print(f"💾 Best model saved!")
        else:
            filename = f'{self.model_name}_checkpoint_epoch_{epoch+1}.pth'
            print(f"💾 Checkpoint saved: epoch {epoch+1}")
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def save_final_model(self, model: torch.nn.Module) -> str:
        """Save final model (state dict only)."""
        filename = f'final_{self.model_name}.pth'
        model_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(model.state_dict(), model_path)
        
        print(f"🎯 Final model saved: {model_path}")
        return model_path
    
    def load_checkpoint(
        self, 
        checkpoint_path: str, 
        model: torch.nn.Module, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
    ) -> Optional[Dict]:
        """Load a checkpoint."""
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            print(f"📂 Loading checkpoint: {os.path.basename(checkpoint_path)}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer state if available
                if optimizer and 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("✅ Optimizer state loaded")
                    except Exception as e:
                        print(f"⚠️ Could not load optimizer state: {e}")
                
                print(f"✅ Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                return checkpoint
            else:
                # Just model weights
                model.load_state_dict(checkpoint)
                print("✅ Model weights loaded")
                return {'epoch': 0, 'train_losses': [], 'val_losses': []}
        
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return None
    
    def find_checkpoints(self) -> list:
        """Find all checkpoints for this model."""
        pattern = os.path.join(self.checkpoint_dir, f'{self.model_name}_checkpoint_*.pth')
        checkpoints = glob.glob(pattern)
        return sorted(checkpoints)
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint."""
        checkpoints = self.find_checkpoints()
        if checkpoints:
            return max(checkpoints, key=os.path.getmtime)
        return None
    
    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, f'best_{self.model_name}.pth')
        return best_path if os.path.exists(best_path) else None