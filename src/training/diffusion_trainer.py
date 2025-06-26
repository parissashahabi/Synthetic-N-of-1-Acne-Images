"""
Diffusion model trainer.
"""
import os
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models.diffusion import DiffusionModel
from utils.checkpoints import CheckpointManager
from utils.visualization import generate_sample_image, save_generation_process
from configs.diffusion_config import DiffusionTrainingConfig


class DiffusionTrainer:
    """Trainer class for diffusion models."""
    
    def __init__(
        self, 
        model: DiffusionModel, 
        config: DiffusionTrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.model.parameters(),
            lr=config.learning_rate
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(config.experiment_dir, "checkpoints"),
            model_name="diffusion"
        )
        
        # Training state
        self.epoch_loss_list = []
        self.val_epoch_loss_list = []
        self.start_epoch = 0
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.n_epochs}"
        )
        
        for step, data in enumerate(progress_bar):
            images = data["image"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.config.num_train_timesteps, (len(images),)
            ).to(self.device)
            
            # Training step with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                # Generate random noise
                noise = torch.randn_like(images).to(self.device)
                
                # Get model prediction
                noise_pred = self.model.inferer(
                    inputs=images,
                    diffusion_model=self.model.model,
                    noise=noise,
                    timesteps=timesteps
                )
                
                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), noise.float())
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        return epoch_loss / (step + 1)
    
    def validate(self, val_loader) -> float:
        """Validate the model."""
        self.model.model.eval()
        val_epoch_loss = 0
        
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                images = data["image"].to(self.device)
                timesteps = torch.randint(
                    0, self.config.num_train_timesteps, (len(images),)
                ).to(self.device)
                
                with autocast(enabled=self.config.mixed_precision):
                    noise = torch.randn_like(images).to(self.device)
                    noise_pred = self.model.inferer(
                        inputs=images,
                        diffusion_model=self.model.model,
                        noise=noise,
                        timesteps=timesteps
                    )
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())
                
                val_epoch_loss += val_loss.item()
        
        return val_epoch_loss / (step + 1)
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"🚀 Starting diffusion training for {self.config.n_epochs} epochs")
        
        total_start = time.time()
        
        for epoch in range(self.start_epoch, self.start_epoch + self.config.n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            self.epoch_loss_list.append(train_loss)
            
            print(f"Epoch {epoch+1}/{self.start_epoch + self.config.n_epochs}, "
                  f"Training loss: {train_loss:.4f}")
            
            # Validation
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate(val_loader)
                self.val_epoch_loss_list.append(val_loss)
                
                print(f"Epoch {epoch+1}, Validation loss: {val_loss:.4f}")
                
                # Generate sample image
                if (epoch + 1) % self.config.sample_interval == 0:
                    sample_dir = os.path.join(self.config.experiment_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    sample_path = os.path.join(
                        sample_dir, f'generated_image_epoch_{epoch+1}.png'
                    )
                    
                    intermediates = generate_sample_image(
                        self.model.model, self.model.inferer, self.model.scheduler,
                        self.device, self.config.img_size, sample_path, epoch
                    )
                    
                    # Save generation process
                    if (epoch + 1) % self.config.process_interval == 0:
                        process_path = os.path.join(
                            sample_dir, f'generation_process_epoch_{epoch+1}.png'
                        )
                        save_generation_process(intermediates, process_path, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_losses=self.epoch_loss_list,
                    val_losses=self.val_epoch_loss_list,
                    config=self.config
                )
        
        total_time = time.time() - total_start
        print(f"🎉 Training completed in {total_time:.2f}s")
        
        # Save final model
        final_path = self.checkpoint_manager.save_final_model(self.model.model)
        return final_path
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model.model, self.optimizer, self.device
        )
        
        if checkpoint:
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.epoch_loss_list = checkpoint.get('train_losses', [])
            self.val_epoch_loss_list = checkpoint.get('val_losses', [])
            
            print(f"✅ Resumed from epoch {self.start_epoch}")
            return True
        return False