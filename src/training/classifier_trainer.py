"""
Classifier model trainer.
"""
import os
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.networks.schedulers.ddpm import DDPMScheduler

from models.classifier import ClassifierModel
from utils.checkpoints import CheckpointManager
from configs.classifier_config import ClassifierTrainingConfig


class ClassifierTrainer:
    """Trainer class for classifier models."""
    
    def __init__(
        self, 
        model: ClassifierModel, 
        config: ClassifierTrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler for noise augmentation
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(config.experiment_dir, "checkpoints"),
            model_name="classifier"
        )
        
        # Training state
        self.epoch_loss_list = []
        self.val_epoch_loss_list = []
        self.val_accuracy_list = []
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, train_loader, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.model.train()
        epoch_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Training Epoch {epoch + 1}"
        )
        
        for step, data in enumerate(progress_bar):
            images = data["image"].to(self.device)
            classes = data["label"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Sample random timesteps for noise augmentation
            timesteps = torch.randint(
                0, self.config.noise_timesteps_train, (len(images),)
            ).to(self.device)
            
            # Training step
            with autocast(enabled=self.config.mixed_precision):
                # Generate random noise
                noise = torch.randn_like(images).to(self.device)
                
                # Add noise to input images
                noisy_img = self.scheduler.add_noise(images, noise, timesteps)
                
                # Get classifier prediction
                pred = self.model.model(noisy_img, timesteps)
                
                # Calculate loss
                loss = F.cross_entropy(pred, classes.long())
            
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
            _, predicted = torch.max(pred, 1)
            total_samples += classes.size(0)
            correct_predictions += (predicted == classes).sum().item()
            
            # Update progress bar
            current_acc = 100.0 * correct_predictions / total_samples
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{current_acc:.2f}%"
            })
        
        avg_loss = epoch_loss / (step + 1)
        avg_accuracy = 100.0 * correct_predictions / total_samples
        
        return avg_loss, avg_accuracy
    
    def validate(self, val_loader) -> tuple:
        """Validate the classifier."""
        self.model.model.eval()
        val_loss = 0
        total_samples = 0
        correct_predictions = 0
        class_correct = [0] * self.model.config.out_channels
        class_total = [0] * self.model.config.out_channels
        
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                images = data["image"].to(self.device)
                classes = data["label"].to(self.device)
                
                # Use minimal noise for validation
                timesteps = torch.randint(
                    0, self.config.noise_timesteps_val, (len(images),)
                ).to(self.device)
                
                with autocast(enabled=self.config.mixed_precision):
                    # For validation, use original images (minimal noise)
                    pred = self.model.model(images, timesteps)
                    loss = F.cross_entropy(pred, classes.long(), reduction="mean")
                
                val_loss += loss.item()
                _, predicted = torch.max(pred, 1)
                total_samples += classes.size(0)
                correct_predictions += (predicted == classes).sum().item()
                
                # Per-class accuracy
                for i in range(classes.size(0)):
                    label = classes[i].item()
                    class_total[label] += 1
                    if predicted[i] == classes[i]:
                        class_correct[label] += 1
        
        avg_val_loss = val_loss / (step + 1)
        avg_accuracy = 100.0 * correct_predictions / total_samples
        
        # Calculate per-class accuracies
        class_accuracies = []
        for i in range(self.model.config.out_channels):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        return avg_val_loss, avg_accuracy, class_accuracies
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"🎓 Starting classifier training for {self.config.n_epochs} epochs")
        
        total_start = time.time()
        
        for epoch in range(self.start_epoch, self.start_epoch + self.config.n_epochs):
            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)
            self.epoch_loss_list.append(train_loss)
            
            print(f"Epoch {epoch + 1}/{self.start_epoch + self.config.n_epochs}")
            print(f"  Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
            
            # Validation
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss, val_accuracy, class_accuracies = self.validate(val_loader)
                self.val_epoch_loss_list.append(val_loss)
                self.val_accuracy_list.append(val_accuracy)
                
                # Check if this is the best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_val_accuracy = val_accuracy
                
                print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
                print(f"  Per-class accuracies:")
                for i, acc in enumerate(class_accuracies):
                    print(f"    Severity Level {i}: {acc:.2f}%")
                
                # Save best model
                if is_best:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        train_losses=self.epoch_loss_list,
                        val_losses=self.val_epoch_loss_list,
                        config=self.config,
                        is_best=True,
                        extra_data={
                            'val_accuracies': self.val_accuracy_list,
                            'best_val_loss': self.best_val_loss,
                            'best_val_accuracy': self.best_val_accuracy
                        }
                    )
            
            # Save regular checkpoints
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_losses=self.epoch_loss_list,
                    val_losses=self.val_epoch_loss_list,
                    config=self.config,
                    extra_data={
                        'val_accuracies': self.val_accuracy_list,
                        'best_val_loss': self.best_val_loss,
                        'best_val_accuracy': self.best_val_accuracy
                    }
                )
        
        total_time = time.time() - total_start
        print(f"🎉 Classifier training completed in {total_time:.2f}s")
        print(f"🌟 Best validation loss: {self.best_val_loss:.4f}")
        print(f"🎯 Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
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
            self.val_accuracy_list = checkpoint.get('val_accuracies', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            
            print(f"✅ Resumed from epoch {self.start_epoch}")
            return True
        return False