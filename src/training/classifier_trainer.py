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
from utils.logging import setup_logging, ExperimentLogger
from utils.config_schemas import ClassifierTrainingConfig


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

        print(f"📦 Initializing Classifier Trainer with config: {config}")
        
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
        
        # Setup logging
        self.logger = setup_logging(
            experiment_dir=config.experiment_dir,
            experiment_name=f"classifier_{int(time.time())}",
            use_wandb=config.use_wandb,
            use_tensorboard=False,
            wandb_project=config.wandb_project,
            config=config.__dict__
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
        
        # Log initial info using correct method names
        if hasattr(self.logger, 'logger'):
            # Get model info manually
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            self.logger.logger.info(f"📊 Classifier Model Information:")
            self.logger.logger.info(f"   Total parameters: {total_params:,}")
            self.logger.logger.info(f"   Trainable parameters: {trainable_params:,}")
            self.logger.logger.info(f"   Model size: {model_size_mb:.1f} MB")
            self.logger.logger.info(f"   Classes: {self.model.config.out_channels} (acne severity levels)")
            
            self.logger.logger.info(f"🖥️ Device Information:")
            self.logger.logger.info(f"   Device: {self.device}")
            if self.device.type == 'cuda':
                self.logger.logger.info(f"   GPU Name: {torch.cuda.get_device_name(self.device)}")
                self.logger.logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
            
            self.logger.logger.info(f"📊 Dataset Information:")
            self.logger.logger.info(f"   Training samples: {len(train_loader.dataset):,}")
            self.logger.logger.info(f"   Validation samples: {len(val_loader.dataset):,}")
            self.logger.logger.info(f"   Batch size: {self.config.batch_size}")
            
            self.logger.logger.log_training_start(self.config.n_epochs)
        
        total_start = time.time()
        
        for epoch in range(self.start_epoch, self.start_epoch + self.config.n_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)
            self.epoch_loss_list.append(train_loss)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch + 1}/{self.start_epoch + self.config.n_epochs}")
            print(f"  Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
            
            # Validation
            val_loss = None
            val_accuracy = None
            class_accuracies = None
            
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
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
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
                    
                    # Log best checkpoint
                    if hasattr(self.logger, 'logger'):
                        self.logger.logger.log_checkpoint(checkpoint_path, epoch, is_best=True)
            
            # Log epoch metrics
            if hasattr(self.logger, 'log_epoch'):
                train_metrics = {"loss": train_loss, "accuracy": train_accuracy}
                val_metrics = None
                if val_loss is not None:
                    val_metrics = {"loss": val_loss, "accuracy": val_accuracy}
                
                self.logger.log_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    lr=self.optimizer.param_groups[0]['lr'],
                    epoch_time=epoch_time
                )
                
                # Log per-class accuracies if available
                if class_accuracies is not None and hasattr(self.logger, 'log_metrics'):
                    class_metrics = {}
                    for i, acc in enumerate(class_accuracies):
                        class_metrics[f"val/class_{i}_accuracy"] = acc
                    self.logger.log_metrics(class_metrics, step=epoch)
                    
            elif hasattr(self.logger, 'log_metrics'):
                # Alternative logging method
                metrics = {
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch_time": epoch_time
                }
                if val_loss is not None:
                    metrics["val/loss"] = val_loss
                    metrics["val/accuracy"] = val_accuracy
                
                self.logger.log_metrics(metrics, step=epoch)
                
                # Log per-class accuracies
                if class_accuracies is not None:
                    class_metrics = {}
                    for i, acc in enumerate(class_accuracies):
                        class_metrics[f"val/class_{i}_accuracy"] = acc
                    self.logger.log_metrics(class_metrics, step=epoch)
            
            # Log GPU memory usage periodically
            if (epoch + 1) % 10 == 0 and hasattr(self.logger, 'logger'):  # Every 10 epochs
                if self.device.type == 'cuda':
                    try:
                        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                        
                        self.logger.logger.info(f"💾 GPU Memory Usage:")
                        self.logger.logger.info(f"   Allocated: {allocated:.2f} GB")
                        self.logger.logger.info(f"   Reserved: {reserved:.2f} GB")
                        self.logger.logger.info(f"   Total: {total:.2f} GB")
                        self.logger.logger.info(f"   Utilization: {(reserved/total)*100:.1f}%")
                    except Exception as e:
                        print(f"⚠️ Could not log GPU memory: {e}")
            
            # Save regular checkpoints
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
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
                
                # Log checkpoint info
                if hasattr(self.logger, 'logger'):
                    self.logger.logger.log_checkpoint(checkpoint_path, epoch, is_best=False)
        
        total_time = time.time() - total_start
        
        if hasattr(self.logger, 'logger'):
            self.logger.logger.log_training_end(self.config.n_epochs)
        
        print(f"🎉 Classifier training completed in {total_time:.2f}s")
        print(f"🌟 Best validation loss: {self.best_val_loss:.4f}")
        print(f"🎯 Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
        # Save final model
        final_path = self.checkpoint_manager.save_final_model(self.model.model)
        
        # Finish logging
        if hasattr(self.logger, 'finish'):
            self.logger.finish()
        
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