"""
Visualization utilities for training and inference.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast


def generate_sample_image(model, inferer, scheduler, device, img_size, save_path, epoch):
    """Generate and save a sample image."""
    model.eval()
    
    # Create noise with correct image size
    noise = torch.randn((1, 3, img_size, img_size)).to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    
    with torch.no_grad():
        with autocast(enabled=True):
            image, intermediates = inferer.sample(
                input_noise=noise,
                diffusion_model=model,
                scheduler=scheduler,
                save_intermediates=True,
                intermediate_steps=100
            )
    
    # Save generated image
    plt.figure(figsize=(8, 8))
    if image.shape[1] == 3:  # RGB
        img_display = image[0].permute(1, 2, 0).cpu().detach().numpy()
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
    else:  # Grayscale
        plt.imshow(image[0, 0].cpu().detach().numpy(), cmap="gray")
    
    plt.axis("off")
    plt.title(f"Generated Image at Epoch {epoch+1}", fontsize=12)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"🖼️ Sample image saved: {save_path}")
    return intermediates


def save_generation_process(intermediates, save_path, epoch):
    """Save the generation process visualization."""
    if not intermediates:
        return
    
    chain = torch.cat(intermediates, dim=-1)
    
    plt.figure(figsize=(20, 4))
    if chain.shape[1] == 3:  # RGB
        chain_display = chain[0].permute(1, 2, 0).cpu().detach().numpy()
        chain_display = np.clip(chain_display, 0, 1)
        plt.imshow(chain_display)
    else:  # Grayscale
        plt.imshow(chain[0, 0].cpu().detach().numpy(), cmap="gray")
    
    plt.axis("off")
    plt.title(f"Generation Process at Epoch {epoch+1}", fontsize=12)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"🔄 Generation process saved: {save_path}")


def show_batch(loader, num_samples=5, title="Sample Images", save_path=None):
    """Visualize a batch of images from the data loader."""
    data = next(iter(loader))
    images = data["image"][:num_samples]
    labels = data["label"][:num_samples]
    
    fig, axes = plt.subplots(1, len(images), figsize=(3*len(images), 4))
    
    # Handle single image case
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Convert from CHW to HWC for display
        if images.shape[1] == 3:  # RGB
            img_to_show = img.permute(1, 2, 0)
        else:  # Grayscale
            img_to_show = img[0]
        
        # Ensure values are in [0,1] range
        img_to_show = torch.clamp(img_to_show, 0, 1)
        
        axes[i].imshow(img_to_show, cmap="gray" if images.shape[1] == 1 else None)
        axes[i].set_title(f"Severity Level {label}", fontsize=10)
        axes[i].axis("off")
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"💾 Batch visualization saved to: {save_path}")
    
    plt.show()


def plot_learning_curves(train_losses, val_losses, val_accuracies=None, save_path=None):
    """Plot and save learning curves."""
    plt.style.use("default")
    
    if val_accuracies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 
             color="C0", linewidth=2.0, label="Training", alpha=0.8)
    
    if val_losses:
        val_epochs = range(len(train_losses) // len(val_losses), 
                          len(train_losses) + 1, 
                          len(train_losses) // len(val_losses))
        val_epochs = val_epochs[:len(val_losses)]
        ax1.plot(val_epochs, val_losses, 
                 color="C1", linewidth=2.0, label="Validation", alpha=0.8, marker='o')
    
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Learning Curves - Loss", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves (if provided)
    if val_accuracies:
        ax2.plot(val_epochs[:len(val_accuracies)], val_accuracies, 
                 color="C2", linewidth=2.0, label="Validation Accuracy", 
                 alpha=0.8, marker='s')
        
        ax2.set_xlabel("Epochs", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title("Learning Curves - Accuracy", fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add best accuracy annotation
        if val_accuracies:
            best_acc = max(val_accuracies)
            best_acc_idx = val_accuracies.index(best_acc)
            best_acc_epoch = val_epochs[best_acc_idx]
            ax2.annotate(f'Best: {best_acc:.2f}%', 
                        xy=(best_acc_epoch, best_acc),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"📈 Learning curves saved: {save_path}")
    
    plt.show()
    return save_path