#!/usr/bin/env python3
"""
Evaluation script for trained models - reads configuration from config.yaml only.
"""
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import AcneDataset
from data.transforms import create_transforms
from models.classifier import ClassifierModel
from utils.checkpoints import CheckpointManager
from utils.config_reader import ConfigReader


def evaluate_classifier(model, test_loader, device, num_classes=4):
    """Evaluate classifier performance."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("🔍 Running evaluation...")
    
    with torch.no_grad():
        for data in test_loader:
            images = data["image"].to(device)
            labels = data["label"].to(device)
            
            # Use minimal noise for evaluation (clean images)
            timesteps = torch.zeros(len(images), dtype=torch.long).to(device)
            
            # Get predictions
            outputs = model.model(images, timesteps)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_labels) * 100
    
    # Per-class accuracy
    class_accuracies = []
    for i in range(num_classes):
        class_mask = all_labels == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_predictions[class_mask] == all_labels[class_mask]) * 100
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved: {save_path}")
    
    plt.show()
    return cm


def plot_class_accuracies(class_accuracies, class_names, save_path=None):
    """Plot per-class accuracies."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Acne Severity Level', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Class accuracies plot saved: {save_path}")
    
    plt.show()


def plot_probability_distribution(probabilities, labels, class_names, save_path=None):
    """Plot probability distribution for each class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i in range(len(class_names)):
        # Get probabilities for this class
        class_probs = probabilities[labels == i, i]
        
        if len(class_probs) > 0:
            axes[i].hist(class_probs, bins=20, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[i].set_title(f'{class_names[i]} - Confidence Distribution', fontweight='bold')
            axes[i].set_xlabel('Predicted Probability')
            axes[i].set_ylabel('Frequency')
            axes[i].set_xlim(0, 1)
            axes[i].grid(alpha=0.3)
            
            # Add mean line
            mean_prob = np.mean(class_probs)
            axes[i].axvline(mean_prob, color='red', linestyle='--', 
                           label=f'Mean: {mean_prob:.3f}')
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, 'No samples', ha='center', va='center', 
                        transform=axes[i].transAxes, fontsize=14)
            axes[i].set_title(f'{class_names[i]} - No Data')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Probability distribution saved: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained classifier')
    
    # Essential arguments only
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to classifier checkpoint')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Load configuration from YAML
    print(f"📖 Loading configuration from: {args.config}")
    try:
        config_reader = ConfigReader(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    # Get configurations
    model_config = config_reader.get_classifier_model_config()
    data_config = config_reader.get_data_config()
    eval_config = config_reader.get_evaluation_config()
    
    # Setup device
    device_setting = eval_config.get('device', 'auto')
    if device_setting == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    
    print(f"🖥️ Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration summary
    print(f"\n📋 Evaluation Configuration:")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data directory: {data_config.dataset_path}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Batch size: {eval_config.get('batch_size', 32)}")
    print(f"   Test split: {eval_config.get('test_split', 0.2)}")
    
    # Setup data
    print("📁 Setting up test data...")
    
    # Create transforms (no augmentation for evaluation)
    _, test_transforms = create_transforms(img_size=128, apply_augmentation=False)
    
    # Create dataset
    full_dataset = AcneDataset(
        data_dir=data_config.dataset_path,
        transform=test_transforms,
        severity_levels=data_config.severity_levels
    )
    
    # Split data (use the test portion)
    test_split = eval_config.get('test_split', 0.2)
    train_size = int((1 - test_split) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    _, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create test loader
    batch_size = eval_config.get('batch_size', 32)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"📊 Test dataset: {len(test_dataset)} samples")
    print(f"📊 Test batches: {len(test_loader)}")
    
    # Load model
    print("🏗️ Loading classifier model...")
    
    model = ClassifierModel(model_config)
    model.to(device)
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=os.path.dirname(args.checkpoint),
        model_name="classifier"
    )
    
    checkpoint = checkpoint_manager.load_checkpoint(
        args.checkpoint, model.model, device=device
    )
    
    if checkpoint is None:
        print("❌ Failed to load checkpoint")
        return 1
    
    # Run evaluation
    print("🚀 Starting evaluation...")
    
    results = evaluate_classifier(model, test_loader, device, 
                                 num_classes=model_config.out_channels)
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print("\nPer-Class Accuracies:")
    
    class_names = [f"Severity {i}" for i in range(model_config.out_channels)]
    for i, (name, acc) in enumerate(zip(class_names, results['class_accuracies'])):
        print(f"  {name}: {acc:.2f}%")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 60)
    report = classification_report(results['labels'], results['predictions'], 
                                 target_names=class_names)
    print(report)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("CLASSIFIER EVALUATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n\n")
        f.write("Per-Class Accuracies:\n")
        for i, (name, acc) in enumerate(zip(class_names, results['class_accuracies'])):
            f.write(f"  {name}: {acc:.2f}%\n")
        f.write(f"\nDetailed Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(report)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['labels'], results['predictions'], 
                         class_names, cm_path)
    
    # Per-class accuracies
    acc_path = os.path.join(args.output_dir, 'class_accuracies.png')
    plot_class_accuracies(results['class_accuracies'], class_names, acc_path)
    
    # Probability distributions
    prob_path = os.path.join(args.output_dir, 'probability_distributions.png')
    plot_probability_distribution(results['probabilities'], results['labels'], 
                                 class_names, prob_path)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 All results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())