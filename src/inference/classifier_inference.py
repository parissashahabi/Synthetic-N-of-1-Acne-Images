"""
Classifier model inference utilities.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union, Tuple
from torch.cuda.amp import autocast
from pathlib import Path
import torch.nn.functional as F
from PIL import Image

from models.classifier import ClassifierModel
from utils.config_schemas import ClassifierModelConfig
from utils.checkpoints import CheckpointManager
from data.transforms import create_transforms


class ClassifierInference:
    """Inference engine for trained classifier models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        model_config: Optional[ClassifierModelConfig] = None
    ):
        """
        Initialize classifier inference engine.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
            model_config: Model configuration (if None, uses default)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Setup configuration
        self.model_config = model_config or ClassifierModelConfig()
        
        # Class names
        self.class_names = [f"Severity {i}" for i in range(self.model_config.out_channels)]
        
        # Initialize model
        self.model = ClassifierModel(self.model_config)
        self.model.to(self.device)
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Setup transforms
        _, self.transforms = create_transforms(img_size=128, apply_augmentation=False)
        
        print(f"🎯 Classifier inference ready on {self.device}")
        print(f"   Classes: {self.class_names}")
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.dirname(self.checkpoint_path),
            model_name="classifier"
        )
        
        checkpoint = checkpoint_manager.load_checkpoint(
            self.checkpoint_path,
            self.model.model,
            device=self.device
        )
        
        if checkpoint is None:
            raise ValueError(f"Failed to load checkpoint: {self.checkpoint_path}")
        
        self.model.model.eval()
        print(f"✅ Loaded checkpoint: {os.path.basename(self.checkpoint_path)}")
    
    def predict(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        return_probabilities: bool = True,
        apply_noise: bool = False,
        noise_level: float = 0.1
    ) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Predict acne severity for a single image.
        
        Args:
            image: Input image (path, PIL Image, tensor, or numpy array)
            return_probabilities: Whether to return class probabilities
            apply_noise: Whether to apply noise during inference
            noise_level: Noise level for inference (0.0 = no noise, 1.0 = max noise)
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Create timesteps for noise (if applicable)
        if apply_noise:
            max_timesteps = 1000
            timesteps = torch.randint(
                0, int(max_timesteps * noise_level), (1,)
            ).to(self.device)
        else:
            timesteps = torch.zeros(1, dtype=torch.long).to(self.device)
        
        # Run inference
        with torch.no_grad():
            with autocast(enabled=True):
                outputs = self.model.model(processed_image, timesteps)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1)
        
        # Convert to numpy
        probabilities_np = probabilities.cpu().numpy()[0]
        predicted_class_int = predicted_class.cpu().item()
        confidence = probabilities_np[predicted_class_int]
        
        result = {
            'predicted_class': predicted_class_int,
            'predicted_class_name': self.class_names[predicted_class_int],
            'confidence': float(confidence)
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities_np
            result['class_probabilities'] = {
                name: float(prob) for name, prob in zip(self.class_names, probabilities_np)
            }
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Image.Image, torch.Tensor, np.ndarray]],
        batch_size: int = 32,
        return_probabilities: bool = True,
        apply_noise: bool = False,
        noise_level: float = 0.1
    ) -> List[Dict[str, Union[int, float, np.ndarray]]]:
        """
        Predict acne severity for a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            apply_noise: Whether to apply noise during inference
            noise_level: Noise level for inference
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = self._predict_batch_internal(
                batch_images, return_probabilities, apply_noise, noise_level
            )
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(
        self,
        batch_images: List[Union[str, Image.Image, torch.Tensor, np.ndarray]],
        return_probabilities: bool,
        apply_noise: bool,
        noise_level: float
    ) -> List[Dict[str, Union[int, float, np.ndarray]]]:
        """Internal batch prediction method."""
        # Preprocess all images
        processed_batch = []
        for image in batch_images:
            processed_image = self._preprocess_image(image)
            processed_batch.append(processed_image)
        
        # Stack into batch
        batch_tensor = torch.cat(processed_batch, dim=0)
        
        # Create timesteps
        if apply_noise:
            max_timesteps = 1000
            timesteps = torch.randint(
                0, int(max_timesteps * noise_level), (len(batch_images),)
            ).to(self.device)
        else:
            timesteps = torch.zeros(len(batch_images), dtype=torch.long).to(self.device)
        
        # Run inference
        with torch.no_grad():
            with autocast(enabled=True):
                outputs = self.model.model(batch_tensor, timesteps)
                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)
        
        # Convert results
        results = []
        for i in range(len(batch_images)):
            probs_np = probabilities[i].cpu().numpy()
            pred_class = predicted_classes[i].cpu().item()
            confidence = probs_np[pred_class]
            
            result = {
                'predicted_class': pred_class,
                'predicted_class_name': self.class_names[pred_class],
                'confidence': float(confidence)
            }
            
            if return_probabilities:
                result['probabilities'] = probs_np
                result['class_probabilities'] = {
                    name: float(prob) for name, prob in zip(self.class_names, probs_np)
                }
            
            results.append(result)
        
        return results
    
    def _preprocess_image(self, image: Union[str, Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference."""
        # Handle different input types
        if isinstance(image, str):
            # Load from file path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # If already a tensor, ensure correct format
            if image.dim() == 2:  # Grayscale
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif image.dim() == 3 and image.shape[0] == 1:  # Grayscale with channel dim
                image = image.repeat(3, 1, 1)
            elif image.dim() == 3 and image.shape[2] == 3:  # HWC format
                image = image.permute(2, 0, 1)
            
            # Convert to PIL for consistent preprocessing
            if image.max() <= 1.0:
                image = (image * 255).byte()
            image = Image.fromarray(image.permute(1, 2, 0).numpy())
        
        # Apply transforms
        sample = {"image": image, "label": 0}  # Dummy label
        transformed = self.transforms(sample)
        
        # Add batch dimension and move to device
        return transformed["image"].unsqueeze(0).to(self.device)
    
    def analyze_image(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        save_path: Optional[str] = None,
        show_gradcam: bool = False
    ) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Analyze an image with detailed visualization.
        
        Args:
            image: Input image
            save_path: Path to save analysis visualization
            show_gradcam: Whether to show Grad-CAM visualization
        
        Returns:
            Analysis results
        """
        # Get prediction
        result = self.predict(image, return_probabilities=True)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2 if not show_gradcam else 3, figsize=(12, 4))
        
        # Original image
        if isinstance(image, str):
            display_image = Image.open(image)
        elif isinstance(image, Image.Image):
            display_image = image
        else:
            # Convert tensor/array to PIL for display
            processed = self._preprocess_image(image)
            display_image = processed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            display_image = np.clip(display_image, 0, 1)
        
        axes[0].imshow(display_image)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # Prediction probabilities
        probabilities = result['probabilities']
        colors = ['red' if i == result['predicted_class'] else 'skyblue' 
                 for i in range(len(self.class_names))]
        
        bars = axes[1].bar(self.class_names, probabilities, color=colors)
        axes[1].set_title(f"Predicted: {result['predicted_class_name']}\nConfidence: {result['confidence']:.3f}")
        axes[1].set_ylabel('Probability')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Grad-CAM visualization (if requested)
        if show_gradcam:
            try:
                gradcam_result = self._generate_gradcam(image, result['predicted_class'])
                axes[2].imshow(gradcam_result)
                axes[2].set_title("Grad-CAM")
                axes[2].axis('off')
            except Exception as e:
                axes[2].text(0.5, 0.5, f"Grad-CAM failed:\n{str(e)}", 
                           ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title("Grad-CAM (Failed)")
                axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Analysis saved: {save_path}")
        
        plt.show()
        
        return result
    
    def _generate_gradcam(self, image: Union[str, Image.Image, torch.Tensor, np.ndarray], target_class: int) -> np.ndarray:
        """Generate Grad-CAM visualization for the prediction."""
        # This is a simplified Grad-CAM implementation
        # For a full implementation, you might want to use libraries like pytorch-gradcam
        
        processed_image = self._preprocess_image(image)
        processed_image.requires_grad_(True)
        
        # Forward pass
        timesteps = torch.zeros(1, dtype=torch.long).to(self.device)
        outputs = self.model.model(processed_image, timesteps)
        
        # Backward pass
        self.model.model.zero_grad()
        outputs[0, target_class].backward()
        
        # Get gradients and activations (simplified)
        gradients = processed_image.grad.data
        
        # Create heatmap (very simplified)
        heatmap = torch.mean(torch.abs(gradients), dim=1).squeeze()
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), 
                               size=(128, 128), mode='bilinear').squeeze()
        
        # Normalize and convert to numpy
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = heatmap.cpu().numpy()
        
        # Create colored heatmap
        import matplotlib.cm as cm
        colored_heatmap = cm.jet(heatmap)[:, :, :3]
        
        return colored_heatmap
    
    def evaluate_uncertainty(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        num_samples: int = 10,
        noise_levels: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate prediction uncertainty using noise-based sampling.
        
        Args:
            image: Input image
            num_samples: Number of samples for uncertainty estimation
            noise_levels: List of noise levels to test
        
        Returns:
            Uncertainty metrics
        """
        if noise_levels is None:
            noise_levels = np.linspace(0.0, 0.5, num_samples)
        
        predictions = []
        probabilities = []
        
        for noise_level in noise_levels:
            result = self.predict(
                image,
                return_probabilities=True,
                apply_noise=True,
                noise_level=noise_level
            )
            predictions.append(result['predicted_class'])
            probabilities.append(result['probabilities'])
        
        # Calculate uncertainty metrics
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Prediction consistency
        most_common_pred = np.bincount(predictions).argmax()
        consistency = np.mean(predictions == most_common_pred)
        
        # Entropy-based uncertainty
        mean_probs = np.mean(probabilities, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
        
        # Variance in probabilities
        prob_variance = np.mean(np.var(probabilities, axis=0))
        
        return {
            'consistency': float(consistency),
            'entropy': float(entropy),
            'probability_variance': float(prob_variance),
            'most_common_prediction': int(most_common_pred),
            'prediction_std': float(np.std(predictions))
        }
    
    def compare_with_groundtruth(
        self,
        image: Union[str, Image.Image, torch.Tensor, np.ndarray],
        true_label: int,
        save_path: Optional[str] = None
    ) -> Dict[str, Union[bool, float, int]]:
        """
        Compare prediction with ground truth label.
        
        Args:
            image: Input image
            true_label: Ground truth label
            save_path: Path to save comparison visualization
        
        Returns:
            Comparison results
        """
        result = self.predict(image, return_probabilities=True)
        
        is_correct = result['predicted_class'] == true_label
        true_class_confidence = result['probabilities'][true_label]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        if isinstance(image, str):
            display_image = Image.open(image)
        elif isinstance(image, Image.Image):
            display_image = image
        else:
            processed = self._preprocess_image(image)
            display_image = processed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            display_image = np.clip(display_image, 0, 1)
        
        axes[0].imshow(display_image)
        status = "✓ Correct" if is_correct else "✗ Incorrect"
        axes[0].set_title(f"Ground Truth: {self.class_names[true_label]}\n{status}")
        axes[0].axis('off')
        
        # Probabilities comparison
        probabilities = result['probabilities']
        colors = ['green' if i == true_label else 'red' if i == result['predicted_class'] else 'skyblue' 
                 for i in range(len(self.class_names))]
        
        bars = axes[1].bar(self.class_names, probabilities, color=colors)
        axes[1].set_title(f"Predicted: {result['predicted_class_name']}\nTrue class confidence: {true_class_confidence:.3f}")
        axes[1].set_ylabel('Probability')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison saved: {save_path}")
        
        plt.show()
        
        comparison_result = {
            'is_correct': is_correct,
            'predicted_class': result['predicted_class'],
            'true_class': true_label,
            'confidence': result['confidence'],
            'true_class_confidence': float(true_class_confidence),
            'prediction_error': abs(result['predicted_class'] - true_label)
        }
        
        return comparison_result
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        
        return {
            'checkpoint_path': self.checkpoint_path,
            'device': str(self.device),
            'model_config': self.model_config.__dict__,
            'num_classes': self.model_config.out_channels,
            'class_names': self.class_names,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


def load_classifier_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> ClassifierInference:
    """
    Convenience function to load a classifier model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        ClassifierInference instance
    """
    return ClassifierInference(checkpoint_path, device)


def batch_predict_from_folder(
    checkpoint_path: str,
    image_folder: str,
    output_csv: str,
    batch_size: int = 32,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> str:
    """
    Predict acne severity for all images in a folder and save results to CSV.
    
    Args:
        checkpoint_path: Path to classifier checkpoint
        image_folder: Folder containing images
        output_csv: Path to save CSV results
        batch_size: Batch size for processing
        image_extensions: Supported image extensions
    
    Returns:
        Path to saved CSV file
    """
    import pandas as pd
    
    # Initialize classifier
    classifier = ClassifierInference(checkpoint_path)
    
    # Find all images
    image_folder = Path(image_folder)
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_folder.glob(f"*{ext}"))
        image_paths.extend(image_folder.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_paths)} images in {image_folder}")
    
    # Process in batches
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_results = classifier.predict_batch(
            [str(path) for path in batch_paths],
            batch_size=len(batch_paths),
            return_probabilities=True
        )
        
        # Add file paths to results
        for path, result in zip(batch_paths, batch_results):
            result['image_path'] = str(path)
            result['image_name'] = path.name
            results.append(result)
        
        print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
    
    # Convert to DataFrame
    df_data = []
    for result in results:
        row = {
            'image_path': result['image_path'],
            'image_name': result['image_name'],
            'predicted_class': result['predicted_class'],
            'predicted_class_name': result['predicted_class_name'],
            'confidence': result['confidence']
        }
        
        # Add individual class probabilities
        for class_name, prob in result['class_probabilities'].items():
            row[f'prob_{class_name.lower().replace(" ", "_")}'] = prob
        
        df_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(df_data)
    df.to_csv(output_csv, index=False)
    
    print(f"💾 Results saved to: {output_csv}")
    print(f"📊 Summary: {len(df)} images processed")
    print(f"   Class distribution:")
    for class_name in classifier.class_names:
        count = df[df['predicted_class_name'] == class_name].shape[0]
        percentage = count / len(df) * 100
        print(f"   {class_name}: {count} images ({percentage:.1f}%)")
    
    return output_csv