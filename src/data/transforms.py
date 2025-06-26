"""
Data transforms for the acne dataset.
"""
import numpy as np
from monai import transforms


def create_transforms(img_size: int, apply_augmentation: bool = True):
    """Create training and validation transforms."""
    
    # Base transforms (common for train and val)
    base_transforms = [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        transforms.Resized(
            keys=["image"],
            spatial_size=(img_size, img_size),
            mode="bilinear",
        ),
    ]
    
    # Augmentation transforms
    augmentation_transforms = []
    if apply_augmentation:
        augmentation_transforms = [
            transforms.Rotated(
                keys=["image"],
                angle=np.pi/2,  # 90 degrees
                keep_size=True,
            ),
            transforms.Flipd(
                keys=["image"],
                spatial_axis=1,  # horizontal flip
            ),
        ]
    
    # Final transforms
    final_transforms = [
        transforms.EnsureTyped(keys=["image"]),
    ]
    
    # Combine all transforms
    all_transforms = base_transforms + augmentation_transforms + final_transforms
    
    train_transforms = transforms.Compose(all_transforms)
    val_transforms = transforms.Compose(all_transforms)  # Same transforms for consistency
    
    return train_transforms, val_transforms