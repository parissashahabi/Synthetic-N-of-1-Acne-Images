"""
Dataset classes for ACNE04 dataset.
"""
import os
import re
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset


class AcneDataset(Dataset):
    """Custom dataset class for ACNE04 dataset with severity levels."""
    
    def __init__(
        self, 
        data_dir: str, 
        transform=None, 
        severity_levels: Optional[List[int]] = None
    ):
        """
        Args:
            data_dir: Directory with the acne04 dataset
            transform: Optional transform to be applied on a sample
            severity_levels: List of severity levels to include.
                           Options: [0, 1, 2, 3, 'all']
                           If None, all levels are included.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Define which severity levels to include
        if severity_levels is None:
            self.severity_folders = [
                "acne0_1024", "acne1_1024", "acne2_1024", "acne3_1024"
            ]
        else:
            self.severity_folders = []
            for level in severity_levels:
                if level == 'all':
                    self.severity_folders.append("all_1024")
                else:
                    self.severity_folders.append(f"acne{level}_1024")
        
        # Collect image paths and labels
        self.image_files = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all image paths and labels from the dataset."""
        for folder in self.severity_folders:
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Folder {folder_path} does not exist.")
                continue
                
            # Extract severity level from folder name
            if 'acne' in folder:
                severity = int(folder.split('_')[0].replace('acne', ''))
            else:
                severity = None  # For "all_1024" folder
                
            # Get all image files in this folder
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                self.image_files.append(file_path)
                
                # Determine severity level
                if severity is None:
                    # Try to extract level from filename
                    match = re.search(r'level(\d+)', file.lower())
                    file_severity = int(match.group(1)) if match else -1
                    self.labels.append(file_severity)
                else:
                    self.labels.append(severity)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        sample = {"image": img_path, "label": label}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def get_statistics(self) -> Dict[int, int]:
        """Get dataset statistics."""
        severity_counts = {}
        for label in self.labels:
            severity_counts[label] = severity_counts.get(label, 0) + 1
        return severity_counts