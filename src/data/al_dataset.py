"""Active Learning dataset utilities."""

from typing import List
import torch
from torch.utils.data import Dataset


class ActiveLearningDataset(Dataset):
    """
    Wrapper dataset for Active Learning.
    
    Manages subsets of data based on indices.
    """
    
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        """
        Args:
            base_dataset: Base dataset to wrap
            indices: List of indices to include
        """
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map to actual index in base dataset
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]
    
    def add_indices(self, new_indices: List[int]):
        """Add more indices to the dataset."""
        self.indices.extend(new_indices)
    
    def remove_indices(self, remove_indices: List[int]):
        """Remove indices from the dataset."""
        self.indices = [idx for idx in self.indices if idx not in remove_indices]
    
    def get_indices(self) -> List[int]:
        """Get current indices."""
        return self.indices.copy()
