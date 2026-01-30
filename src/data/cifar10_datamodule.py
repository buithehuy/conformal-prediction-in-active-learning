"""CIFAR-10 DataModule for Active Learning with PyTorch Lightning."""

from typing import Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np


class CIFAR10DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CIFAR-10 with Active Learning support.
    
    This module manages:
    - Initial labeled set
    - Calibration set (for conformal prediction)
    - Unlabeled pool
    - Test set
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
        std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010),
        augmentation: dict = None,
        al_splits: dict = None,
        shuffle: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # Data transforms
        self.transform_train = None
        self.transform_test = None
        
        # Datasets
        self.train_dataset = None
        self.test_dataset = None
        
        # AL indices
        self.labeled_idx: List[int] = []
        self.calibration_idx: List[int] = []
        self.pool_idx: List[int] = []
        
        # AL splits configuration
        al_config = al_splits or {}
        self.initial_labeled = al_config.get('initial_labeled', 5000)
        self.calibration_size = al_config.get('calibration_size', 5000)
        self.total_train_samples = al_config.get('total_train_samples', 50000)
    
    def prepare_data(self):
        """Download CIFAR-10 dataset."""
        datasets.CIFAR10(root=self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.hparams.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets and AL splits."""
        
        # Build transforms
        self._build_transforms()
        
        if stage == "fit" or stage is None:
            # Load full training dataset
            full_train_dataset = datasets.CIFAR10(
                root=self.hparams.data_dir,
                train=True,
                transform=self.transform_train,
                download=False
            )
            
            # Create AL splits if not already initialized
            if not self.labeled_idx:
                self._initialize_al_splits()
            
            # Store the full dataset
            self.train_dataset = full_train_dataset
        
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                root=self.hparams.data_dir,
                train=False,
                transform=self.transform_test,
                download=False
            )
    
    def _build_transforms(self):
        """Build train and test transforms."""
        
        aug_config = self.hparams.augmentation or {}
        
        # Training transforms with augmentation
        transform_list = []
        
        if aug_config.get('random_crop', True):
            transform_list.append(transforms.RandomCrop(32, padding=4))
        
        if aug_config.get('random_horizontal_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if aug_config.get('color_jitter', True):
            transform_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            )
        
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(self.hparams.mean, self.hparams.std))
        
        if aug_config.get('random_erasing', True):
            prob = aug_config.get('erasing_probability', 0.5)
            transform_list.append(
                transforms.RandomErasing(p=prob, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            )
        
        self.transform_train = transforms.Compose(transform_list)
        
        # Test transforms (no augmentation)
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.hparams.mean, self.hparams.std)
        ])
    
    def _initialize_al_splits(self):
        """Initialize Active Learning data splits."""
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create random permutation of indices
        all_indices = np.random.permutation(self.total_train_samples)
        
        # Split into: labeled, calibration, pool
        self.labeled_idx = all_indices[:self.initial_labeled].tolist()
        self.calibration_idx = all_indices[
            self.initial_labeled:self.initial_labeled + self.calibration_size
        ].tolist()
        self.pool_idx = all_indices[
            self.initial_labeled + self.calibration_size:
        ].tolist()
        
        print(f"AL splits initialized:")
        print(f"  Labeled: {len(self.labeled_idx)}")
        print(f"  Calibration: {len(self.calibration_idx)}")
        print(f"  Pool: {len(self.pool_idx)}")
    
    def add_to_labeled(self, indices: List[int]):
        """
        Add samples from pool to labeled set.
        
        Args:
            indices: Pool indices to add to labeled set
        """
        # Remove from pool and add to labeled
        for idx in indices:
            if idx in self.pool_idx:
                self.pool_idx.remove(idx)
                self.labeled_idx.append(idx)
        
        print(f"Added {len(indices)} samples to labeled set")
        print(f"  Labeled: {len(self.labeled_idx)}")
        print(f"  Pool: {len(self.pool_idx)}")
    
    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for labeled training set."""
        labeled_dataset = Subset(self.train_dataset, self.labeled_idx)
        return DataLoader(
            labeled_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for calibration set (used as validation)."""
        calibration_dataset = Subset(self.train_dataset, self.calibration_idx)
        return DataLoader(
            calibration_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False
        )
    
    def pool_dataloader(self) -> DataLoader:
        """Return DataLoader for unlabeled pool."""
        # Use test transform for pool (no augmentation for uncertainty estimation)
        pool_dataset = datasets.CIFAR10(
            root=self.hparams.data_dir,
            train=True,
            transform=self.transform_test,  # No augmentation
            download=False
        )
        pool_subset = Subset(pool_dataset, self.pool_idx)
        
        return DataLoader(
            pool_subset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False
        )
    
    def get_pool_indices(self) -> List[int]:
        """Get current pool indices."""
        return self.pool_idx.copy()
    
    def get_labeled_indices(self) -> List[int]:
        """Get current labeled indices."""
        return self.labeled_idx.copy()
    
    def get_calibration_indices(self) -> List[int]:
        """Get calibration indices."""
        return self.calibration_idx.copy()
