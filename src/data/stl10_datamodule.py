"""STL-10 DataModule for Active Learning with PyTorch Lightning."""

from typing import Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np


class STL10DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for STL-10 with Active Learning support.
    
    STL-10 has 10 classes with 500 labeled training images, 800 test images,
    and 100,000 unlabeled images (96x96 RGB).
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        mean: Tuple[float, float, float] = (0.4467, 0.4398, 0.4066),
        std: Tuple[float, float, float] = (0.2603, 0.2566, 0.2713),
        augmentation: dict = None,
        al_splits: dict = None,
        shuffle: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.transform_train = None
        self.transform_test = None
        
        self.train_dataset = None
        self.test_dataset = None
        
        self.labeled_idx: List[int] = []
        self.calibration_idx: List[int] = []
        self.pool_idx: List[int] = []
        
        al_config = al_splits or {}
        self.initial_labeled = al_config.get('initial_labeled', 1000)
        self.calibration_size = al_config.get('calibration_size', 1000)
        self.total_train_samples = al_config.get('total_train_samples', 5000)
    
    def prepare_data(self):
        """Download STL-10 dataset."""
        datasets.STL10(root=self.hparams.data_dir, split='train', download=True)
        datasets.STL10(root=self.hparams.data_dir, split='test', download=True)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets and AL splits."""
        self._build_transforms()
        
        if stage == "fit" or stage is None:
            full_train_dataset = datasets.STL10(
                root=self.hparams.data_dir,
                split='train',
                transform=self.transform_train,
                download=False
            )
            
            if not self.labeled_idx:
                self._initialize_al_splits()
            
            self.train_dataset = full_train_dataset
        
        if stage == "test" or stage is None:
            self.test_dataset = datasets.STL10(
                root=self.hparams.data_dir,
                split='test',
                transform=self.transform_test,
                download=False
            )
    
    def _build_transforms(self):
        """Build train and test transforms for STL-10 (96x96 images)."""
        aug_config = self.hparams.augmentation or {}
        
        transform_list = []
        if aug_config.get('random_crop', True):
            crop_size = aug_config.get('crop_size', 96)
            padding = aug_config.get('padding', 12)
            transform_list.append(transforms.RandomCrop(crop_size, padding=padding))
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
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.hparams.mean, self.hparams.std)
        ])
    
    def _initialize_al_splits(self):
        """Initialize Active Learning data splits."""
        np.random.seed(42)
        all_indices = np.random.permutation(self.total_train_samples)
        
        self.labeled_idx = all_indices[:self.initial_labeled].tolist()
        self.calibration_idx = all_indices[
            self.initial_labeled:self.initial_labeled + self.calibration_size
        ].tolist()
        self.pool_idx = all_indices[
            self.initial_labeled + self.calibration_size:
        ].tolist()
        
        print(f"STL-10 AL splits initialized:")
        print(f"  Labeled: {len(self.labeled_idx)}")
        print(f"  Calibration: {len(self.calibration_idx)}")
        print(f"  Pool: {len(self.pool_idx)}")
    
    def add_to_labeled(self, indices: List[int]):
        """Add samples from pool to labeled set."""
        for idx in indices:
            if idx in self.pool_idx:
                self.pool_idx.remove(idx)
                self.labeled_idx.append(idx)
        
        print(f"Added {len(indices)} samples to labeled set")
        print(f"  Labeled: {len(self.labeled_idx)}, Pool: {len(self.pool_idx)}")
    
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
        """Return DataLoader for calibration set."""
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
        pool_dataset = datasets.STL10(
            root=self.hparams.data_dir,
            split='train',
            transform=self.transform_test,
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
