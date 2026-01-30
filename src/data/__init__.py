"""Data module initialization."""

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.data.cifar100_datamodule import CIFAR100DataModule
from src.data.stl10_datamodule import STL10DataModule
from src.data.svhn_datamodule import SVHNDataModule
from src.data.al_dataset import ActiveLearningDataset

__all__ = [
    "CIFAR10DataModule",
    "CIFAR100DataModule", 
    "STL10DataModule",
    "SVHNDataModule",
    "ActiveLearningDataset"
]
