"""Basic tests for data module."""

import pytest
import torch
from src.data import CIFAR10DataModule


def test_cifar10_datamodule_init():
    """Test CIFAR10DataModule initialization."""
    dm = CIFAR10DataModule(data_dir="./data", batch_size=128)
    assert dm.hparams.batch_size == 128
    assert dm.hparams.data_dir == "./data"


def test_cifar10_datamodule_setup():
    """Test CIFAR10DataModule setup creates AL splits."""
    dm = CIFAR10DataModule(
        data_dir="./data",
        batch_size=128,
        al_splits={
            "initial_labeled": 1000,
            "calibration_size": 500,
            "total_train_samples": 50000
        }
    )
    dm.prepare_data()
    dm.setup("fit")
    
    assert len(dm.labeled_idx) == 1000
    assert len(dm.calibration_idx) == 500
    assert len(dm.pool_idx) == 50000 - 1000 - 500


def test_cifar10_datamodule_dataloaders():
    """Test CIFAR10DataModule creates dataloaders."""
    dm = CIFAR10DataModule(data_dir="./data", batch_size=64)
    dm.prepare_data()
    dm.setup("fit")
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    pool_loader = dm.pool_dataloader()
    
    assert train_loader is not None
    assert val_loader is not None
    assert pool_loader is not None
