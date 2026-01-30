"""ResNet model module for PyTorch Lightning."""

from typing import Any, Dict
import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import Accuracy, MaxMetric, MeanMetric


class ResNetModule(pl.LightningModule):
    """
    PyTorch Lightning module for ResNet18 on CIFAR-10.
    
    Adapted for 32x32 images with modified conv1 and removed maxpool.
    """
    
    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
        cifar_modification: bool = True,
        optimizer: dict = None,
        scheduler: dict = None,
        compile: bool = False,
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        # Build model
        self.model = self._build_model()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Track best validation accuracy
        self.val_acc_best = MaxMetric()
        
        # Average losses
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
    
    def _build_model(self) -> nn.Module:
        """Build ResNet model with CIFAR-10 modifications."""
        
        # Load ResNet
        if self.hparams.arch == "resnet18":
            if self.hparams.pretrained:
                model = models.resnet18(weights='IMAGENET1K_V1')
            else:
                model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {self.hparams.arch}")
        
        # Modify for CIFAR-10 (32x32 images)
        if self.hparams.cifar_modification:
            # Replace first conv layer: 7x7 stride 2 -> 3x3 stride 1
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            # Remove maxpool layer
            model.maxpool = nn.Identity()
        
        # Replace final FC layer
        model.fc = nn.Linear(model.fc.in_features, self.hparams.num_classes)
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def model_step(self, batch: Any) -> tuple:
        """Perform a single model step."""
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def training_step(self, batch: Any, batch_idx: int):
        """Training step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        
        # Log metrics (multiply accuracy by 100 for percentage display)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc * 100.0, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        
        # Log metrics (multiply accuracy by 100 for percentage display)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc * 100.0, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
    
    def test_step(self, batch: Any, batch_idx: int):
        """Test step."""
        loss, preds, targets = self.model_step(batch)
        
        # Update metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        
        # Log metrics (multiply accuracy by 100 for percentage display)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc * 100.0, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        pass
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        
        # Build optimizer
        if self.hparams.optimizer is None:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=0.0005
            )
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())
        
        # Return optimizer if no scheduler
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        
        # Build scheduler
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get softmax probabilities for input.
        
        Args:
            x: Input tensor
            
        Returns:
            Softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
