"""Conformal Prediction utilities."""

import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F


def get_probabilities(model, dataloader, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get softmax probabilities for all samples in dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing samples
        device: Device to run inference on
        
    Returns:
        probs: Tensor of shape (N, num_classes) with softmax probabilities
        labels: Tensor of shape (N,) with true labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(y)
    
    return torch.cat(all_probs), torch.cat(all_labels)


def compute_qhat(model, calib_loader, alpha: float = 0.1, device: str = "cuda") -> float:
    """
    Compute calibrated quantile threshold for conformal prediction.
    
    Args:
        model: Trained PyTorch model
        calib_loader: DataLoader for calibration set
        alpha: Significance level (1 - alpha = coverage)
        device: Device to run inference on
        
    Returns:
        qhat: Calibrated quantile value
    """
    probs, labels = get_probabilities(model, calib_loader, device)
    
    # Non-conformity score: 1 - probability of true class
    n_samples = len(labels)
    true_class_probs = probs[torch.arange(n_samples), labels]
    scores = 1 - true_class_probs
    
    # Compute quantile
    k = int(np.ceil((n_samples + 1) * (1 - alpha)))
    sorted_scores, _ = torch.sort(scores)
    qhat = sorted_scores[min(k - 1, n_samples - 1)].item()
    
    return qhat


def get_prediction_sets(probs: torch.Tensor, qhat: float) -> torch.Tensor:
    """
    Get prediction sets for each sample.
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        qhat: Calibrated quantile threshold
        
    Returns:
        pred_sets: Boolean tensor (N, num_classes) indicating which classes are in prediction set
    """
    return probs >= (1 - qhat)


def compute_coverage(pred_sets: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute empirical coverage of prediction sets.
    
    Args:
        pred_sets: Boolean tensor (N, num_classes)
        labels: True labels (N,)
        
    Returns:
        coverage: Fraction of samples where true label is in prediction set
    """
    n_samples = len(labels)
    covered = pred_sets[torch.arange(n_samples), labels]
    return covered.float().mean().item()


def compute_avg_set_size(pred_sets: torch.Tensor) -> float:
    """
    Compute average prediction set size.
    
    Args:
        pred_sets: Boolean tensor (N, num_classes)
        
    Returns:
        avg_size: Average number of classes in prediction sets
    """
    return pred_sets.sum(dim=1).float().mean().item()
