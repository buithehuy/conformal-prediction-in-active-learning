"""Active Learning Acquisition Strategies."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


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


# =============================================================================
# Acquisition Functions
# =============================================================================

def random_sampling(probs: torch.Tensor, budget: int, **kwargs) -> torch.Tensor:
    """
    Random sampling - baseline strategy.
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        
    Returns:
        indices: Selected sample indices
    """
    n_samples = probs.shape[0]
    scores = torch.rand(n_samples)
    _, indices = torch.topk(scores, budget)
    return indices


def entropy_sampling(probs: torch.Tensor, budget: int, **kwargs) -> torch.Tensor:
    """
    Entropy-based sampling - select samples with highest entropy.
    High entropy = model is uncertain.
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        
    Returns:
        indices: Selected sample indices
    """
    # Entropy = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-9)
    entropy = -(probs * log_probs).sum(dim=1)
    _, indices = torch.topk(entropy, budget)
    return indices


def least_confidence_sampling(probs: torch.Tensor, budget: int, **kwargs) -> torch.Tensor:
    """
    Least confidence sampling - select samples where model has lowest max probability.
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        
    Returns:
        indices: Selected sample indices
    """
    # Uncertainty = 1 - max(p)
    max_probs, _ = probs.max(dim=1)
    uncertainty = 1 - max_probs
    _, indices = torch.topk(uncertainty, budget)
    return indices


def margin_sampling(probs: torch.Tensor, budget: int, **kwargs) -> torch.Tensor:
    """
    Margin sampling - select samples with smallest margin between top-2 predictions.
    Small margin = model is uncertain between two classes.
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        
    Returns:
        indices: Selected sample indices
    """
    # Sort probabilities in descending order
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    # Margin = difference between top-1 and top-2 probabilities
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    # Smaller margin = more uncertain, so we want to minimize margin
    # Equivalent to maximizing negative margin
    _, indices = torch.topk(-margin, budget)
    return indices


def cp_set_size_sampling(probs: torch.Tensor, budget: int, qhat: float, **kwargs) -> torch.Tensor:
    """
    Conformal Prediction set size sampling (Original strategy).
    Select samples with largest prediction set size (most uncertain according to CP).
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        qhat: Calibrated quantile threshold
        
    Returns:
        indices: Selected sample indices
    """
    # Prediction set: classes with prob >= 1 - qhat
    pred_sets = probs >= (1 - qhat)
    set_sizes = pred_sets.sum(dim=1).float()
    _, indices = torch.topk(set_sizes, budget)
    return indices


def cp_v_shaped_sampling(probs: torch.Tensor, budget: int, qhat: float, **kwargs) -> torch.Tensor:
    """
    Conformal Prediction V-shaped sampling (NEW strategy).
    
    Prioritizes BOTH extremes:
    - set_size = 0: Model is overconfident but possibly WRONG (no class passes threshold)
    - set_size > 1: Model is uncertain (multiple classes pass threshold)
    
    set_size = 1 is least informative (model is confident and likely correct).
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        qhat: Calibrated quantile threshold
        
    Returns:
        indices: Selected sample indices
    """
    num_classes = probs.shape[1]
    
    # Prediction set: classes with prob >= 1 - qhat
    pred_sets = probs >= (1 - qhat)
    set_sizes = pred_sets.sum(dim=1).float()
    
    # V-shaped scoring: 
    # - set_size = 0 gets highest score (num_classes + 1)
    # - set_size = 1 gets lowest score (0)
    # - set_size > 1 gets increasing score
    # This prioritizes both overconfident errors (0) and uncertain samples (large)
    
    uncertainty_score = torch.where(
        set_sizes == 0,
        torch.tensor(num_classes + 1.0),  # Highest priority: overconfident but possibly wrong
        torch.where(
            set_sizes == 1,
            torch.tensor(0.0),  # Lowest priority: confident and likely correct
            set_sizes  # Medium-high priority: uncertain (2, 3, 4, ...)
        )
    )
    
    _, indices = torch.topk(uncertainty_score, budget)
    return indices


def combined_entropy_cp_sampling(
    probs: torch.Tensor, 
    budget: int, 
    qhat: float,
    entropy_weight: float = 0.5,
    cp_weight: float = 0.5,
    **kwargs
) -> torch.Tensor:
    """
    Combined strategy: weighted combination of entropy and CP set size (Original).
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        qhat: Calibrated quantile threshold
        entropy_weight: Weight for entropy component
        cp_weight: Weight for CP component
        
    Returns:
        indices: Selected sample indices
    """
    # Normalize entropy to [0, 1]
    log_probs = torch.log(probs + 1e-9)
    entropy = -(probs * log_probs).sum(dim=1)
    num_classes = probs.shape[1]
    max_entropy = np.log(num_classes)
    entropy_normalized = entropy / max_entropy
    
    # Normalize CP set size to [0, 1]
    pred_sets = probs >= (1 - qhat)
    set_sizes = pred_sets.sum(dim=1).float()
    set_sizes_normalized = set_sizes / num_classes
    
    # Combined score
    combined_score = entropy_weight * entropy_normalized + cp_weight * set_sizes_normalized
    
    _, indices = torch.topk(combined_score, budget)
    return indices


def combined_v_shaped_sampling(
    probs: torch.Tensor, 
    budget: int, 
    qhat: float,
    entropy_weight: float = 0.5,
    cp_weight: float = 0.5,
    **kwargs
) -> torch.Tensor:
    """
    Combined strategy: weighted combination of entropy and V-shaped CP score (NEW).
    
    CP component uses V-shaped scoring:
    - set_size = 0: Highest uncertainty (overconfident but possibly wrong)
    - set_size = 1: Lowest uncertainty (confident and likely correct)
    - set_size > 1: Medium-high uncertainty (uncertain)
    
    Args:
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        qhat: Calibrated quantile threshold
        entropy_weight: Weight for entropy component
        cp_weight: Weight for CP component
        
    Returns:
        indices: Selected sample indices
    """
    num_classes = probs.shape[1]
    
    # Normalize entropy to [0, 1]
    log_probs = torch.log(probs + 1e-9)
    entropy = -(probs * log_probs).sum(dim=1)
    max_entropy = np.log(num_classes)
    entropy_normalized = entropy / max_entropy
    
    # V-shaped CP score normalized to [0, 1]
    pred_sets = probs >= (1 - qhat)
    set_sizes = pred_sets.sum(dim=1).float()
    
    # V-shaped: set_size=0 -> 1.0, set_size=1 -> 0.0, set_size=K -> K/num_classes
    cp_score = torch.where(
        set_sizes == 0,
        torch.tensor(1.0),  # Highest: overconfident errors
        torch.where(
            set_sizes == 1,
            torch.tensor(0.0),  # Lowest: confident and correct
            set_sizes / num_classes  # Medium: uncertain
        )
    )
    
    # Combined score
    combined_score = entropy_weight * entropy_normalized + cp_weight * cp_score
    
    _, indices = torch.topk(combined_score, budget)
    return indices


# =============================================================================
# Strategy Selector
# =============================================================================

STRATEGY_FUNCTIONS = {
    "random": random_sampling,
    "entropy": entropy_sampling,
    "least_confidence": least_confidence_sampling,
    "margin": margin_sampling,
    "cp_size": cp_set_size_sampling,
    "cp_v_shaped": cp_v_shaped_sampling,
    "combined": combined_entropy_cp_sampling,
    "combined_v_shaped": combined_v_shaped_sampling
}


def get_acquisition_function(strategy_name: str):
    """
    Get acquisition function by name.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Acquisition function
    """
    if strategy_name not in STRATEGY_FUNCTIONS:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(STRATEGY_FUNCTIONS.keys())}"
        )
    return STRATEGY_FUNCTIONS[strategy_name]


def select_samples(
    strategy_name: str,
    probs: torch.Tensor,
    budget: int,
    qhat: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    """
    Select samples using specified strategy.
    
    Args:
        strategy_name: Name of the acquisition strategy
        probs: Softmax probabilities (N, num_classes)
        budget: Number of samples to select
        qhat: Calibrated quantile (required for CP-based strategies)
        **kwargs: Additional arguments for strategy
        
    Returns:
        indices: Selected sample indices
    """
    acquisition_fn = get_acquisition_function(strategy_name)
    
    # Check if strategy requires qhat
    cp_strategies = ["cp_size", "cp_v_shaped", "combined", "combined_v_shaped"]
    if strategy_name in cp_strategies and qhat is None:
        raise ValueError(f"Strategy '{strategy_name}' requires qhat parameter")
    
    return acquisition_fn(probs, budget, qhat=qhat, **kwargs)
