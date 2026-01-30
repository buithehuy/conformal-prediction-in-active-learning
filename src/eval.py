"""Evaluation script for trained Active Learning models."""

import os
from pathlib import Path
from typing import Dict

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import json

from src.data import CIFAR10DataModule
from src.models import ResNetModule
from src.utils import (
    set_seed,
    disable_warnings,
    get_probabilities,
    compute_qhat,
    get_prediction_sets,
    compute_coverage,
    compute_avg_set_size
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    """
    Main evaluation function.
    
    Args:
        cfg: Hydra configuration
    """
    
    # Print config if requested
    if cfg.get("print_config"):
        print(OmegaConf.to_yaml(cfg))
    
    # Disable warnings if requested
    if cfg.get("disable_python_warnings", True):
        disable_warnings()
    
    # Set random seed
    if cfg.get("seed"):
        set_seed(cfg.seed)
        pl.seed_everything(cfg.seed, workers=True)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check checkpoint path
    if cfg.ckpt_path is None:
        raise ValueError("Please provide checkpoint path via: ckpt_path=/path/to/checkpoint.ckpt")
    
    if not os.path.exists(cfg.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")
    
    print(f"\nLoading checkpoint: {cfg.ckpt_path}")
    
    # Initialize data module
    print("\nInitializing CIFAR-10 DataModule...")
    datamodule: CIFAR10DataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("test")
    
    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    model = ResNetModule.load_from_checkpoint(cfg.ckpt_path)
    model.eval()
    model = model.to(device)
    
    # Initialize trainer
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)
    test_acc = test_results[0]["test/acc"]
    test_loss = test_results[0]["test/loss"]
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    
    # Conformal Prediction evaluation
    print("\nComputing Conformal Prediction metrics...")
    cp_alpha = cfg.conformal.alpha
    
    # Setup calibration if needed
    datamodule.setup("fit")
    calib_loader = datamodule.val_dataloader()
    qhat = compute_qhat(model.model, calib_loader, alpha=cp_alpha, device=device)
    
    print(f"  qhat (alpha={cp_alpha}): {qhat:.4f}")
    
    # Evaluate CP on test set
    test_loader = datamodule.test_dataloader()
    test_probs, test_labels = get_probabilities(model.model, test_loader, device=device)
    test_pred_sets = get_prediction_sets(test_probs, qhat)
    
    cp_coverage = compute_coverage(test_pred_sets, test_labels)
    cp_avg_set_size = compute_avg_set_size(test_pred_sets)
    
    print(f"  Coverage: {cp_coverage:.4f} (target: {1-cp_alpha:.4f})")
    print(f"  Avg Set Size: {cp_avg_set_size:.2f}")
    
    # Save evaluation results
    results = {
        "checkpoint": str(cfg.ckpt_path),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "cp_alpha": cp_alpha,
        "cp_qhat": qhat,
        "cp_coverage": cp_coverage,
        "cp_avg_set_size": cp_avg_set_size
    }
    
    # Save results
    results_dir = Path(cfg.paths.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()
