"""Main training script for Active Learning with Hydra configuration."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import json

from src.data import CIFAR10DataModule
from src.models import ResNetModule
from src.utils import (
    set_seed,
    disable_warnings,
    get_probabilities,
    select_samples,
    compute_qhat,
    get_prediction_sets,
    compute_coverage,
    compute_avg_set_size
)


def instantiate_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Instantiate callbacks from config."""
    callbacks = []
    
    if "callbacks" in cfg and cfg.callbacks:
        for _, cb_conf in cfg.callbacks.items():
            if cb_conf is None or "_target_" not in cb_conf:
                continue
            
            # Skip disabled callbacks
            if "enable" in cb_conf and not cb_conf.enable:
                continue
                
            callbacks.append(hydra.utils.instantiate(cb_conf))
    
    return callbacks


def instantiate_loggers(cfg: DictConfig) -> List:
    """Instantiate loggers from config."""
    loggers = []
    
    if "logger" in cfg and cfg.logger:
        for _, lg_conf in cfg.logger.items():
            if lg_conf is not None and "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))
    
    return loggers


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """
    Main training function for Active Learning.
    
    Args:
        cfg: Hydra configuration
    """
    
    # Print config if requested
    if cfg.get("print_config"):
        print(OmegaConf.to_yaml(cfg))
    
    # Disable warnings if requested
    if cfg.get("disable_python_warnings"):
        disable_warnings()
    
    # Set random seed
    if cfg.get("seed"):
        set_seed(cfg.seed)
        pl.seed_everything(cfg.seed, workers=True)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize data module
    print("\n" + "="*60)
    print("Initializing DataModule")
    print("="*60)
    datamodule: CIFAR10DataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    # Get AL configuration
    al_cfg = cfg.al
    strategy_name = al_cfg.strategy
    num_rounds = al_cfg.num_rounds
    budget_per_round = al_cfg.budget_per_round
    
    # Conformal prediction configuration
    cp_alpha = cfg.conformal.alpha
    
    # Combined strategy weights (if applicable)
    combined_weights = cfg.get("combined_weights", {})
    entropy_weight = combined_weights.get("entropy_weight", 0.5)
    cp_weight = combined_weights.get("cp_weight", 0.5)
    
    # Compact logging mode
    compact_logging = cfg.get("compact_logging", False)
    
    if not compact_logging:
        print(f"\nActive Learning Configuration:")
        print(f"  Strategy: {strategy_name}")
        print(f"  Num rounds: {num_rounds}")
        print(f"  Budget per round: {budget_per_round}")
        print(f"  CP alpha: {cp_alpha}")
        if "combined" in strategy_name:
            print(f"  Entropy weight: {entropy_weight}")
            print(f"  CP weight: {cp_weight}")
    else:
        print("\n" + "="*70)
        print(f"{strategy_name.upper()}")
        print("="*70)
    
    # Results tracking
    results = {
        "round": [],
        "labeled_samples": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "cp_coverage": [],
        "cp_avg_set_size": []
    }
    
    # Active Learning Loop
    if not compact_logging:
        print("\n" + "="*60)
        print("Starting Active Learning Loop")
        print("="*60)
    
    for round_idx in range(num_rounds):
        if not compact_logging:
            print(f"\n{'='*60}")
            print(f"Round {round_idx + 1}/{num_rounds}")
            print(f"Labeled samples: {len(datamodule.labeled_idx)}")
            print(f"{'='*60}")
        
        # Initialize model for this round
        model: ResNetModule = hydra.utils.instantiate(cfg.model)
        
        # Setup callbacks
        callbacks = instantiate_callbacks(cfg)
        
        # Setup loggers
        loggers = instantiate_loggers(cfg)
        
        # Initialize trainer
        if compact_logging:
            # Disable verbose logging in compact mode
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer,
                callbacks=callbacks,
                logger=loggers,
                enable_progress_bar=False,
                enable_model_summary=False
            )
        else:
            trainer: Trainer = hydra.utils.instantiate(
                cfg.trainer,
                callbacks=callbacks,
                logger=loggers
            )
        
        # Train model
        if not compact_logging:
            print(f"\nTraining model...")
        trainer.fit(model, datamodule=datamodule)
        
        # Get training and validation metrics
        # Run validation to get updated metrics
        val_results = trainer.validate(model, datamodule=datamodule, verbose=False)
        val_acc = val_results[0]["val/acc"]
        val_loss = val_results[0]["val/loss"]
        
        # Get train accuracy from model's metric (last computed value)
        train_acc = model.train_acc.compute() * 100.0  # Convert to percentage
        
        # Evaluate on test set
        if not compact_logging:
            print(f"\nEvaluating on test set...")
        test_results = trainer.test(model, datamodule=datamodule, verbose=False)
        test_acc = test_results[0]["test/acc"]
        
        # Compute Conformal Prediction metrics
        if not compact_logging:
            print(f"  Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc:   {val_acc:.2f}%")
            print(f"  Test Acc:  {test_acc:.2f}%")
            print(f"\nComputing Conformal Prediction metrics...")
        model.eval()
        model = model.to(device)
        
        # Compute qhat on calibration set
        calib_loader = datamodule.val_dataloader()
        qhat = compute_qhat(model.model, calib_loader, alpha=cp_alpha, device=device)
        
        # Evaluate CP on test set
        test_loader = datamodule.test_dataloader()
        test_probs, test_labels = get_probabilities(model.model, test_loader, device=device)
        test_pred_sets = get_prediction_sets(test_probs, qhat)
        cp_coverage = compute_coverage(test_pred_sets, test_labels)
        cp_avg_set_size = compute_avg_set_size(test_pred_sets)
        
        if compact_logging:
            # Compact single-line output
            total_pool = len(datamodule.labeled_idx) + len(datamodule.pool_idx)
            trained_pct = (len(datamodule.labeled_idx) / total_pool) * 100
            zero_count = sum(1 for pred_set in test_pred_sets if len(pred_set) == 0)
            print(f"R{round_idx:2d}: Acc={test_acc:5.2f}% | Trained={len(datamodule.labeled_idx):5d} ({trained_pct:4.1f}%) | Cov={cp_coverage:.3f} | AvgSet={cp_avg_set_size:.2f} | Zero={zero_count}")
        else:
            print(f"  CP Coverage: {cp_coverage:.4f}")
            print(f"  CP Avg Set Size: {cp_avg_set_size:.2f}")
        
        # Store results
        results["round"].append(round_idx + 1)
        results["labeled_samples"].append(len(datamodule.labeled_idx))
        results["train_acc"].append(float(train_acc))
        results["val_acc"].append(float(val_acc))
        results["test_acc"].append(float(test_acc))
        results["cp_coverage"].append(cp_coverage)
        results["cp_avg_set_size"].append(cp_avg_set_size)
        
        # Break if this is the last round
        if round_idx == num_rounds - 1:
            break
        
        # Acquisition: Select samples from pool
        if not compact_logging:
            print(f"\nAcquiring {budget_per_round} samples using '{strategy_name}' strategy...")
        pool_loader = datamodule.pool_dataloader()
        pool_probs, _ = get_probabilities(model.model, pool_loader, device=device)
        
        # Select samples
        selected_pool_indices = select_samples(
            strategy_name=strategy_name,
            probs=pool_probs,
            budget=budget_per_round,
            qhat=qhat if "cp" in strategy_name or "combined" in strategy_name else None,
            entropy_weight=entropy_weight,
            cp_weight=cp_weight
        )
        
        # Map pool indices to dataset indices
        pool_dataset_indices = [datamodule.pool_idx[i] for i in selected_pool_indices.tolist()]
        
        # Add to labeled set
        datamodule.add_to_labeled(pool_dataset_indices)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    
    results_dir = Path(cfg.paths.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / f"results_{strategy_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Final labeled samples: {len(datamodule.labeled_idx)}")
    print(f"Final test accuracy: {results['test_acc'][-1]:.2f}%")
    print(f"Final CP coverage: {results['cp_coverage'][-1]:.4f}")
    print(f"Final CP avg set size: {results['cp_avg_set_size'][-1]:.2f}")
    
    return results


if __name__ == "__main__":
    main()
