# Active Learning with Conformal Prediction

This project implements Active Learning (AL) with Conformal Prediction (CP) for image classification across multiple datasets. The project follows the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) structure for clean, modular, and reproducible ML experiments.

## Features

- **4 Datasets Supported**:
  - CIFAR-10 (10 classes, 50K train, 32x32)
  - CIFAR-100 (100 classes, 50K train, 32x32)
  - STL-10 (10 classes, 5K labeled train, 96x96)
  - SVHN (10 digit classes, 73K train, 32x32)

- **8 Active Learning Strategies**:
  - Random Sampling (baseline)
  - Entropy Sampling
  - Least Confidence
  - Margin Sampling
  - CP Set Size
  - CP V-Shaped (novel strategy)
  - Combined (Entropy + CP)
  - Combined V-Shaped (Entropy + CP V-Shaped)

- **Conformal Prediction Integration**: Uncertainty quantification with guaranteed coverage
- **PyTorch Lightning**: Structured training loops with automatic logging
- **Hydra Configuration**: Flexible experiment management via YAML configs
- **Modular Architecture**: Clean separation of data, models, and utilities

## Quick Start Guide

📖 **For detailed usage instructions in Vietnamese, see [USAGE_GUIDE.md](USAGE_GUIDE.md)**

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd main_backup_1

# Install in development mode
pip install -e ".[dev]"

# Or use make
make setup
```

## Quick Start

### CIFAR-10

### Train with a specific AL strategy

```bash
# Random sampling (baseline)
python src/train.py experiment=al_random

# Entropy sampling
python src/train.py experiment=al_entropy

# CP-based strategies
python src/train.py experiment=al_cp_size
python src/train.py experiment=al_cp_v_shaped

# Combined strategies
python src/train.py experiment=al_combined
python src/train.py experiment=al_combined_v_shaped
```

### CIFAR-100

```bash
# Requires model config for 100 classes
python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_entropy
```

### STL-10

```bash
# 96x96 images, requires specific model config
python src/train.py data=stl10 model=resnet18_stl10 experiment=al_cp_size
```

### SVHN

```bash
# Street View House Numbers dataset
python src/train.py data=svhn experiment=al_combined
```

### Run multiple strategies (multirun)

```bash
# CIFAR-10
python src/train.py -m experiment=al_random,al_entropy,al_cp_size,al_cp_v_shaped

# All datasets with all strategies
make run-all  # or: powershell scripts/run_all_experiments.ps1
```

### Quick Test

```bash
# Fast test to verify setup
make quick-test  # or: powershell scripts/quick_test.ps1
```

### Override configuration

```bash
# Change number of AL rounds
python src/train.py experiment=al_entropy al.num_rounds=10

# Change budget per round
python src/train.py experiment=al_cp_size al.budget_per_round=1000

# Change random seed
python src/train.py seed=123
```

### Evaluate a checkpoint

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

## Project Structure

```
main_backup_1/
├── configs/                  # Hydra configuration files
│   ├── callbacks/           # Callback configs
│   ├── data/                # Data configs (CIFAR-10)
│   ├── experiment/          # Experiment configs (AL strategies)
│   ├── hydra/               # Hydra runtime settings
│   ├── logger/              # Logger configs (TensorBoard)
│   ├── model/               # Model configs (ResNet18)
│   ├── paths/               # Project paths
│   ├── trainer/             # PyTorch Lightning Trainer configs
│   ├── train.yaml           # Main training config
│   └── eval.yaml            # Main evaluation config
│
├── src/                      # Source code
│   ├── data/                # Data modules
│   │   ├── cifar10_datamodule.py
│   │   └── al_dataset.py
│   ├── models/              # Model modules
│   │   └── resnet_module.py
│   ├── utils/               # Utilities
│   │   ├── al_strategies.py      # Acquisition functions
│   │   ├── conformal.py          # CP utilities
│   │   └── utils.py              # General utilities
│   ├── train.py             # Main training script
│   └── eval.py              # Evaluation script
│
├── notebooks/                # Jupyter notebooks (original)
├── data/                     # Dataset storage
├── logs/                     # Experiment logs
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Project configuration
├── Makefile                  # Common commands
└── README.md                 # This file
```

## Configuration System

The project uses Hydra for configuration management. Configurations are composed from multiple YAML files:

- **Data**: `configs/data/cifar10.yaml` - Dataset settings, augmentation, AL splits
- **Model**: `configs/model/resnet18.yaml` - Architecture, optimizer, scheduler
- **Trainer**: `configs/trainer/default.yaml` - Training settings (epochs, devices)
- **Experiment**: `configs/experiment/*.yaml` - AL strategy specific settings

## Active Learning Workflow

1. **Initialize**: Start with a small labeled set (default: 5000 samples)
2. **Train**: Train ResNet18 on current labeled set
3. **Calibrate**: Compute conformal prediction threshold (qhat) on calibration set
4. **Acquire**: Select most informative samples from unlabeled pool using AL strategy
5. **Repeat**: Add selected samples to labeled set and repeat

## Conformal Prediction

Conformal Prediction provides uncertainty quantification with guaranteed coverage:

- **Coverage**: Probability that true label is in prediction set (default: 90%)
- **Set Size**: Number of classes in prediction set (smaller = more efficient)
- **Alpha**: Miscoverage rate (default: 0.1 for 90% coverage)

## Results

Results are saved in `logs/runs/YYYY-MM-DD/HH-MM-SS/`:
- `results_{strategy}.json` - Training metrics per round
- TensorBoard logs for visualization

View logs with:
```bash
tensorboard --logdir logs/runs
```

## Development

### Run tests
```bash
make test
```

### Format code
```bash
make format
```

### Lint code
```bash
make lint
```

## Citation

If you use this code, please cite the original lightning-hydra-template:

```bibtex
@misc{lightning-hydra-template,
  author = {Łukasz Zalewski},
  title = {Lightning-Hydra-Template},
  url = {https://github.com/ashleve/lightning-hydra-template},
  year = {2021}
}
```

## License

MIT License
