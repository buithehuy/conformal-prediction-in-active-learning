.PHONY: help setup install train test clean

help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies in development mode"
	@echo "  make install        - Install package in production mode"
	@echo "  make test           - Run tests"
	@echo "  make quick-test     - Quick test with all datasets"
	@echo ""
	@echo "Training commands:"
	@echo "  make train          - Run training with default config"
	@echo "  make train-random   - Run training with random sampling (CIFAR-10)"
	@echo "  make train-entropy  - Run training with entropy sampling (CIFAR-10)"
	@echo "  make train-all      - Run all strategies (CIFAR-10)"
	@echo ""
	@echo "Multi-dataset commands:"
	@echo "  make train-cifar100 - Run on CIFAR-100"
	@echo "  make train-stl10    - Run on STL-10"
	@echo "  make train-svhn     - Run on SVHN"
	@echo "  make run-all        - Run ALL datasets + ALL strategies"
	@echo ""
	@echo "Utility commands:"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make format         - Format code with black and isort"
	@echo "  make lint           - Run flake8 linter"

setup:
	pip install -e ".[dev]"
	pre-commit install

install:
	pip install -e .

train:
	python src/train.py

train-random:
	python src/train.py experiment=al_random

train-entropy:
	python src/train.py experiment=al_entropy

train-cp-size:
	python src/train.py experiment=al_cp_size

train-cp-v:
	python src/train.py experiment=al_cp_v_shaped

train-combined:
	python src/train.py experiment=al_combined

train-all:
	python src/train.py -m experiment=al_random,al_entropy,al_cp_size,al_cp_v_shaped

# Multi-dataset commands
train-cifar100:
	python src/train.py data=cifar100 model=resnet18_cifar100 experiment=al_entropy

train-stl10:
	python src/train.py data=stl10 model=resnet18_stl10 experiment=al_entropy

train-svhn:
	python src/train.py data=svhn experiment=al_entropy

# Run all experiments
run-all:
	powershell -ExecutionPolicy Bypass -File scripts/run_all_experiments.ps1

quick-test:
	powershell -ExecutionPolicy Bypass -File scripts/quick_test.ps1

test:
	pytest tests/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .tox/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
