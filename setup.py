"""Setup script for Active Learning with Conformal Prediction project."""

from setuptools import setup, find_packages

setup(
    name="conformal-prediction-in-active-learning",
    version="0.1.0",
    description="Active Learning with Conformal Prediction on CIFAR-10",
    author="buithehuy",
    author_email="[EMAIL_ADDRESS]",
    url="https://github.com/buithehuy/conformal-prediction-in-active-learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pre-commit>=3.3.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
