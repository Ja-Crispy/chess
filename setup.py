"""Setup script for minimal chess models research."""

from setuptools import setup, find_packages

setup(
    name="minimal-chess-models",
    version="0.1.0",
    description="Minimal, high-performance chess models research",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "python-chess>=1.9.4",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "h5py>=3.10.0",
        "datasets>=2.16.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "ruff>=0.1.9",
        ]
    }
)
