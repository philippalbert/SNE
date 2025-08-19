# SNE

A naive implementation of the Stochastic Neighbor Embedding (SNE) algorithm using PyTorch for dimensionality reduction.

## Overview

SNE is a dimensionality reduction technique that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. This implementation provides a PyTorch-based approach to SNE, optimizing the Kullback-Leibler divergence between high-dimensional and low-dimensional probability distributions.

## Features

- Pure PyTorch implementation for GPU acceleration support
- Configurable number of output dimensions
- Adjustable learning rate and iteration count
- Clean, object-oriented API with attrs-based configuration

## Installation

### Requirements

- Python 3.12
- PyTorch
- NumPy
- Pandas
- Plotly
- attrs

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/philippalbert/SNE.git
cd SNE
```

2. Install the package in development mode:

```bash
pip install -e .
```

3. For development with all dependencies:

```bash
pip install -e ".[all]"
```

### Install Dependencies Only

If you want to install just the runtime dependencies:

```bash
pip install -e ".[dev]"      # Development tools (pre-commit, ruff)
pip install -e ".[testing]"  # Testing tools (pytest, pytest-cov)
pip install -e ".[all]"      # All dependencies
```

## Usage

```python
import torch
from src.SNE import SNE

# Create sample high-dimensional data
data = torch.randn(100, 50)  # 100 samples, 50 dimensions

# Initialize SNE with custom parameters
sne = SNE(
    n_components=2,      # Output dimensions
    learning_rate=10.0,  # Learning rate for optimization
    iterations=100       # Number of optimization iterations
)

# Fit and transform the data
embedding = sne.train(data)

# embedding is now a numpy array of shape (100, 2)
print(f"Original shape: {data.shape}")
print(f"Embedded shape: {embedding.shape}")
```

## References

The implementation is based on:

- Laurens van der Maaten, Geoffrey Hinton, 2008. "Visualizing Data using t-SNE". Journal of Machine Learning Research 9:2579-2605.
