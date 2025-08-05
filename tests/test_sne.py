"""Test script for sne implementation."""

import sys
import os
import torch
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from SNE import SNE


def test_train():
    """Test training procedure.

    Compare the resulting dimensions. Values of the 2 resulting dimensions
    should be in the 2 defined groups - the unshifted and shifted.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define number of observations (rows) and dimensions (columns).
    dim = 10  # Reduced for faster testing
    n_obs = 12

    # Create tensor with two distinct groups
    tensor = torch.randn((n_obs, dim))
    # Make the first half of observations very different from the second half
    tensor[0 : int(n_obs / 2), :] += 10  # Large shift for clear separation

    # Apply SNE with more iterations for better convergence
    sne = SNE(iterations=50, learning_rate=5.0)
    y_hat = sne.train(tensor)

    # Check output shape
    assert y_hat.shape == (n_obs, 2), f"Expected shape ({n_obs}, 2), got {y_hat.shape}"

    # We expect that the first half of observations (which had shifted features)
    # should be separated from the second half in the 2D embedding
    first_half = y_hat[0 : int(n_obs / 2)]
    second_half = y_hat[int(n_obs / 2) :]

    # Calculate centroids of each group
    centroid_first = np.mean(first_half, axis=0)
    centroid_second = np.mean(second_half, axis=0)

    # Distance between centroids should be significant (reduced threshold)
    centroid_distance = np.linalg.norm(centroid_first - centroid_second)
    print(f"Centroid distance: {centroid_distance}")
    assert centroid_distance > 1.0, (
        f"Centroid distance {centroid_distance} is too small"
    )

    # Additional check: verify that the algorithm produces reasonable output
    assert not np.any(np.isnan(y_hat)), "Output contains NaN values"
    assert not np.any(np.isinf(y_hat)), "Output contains infinite values"


def test_sne_initialization():
    """Test SNE class initialization with different parameters."""
    # Test default initialization
    sne_default = SNE()
    assert sne_default.n_components == 2
    assert sne_default.learning_rate == 10.0
    assert sne_default.iterations == 100

    # Test custom initialization
    sne_custom = SNE(n_components=3, learning_rate=5.0, iterations=50)
    assert sne_custom.n_components == 3
    assert sne_custom.learning_rate == 5.0
    assert sne_custom.iterations == 50


def test_conditional_probability_p_j_i():
    """Test the conditional probability calculation in high-dimensional space."""
    # Create simple test data
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32)

    # Test probability calculation
    prob = SNE._conditional_probability_p_j_i(x, i=0, j=1, sigma=1.0)

    # Probability should be a scalar tensor between 0 and 1
    assert isinstance(prob, torch.Tensor)
    assert prob.dim() == 0  # scalar
    assert 0 <= prob <= 1


def test_conditional_probability_q_j_i():
    """Test the conditional probability calculation in low-dimensional space."""
    # Create simple test data
    y = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    # Test probability calculation
    prob = SNE._conditional_probability_q_j_i(y, i=0, j=1, sigma=1.0)

    # Probability should be a scalar tensor between 0 and 1
    assert isinstance(prob, torch.Tensor)
    assert prob.dim() == 0  # scalar
    assert 0 <= prob <= 1


def test_train_input_validation():
    """Test that train method handles different input types correctly."""
    sne = SNE(iterations=5, learning_rate=1.0)

    # Test with numpy array
    np_data = np.random.randn(10, 5)
    result_np = sne.train(np_data)
    assert result_np.shape == (10, 2)

    # Test with torch tensor
    torch_data = torch.randn(10, 5)
    result_torch = sne.train(torch_data)
    assert result_torch.shape == (10, 2)

    # Test with list
    list_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result_list = sne.train(list_data)
    assert result_list.shape == (3, 2)


def test_train_small_dataset():
    """Test training with a very small dataset."""
    sne = SNE(iterations=5, learning_rate=1.0)

    # Create minimal dataset
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    result = sne.train(data)

    assert result.shape == (3, 2)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_reproducibility():
    """Test that results are reproducible with same random seed."""
    torch.manual_seed(42)
    np.random.seed(42)

    data = torch.randn(10, 5)
    sne1 = SNE(iterations=5, learning_rate=1.0)

    torch.manual_seed(42)
    result1 = sne1.train(data.clone())

    torch.manual_seed(42)
    sne2 = SNE(iterations=5, learning_rate=1.0)
    result2 = sne2.train(data.clone())

    # Results should be similar (allowing for small numerical differences)
    assert np.allclose(result1, result2, rtol=1e-3, atol=1e-3)
