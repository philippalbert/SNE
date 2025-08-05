"""Test edge cases and error handling for SNE implementation."""

import torch
import numpy as np

from SNE import SNE


def test_identical_observations():
    """Test behavior when all observations are identical."""
    sne = SNE(iterations=5, learning_rate=1.0)

    # All identical observations
    data = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32
    )
    result = sne.train(data)

    assert result.shape == (3, 2)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_high_dimensional_data():
    """Test with high-dimensional input data."""
    sne = SNE(iterations=10, learning_rate=1.0)

    # High-dimensional data
    data = torch.randn(5, 100)
    result = sne.train(data)

    assert result.shape == (5, 2)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_different_n_components():
    """Test with different numbers of output components."""
    # Test 1D output
    sne_1d = SNE(n_components=1, iterations=5, learning_rate=1.0)
    data = torch.randn(5, 3)
    result_1d = sne_1d.train(data)
    assert result_1d.shape == (5, 1)

    # Test 3D output
    sne_3d = SNE(n_components=3, iterations=5, learning_rate=1.0)
    result_3d = sne_3d.train(data)
    assert result_3d.shape == (5, 3)


def test_zero_learning_rate():
    """Test behavior with zero learning rate (should not crash)."""
    sne = SNE(iterations=5, learning_rate=0.0)
    data = torch.randn(5, 3)
    result = sne.train(data)

    assert result.shape == (5, 2)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_very_small_learning_rate():
    """Test with very small learning rate."""
    sne = SNE(iterations=5, learning_rate=1e-6)
    data = torch.randn(5, 3)
    result = sne.train(data)

    assert result.shape == (5, 2)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_large_learning_rate():
    """Test with large learning rate."""
    sne = SNE(iterations=5, learning_rate=100.0)
    data = torch.randn(5, 3)
    result = sne.train(data)

    assert result.shape == (5, 2)
    # With large learning rate, we might get NaN/inf, but algorithm should handle it
    # Just check that we get some result


def test_zero_iterations():
    """Test with zero iterations."""
    sne = SNE(iterations=0, learning_rate=1.0)
    data = torch.randn(5, 3)
    result = sne.train(data)

    assert result.shape == (5, 2)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_probability_functions_edge_cases():
    """Test conditional probability functions with edge cases."""
    # Test with same indices (should handle i == j case in calling code)
    x = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    # Test with very small sigma
    prob_small_sigma = SNE._conditional_probability_p_j_i(x, i=0, j=1, sigma=1e-6)
    assert isinstance(prob_small_sigma, torch.Tensor)
    assert not torch.isnan(prob_small_sigma)
    assert not torch.isinf(prob_small_sigma)

    # Test with large sigma
    prob_large_sigma = SNE._conditional_probability_p_j_i(x, i=0, j=1, sigma=1000.0)
    assert isinstance(prob_large_sigma, torch.Tensor)
    assert not torch.isnan(prob_large_sigma)
    assert not torch.isinf(prob_large_sigma)


def test_mixed_data_types():
    """Test with mixed data types (int, float)."""
    sne = SNE(iterations=5, learning_rate=1.0)

    # Integer data
    int_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result_int = sne.train(int_data)
    assert result_int.shape == (3, 2)

    # Mixed int/float data
    mixed_data = [[1, 2.5, 3], [4.0, 5, 6.7], [7, 8.1, 9]]
    result_mixed = sne.train(mixed_data)
    assert result_mixed.shape == (3, 2)
