"""Test script for sne implementation."""

import torch
import numpy as np

from SNE import SNE


def test_train():
    """Test training procedure.

    Compare the resulting dimensions. Values of the 2 resulting dimensions
    should be in the 2 defined groups - the unshifted and shifted.
    """
    # Define number of observations (rows) and dimensions (columns).
    dim = 30
    n_obs = 20

    # Create tensor that is randomly distributed and shift distribution of
    # subset of data points.
    tensor = torch.randn((n_obs, dim))
    tensor[:, 0 : int(dim / 2)] += 100

    # Apply SNE.
    sne = SNE(iterations=10, learning_rate=1)
    y_hat = sne.train(tensor)
    y_hat[0 : int(n_obs / 2)]

    # We expect that the first half of entries of each output vector is
    # rather far away.
    assert np.linalg.norm(y_hat[0 : int(dim / 2), 0] - y_hat[int(dim / 2) :, 0]) > 10
    assert np.linalg.norm(y_hat[0 : int(dim / 2), 1] - y_hat[int(dim / 2) :, 1]) > 10
