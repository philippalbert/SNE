"""Naive implementation of the Stochastic Neighbor Embedding algorithm using torch."""

import torch
from attrs import field, define


@define
class SNE:
    """
    A class implementing Stochastic Neighbor Embedding (SNE) for dimensionality reduction.

    References
    ----------
    .. [1] Laurens van der Maaten, Geoffrey Hinton, 2008. "Visualizing Data using t-SNE".
           DOI: Visualizing Data using t-SNE. Available at: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

    """  # noqa: E501

    n_components: int = field(default=2)
    learning_rate: float = field(default=10.0)
    iterations: int = field(default=100)

    @staticmethod
    def _conditional_probability_p_j_i(
        x: torch.tensor, i: int, j: int, sigma: float = 2
    ):
        """Compute the conditional probability P(j|i) in the high-dimensional space.

        Args:
            x (torch.tensor): High-dimensional data matrix of shape (dim, n_obs).
            i (int): Index of the reference point.
            j (int): Index of the target point.
            sigma (float): Bandwidth parameter for the Gaussian kernel. Default is 2.

        Returns:
            torch.tensor: The conditional probability P(j|i).
        """

        neg_norm = -torch.linalg.norm(x[:, i] - x[:, j], ord=None)
        numerator = torch.exp(neg_norm / 2 / (sigma**2))
        denominator_vec = torch.zeros((x.shape[1]))
        for k in range(x.shape[1]):
            if i != k:
                denominator_vec[k] = torch.exp(
                    -torch.norm(x[:, i] - x[:, k]) / 2 / (sigma**2)
                )

        return numerator / (denominator_vec.sum() + 1e-8)

    @staticmethod
    def _conditional_probability_q_j_i(
        y: torch.tensor, i: int, j: int, sigma: float = 2
    ):
        """Compute the conditional probability Q(j|i) in the low-dimensional space.

        Args:
            y (torch.tensor): Low-dimensional data matrix of shape (dim, n_obs).
            i (int): Index of the reference point.
            j (int): Index of the target point.
            sigma (float): Bandwidth parameter for the Gaussian kernel. Default is 2.

        Returns:
            torch.tensor: The conditional probability Q(j|i).
        """

        neg_norm = -torch.linalg.norm(y[:, i] - y[:, j], ord=None)
        numerator = torch.exp(neg_norm)
        denominator_vec = torch.zeros((y.shape[1]))
        for k in range(y.shape[1]):
            if i != k:
                denominator_vec[k] = torch.exp(-torch.norm(y[:, i] - y[:, k]))

        return numerator / (denominator_vec.sum() + 1e-8)

    def train(self, x):
        """Optimize the Kullback-Leibler divergence between the high- and low-dimensional distributions.

        Args:
            x (torch.tensor): High-dimensional data matrix of shape (n_obs, dim).

        Returns:
            numpy.ndarray: Optimized low-dimensional representation of the data with shape (n_obs, n_components).
        """
        # Convert input to tensor properly
        if isinstance(x, torch.Tensor):
            x = x.detach().clone().float()
        else:
            x = torch.tensor(x, dtype=torch.float32)

        # Transpose to match expected format (dim, n_obs)
        x = x.t()
        x.requires_grad_(False)

        n_obs = x.shape[1]
        y = torch.randn(self.n_components, n_obs, requires_grad=True)
        optimizer = torch.optim.Adam([y], lr=self.learning_rate)

        for epoch in range(self.iterations):
            optimizer.zero_grad()
            loss = 0
            for i in range(n_obs):
                for j in range(n_obs):
                    if i != j:  # Skip self-comparison
                        p_j_i = self._conditional_probability_p_j_i(x, i, j)
                        q_j_i = self._conditional_probability_q_j_i(y, i, j)
                        # Add small epsilon to prevent log(0)
                        loss += p_j_i * torch.log(p_j_i / (q_j_i + 1e-8))
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == self.iterations - 1:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

        return y.t().detach().numpy()
