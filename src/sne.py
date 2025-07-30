"""Naive implementation of the Stochastic Neighbor Embedding algorithm using torch."""

import torch
import plotly.express as px
from attrs import field, define


@define
class SNE:
    """
    A class implementing Stochastic Neighbor Embedding (SNE) for dimensionality reduction.
    """

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
        for k in range(x.shape[1] - 1):
            if i != k:
                denominator_vec[k] = torch.exp(
                    -torch.norm(x[:, i] - x[:, k]) / 2 / (sigma**2)
                )

        return numerator / denominator_vec.sum()

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
        for k in range(y.shape[1] - 1):
            if i != k:
                denominator_vec[k] = torch.exp(-torch.norm(y[:, i] - y[:, k]))

        return numerator / denominator_vec.sum()

    def train(self, x):
        """Optimize the Kullback-Leibler divergence between the high- and low-dimensional distributions.

        Args:
            x (torch.tensor): High-dimensional data matrix of shape (dim, n_obs).

        Returns:
            torch.tensor: Optimized low-dimensional representation of the data.
        """

        x = torch.tensor(x, dtype=torch.float32, requires_grad=False)
        n_obs = x.shape[1]
        y = torch.randn(2, n_obs, requires_grad=True)
        optimizer = torch.optim.Adam([y], lr=self.learning_rate)

        for epoch in range(self.iterations):
            optimizer.zero_grad()
            loss = 0
            for i in range(n_obs):
                for j in range(n_obs):
                    p_j_i = self._conditional_probability_p_j_i(x, i, j)
                    q_j_i = self._conditional_probability_q_j_i(y, i, j)
                    loss += p_j_i * torch.log(p_j_i / q_j_i)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        return y.t().detach().numpy()


if __name__ == "__main__":
    # tensor = torch.tensor([[1,2,3], [4,5,6], [7,8,9]], dtype=torch.float32)
    # sne = SNE(tensor)
    # print(sne._conditional_probability_p_j_i(tensor, 1, 2))
    dim = 10
    n_obs = 30
    tensor = torch.randn((dim, n_obs))
    tensor[:, 0 : int(n_obs / 2)] += 100
    print(f"{tensor=}")
    sne = SNE()
    y_hat = sne._kullback_leibler(tensor)
    print(tensor.grad)

    print(f"{y_hat=}")
    fig = px.scatter(
        x=y_hat[:, 0],
        y=y_hat[:, 1],
        color=["b"] * int(n_obs / 2) + ["r"] * int(n_obs / 2),
    )
    fig.show()
