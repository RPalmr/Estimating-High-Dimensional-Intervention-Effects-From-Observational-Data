"""Random Gaussian structural equation models used in Section 6.1."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class GaussianDAG:
    """weights[parent, child] is the coefficient of that structural equation."""

    weights: NDArray[np.float64]
    noise_variances: NDArray[np.float64]

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        variances = np.asarray(self.noise_variances, dtype=float)
        if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
            raise ValueError("Weights must be a square matrix of at least two variables")
        if not np.all(np.isfinite(w)) or np.any(np.tril(w) != 0):
            raise ValueError("Simulation weights must be finite and strictly upper triangular")
        if variances.shape != (len(w),) or not np.all(np.isfinite(variances)):
            raise ValueError("Noise variances must match the graph dimension")
        if np.any(variances <= 0):
            raise ValueError("Noise variances must be positive")
        object.__setattr__(self, "weights", w.copy())
        object.__setattr__(self, "noise_variances", variances.copy())

    @property
    def covariance(self) -> NDArray[np.float64]:
        transform = solve_triangular(
            np.eye(len(self.weights)) - self.weights.T,
            np.eye(len(self.weights)),
            lower=True,
        )
        return (transform * self.noise_variances) @ transform.T

    @property
    def total_effects(self) -> NDArray[np.float64]:
        return solve_triangular(
            np.eye(len(self.weights)) - self.weights,
            np.eye(len(self.weights)),
            lower=False,
        ) - np.eye(len(self.weights))

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        if n < 4:
            raise ValueError("At least four observations are required")
        data = rng.normal(size=(n, len(self.weights))) * np.sqrt(self.noise_variances)
        for child in range(len(self.weights)):
            parents = np.flatnonzero(self.weights[:, child])
            if len(parents):
                data[:, child] += data[:, parents] @ self.weights[parents, child]
        return data


def random_dag(
    vertices: int,
    expected_degree: float,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> GaussianDAG:
    """Ordered Bernoulli DAG, probability en/(block_size-1), U[1,2] edge weights."""
    block = vertices if block_size is None else block_size
    if vertices < 2 or block < 2 or vertices % block or not 0 <= expected_degree <= block - 1:
        raise ValueError("Invalid graph size, block size, or expected degree")
    weights = np.zeros((vertices, vertices))
    for start in range(0, vertices, block):
        edges = np.triu(rng.random((block, block)) < expected_degree / (block - 1), 1)
        weights[start : start + block, start : start + block] = edges * rng.uniform(
            1, 2, (block, block)
        )
    return GaussianDAG(weights, np.ones(vertices))


def paper_example(number: int) -> GaussianDAG:
    """Examples 2.1 and 2.2 in topological order X2, X1, X3, Y."""
    if number not in (1, 2):
        raise ValueError("Example must be 1 or 2")
    weights = np.zeros((4, 4))
    weights[0, 1:3] = 0.8
    weights[:, 3] = [2, -1, -1, 0] if number == 1 else [0, 1, 1, 0]
    return GaussianDAG(weights, np.array([1, 0.36, 0.36, 1]))
