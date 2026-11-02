"""Gaussian DAG likelihood and the BIC expression printed in Section 4.1."""

from dataclasses import dataclass
from math import log, pi

import numpy as np
from numpy.typing import ArrayLike

from ida.graph import PDAG, consistent_extension
from ida.pc import PCResult, gaussian_pc, validate_covariance


@dataclass(frozen=True)
class BICFit:
    alpha: float
    score: float
    result: PCResult


def gaussian_mle(covariance: ArrayLike, graph: PDAG) -> tuple[np.ndarray, float]:
    """Input covariance uses denominator n; return fitted covariance and logdet."""
    cov = validate_covariance(covariance)
    if len(cov) != graph.size:
        raise ValueError("Graph dimension does not match covariance")
    graph.topological_order()
    coefficients = np.zeros_like(cov)
    residuals = np.diag(cov).copy()
    for x in range(graph.size):
        parents = graph.parents(x)
        if parents:
            beta = np.linalg.solve(cov[np.ix_(parents, parents)], cov[list(parents), x])
            coefficients[list(parents), x] = beta
            residuals[x] -= cov[x, list(parents)] @ beta
    if np.any(residuals <= 0):
        raise ValueError("Gaussian likelihood has nonpositive residual variance")
    transform = np.linalg.solve(np.eye(len(cov)) - coefficients.T, np.eye(len(cov)))
    return (transform * residuals) @ transform.T, float(np.log(residuals).sum())


def paper_bic(covariance_mle: ArrayLike, n: int, graph: PDAG) -> float:
    """Count nonzero upper-triangular fitted covariance entries, as printed in the paper."""
    if n < 4:
        raise ValueError("Sample size must be at least four")
    fitted, logdet = gaussian_mle(covariance_mle, graph)
    minus_twice_loglik = n * (graph.size * (1 + log(2 * pi)) + logdet)
    parameters = np.count_nonzero(np.abs(np.triu(fitted)) > 1e-10) + graph.size
    return float(minus_twice_loglik + log(n) * parameters)


def select_alpha(data: ArrayLike, alphas: tuple[float, ...]) -> tuple[BICFit, tuple[BICFit, ...]]:
    values = np.asarray(data, dtype=float)
    if not alphas:
        raise ValueError("Supply at least one alpha candidate")
    if values.ndim != 2 or len(values) < 4 or not np.all(np.isfinite(values)):
        raise ValueError("Data must be finite with at least four rows")
    cov = np.cov(values, rowvar=False, ddof=0)
    fits = []
    for alpha in alphas:
        result = gaussian_pc(cov, len(values), alpha)
        extension = consistent_extension(result.graph)
        if extension is None:
            raise ValueError("BIC requires an extendable graph; repair the PC result explicitly")
        fits.append(BICFit(alpha, paper_bic(cov, len(values), extension), result))
    return min(fits, key=lambda fit: fit.score), tuple(fits)
