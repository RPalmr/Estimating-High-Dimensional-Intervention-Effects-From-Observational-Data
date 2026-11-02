"""Original, order-dependent Gaussian PC with Fisher Z conditional tests."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numba import njit, types
from numba.typed import Dict
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ida.graph import PDAG, meek_closure


@dataclass
class PCResult:
    graph: PDAG
    separating_sets: dict[tuple[int, int], tuple[int, ...]]
    tests: int
    max_order: int
    conflicting_orientations: int
    requested_colliders: tuple[tuple[int, int, int], ...] = ()


def validate_covariance(covariance: ArrayLike) -> NDArray[np.float64]:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        raise ValueError("Covariance must be square with at least two variables")
    if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T):
        raise ValueError("Covariance must be finite and symmetric")
    if np.any(np.diag(cov) <= 0):
        raise ValueError("All variables must have positive variance")
    scale = np.sqrt(np.diag(cov))
    correlation = cov / scale[:, None] / scale[None, :]
    if np.any(np.abs(correlation) > 1 + 1e-8):
        raise ValueError("Covariance contains invalid correlations")
    return cov


def sample_covariance(data: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 2:
        raise ValueError("Data must have at least four rows and two columns")
    if not np.all(np.isfinite(values)):
        raise ValueError("Data must be finite")
    return validate_covariance(np.cov(values, rowvar=False))


@njit(cache=True)
def _partial(correlation, x, y, conditioning):
    k = len(conditioning)
    if k == 0:
        return correlation[x, y]
    if k == 1:
        z = conditioning[0]
        denominator = np.sqrt(max(0.0, (1 - correlation[x, z] ** 2) * (1 - correlation[y, z] ** 2)))
        if denominator < 1e-14:
            raise ValueError("Degenerate partial correlation")
        return (correlation[x, y] - correlation[x, z] * correlation[y, z]) / denominator
    indices = np.empty(k + 2, dtype=np.int64)
    indices[0], indices[1] = x, y
    indices[2:] = conditioning
    sub = np.empty((k + 2, k + 2))
    for a in range(k + 2):
        for b in range(k + 2):
            sub[a, b] = correlation[indices[a], indices[b]]
    precision = np.linalg.pinv(sub)
    denominator = np.sqrt(precision[0, 0] * precision[1, 1])
    if denominator < 1e-14:
        raise ValueError("Degenerate partial correlation")
    return -precision[0, 1] / denominator


@njit(cache=True)
def _skeleton(correlation, n, cutoff, tolerance, population):
    p = len(correlation)
    adjacency = np.ones((p, p), dtype=np.bool_)
    for x in range(p):
        adjacency[x, x] = False
    sepsets = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    tests = 0
    order = 0
    while True:
        eligible = False
        for x in range(p):
            for y in range(p):
                if not adjacency[x, y]:
                    continue
                neighbors = np.empty(0, dtype=np.int64)
                if order:
                    neighbors = np.array(
                        [z for z in range(p) if adjacency[x, z] and z != y], dtype=np.int64
                    )
                if len(neighbors) < order:
                    continue
                eligible = True
                if not population and n - order - 3 <= 0:
                    raise ValueError("Insufficient sample size for the required PC test")
                selection = np.arange(order)
                while True:
                    conditioning = neighbors[selection].copy()
                    rho = _partial(correlation, x, y, conditioning)
                    tests += 1
                    if population:
                        independent = abs(rho) <= tolerance
                    else:
                        rho = min(1 - 1e-15, max(-1 + 1e-15, rho))
                        independent = abs(np.arctanh(rho)) * np.sqrt(n - order - 3) <= cutoff
                    if independent:
                        adjacency[x, y] = adjacency[y, x] = False
                        if order:
                            key = min(x, y) * p + max(x, y)
                            sepsets[key] = conditioning
                        break
                    position = order - 1
                    while (
                        position >= 0 and selection[position] == len(neighbors) - order + position
                    ):
                        position -= 1
                    if position < 0:
                        break
                    selection[position] += 1
                    for j in range(position + 1, order):
                        selection[j] = selection[j - 1] + 1
        if not eligible:
            break
        order += 1
    return adjacency, sepsets, tests, max(0, order - 1)


def _orient(adjacency, sepsets) -> tuple[PDAG, int, tuple[tuple[int, int, int], ...]]:
    graph = PDAG(adjacency)
    conflicts = 0
    requests = []
    for y in range(graph.size):
        for x, z in combinations(np.flatnonzero(adjacency[y]).tolist(), 2):
            if adjacency[x, z] or y in sepsets.get((x, z), ()):
                continue
            requests.append((x, y, z))
            for source in (x, z):
                conflicts += int(graph.directed(y, source))
                graph.orient(source, y)
    return meek_closure(graph), conflicts, tuple(requests)


def gaussian_pc(covariance: ArrayLike, n: int, alpha: float = 0.01) -> PCResult:
    """Estimate a PDAG; conflicting colliders overwrite by vertex order (Section 4.2)."""
    cov = validate_covariance(covariance)
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("Sample size must be an integer of at least four")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha must lie strictly between zero and one")
    return _run(cov, n, float(norm.isf(alpha / 2)), 0.0, False)


def population_pc(covariance: ArrayLike, tolerance: float = 1e-10) -> PCResult:
    """Oracle Gaussian PC, interpreting partial correlations near zero as independence."""
    cov = validate_covariance(covariance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    return _run(cov, 0, 0.0, tolerance, True)


def _run(cov, n, cutoff, tolerance, population) -> PCResult:
    scale = np.sqrt(np.diag(cov))
    correlation = np.ascontiguousarray(cov / scale[:, None] / scale[None, :])
    adjacency, sparse_sets, tests, order = _skeleton(correlation, n, cutoff, tolerance, population)
    p = len(cov)
    sepsets = {divmod(int(key), p): tuple(value.tolist()) for key, value in sparse_sets.items()}
    graph, conflicts, requests = _orient(adjacency, sepsets)
    return PCResult(graph, sepsets, tests, order, conflicts, requests)
