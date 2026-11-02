"""Algorithms 1–3 and the parent-adjusted regression in equation (4)."""

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ida.graph import PDAG, dag_extensions
from ida.pc import validate_covariance


@dataclass(frozen=True)
class EffectSet:
    values: tuple[float, ...]
    parent_sets: tuple[tuple[int, ...], ...]

    @property
    def minimum_absolute(self) -> float:
        if not self.values:
            raise ValueError("No admissible effects")
        return min(abs(value) for value in self.values)

    @property
    def bounds(self) -> tuple[float, float]:
        if not self.values:
            raise ValueError("No admissible effects")
        return min(self.values), max(self.values)


def _beta(cov: NDArray[np.float64], x: int, y: int, parents: tuple[int, ...]) -> float:
    if y in parents:
        return 0.0
    predictors = (x, *parents)
    block = cov[np.ix_(predictors, predictors)]
    if np.linalg.matrix_rank(block) != len(predictors):
        raise ValueError("Singular parent-adjusted regression")
    return float(np.linalg.solve(block, cov[list(predictors), y])[0])


def regression_effect(covariance: ArrayLike, x: int, y: int, parents: Sequence[int]) -> float:
    cov = validate_covariance(covariance)
    if x == y or not 0 <= x < len(cov) or not 0 <= y < len(cov):
        raise ValueError("Exposure and response must be distinct valid indices")
    pa = tuple(sorted(parents))
    if len(set(pa)) != len(pa) or x in pa or any(not 0 <= v < len(cov) for v in pa):
        raise ValueError("Invalid parent set")
    return _beta(cov, x, y, pa)


def local_parent_sets(graph: PDAG, x: int) -> Iterator[tuple[int, ...]]:
    """Reject every new unshielded collider at x, including parent–sibling pairs."""
    graph.check_vertex(x)
    parents, siblings = graph.parents(x), graph.siblings(x)
    for size in range(len(siblings) + 1):
        for subset in combinations(siblings, size):
            if any(not graph.adjacent(a, b) for a, b in combinations(subset, 2)):
                continue
            if any(not graph.adjacent(a, b) for a in subset for b in parents):
                continue
            yield tuple(sorted((*parents, *subset)))


def _inputs(covariance, graph, y, exposures):
    cov = validate_covariance(covariance)
    if len(cov) != graph.size:
        raise ValueError("Covariance and graph dimensions differ")
    graph.check_vertex(y)
    xs = tuple(x for x in range(graph.size) if x != y) if exposures is None else tuple(exposures)
    if len(set(xs)) != len(xs) or y in xs:
        raise ValueError("Exposures must be unique and exclude the response")
    for x in xs:
        graph.check_vertex(x)
    return cov, xs


def local_ida(
    covariance: ArrayLike,
    graph: PDAG,
    y: int,
    exposures: Sequence[int] | None = None,
) -> dict[int, EffectSet]:
    cov, xs = _inputs(covariance, graph, y, exposures)
    result = {}
    for x in xs:
        parents = tuple(local_parent_sets(graph, x))
        result[x] = EffectSet(tuple(_beta(cov, x, y, pa) for pa in parents), parents)
    return result


def basic_ida(
    covariance: ArrayLike,
    graph: PDAG,
    y: int,
    exposures: Sequence[int] | None = None,
    limit: int | None = None,
) -> NDArray[np.float64]:
    """Rows follow exposures; columns retain the common DAG identity."""
    cov, xs = _inputs(covariance, graph, y, exposures)
    columns = [
        [_beta(cov, x, y, dag.parents(x)) for x in xs] for dag in dag_extensions(graph, limit=limit)
    ]
    if not columns:
        raise ValueError("PDAG has no consistent DAG extension; repair it explicitly")
    return np.asarray(columns, dtype=float).T


def grouped_ida(
    covariance: ArrayLike,
    graph: PDAG,
    y: int,
    exposures: Sequence[int] | None = None,
    limit: int | None = None,
) -> dict[int, EffectSet]:
    """Algorithm 2 counts extensions per parent set, preserving global multiplicities."""
    cov, xs = _inputs(covariance, graph, y, exposures)
    counts: dict[int, Counter] = {x: Counter() for x in xs}
    for dag in dag_extensions(graph, limit=limit):
        for x in xs:
            counts[x][dag.parents(x)] += 1
    if xs and not counts[xs[0]]:
        raise ValueError("PDAG has no consistent DAG extension; repair it explicitly")
    result = {}
    for x in xs:
        parents = tuple(pa for pa, count in sorted(counts[x].items()) for _ in range(count))
        result[x] = EffectSet(tuple(_beta(cov, x, y, pa) for pa in parents), parents)
    return result
