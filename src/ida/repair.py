"""Explicit, auditable sample-PDAG repairs following Section 4.2."""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from ida.graph import PDAG, colliders, consistent_extension, dag_to_cpdag, meek_closure


@dataclass(frozen=True)
class RepairResult:
    graph: PDAG
    strategy: str
    destroyed_colliders: int


def repair_pdag(
    graph: PDAG,
    rng: np.random.Generator,
    requested_colliders: tuple[tuple[int, int, int], ...] = (),
) -> RepairResult:
    """Try conflicting directions, minimum collider deletion, then a random skeleton DAG."""
    if consistent_extension(graph) is not None:
        return RepairResult(graph.copy(), "unchanged", 0)
    skeleton = graph.adjacency | graph.adjacency.T
    original = sorted(colliders(graph))
    arrows = {(source, y) for x, y, z in requested_colliders for source in (x, z)}
    conflicting_edges = sorted({tuple(sorted((x, y))) for x, y in arrows if (y, x) in arrows})
    for directions in product((False, True), repeat=len(conflicting_edges)):
        if not conflicting_edges:
            break
        candidate = graph.copy()
        for (a, b), reverse in zip(conflicting_edges, directions, strict=True):
            candidate.orient(b, a) if reverse else candidate.orient(a, b)
        extension = consistent_extension(candidate)
        if extension is not None:
            return RepairResult(dag_to_cpdag(extension.adjacency), "rearranged", 0)
    for removed in range(1, len(original) + 1):
        for deleted in combinations(range(len(original)), removed):
            candidate = PDAG(skeleton)
            valid = True
            for index, (x, y, z) in enumerate(original):
                if index in deleted:
                    continue
                for source in (x, z):
                    if candidate.directed(y, source):
                        valid = False
                        break
                    candidate.orient(source, y)
                if not valid:
                    break
            if not valid:
                continue
            candidate = meek_closure(candidate)
            extension = consistent_extension(candidate)
            if extension is not None:
                return RepairResult(dag_to_cpdag(extension.adjacency), "deleted", removed)
    order = rng.permutation(graph.size)
    rank = np.argsort(order)
    adjacency = skeleton & (rank[:, None] < rank[None, :])
    return RepairResult(dag_to_cpdag(adjacency), "random_skeleton_dag", len(original))
