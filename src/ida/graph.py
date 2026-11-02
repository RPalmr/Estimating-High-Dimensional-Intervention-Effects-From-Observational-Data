"""PDAG representation, orientation, and exact consistent DAG extensions."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolMatrix = NDArray[np.bool_]


@dataclass
class PDAG:
    """A[i,j] encodes i -> j; two entries encode the undirected edge i - j."""

    adjacency: BoolMatrix

    def __post_init__(self) -> None:
        a = np.asarray(self.adjacency)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError("Adjacency must be a nonempty square matrix")
        if not np.all((a == 0) | (a == 1)) or np.any(np.diag(a)):
            raise ValueError("Adjacency must be binary with no self edges")
        self.adjacency = a.astype(bool, copy=True)

    @property
    def size(self) -> int:
        return self.adjacency.shape[0]

    def check_vertex(self, vertex: int) -> None:
        if not isinstance(vertex, (int, np.integer)) or not 0 <= vertex < self.size:
            raise ValueError("Vertex index outside graph")

    def adjacent(self, x: int, y: int) -> bool:
        return bool(self.adjacency[x, y] or self.adjacency[y, x])

    def directed(self, x: int, y: int) -> bool:
        return bool(self.adjacency[x, y] and not self.adjacency[y, x])

    def parents(self, x: int) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.adjacency[:, x] & ~self.adjacency[x, :]).tolist())

    def children(self, x: int) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.adjacency[x, :] & ~self.adjacency[:, x]).tolist())

    def siblings(self, x: int) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.adjacency[:, x] & self.adjacency[x, :]).tolist())

    def neighbors(self, x: int) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.adjacency[:, x] | self.adjacency[x, :]).tolist())

    def orient(self, x: int, y: int) -> None:
        if not self.adjacent(x, y):
            raise ValueError("Cannot orient an absent edge")
        self.adjacency[x, y], self.adjacency[y, x] = True, False

    def copy(self) -> "PDAG":
        return PDAG(self.adjacency)

    def topological_order(self) -> tuple[int, ...]:
        if np.any(self.adjacency & self.adjacency.T):
            raise ValueError("Topological order requires a directed graph")
        degrees = self.adjacency.sum(axis=0).astype(int)
        pending = np.flatnonzero(degrees == 0).tolist()
        order = []
        while pending:
            x = int(pending.pop())
            order.append(x)
            for y in self.children(x):
                degrees[y] -= 1
                if degrees[y] == 0:
                    pending.append(y)
        if len(order) != self.size:
            raise ValueError("Graph contains a directed cycle")
        return tuple(order)


def colliders(graph: PDAG) -> set[tuple[int, int, int]]:
    return {
        (x, y, z)
        for y in range(graph.size)
        for x, z in combinations(graph.parents(y), 2)
        if not graph.adjacent(x, z)
    }


def meek_closure(graph: PDAG) -> PDAG:
    """Apply Meek's three orientation rules, in vertex order, to convergence."""
    g = graph.copy()
    changed = True
    while changed:
        changed = False
        for a in range(g.size):
            for b in g.siblings(a):
                if not g.adjacency[b, a]:
                    continue
                r1 = any(not g.adjacent(c, b) for c in g.parents(a))
                r2 = any(g.directed(c, b) for c in g.children(a))
                candidates = [c for c in g.siblings(a) if g.directed(c, b)]
                r3 = any(not g.adjacent(c, d) for c, d in combinations(candidates, 2))
                if r1 or r2 or r3:
                    g.orient(a, b)
                    changed = True
    return g


def dag_to_cpdag(adjacency: ArrayLike) -> PDAG:
    dag = PDAG(np.asarray(adjacency))
    dag.topological_order()
    g = PDAG(dag.adjacency | dag.adjacency.T)
    for x, y, z in sorted(colliders(dag)):
        g.orient(x, y)
        g.orient(z, y)
    return meek_closure(g)


def consistent_extension(graph: PDAG) -> PDAG | None:
    """Dor–Tarsi sink elimination, without adding unshielded colliders."""
    result = graph.copy()
    active = set(range(graph.size))
    while active:
        found = False
        for x in sorted(active):
            if any(y in active for y in result.children(x)):
                continue
            siblings = set(result.siblings(x)) & active
            adjacent = (set(result.neighbors(x)) & active) - {x}
            if any(not result.adjacent(y, z) for y in siblings for z in adjacent - {y}):
                continue
            for y in siblings:
                result.orient(y, x)
            active.remove(x)
            found = True
            break
        if not found:
            return None
    return result


def dag_extensions(graph: PDAG, limit: int | None = None) -> Iterator[PDAG]:
    """Enumerate every consistent DAG once; refuse silent truncation."""
    count = 0

    def visit(g: PDAG) -> Iterator[PDAG]:
        nonlocal count
        if consistent_extension(g) is None:
            return
        edge = next(((x, y) for x in range(g.size) for y in g.siblings(x)), None)
        if edge is None:
            count += 1
            if limit is not None and count > limit:
                raise RuntimeError("DAG enumeration limit exceeded")
            yield g
            return
        x, y = edge
        allowed = colliders(g)
        for u, v in ((x, y), (y, x)):
            candidate = g.copy()
            candidate.orient(u, v)
            if colliders(candidate) == allowed:
                yield from visit(candidate)

    yield from visit(graph.copy())
