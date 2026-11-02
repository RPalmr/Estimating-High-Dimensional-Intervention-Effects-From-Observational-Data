"""Independent numerical and exhaustive graph checks for the paper's claims."""

from collections import Counter
from itertools import combinations, product

import numpy as np
import pytest

from ida.effects import basic_ida, grouped_ida, local_ida, regression_effect
from ida.graph import PDAG, colliders, consistent_extension, dag_extensions, dag_to_cpdag
from ida.pc import _partial, gaussian_pc, population_pc, sample_covariance
from ida.repair import repair_pdag
from ida.simulation import GaussianDAG, paper_example, random_dag
from ida.tuning import gaussian_mle, paper_bic


@pytest.mark.parametrize("number, expected", [(1, [0.4, -1, -1]), (2, [1.6, 1, 1])])
def test_paper_worked_examples(number, expected):
    model = paper_example(number)
    graph = PDAG(model.weights != 0)
    observed = [regression_effect(model.covariance, x, 3, graph.parents(x)) for x in range(3)]
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(model.total_effects[:3, 3], expected)


def test_response_parent_effect_is_exactly_zero():
    assert regression_effect(np.eye(3), 0, 2, [2]) == 0


def test_regression_matches_ordinary_least_squares_with_intercept():
    rng = np.random.default_rng(45)
    data = rng.normal(size=(70, 5)) + np.arange(5)
    data[:, 4] += 2 * data[:, 0] - data[:, 1]
    design = np.column_stack([np.ones(len(data)), data[:, [0, 1, 3]]])
    expected = np.linalg.lstsq(design, data[:, 4], rcond=None)[0][1]
    assert regression_effect(np.cov(data, rowvar=False), 0, 4, [1, 3]) == pytest.approx(expected)


def test_global_multiplicities_differ_from_local():
    adjacency = np.zeros((5, 5), dtype=bool)
    for x, y in [(0, 1), (0, 3), (1, 2)]:
        adjacency[x, y] = adjacency[y, x] = True
    adjacency[2, 4] = adjacency[3, 4] = True
    graph = PDAG(adjacency)
    extensions = list(dag_extensions(graph))
    assert len(extensions) == 4
    assert Counter(dag.parents(0) for dag in extensions) == Counter({(): 1, (1,): 2, (3,): 1})
    rng = np.random.default_rng(13)
    a = rng.normal(size=(5, 5))
    covariance = a @ a.T + np.eye(5)
    global_values = basic_ida(covariance, graph, 4, [0])[0]
    grouped = grouped_ida(covariance, graph, 4, [0])[0]
    local = local_ida(covariance, graph, 4, [0])[0]
    assert len(local.values) == 3
    np.testing.assert_allclose(sorted(global_values), sorted(grouped.values))
    np.testing.assert_allclose(np.unique(global_values), np.unique(local.values))


def test_all_four_vertex_markov_equivalence_classes_exhaustively():
    """Brute-force every labelled DAG, independently of our extension algorithm."""
    pairs = list(combinations(range(4), 2))
    classes = {}
    for states in product(range(3), repeat=len(pairs)):
        adjacency = np.zeros((4, 4), dtype=bool)
        for (x, y), state in zip(pairs, states, strict=True):
            if state == 1:
                adjacency[x, y] = True
            elif state == 2:
                adjacency[y, x] = True
        graph = PDAG(adjacency)
        try:
            graph.topological_order()
        except ValueError:
            continue
        key = ((adjacency | adjacency.T).tobytes(), tuple(sorted(colliders(graph))))
        classes.setdefault(key, []).append(adjacency)
    assert sum(map(len, classes.values())) == 543
    for members in classes.values():
        expected = np.all(members, axis=0)
        skeleton = members[0] | members[0].T
        reversible = skeleton & ~(expected | expected.T)
        expected |= reversible
        cpdag = dag_to_cpdag(members[0])
        np.testing.assert_array_equal(cpdag.adjacency, expected)
        actual = {dag.adjacency.tobytes() for dag in dag_extensions(cpdag)}
        assert actual == {member.tobytes() for member in members}
        cov = np.eye(4) + 0.2 * np.ones((4, 4))
        for x in range(3):
            global_values = basic_ida(cov, cpdag, 3, [x])[0]
            local_values = local_ida(cov, cpdag, 3, [x])[x].values
            np.testing.assert_allclose(np.unique(global_values), np.unique(local_values))


@pytest.mark.parametrize("seed", range(8))
def test_population_pc_recovers_random_faithful_dag(seed):
    model = random_dag(7, 2, np.random.default_rng(seed))
    expected = dag_to_cpdag(model.weights != 0)
    actual = population_pc(model.covariance).graph
    np.testing.assert_array_equal(actual.adjacency, expected.adjacency)


def test_partial_correlation_matches_residual_correlation():
    data = np.random.default_rng(3).normal(size=(100, 6))
    data[:, 0] += data[:, 2] + 0.7 * data[:, 1]
    correlation = np.corrcoef(data, rowvar=False)
    for conditioning in [[], [2], [2, 3], [2, 3, 4]]:
        design = np.column_stack([np.ones(len(data)), data[:, conditioning]])
        residuals = data[:, :2] - design @ np.linalg.lstsq(design, data[:, :2], rcond=None)[0]
        expected = np.corrcoef(residuals, rowvar=False)[0, 1]
        actual = _partial(correlation, 0, 1, np.array(conditioning, dtype=np.int64))
        assert actual == pytest.approx(expected)


def test_gaussian_pc_fisher_z_boundary():
    rho = 0.2
    covariance = np.array([[1, rho], [rho, 1]])
    assert not gaussian_pc(covariance, 20, 0.01).graph.adjacent(0, 1)
    assert gaussian_pc(covariance, 1000, 0.01).graph.adjacent(0, 1)


def test_nonextendable_cycle_is_accepted_locally_and_rejected_globally():
    adjacency = np.zeros((4, 4), dtype=bool)
    for x, y in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        adjacency[x, y] = adjacency[y, x] = True
    graph = PDAG(adjacency)
    assert consistent_extension(graph) is None
    assert local_ida(np.eye(4), graph, 3)[0].values
    with pytest.raises(ValueError, match="no consistent"):
        basic_ida(np.eye(4), graph, 3)
    repaired = repair_pdag(graph, np.random.default_rng(1))
    assert repaired.strategy == "random_skeleton_dag"
    assert consistent_extension(repaired.graph) is not None
    np.testing.assert_array_equal(repaired.graph.adjacency | repaired.graph.adjacency.T, adjacency)


def test_mle_and_printed_bic_formula():
    model = paper_example(1)
    graph = PDAG(model.weights != 0)
    covariance, logdet = gaussian_mle(model.covariance, graph)
    np.testing.assert_allclose(covariance, model.covariance)
    assert logdet == pytest.approx(np.linalg.slogdet(model.covariance)[1])
    expected = 100 * (4 * (1 + np.log(2 * np.pi)) + logdet) + np.log(100) * 14
    assert paper_bic(model.covariance, 100, graph) == pytest.approx(expected)


def test_invalid_inputs_and_limits():
    with pytest.raises(ValueError):
        PDAG(np.eye(3))
    with pytest.raises(ValueError):
        gaussian_pc(np.eye(3), 100, alpha=1)
    with pytest.raises(ValueError):
        sample_covariance(np.ones((10, 3)))
    with pytest.raises(ValueError):
        regression_effect(np.ones((3, 3)), 0, 2, [1])
    with pytest.raises(ValueError):
        GaussianDAG(np.ones((3, 3)), np.ones(3))
    graph = PDAG(np.ones((3, 3)) - np.eye(3))
    with pytest.raises(RuntimeError, match="limit exceeded"):
        list(dag_extensions(graph, limit=1))


def test_local_rejects_new_collider_between_parent_and_sibling():
    adjacency = np.zeros((4, 4), dtype=bool)
    adjacency[0, 1] = True
    adjacency[1, 2] = adjacency[2, 1] = True
    result = local_ida(np.eye(4), PDAG(adjacency), 3, [1])[1]
    assert result.parent_sets == ((0,),)


def test_sample_pc_is_deterministic_and_saves_collider_requests():
    model = paper_example(1)
    data = model.sample(2000, np.random.default_rng(8))
    first = gaussian_pc(sample_covariance(data), len(data))
    second = gaussian_pc(sample_covariance(data), len(data))
    np.testing.assert_array_equal(first.graph.adjacency, second.graph.adjacency)
    assert first.requested_colliders
    assert first.separating_sets == second.separating_sets


def test_repaired_graph_preserves_skeleton_and_is_extendable():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        model = random_dag(8, 3, rng)
        fitted = gaussian_pc(sample_covariance(model.sample(100, rng)), 100)
        repaired = repair_pdag(fitted.graph, rng, fitted.requested_colliders)
        assert consistent_extension(repaired.graph) is not None
        np.testing.assert_array_equal(
            repaired.graph.adjacency | repaired.graph.adjacency.T,
            fitted.graph.adjacency | fitted.graph.adjacency.T,
        )
