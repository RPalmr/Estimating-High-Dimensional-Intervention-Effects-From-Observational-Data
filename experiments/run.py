"""Run the paper protocols and a separate random Gaussian DAG experiment."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from ida.effects import basic_ida, local_ida
from ida.graph import dag_to_cpdag
from ida.pc import gaussian_pc, sample_covariance
from ida.repair import repair_pdag
from ida.simulation import random_dag


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metadata(seed: int) -> dict:
    return {
        "seed": seed,
        "time_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "numba", "scikit-learn", "matplotlib")
        },
        "paper": "arXiv:0810.4214v3",
    }


def errors(estimated, truth) -> dict:
    absolute, target = np.abs(estimated), np.abs(truth)
    return {
        "e2_ave": float((np.mean(absolute) - np.mean(target)) ** 2),
        "e2_min": float((np.min(absolute) - np.min(target)) ** 2),
    }


def simulations(output: Path, seed: int, setting: str, repetitions: int | None) -> None:
    size, n, default = {"1a": (10, 20, 1000), "1b": (10, 2000, 1000), "2": (1000, 100, 100)}[
        setting
    ]
    reps = default if repetitions is None else repetitions
    result = {
        **metadata(seed),
        "setting": setting,
        "vertices": size,
        "n": n,
        "expected_degree": 4,
        "alpha": 0.01,
        "requested_repetitions": reps,
        "paper_repetitions": default,
        "rows": [],
    }
    for replicate in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        model = random_dag(size, 4, rng, block_size=10)
        x, y = rng.choice(10, size=2, replace=False).tolist()
        truth_model = model.weights[:10, :10]
        truth_cov = type(model)(truth_model, np.ones(10)).covariance
        truth = basic_ida(truth_cov, dag_to_cpdag(truth_model != 0), y, [x])[0]
        data = model.sample(n, rng)
        covariance = sample_covariance(data)
        started = perf_counter()
        fitted = gaussian_pc(covariance, n, 0.01)
        local = local_ida(covariance, fitted.graph, y, [x])[x].values
        row = {
            "replicate": replicate,
            "x": x,
            "y": y,
            "local": errors(local, truth),
            "local_seconds": perf_counter() - started,
            "conflicts": fitted.conflicting_orientations,
            "pc_tests": fitted.tests,
        }
        if setting != "2":
            started = perf_counter()
            repaired = repair_pdag(fitted.graph, rng, fitted.requested_colliders)
            basic = basic_ida(covariance, repaired.graph, y, [x])[0]
            row["basic"] = errors(basic, truth)
            row["basic_enumeration_seconds"] = perf_counter() - started
            row["repair"] = repaired.strategy
        result["rows"].append(row)
        write_json(output / f"simulation-{setting}.json", result)
        if replicate % 10 == 0 or replicate == reps - 1:
            print(f"Simulation {setting}: {replicate + 1}/{reps}", flush=True)
    result["complete"] = True
    write_json(output / f"simulation-{setting}.json", result)


def density(output: Path, seed: int, repetitions: int | None) -> None:
    reps = 50 if repetitions is None else repetitions
    result = {**metadata(seed), "n": 1000, "vertices": 8, "expected_degree": 3, "panels": []}
    for panel in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([seed, panel]))
        model = random_dag(8, 3, rng)
        x, y = rng.choice(8, size=2, replace=False).tolist()
        truth = basic_ida(model.covariance, dag_to_cpdag(model.weights != 0), y, [x])[0]
        record = {"x": x, "y": y, "weights": model.weights.tolist(), "truth": truth.tolist()}
        record["replicates"] = []
        for replicate in range(reps):
            covariance = sample_covariance(model.sample(1000, rng))
            fitted = gaussian_pc(covariance, 1000)
            repaired = repair_pdag(fitted.graph, rng, fitted.requested_colliders)
            record["replicates"].append(
                {
                    "replicate": replicate,
                    "local": local_ida(covariance, fitted.graph, y, [x])[x].values,
                    "basic": basic_ida(covariance, repaired.graph, y, [x])[0].tolist(),
                    "repair": repaired.strategy,
                }
            )
        result["panels"].append(record)
        write_json(output / "density.json", result)
        print(f"Density panels: {panel + 1}/4", flush=True)
    result["complete"] = True
    write_json(output / "density.json", result)


def random_dataset(output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    model = random_dag(30, 3, rng)
    data = model.sample(500, rng)
    x, y = rng.choice(30, size=2, replace=False).tolist()
    covariance = sample_covariance(data)
    fitted = gaussian_pc(covariance, len(data))
    truth_graph = dag_to_cpdag(model.weights != 0)
    estimated = local_ida(covariance, fitted.graph, y)
    population = local_ida(model.covariance, truth_graph, y)
    np.savez_compressed(output / "random-dataset.npz", data=data, weights=model.weights, x=x, y=y)
    rows = []
    for exposure in estimated:
        rows.append(
            {
                "exposure": exposure,
                "estimated_values": estimated[exposure].values,
                "population_values": population[exposure].values,
                "true_effect": float(model.total_effects[exposure, y]),
                "estimated_minimum_absolute": estimated[exposure].minimum_absolute,
                "population_minimum_absolute": population[exposure].minimum_absolute,
            }
        )
    write_json(
        output / "random.json",
        {
            **metadata(seed),
            "complete": True,
            "vertices": 30,
            "n": 500,
            "expected_degree": 3,
            "y": y,
            "random_exposure": x,
            "alpha": 0.01,
            "dataset_sha256": hashlib.sha256(
                (output / "random-dataset.npz").read_bytes()
            ).hexdigest(),
            "weights": model.weights.tolist(),
            "rows": rows,
        },
    )
    print("Additional random dataset complete", flush=True)


def read_riboflavin(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="") as source:
        rows = list(csv.reader(source))
    header = rows[0]
    if header[0] != "y":
        raise ValueError("Riboflavin CSV must put response 'y' first")
    data = np.asarray(rows[1:], dtype=float)
    if data.shape != (71, 4089) or not np.all(np.isfinite(data)):
        raise ValueError("Expected the complete finite 71 by 4089 riboflavin dataset")
    return np.column_stack([data[:, 1:], data[:, 0]]), header[1:]


def riboflavin(output: Path, seed: int, path: Path, resume: bool = False) -> None:
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data, genes = read_riboflavin(path)
    data[:, :-1] = (data[:, :-1] - data[:, :-1].mean(axis=0)) / data[:, :-1].std(axis=0, ddof=1)
    covariance = sample_covariance(data)
    result = {
        **metadata(seed),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n": 71,
        "genes": 4088,
        "ambiguity": [],
        "bootstrap": [],
    }
    checkpoint = output / "riboflavin.json"
    if resume and checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        for key in ("seed", "source_sha256", "n", "genes"):
            if previous.get(key) != result[key]:
                raise ValueError(f"Checkpoint has incompatible {key}")
        result = previous
    for alpha in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        if any(row["alpha"] == alpha for row in result["ambiguity"]):
            continue
        started = perf_counter()
        fitted = gaussian_pc(covariance, len(data), alpha)
        effects = local_ida(covariance, fitted.graph, 4088)
        ambiguity = np.array([len(np.unique(np.round(e.values, 10))) for e in effects.values()])
        result["ambiguity"].append(
            {
                "alpha": alpha,
                "counts": {str(int(a)): int(np.sum(ambiguity == a)) for a in np.unique(ambiguity)},
                "seconds": perf_counter() - started,
                "pc_tests": fitted.tests,
                "max_order": fitted.max_order,
                "conflicts": fitted.conflicting_orientations,
            }
        )
        write_json(output / "riboflavin.json", result)
        print(f"Riboflavin alpha {alpha} complete", flush=True)
    for replicate in range(len(result["bootstrap"]), 10):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        sample = data[rng.integers(0, len(data), size=len(data))].copy()
        sample[:, :-1] /= sample[:, :-1].std(axis=0, ddof=1)
        cov = sample_covariance(sample)
        fitted = gaussian_pc(cov, len(sample), 0.01)
        effects = local_ida(cov, fitted.graph, 4088)
        result["bootstrap"].append([effects[x].minimum_absolute for x in range(4088)])
        write_json(output / "riboflavin.json", result)
        print(f"Riboflavin bootstrap {replicate + 1}/10 complete", flush=True)
    scores = np.median(result["bootstrap"], axis=0)
    folds = KFold(n_splits=10, shuffle=True, random_state=seed)
    lasso = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("fit", Lasso(max_iter=100000))]),
        {"fit__alpha": np.geomspace(1e-4, 10, 100)},
        scoring="neg_mean_squared_error",
        cv=folds,
    )
    lasso.fit(data[:, :-1], data[:, -1])
    coefficients = lasso.best_estimator_.named_steps["fit"].coef_
    causal_top = np.argsort(-scores, kind="stable")[:10]
    lasso_top = np.argsort(-np.abs(coefficients), kind="stable")[:10]
    result["scores"] = dict(zip(genes, scores.tolist(), strict=True))
    result["lasso_coefficients"] = dict(zip(genes, coefficients.tolist(), strict=True))
    result["top10_overlap"] = sorted(
        set(genes[i] for i in causal_top) & set(genes[i] for i in lasso_top)
    )
    result["lasso_alpha"] = float(lasso.best_params_["fit__alpha"])
    result["lasso_protocol"] = "10-fold shuffled CV; scaling fitted within each training fold"
    result["local_fdr"] = {
        "status": "unspecified_in_paper",
        "reason": "The paper leaves empirical-null, density-fit, and tuning choices unspecified",
    }
    result["complete"] = True
    write_json(output / "riboflavin.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("simulation", "density", "riboflavin", "random"))
    parser.add_argument("--setting", choices=("1a", "1b", "2"), default="1a")
    parser.add_argument("--seed", type=int, default=4214)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/full"))
    parser.add_argument("--data", type=Path, default=Path("data/riboflavin.csv"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repetitions is not None and args.repetitions < 1:
        parser.error("Repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.experiment == "simulation":
        simulations(args.output, args.seed, args.setting, args.repetitions)
    elif args.experiment == "density":
        density(args.output, args.seed, args.repetitions)
    elif args.experiment == "riboflavin":
        riboflavin(args.output, args.seed, args.data, args.resume)
    else:
        random_dataset(args.output, args.seed)


if __name__ == "__main__":
    main()
