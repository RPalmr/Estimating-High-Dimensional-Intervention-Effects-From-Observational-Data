"""Table 1 protocol, with an explicit per-replicate global-enumeration time budget."""

import argparse
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from experiments.run import metadata, write_json
from ida.effects import basic_ida, local_ida
from ida.pc import gaussian_pc, sample_covariance
from ida.repair import repair_pdag
from ida.simulation import random_dag


def _basic_worker(covariance, graph, y, seed, connection) -> None:
    try:
        started = perf_counter()
        repaired = repair_pdag(graph, np.random.default_rng(seed))
        basic_ida(covariance, repaired.graph, y)
        connection.send({"seconds": perf_counter() - started, "repair": repaired.strategy})
    except Exception as error:
        connection.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def runtime(output: Path, seed: int, timeout: float, repetitions: int = 10) -> None:
    result = {
        **metadata(seed),
        "n": 1000,
        "alpha": 0.01,
        "expected_degree": 3,
        "repetitions": repetitions,
        "timeout_seconds": timeout,
        "paper_timeout_seconds": 48 * 3600,
        "timing_scope": "PC plus all-exposure effects; excludes covariance, generation, and JIT",
        "rows": [],
    }
    gaussian_pc(np.eye(2), 1000)
    context = mp.get_context("spawn")
    for p in (4, 9, 14, 29, 49, 99):
        record: dict[str, Any] = {
            "p": p,
            "local_seconds": [],
            "basic_seconds": [],
            "basic_status": "complete",
        }
        for replicate in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, p, replicate]))
            model = random_dag(p + 1, 3, rng)
            y = int(rng.integers(p + 1))
            covariance = sample_covariance(model.sample(1000, rng))
            started = perf_counter()
            fitted = gaussian_pc(covariance, 1000)
            pc_seconds = perf_counter() - started
            started = perf_counter()
            local_ida(covariance, fitted.graph, y)
            record["local_seconds"].append(pc_seconds + perf_counter() - started)
            if record["basic_status"] == "complete":
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_basic_worker,
                    args=(covariance, fitted.graph, y, seed + replicate, sender),
                )
                process.start()
                sender.close()
                if receiver.poll(timeout):
                    message = receiver.recv()
                    if "error" in message:
                        process.join()
                        raise RuntimeError(message["error"])
                    record["basic_seconds"].append(pc_seconds + message["seconds"])
                else:
                    process.terminate()
                    record["basic_status"] = "timeout"
                process.join()
                receiver.close()
            print(f"Runtime p={p}: {replicate + 1}/{repetitions}", flush=True)
        for method in ("local", "basic"):
            times = record[f"{method}_seconds"]
            if len(times) == repetitions:
                record[f"{method}_mean"] = float(np.mean(times))
                record[f"{method}_se"] = float(np.std(times, ddof=1) / np.sqrt(repetitions))
            else:
                record[f"{method}_mean"] = None
                record[f"{method}_se"] = None
        result["rows"].append(record)
        write_json(output / "runtime.json", result)
    result["complete"] = True
    write_json(output / "runtime.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/full"))
    parser.add_argument("--seed", type=int, default=4214)
    parser.add_argument("--timeout", type=float, default=48 * 3600)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.timeout <= 0 or args.repetitions < 2:
        parser.error("Timeout must be positive and at least two replicates are needed")
    runtime(args.output, args.seed, args.timeout, args.repetitions)


if __name__ == "__main__":
    main()
