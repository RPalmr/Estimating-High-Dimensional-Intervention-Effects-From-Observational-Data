"""Create reviewable figures and a numerical report from recorded experiment outputs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def load(directory: Path, filename: str) -> dict:
    return json.loads((directory / filename).read_text())


def report(directory: Path) -> None:
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    lines = [
        "Experiment results",
        "==================",
        "",
        "These are new runs of the paper's protocols, not its original numerical outputs. "
        "Each JSON file records the seed, environment, settings, and individual replicates.",
        "",
        "| Setting | n | Vertices | Replicates | Method | Median squared minimum error | "
        "Mean squared minimum error |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels, minimum, average = [], [], []
    for setting in ("1a", "1b", "2"):
        source = directory / f"simulation-{setting}.json"
        if not source.exists():
            continue
        result = load(directory, source.name)
        for method in ("basic", "local"):
            rows = [row[method] for row in result["rows"] if method in row]
            if not rows:
                continue
            labels.append(f"{method[0].upper()}{setting}")
            minimum.append([row["e2_min"] for row in rows])
            average.append([row["e2_ave"] for row in rows])
            lines.append(
                f"| {setting} | {result['n']} | {result['vertices']} | {len(rows)} | "
                f"{method} | {np.median(minimum[-1]):.6g} | {np.mean(minimum[-1]):.6g} |"
            )
    for ax, values, title in zip(
        axes,
        (average, minimum),
        ("Squared mean absolute error", "Squared minimum absolute error"),
        strict=True,
    ):
        if values:
            ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel("Squared error")
    fig.savefig(directory / "simulation-errors.png")
    plt.close(fig)
    if (directory / "density.json").exists():
        result = load(directory, "density.json")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, panel in zip(axes.flat, result["panels"], strict=True):
            for method in ("basic", "local"):
                density_values = np.concatenate([row[method] for row in panel["replicates"]])
                if np.ptp(density_values) > 1e-10:
                    grid = np.linspace(density_values.min() - 0.1, density_values.max() + 0.1, 400)
                    ax.plot(grid, gaussian_kde(density_values)(grid), label=method)
                else:
                    ax.axvline(density_values[0], label=f"{method} (point mass)")
            for value in np.unique(np.round(panel["truth"], 10)):
                ax.axvline(value, color="black", alpha=0.35, linestyle=":")
            ax.set_xlabel("Estimated effect")
            ax.set_ylabel("Density")
            ax.legend()
        fig.savefig(directory / "effect-densities.png")
        plt.close(fig)
        lines.extend(
            [
                "",
                "The density experiment uses four seeded DAGs, each with 8 vertices, "
                "expected degree 3, n=1,000, and 50 replicates. The original four selected "
                "DAGs were not supplied with the paper; these are independent realizations.",
            ]
        )
    if (directory / "runtime.json").exists():
        result = load(directory, "runtime.json")
        lines.extend(
            [
                "",
                f"Runtime uses {result['timeout_seconds']:g} seconds per global enumeration, "
                "rather than the paper's 48-hour timeout. A missing global mean means "
                "at least one replicate exceeded this run's budget. Times exclude process "
                "startup and JIT compilation; they include PC and all-exposure IDA.",
                "",
                "| Covariates | Basic mean seconds (SE) | Local mean seconds (SE) |",
                "| ---: | ---: | ---: |",
            ]
        )
        for row in result["rows"]:
            basic = (
                "timeout"
                if row["basic_mean"] is None
                else f"{row['basic_mean']:.4g} ({row['basic_se']:.3g})"
            )
            lines.append(
                f"| {row['p']} | {basic} | {row['local_mean']:.4g} ({row['local_se']:.3g}) |"
            )
    if (directory / "riboflavin.json").exists():
        result = load(directory, "riboflavin.json")
        lines.extend(
            [
                "",
                "Riboflavin uses all 71 observations and 4,088 covariates. "
                "Ambiguities count numerically distinct effects rounded to 10 decimal places.",
                "",
                "| Alpha | Ambiguity 1 | Ambiguity 2 | Ambiguity 3 | Ambiguity 4 | Ambiguity 5 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["ambiguity"]:
            fractions = " | ".join(
                f"{row['counts'].get(str(a), 0) / 4088:.4f}" for a in range(1, 6)
            )
            lines.append(f"| {row['alpha']} | {fractions} |")
        lines.extend(["", f"Completed bootstrap samples: {len(result['bootstrap'])}/10."])
        if "scores" in result:
            genes = sorted(result["scores"], key=lambda gene: -result["scores"][gene])
            lines.extend(["", "| Gene | Causal score |", "| --- | ---: |"])
            lines.extend(f"| {gene} | {result['scores'][gene]:.6g} |" for gene in genes[:10])
            lines.extend(
                [
                    "",
                    f"Top-ten overlap with prediction-tuned Lasso: {result['top10_overlap']}.",
                    "",
                    "No local-FDR cutoff is reported: the paper leaves the null distribution, "
                    "density estimation, and tuning choices unspecified. A cutoff would require "
                    "additional assumptions. The causal scores and their histogram are retained.",
                ]
            )
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            ax.hist(list(result["scores"].values()), bins=50, density=True)
            ax.set_xlabel("Median minimum absolute effect across 10 bootstrap samples")
            ax.set_ylabel("Density")
            fig.savefig(directory / "riboflavin-scores.png")
            plt.close(fig)
    if (directory / "random.json").exists():
        result = load(directory, "random.json")
        mse = np.mean(
            [
                (row["estimated_minimum_absolute"] - row["population_minimum_absolute"]) ** 2
                for row in result["rows"]
            ]
        )
        lines.extend(
            [
                "",
                "The additional random dataset has 30 vertices, expected degree 3, and n=500. "
                f"All 29 exposures on response {result['y']} were evaluated. "
                f"Mean squared lower-bound error: {mse:.6g}. "
                "The generated data and weights are saved in random-dataset.npz.",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation assumes a faithful linear Gaussian causal DAG and no unmeasured "
            "confounding. Observational rankings require experimental validation.",
        ]
    )
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/full"))
    args = parser.parse_args()
    report(args.input)


if __name__ == "__main__":
    main()
