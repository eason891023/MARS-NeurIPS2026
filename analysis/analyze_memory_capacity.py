#!/usr/bin/env python3
"""Temporal-expressivity statistics and plots from analysis/memory_capacity_eval.py output."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402

# Raw output of analysis/memory_capacity_eval.py (shipped: results/analysis/memory_capacity_raw.json);
# the analyzed file is shipped as results/analysis/memory_capacity.json.
DEFAULT_INPUT = RESULTS / "analysis" / "memory_capacity_raw.json"
DEFAULT_OUTPUT = OUTPUT_ROOT / "analysis" / "memory_capacity"
MODES = ["clean", "noise_aug", "asar"]
LABELS = {"clean": "Clean", "noise_aug": "NoiseAug", "asar": "MARS"}
COLORS = {"clean": "#4477AA", "noise_aug": "#CCBB44", "asar": "#228833"}


def mean_ci(values: list[float], seed: int) -> dict:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = array[
        rng.integers(0, len(array), size=(50_000, len(array)))
    ].mean(1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "bootstrap_95ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def exact_sign_flip_p(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=float)
    observed = abs(float(values.mean()))
    null = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        null.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def validate(records: list[dict], architectures: list[str]) -> None:
    seeds = [42, 52, 72, 92, 102]
    lengths = [5, 10, 20, 30, 40, 50]
    lags = [1, 2, 5, 10, 20]
    expected_context = {
        (architecture, mode, seed, kind, length)
        for architecture in architectures
        for mode in MODES
        for seed in seeds
        for kind in ["prefix", "suffix"]
        for length in lengths
    }
    expected_probe = {
        (architecture, mode, seed, lag)
        for architecture in architectures
        for mode in MODES
        for seed in seeds
        for lag in lags
    }
    actual_context = {
        (
            r["architecture"],
            r["mode"],
            r["seed"],
            r["kind"],
            r["length"],
        )
        for r in records
        if r["test"] == "context" and r["architecture"] in architectures
    }
    actual_probe = {
        (r["architecture"], r["mode"], r["seed"], r["lag"])
        for r in records
        if r["test"] == "lag_probe" and r["architecture"] in architectures
    }
    missing_context = expected_context - actual_context
    missing_probe = expected_probe - actual_probe
    if missing_context or missing_probe:
        raise RuntimeError(
            f"Incomplete formal memory evaluation: "
            f"{len(missing_context)} context and {len(missing_probe)} probe "
            "records missing"
        )


def analyze(records: list[dict], architectures: list[str]) -> dict:
    context = {
        (
            r["architecture"],
            r["mode"],
            r["seed"],
            r["kind"],
            r["length"],
        ): float(r["accuracy"])
        for r in records
        if r["test"] == "context"
    }
    probe = {
        (r["architecture"], r["mode"], r["seed"], r["lag"]): r
        for r in records
        if r["test"] == "lag_probe"
    }
    seeds = [42, 52, 72, 92, 102]
    lengths = [5, 10, 20, 30, 40, 50]
    lags = [1, 2, 5, 10, 20]
    output: dict = {
        "metadata": {
            "architectures": architectures,
            "modes": MODES,
            "seeds": seeds,
            "context_lengths": lengths,
            "probe_lags": lags,
            "task_horizon_definition": (
                "smallest retained prefix/suffix length reaching at least 95% "
                "of that checkpoint's full-length accuracy"
            ),
        },
        "context": {},
        "probe": {},
        "paired_mars_minus_clean": {},
    }
    for architecture in architectures:
        output["context"][architecture] = {}
        output["probe"][architecture] = {}
        output["paired_mars_minus_clean"][architecture] = {
            "context": {},
            "probe": {},
        }
        for mode in MODES:
            output["context"][architecture][mode] = {}
            for kind in ["prefix", "suffix"]:
                curves = {
                    length: [
                        context[(architecture, mode, seed, kind, length)]
                        for seed in seeds
                    ]
                    for length in lengths
                }
                horizons = []
                for seed_index, seed in enumerate(seeds):
                    full = curves[50][seed_index]
                    threshold = 0.95 * full
                    horizon = next(
                        (
                            length
                            for length in lengths
                            if curves[length][seed_index] >= threshold
                        ),
                        50,
                    )
                    horizons.append(horizon)
                output["context"][architecture][mode][kind] = {
                    "accuracy_by_length": {
                        str(length): mean_ci(
                            curves[length],
                            seed=25_072_500 + length,
                        )
                        for length in lengths
                    },
                    "horizon95_bins": mean_ci(
                        horizons,
                        seed=25_072_600
                        + (0 if kind == "prefix" else 1),
                    ),
                    "per_seed_horizon95_bins": {
                        str(seed): horizon
                        for seed, horizon in zip(seeds, horizons)
                    },
                }
            output["probe"][architecture][mode] = {
                str(lag): {
                    "r2": mean_ci(
                        [
                            float(probe[(architecture, mode, seed, lag)]["r2"])
                            for seed in seeds
                        ],
                        seed=25_072_700 + lag,
                    ),
                    "pearson_r": mean_ci(
                        [
                            float(
                                probe[(architecture, mode, seed, lag)][
                                    "pearson_r"
                                ]
                            )
                            for seed in seeds
                        ],
                        seed=25_072_800 + lag,
                    ),
                }
                for lag in lags
            }

        for kind in ["prefix", "suffix"]:
            output["paired_mars_minus_clean"][architecture]["context"][kind] = {}
            for length in lengths:
                differences = [
                    context[(architecture, "asar", seed, kind, length)]
                    - context[(architecture, "clean", seed, kind, length)]
                    for seed in seeds
                ]
                output["paired_mars_minus_clean"][architecture]["context"][kind][
                    str(length)
                ] = {
                    **mean_ci(differences, seed=25_073_000 + length),
                    "exact_two_sided_sign_flip_p": exact_sign_flip_p(
                        differences
                    ),
                }
        for lag in lags:
            differences = [
                float(probe[(architecture, "asar", seed, lag)]["r2"])
                - float(probe[(architecture, "clean", seed, lag)]["r2"])
                for seed in seeds
            ]
            output["paired_mars_minus_clean"][architecture]["probe"][str(lag)] = {
                **mean_ci(differences, seed=25_074_000 + lag),
                "exact_two_sided_sign_flip_p": exact_sign_flip_p(differences),
            }
    return output


def make_plot(results: dict, output_dir: Path) -> None:
    architectures = results["metadata"]["architectures"]
    lengths = results["metadata"]["context_lengths"]
    lags = results["metadata"]["probe_lags"]
    fig, axes = plt.subplots(
        len(architectures),
        3,
        figsize=(11.2, 3.3 * len(architectures)),
        squeeze=False,
    )
    for row, architecture in enumerate(architectures):
        for mode in MODES:
            for column, kind in enumerate(["prefix", "suffix"]):
                values = results["context"][architecture][mode][kind][
                    "accuracy_by_length"
                ]
                means = [values[str(length)]["mean"] for length in lengths]
                stds = [values[str(length)]["std"] for length in lengths]
                axes[row, column].plot(
                    lengths,
                    means,
                    marker="o",
                    color=COLORS[mode],
                    label=LABELS[mode],
                )
                axes[row, column].fill_between(
                    lengths,
                    np.asarray(means) - np.asarray(stds),
                    np.asarray(means) + np.asarray(stds),
                    color=COLORS[mode],
                    alpha=0.10,
                )
            values = results["probe"][architecture][mode]
            means = [values[str(lag)]["r2"]["mean"] for lag in lags]
            stds = [values[str(lag)]["r2"]["std"] for lag in lags]
            axes[row, 2].plot(
                lags,
                means,
                marker="o",
                color=COLORS[mode],
                label=LABELS[mode],
            )
            axes[row, 2].fill_between(
                lags,
                np.asarray(means) - np.asarray(stds),
                np.asarray(means) + np.asarray(stds),
                color=COLORS[mode],
                alpha=0.10,
            )
        axes[row, 0].set_title(f"{architecture.upper()}: prefix context")
        axes[row, 1].set_title(f"{architecture.upper()}: suffix context")
        axes[row, 2].set_title(f"{architecture.upper()}: raw-count trace probe")
        axes[row, 0].set_ylabel("Accuracy (%)")
        axes[row, 0].set_xlabel("Retained bins")
        axes[row, 1].set_xlabel("Retained bins")
        axes[row, 2].set_xlabel("Lag (bins)")
        axes[row, 2].set_ylabel("$R^2$")
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "temporal_expressivity.png", dpi=300)
    fig.savefig(output_dir / "temporal_expressivity.pdf")
    plt.close(fig)


def write_markdown(results: dict, output_dir: Path) -> None:
    lines = [
        "# Temporal expressivity",
        "",
        "The context curves measure task-relevant dependence on retained input "
        "history. The frozen-state probe is deliberately different: it decodes "
        "the raw past event count, so lower long-lag R² means less generic trace "
        "memory, not necessarily less task-relevant temporal computation.",
        "",
        "## 95%-of-full task-context horizon",
        "",
        "| Architecture | Method | Prefix horizon (bins) | Suffix horizon (bins) |",
        "|---|---|---:|---:|",
    ]
    for architecture in results["metadata"]["architectures"]:
        for mode in MODES:
            prefix = results["context"][architecture][mode]["prefix"][
                "horizon95_bins"
            ]
            suffix = results["context"][architecture][mode]["suffix"][
                "horizon95_bins"
            ]
            lines.append(
                f"| {architecture.upper()} | {LABELS[mode]} | "
                f"{prefix['mean']:.1f} ± {prefix['std']:.1f} | "
                f"{suffix['mean']:.1f} ± {suffix['std']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Frozen-state raw-event-count probe",
            "",
            "| Architecture | Method | lag 1 R² | lag 5 R² | lag 10 R² | "
            "lag 20 R² |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for architecture in results["metadata"]["architectures"]:
        for mode in MODES:
            values = results["probe"][architecture][mode]
            cells = [
                f"{values[str(lag)]['r2']['mean']:.3f} ± "
                f"{values[str(lag)]['r2']['std']:.3f}"
                for lag in [1, 5, 10, 20]
            ]
            lines.append(
                f"| {architecture.upper()} | {LABELS[mode]} | "
                + " | ".join(cells)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The results do not show that MARS preserves every "
            "notion of memory unchanged. The task-context horizon, temporal "
            "shuffle/rate-repeat tests, and reset-interval tests establish "
            "whether the classifier still needs long ordered context. The "
            "raw-count probe separately tests how much nuisance-level input "
            "history remains linearly recoverable. If MARS retains the former "
            "while reducing the latter, the correct conclusion is selective "
            "temporal compression—not a memory-free/static solution, but a "
            "measurable robustness–generic-memory trade-off.",
            "",
        ]
    )
    (output_dir / "TEMPORAL_EXPRESSIVITY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--architectures", nargs="+", default=["ff", "rsnn"]
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    records = data["records"]
    validate(records, args.architectures)
    results = analyze(records, args.architectures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    make_plot(results, args.output_dir)
    write_markdown(results, args.output_dir)
    print(f"Completed memory analysis -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
