#!/usr/bin/env python3
"""Completeness-gated analysis of SSC temporal-expressivity tests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ["clean", "noise_aug", "asar"]
LABELS = {"clean": "Clean", "noise_aug": "NoiseAug", "asar": "MARS"}
COLORS = {"clean": "#4477AA", "noise_aug": "#CCBB44", "asar": "#228833"}


def validate(payload: dict) -> None:
    metadata = payload["metadata"]
    records = payload["records"]
    expected = {
        (mode, int(seed), condition)
        for mode in metadata["modes"]
        for seed in metadata["seeds"]
        for condition in metadata["conditions"]
    }
    keys = [
        (row["mode"], int(row["seed"]), row["condition"]) for row in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate SSC temporal keys")
    if set(keys) != expected:
        raise RuntimeError(
            f"SSC temporal: {len(set(keys))}/{len(expected)} keys; "
            f"missing={len(expected-set(keys))}, extra={len(set(keys)-expected)}"
        )


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values)
    return float(array.mean()), (
        float(array.std(ddof=1)) if len(array) > 1 else 0.0
    )


def analyze(payload: dict) -> dict:
    records = payload["records"]
    metadata = payload["metadata"]
    lookup = {
        (row["mode"], int(row["seed"]), row["condition"]): float(row["accuracy"])
        for row in records
    }
    seeds = [int(seed) for seed in metadata["seeds"]]
    conditions = metadata["conditions"]
    summary = {}
    for mode in MODES:
        original = [lookup[(mode, seed, "original")] for seed in seeds]
        for condition in conditions:
            accuracy = [lookup[(mode, seed, condition)] for seed in seeds]
            degradation = [
                base - changed for base, changed in zip(original, accuracy)
            ]
            acc_mean, acc_std = mean_std(accuracy)
            drop_mean, drop_std = mean_std(degradation)
            summary[f"{mode}|{condition}"] = {
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "degradation_mean": drop_mean,
                "degradation_std": drop_std,
            }

    horizons = {}
    for mode in MODES:
        for kind in ["prefix", "suffix"]:
            lengths = sorted(
                int(condition.removeprefix(kind))
                for condition in conditions
                if condition.startswith(kind)
            )
            per_seed = []
            for seed in seeds:
                threshold = 0.95 * lookup[(mode, seed, "original")]
                qualifying = [
                    length
                    for length in lengths
                    if lookup[(mode, seed, f"{kind}{length}")] >= threshold
                ]
                per_seed.append(min(qualifying) if qualifying else max(lengths))
            mean, std = mean_std(per_seed)
            horizons[f"{mode}|{kind}"] = {
                "mean": mean,
                "std": std,
                "per_seed": per_seed,
            }
    return {
        "metadata": metadata,
        "summary": summary,
        "horizons_95pct_original": horizons,
    }


def plot(result: dict, output: Path) -> None:
    conditions = result["metadata"]["conditions"]
    context_lengths = sorted(
        int(condition.removeprefix("prefix"))
        for condition in conditions
        if condition.startswith("prefix")
    )
    jitter = sorted(
        int(condition.removeprefix("jitter"))
        for condition in conditions
        if condition.startswith("jitter")
    )
    windows = sorted(
        int(condition.removeprefix("local_shuffle"))
        for condition in conditions
        if condition.startswith("local_shuffle")
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for mode in MODES:
        prefix = [
            result["summary"][f"{mode}|prefix{length}"]["accuracy_mean"]
            for length in context_lengths
        ]
        prefix_std = [
            result["summary"][f"{mode}|prefix{length}"]["accuracy_std"]
            for length in context_lengths
        ]
        axes[0].errorbar(
            context_lengths,
            prefix,
            yerr=prefix_std,
            marker="o",
            color=COLORS[mode],
            label=LABELS[mode],
        )
        jitter_acc = [
            result["summary"][f"{mode}|jitter{radius}"]["accuracy_mean"]
            for radius in jitter
        ]
        axes[1].plot(
            jitter, jitter_acc, marker="o", color=COLORS[mode], label=LABELS[mode]
        )
        local = [
            result["summary"][f"{mode}|local_shuffle{window}"]["accuracy_mean"]
            for window in windows
        ]
        axes[2].plot(
            windows, local, marker="o", color=COLORS[mode], label=LABELS[mode]
        )
    axes[0].set_xlabel("Retained prefix (of 250 bins)")
    axes[0].set_ylabel("SSC accuracy (%)")
    axes[0].set_title("Long task context")
    axes[1].set_xlabel("Time-bin jitter radius")
    axes[1].set_ylabel("SSC accuracy (%)")
    axes[1].set_title("Fine timing perturbation")
    axes[2].set_xlabel("Local shuffle window")
    axes[2].set_ylabel("SSC accuracy (%)")
    axes[2].set_title("Destroy timing, preserve coarse order")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(output / "ssc_temporal_expressivity.png", dpi=220)
    figure.savefig(output / "ssc_temporal_expressivity.pdf")
    plt.close(figure)


def write_markdown(result: dict, output: Path, record_count: int) -> None:
    lines = [
        "# SSC temporal expressivity",
        "",
        f"Exact formal records: `{record_count}`. Mean±SD is across five "
        "independent training seeds.",
        "",
        "## Destructive temporal controls",
        "",
        "| Method | Original | Reverse | Full shuffle | Rate repeat |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        values = []
        for condition in ["original", "reverse", "shuffle", "rate_repeat"]:
            row = result["summary"][f"{mode}|{condition}"]
            values.append(f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}")
        lines.append(f"| {LABELS[mode]} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 95%-of-original context horizon",
            "",
            "| Method | Prefix horizon | Suffix horizon |",
            "|---|---:|---:|",
        ]
    )
    for mode in MODES:
        prefix = result["horizons_95pct_original"][f"{mode}|prefix"]
        suffix = result["horizons_95pct_original"][f"{mode}|suffix"]
        lines.append(
            f"| {LABELS[mode]} | {prefix['mean']:.1f}±{prefix['std']:.1f} | "
            f"{suffix['mean']:.1f}±{suffix['std']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Fine temporal resolution",
            "",
            "| Method | Jitter 1 | Jitter 2 | Jitter 5 | Jitter 10 | Local shuffle 2 | 5 | 10 | 25 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        conditions = [
            "jitter1",
            "jitter2",
            "jitter5",
            "jitter10",
            "local_shuffle2",
            "local_shuffle5",
            "local_shuffle10",
            "local_shuffle25",
        ]
        values = [
            f"{result['summary'][f'{mode}|{condition}']['accuracy_mean']:.3f}"
            for condition in conditions
        ]
        lines.append(f"| {LABELS[mode]} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "SSC is a 35-class spoken-command task with learned recurrent delays. "
            "A large shuffle/rate-repeat penalty and a long context horizon rule "
            "out static rate accumulation. Jitter and local-shuffle curves "
            "measure timing resolution; they do not by themselves prove that "
            "every generic memory trace is preserved.",
            "",
        ]
    )
    (output / "SSC_TEMPORAL_EXPRESSIVITY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    validate(payload)
    result = analyze(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    plot(result, args.output_dir)
    write_markdown(result, args.output_dir, len(payload["records"]))
    print(f"PASS -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
