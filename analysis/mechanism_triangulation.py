#!/usr/bin/env python3
"""Triangulation of post-hoc sensitivity energy and beta robustness (Clean / Noise Aug / MARS).

This analysis reuses the post-hoc sensitivity/deployment evaluations of the main checkpoints
(results/mechanism/asar_vs_robustness.json, results/mechanism/ssc_scatter_points.json).
It focuses on three interventions:

* Clean: no robustness intervention;
* NoiseAug: an independent, perturbation-based intervention;
* MARS: the direct sensitivity intervention proposed in the paper.

The pooled correlation is residualized within dataset so that different raw
sensitivity scales across architectures cannot create a spurious association.
Output: outputs/analysis/mechanism_triangulation/ (shipped copy of results.json:
results/analysis/mechanism_triangulation.json; source of the 4.67x / 78.2x / 16.7x fold reductions).
Usage : python3 analysis/mechanism_triangulation.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402

OUTPUT_DIR = OUTPUT_ROOT / "analysis" / "mechanism_triangulation"
SHD_DVS_CIFAR = RESULTS / "mechanism" / "asar_vs_robustness.json"
SSC = RESULTS / "mechanism" / "ssc_scatter_points.json"

SENSITIVE = ["SHD-FC", "SHD-RSNN", "SSC"]
CONTROLS = ["DVS", "CIFAR10"]
MODES = ["clean", "noise_aug", "asar"]
LABELS = {"clean": "Clean", "noise_aug": "NoiseAug", "asar": "MARS"}
COLORS = {"clean": "#4477AA", "noise_aug": "#CCBB44", "asar": "#228833"}


def bootstrap_ci(
    values: np.ndarray, *, samples: int = 50_000, seed: int = 25_072_500
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def safe_wilcoxon_greater(left: np.ndarray, right: np.ndarray) -> float:
    """Test whether paired left values are greater than right values."""
    differences = left - right
    if np.allclose(differences, 0):
        return 1.0
    return float(
        wilcoxon(
            left,
            right,
            alternative="greater",
            zero_method="wilcox",
        ).pvalue
    )


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def load_records() -> list[dict]:
    records = json.loads(SHD_DVS_CIFAR.read_text()) + json.loads(SSC.read_text())
    selected = [
        row
        for row in records
        if row["mode"] in MODES and row["dataset"] in SENSITIVE + CONTROLS
    ]
    keys = [(r["dataset"], r["mode"], r["seed"]) for r in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate dataset/mode/seed records found")
    expected = {
        (dataset, mode, seed)
        for dataset in SENSITIVE + CONTROLS
        for mode in MODES
        for seed in [42, 52, 72, 92, 102]
    }
    missing = sorted(expected - set(keys))
    if missing:
        raise RuntimeError(f"Missing records: {missing}")
    return selected


def analyze(records: list[dict]) -> dict:
    lookup = {
        (r["dataset"], r["mode"], r["seed"]): r
        for r in records
    }
    seeds = sorted({r["seed"] for r in records})
    per_dataset: dict[str, dict] = {}
    correlations: dict[str, dict] = {}

    for dataset in SENSITIVE + CONTROLS:
        per_dataset[dataset] = {}
        subset = [r for r in records if r["dataset"] == dataset]
        x = np.log10([r["asar_loss"] for r in subset])
        y = np.asarray([r["acc_drop"] for r in subset], dtype=float)
        correlations[dataset] = {
            "n": len(subset),
            "pearson_log_sensitivity_vs_drop": float(pearsonr(x, y).statistic),
            "pearson_p_two_sided": float(pearsonr(x, y).pvalue),
            "spearman_sensitivity_vs_drop": float(
                spearmanr(x, y).statistic
            ),
            "spearman_p_two_sided": float(spearmanr(x, y).pvalue),
        }
        for mode in MODES:
            rows = [r for r in subset if r["mode"] == mode]
            sensitivity = [float(r["asar_loss"]) for r in rows]
            drops = [float(r["acc_drop"]) for r in rows]
            per_dataset[dataset][mode] = {
                "n": len(rows),
                "sensitivity_arithmetic": mean_std(sensitivity),
                "sensitivity_geometric_mean": float(
                    np.exp(np.mean(np.log(sensitivity)))
                ),
                "drop_pp": mean_std(drops),
            }

    paired: dict[str, dict] = {}
    for mode in ["noise_aug", "asar"]:
        clean_sensitivity = []
        method_sensitivity = []
        clean_drop = []
        method_drop = []
        by_dataset: dict[str, dict] = {}
        for dataset in SENSITIVE:
            dataset_log_ratios = []
            dataset_drop_improvements = []
            for seed in seeds:
                clean = lookup[(dataset, "clean", seed)]
                method = lookup[(dataset, mode, seed)]
                clean_sensitivity.append(float(clean["asar_loss"]))
                method_sensitivity.append(float(method["asar_loss"]))
                clean_drop.append(float(clean["acc_drop"]))
                method_drop.append(float(method["acc_drop"]))
                dataset_log_ratios.append(
                    np.log(float(clean["asar_loss"]) / float(method["asar_loss"]))
                )
                dataset_drop_improvements.append(
                    float(clean["acc_drop"]) - float(method["acc_drop"])
                )
            by_dataset[dataset] = {
                "sensitivity_fold_reduction_geometric_mean": float(
                    np.exp(np.mean(dataset_log_ratios))
                ),
                "drop_improvement_pp_mean": float(
                    np.mean(dataset_drop_improvements)
                ),
            }

        clean_sensitivity_array = np.asarray(clean_sensitivity)
        method_sensitivity_array = np.asarray(method_sensitivity)
        clean_drop_array = np.asarray(clean_drop)
        method_drop_array = np.asarray(method_drop)
        log_ratio = np.log(clean_sensitivity_array / method_sensitivity_array)
        drop_improvement = clean_drop_array - method_drop_array
        log_ci = bootstrap_ci(log_ratio)
        drop_ci = bootstrap_ci(drop_improvement, seed=25_072_501)
        paired[mode] = {
            "n_dataset_seed_pairs": len(log_ratio),
            "sensitivity_fold_reduction_geometric_mean": float(
                np.exp(log_ratio.mean())
            ),
            "sensitivity_fold_reduction_95ci": [
                float(np.exp(log_ci[0])),
                float(np.exp(log_ci[1])),
            ],
            "sensitivity_wilcoxon_one_sided_p": safe_wilcoxon_greater(
                clean_sensitivity_array, method_sensitivity_array
            ),
            "drop_improvement_pp_mean": float(drop_improvement.mean()),
            "drop_improvement_pp_95ci": list(drop_ci),
            "drop_wilcoxon_one_sided_p": safe_wilcoxon_greater(
                clean_drop_array, method_drop_array
            ),
            "by_dataset": by_dataset,
        }

    noise_sensitivity = []
    mars_sensitivity = []
    noise_drop = []
    mars_drop = []
    direct_by_dataset = {}
    for dataset in SENSITIVE:
        dataset_log_ratios = []
        dataset_drop_advantages = []
        for seed in seeds:
            noise = lookup[(dataset, "noise_aug", seed)]
            mars = lookup[(dataset, "asar", seed)]
            noise_sensitivity.append(float(noise["asar_loss"]))
            mars_sensitivity.append(float(mars["asar_loss"]))
            noise_drop.append(float(noise["acc_drop"]))
            mars_drop.append(float(mars["acc_drop"]))
            dataset_log_ratios.append(
                np.log(float(noise["asar_loss"]) / float(mars["asar_loss"]))
            )
            dataset_drop_advantages.append(
                float(noise["acc_drop"]) - float(mars["acc_drop"])
            )
        direct_by_dataset[dataset] = {
            "mars_sensitivity_fold_reduction_vs_noise_aug": float(
                np.exp(np.mean(dataset_log_ratios))
            ),
            "mars_drop_advantage_pp_mean": float(
                np.mean(dataset_drop_advantages)
            ),
        }
    noise_sensitivity_array = np.asarray(noise_sensitivity)
    mars_sensitivity_array = np.asarray(mars_sensitivity)
    noise_drop_array = np.asarray(noise_drop)
    mars_drop_array = np.asarray(mars_drop)
    direct_log_ratio = np.log(
        noise_sensitivity_array / mars_sensitivity_array
    )
    direct_drop_advantage = noise_drop_array - mars_drop_array
    direct_log_ci = bootstrap_ci(direct_log_ratio, seed=25_072_502)
    direct_drop_ci = bootstrap_ci(direct_drop_advantage, seed=25_072_503)
    mars_vs_noise_aug = {
        "n_dataset_seed_pairs": len(direct_log_ratio),
        "mars_sensitivity_fold_reduction_geometric_mean": float(
            np.exp(direct_log_ratio.mean())
        ),
        "mars_sensitivity_fold_reduction_95ci": [
            float(np.exp(direct_log_ci[0])),
            float(np.exp(direct_log_ci[1])),
        ],
        "sensitivity_wilcoxon_one_sided_p": safe_wilcoxon_greater(
            noise_sensitivity_array, mars_sensitivity_array
        ),
        "mars_drop_advantage_pp_mean": float(direct_drop_advantage.mean()),
        "mars_drop_advantage_pp_95ci": list(direct_drop_ci),
        "drop_wilcoxon_one_sided_p": safe_wilcoxon_greater(
            noise_drop_array, mars_drop_array
        ),
        "by_dataset": direct_by_dataset,
    }

    sensitive_records = [r for r in records if r["dataset"] in SENSITIVE]
    residual_log_sensitivity = []
    residual_drop = []
    for dataset in SENSITIVE:
        subset = [r for r in sensitive_records if r["dataset"] == dataset]
        x = np.log10([r["asar_loss"] for r in subset])
        y = np.asarray([r["acc_drop"] for r in subset], dtype=float)
        residual_log_sensitivity.extend((x - x.mean()).tolist())
        residual_drop.extend((y - y.mean()).tolist())
    pooled_pearson = pearsonr(residual_log_sensitivity, residual_drop)
    pooled_spearman = spearmanr(residual_log_sensitivity, residual_drop)

    # Stronger anti-circularity check: MARS checkpoints are excluded. NoiseAug
    # was not trained with the Ψ objective, so its change relative to Clean is
    # an independent intervention on the proposed diagnostic.
    non_mars_residual_log_sensitivity = []
    non_mars_residual_drop = []
    for dataset in SENSITIVE:
        subset = [
            r
            for r in sensitive_records
            if r["dataset"] == dataset
            and r["mode"] in ["clean", "noise_aug"]
        ]
        x = np.log10([r["asar_loss"] for r in subset])
        y = np.asarray([r["acc_drop"] for r in subset], dtype=float)
        non_mars_residual_log_sensitivity.extend((x - x.mean()).tolist())
        non_mars_residual_drop.extend((y - y.mean()).tolist())
    non_mars_pearson = pearsonr(
        non_mars_residual_log_sensitivity, non_mars_residual_drop
    )
    non_mars_spearman = spearmanr(
        non_mars_residual_log_sensitivity, non_mars_residual_drop
    )

    monotonic_sensitivity = []
    monotonic_drop = []
    for dataset in SENSITIVE:
        for seed in seeds:
            clean = lookup[(dataset, "clean", seed)]
            noise = lookup[(dataset, "noise_aug", seed)]
            mars = lookup[(dataset, "asar", seed)]
            monotonic_sensitivity.append(
                float(mars["asar_loss"])
                < float(noise["asar_loss"])
                < float(clean["asar_loss"])
            )
            monotonic_drop.append(
                float(mars["acc_drop"])
                <= float(noise["acc_drop"])
                < float(clean["acc_drop"])
            )

    return {
        "metadata": {
            "source_files": [str(SHD_DVS_CIFAR.relative_to(ROOT)) if SHD_DVS_CIFAR.is_relative_to(ROOT) else str(SHD_DVS_CIFAR),
                             str(SSC.relative_to(ROOT)) if SSC.is_relative_to(ROOT) else str(SSC)],
            "selected_modes": MODES,
            "sensitive_datasets": SENSITIVE,
            "control_datasets": CONTROLS,
            "seeds": seeds,
            "sensitivity": "post-hoc time-averaged MARS energy",
            "deployment_outcome": "accuracy drop in percentage points at beta CV=0.2",
            "bootstrap_samples": 50_000,
        },
        "per_dataset": per_dataset,
        "paired_sensitive_task_effects": paired,
        "mars_vs_noise_aug": mars_vs_noise_aug,
        "correlations": {
            **correlations,
            "sensitive_pooled_within_dataset_residualized": {
                "n": len(residual_drop),
                "pearson": float(pooled_pearson.statistic),
                "pearson_p_two_sided": float(pooled_pearson.pvalue),
                "spearman": float(pooled_spearman.statistic),
                "spearman_p_two_sided": float(pooled_spearman.pvalue),
            },
            "sensitive_non_mars_clean_noiseaug_within_dataset_residualized": {
                "n": len(non_mars_residual_drop),
                "pearson": float(non_mars_pearson.statistic),
                "pearson_p_two_sided": float(non_mars_pearson.pvalue),
                "spearman": float(non_mars_spearman.statistic),
                "spearman_p_two_sided": float(non_mars_spearman.pvalue),
            },
        },
        "triangulation_order": {
            "sensitivity_MARS_lt_NoiseAug_lt_Clean": {
                "count": int(sum(monotonic_sensitivity)),
                "total": len(monotonic_sensitivity),
            },
            "drop_MARS_le_NoiseAug_lt_Clean": {
                "count": int(sum(monotonic_drop)),
                "total": len(monotonic_drop),
            },
        },
    }


def plot(records: list[dict], results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    x = np.arange(len(SENSITIVE))
    width = 0.24

    for index, mode in enumerate(MODES):
        relative = []
        drop_means = []
        drop_sems = []
        for dataset in SENSITIVE:
            gm = results["per_dataset"][dataset][mode][
                "sensitivity_geometric_mean"
            ]
            clean_gm = results["per_dataset"][dataset]["clean"][
                "sensitivity_geometric_mean"
            ]
            relative.append(gm / clean_gm)
            drop_stats = results["per_dataset"][dataset][mode]["drop_pp"]
            drop_means.append(drop_stats["mean"])
            drop_sems.append(drop_stats["std"] / np.sqrt(5))
        offset = (index - 1) * width
        axes[0].bar(
            x + offset,
            relative,
            width,
            color=COLORS[mode],
            label=LABELS[mode],
        )
        axes[1].bar(
            x + offset,
            drop_means,
            width,
            yerr=drop_sems,
            capsize=2,
            color=COLORS[mode],
        )

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Post-hoc sensitivity / Clean")
    axes[0].set_title("(a) Sensitivity intervention")
    axes[0].axhline(1.0, color="0.4", lw=0.8, ls="--")
    axes[1].set_ylabel("Accuracy drop @ CV=0.2 (pp)")
    axes[1].set_title("(b) Deployment consequence")
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(SENSITIVE)
        ax.grid(axis="y", alpha=0.2)

    plot_records = [r for r in records if r["dataset"] in SENSITIVE]
    for dataset, marker in zip(SENSITIVE, ["o", "s", "^"]):
        subset = [r for r in plot_records if r["dataset"] == dataset]
        log_x = np.log10([r["asar_loss"] for r in subset])
        y = np.asarray([r["acc_drop"] for r in subset], dtype=float)
        residual_x = log_x - log_x.mean()
        residual_y = y - y.mean()
        for mode in MODES:
            mask = np.asarray([r["mode"] == mode for r in subset])
            axes[2].scatter(
                residual_x[mask],
                residual_y[mask],
                s=38,
                marker=marker,
                color=COLORS[mode],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.4,
            )
    pooled = results["correlations"][
        "sensitive_pooled_within_dataset_residualized"
    ]
    all_x = []
    all_y = []
    for dataset in SENSITIVE:
        subset = [r for r in plot_records if r["dataset"] == dataset]
        dataset_x = np.log10([r["asar_loss"] for r in subset])
        dataset_y = np.asarray([r["acc_drop"] for r in subset], dtype=float)
        all_x.extend((dataset_x - dataset_x.mean()).tolist())
        all_y.extend((dataset_y - dataset_y.mean()).tolist())
    slope, intercept = np.polyfit(all_x, all_y, 1)
    line_x = np.linspace(min(all_x), max(all_x), 100)
    axes[2].plot(line_x, slope * line_x + intercept, color="0.2", lw=1.2)
    axes[2].axhline(0, color="0.6", lw=0.7, ls="--")
    axes[2].axvline(0, color="0.6", lw=0.7, ls="--")
    axes[2].set_xlabel("Within-dataset residual log sensitivity")
    axes[2].set_ylabel("Within-dataset residual drop (pp)")
    axes[2].set_title(f"(c) Mechanism link: r={pooled['pearson']:.2f}")
    axes[2].grid(alpha=0.2)

    fig.legend(
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUTPUT_DIR / "mechanism_triangulation.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "mechanism_triangulation.pdf")
    plt.close(fig)


def write_markdown(results: dict) -> None:
    paired = results["paired_sensitive_task_effects"]
    direct = results["mars_vs_noise_aug"]
    pooled = results["correlations"][
        "sensitive_pooled_within_dataset_residualized"
    ]
    non_mars = results["correlations"][
        "sensitive_non_mars_clean_noiseaug_within_dataset_residualized"
    ]
    order = results["triangulation_order"]
    lines = [
        "# Mechanism triangulation",
        "",
        "This is a post-hoc analysis of the main training checkpoints. It uses "
        "three temporally sensitive settings (SHD-FC, SHD-RSNN, SSC), five "
        "seeds, and the independently trained Clean, NoiseAug, and MARS models.",
        "",
        "## Main paired results",
        "",
        "| Intervention vs Clean | Ψ fold reduction (95% bootstrap CI) | "
        "Drop improvement at CV=0.2 (95% bootstrap CI) | one-sided paired p "
        "(Ψ / drop) |",
        "|---|---:|---:|---:|",
    ]
    for mode in ["noise_aug", "asar"]:
        row = paired[mode]
        psi_ci = row["sensitivity_fold_reduction_95ci"]
        drop_ci = row["drop_improvement_pp_95ci"]
        lines.append(
            f"| {LABELS[mode]} | "
            f"{row['sensitivity_fold_reduction_geometric_mean']:.2f}× "
            f"[{psi_ci[0]:.2f}, {psi_ci[1]:.2f}] | "
            f"{row['drop_improvement_pp_mean']:.2f} pp "
            f"[{drop_ci[0]:.2f}, {drop_ci[1]:.2f}] | "
            f"{row['sensitivity_wilcoxon_one_sided_p']:.4g} / "
            f"{row['drop_wilcoxon_one_sided_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "Relative to NoiseAug itself, MARS further lowers Ψ by "
            f"{direct['mars_sensitivity_fold_reduction_geometric_mean']:.2f}× "
            f"(95% CI "
            f"[{direct['mars_sensitivity_fold_reduction_95ci'][0]:.2f}, "
            f"{direct['mars_sensitivity_fold_reduction_95ci'][1]:.2f}]) and "
            f"reduces deployment drop by an average "
            f"{direct['mars_drop_advantage_pp_mean']:.2f} pp (95% CI "
            f"[{direct['mars_drop_advantage_pp_95ci'][0]:.2f}, "
            f"{direct['mars_drop_advantage_pp_95ci'][1]:.2f}]; one-sided "
            f"paired Wilcoxon p="
            f"{direct['drop_wilcoxon_one_sided_p']:.4g}). This average is "
            "driven mainly by SHD-FC/RSNN; the two methods are effectively "
            "tied on SSC.",
            "",
            "After removing dataset-specific mean offsets, post-hoc log Ψ "
            f"correlates with deployment drop across the 45 sensitive-task "
            f"checkpoints (Pearson r={pooled['pearson']:.3f}, "
            f"p={pooled['pearson_p_two_sided']:.3g}; "
            f"Spearman ρ={pooled['spearman']:.3f}, "
            f"p={pooled['spearman_p_two_sided']:.3g}).",
            "",
            "This association is not dependent on including the checkpoints "
            "trained to minimize Ψ. Restricting the analysis to Clean and "
            "NoiseAug—the independent, non-Ψ-trained intervention—gives "
            f"Pearson r={non_mars['pearson']:.3f}, "
            f"p={non_mars['pearson_p_two_sided']:.3g} and Spearman "
            f"ρ={non_mars['spearman']:.3f}, "
            f"p={non_mars['spearman_p_two_sided']:.3g} (n={non_mars['n']}).",
            "",
            f"The strict Ψ ordering MARS < NoiseAug < Clean holds for "
            f"{order['sensitivity_MARS_lt_NoiseAug_lt_Clean']['count']}/"
            f"{order['sensitivity_MARS_lt_NoiseAug_lt_Clean']['total']} "
            "dataset-seed pairs. The corresponding deployment-drop ordering "
            f"MARS ≤ NoiseAug < Clean holds for "
            f"{order['drop_MARS_le_NoiseAug_lt_Clean']['count']}/"
            f"{order['drop_MARS_le_NoiseAug_lt_Clean']['total']} pairs.",
            "",
            "## Dataset-level values",
            "",
            "| Dataset | Method | Ψ geometric mean | Drop mean ± SD (pp) |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset in SENSITIVE + CONTROLS:
        for mode in MODES:
            row = results["per_dataset"][dataset][mode]
            lines.append(
                f"| {dataset} | {LABELS[mode]} | "
                f"{row['sensitivity_geometric_mean']:.4g} | "
                f"{row['drop_pp']['mean']:.3f} ± "
                f"{row['drop_pp']['std']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The analysis is triangulating evidence, not a standalone proof of "
            "causality. MARS directly minimizes Ψ and yields low deployment "
            "drop. NoiseAug is not derived from the membrane-sensitivity "
            "recursion, yet independently lowers Ψ and also improves robustness "
            "on the sensitive tasks. Together with the within-dataset "
            "checkpoint correlation, these intervention and observational "
            "results support Ψ as the mechanism-level diagnostic. On the "
            "low-sensitivity DVS/CIFAR controls, all drops remain small, so Ψ "
            "is not presented as a task-agnostic accuracy predictor.",
            "",
        ]
    )
    (OUTPUT_DIR / "MECHANISM_TRIANGULATION.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    results = analyze(records)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    plot(records, results)
    write_markdown(results)
    print(
        f"Completed mechanism triangulation from {len(records)} checkpoints -> "
        f"{OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
