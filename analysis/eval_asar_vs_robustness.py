"""
Post-hoc evaluation: compute ASAR loss (||Ψ||²) for all trained checkpoints
and correlate with β-variation robustness drop.

Produces a scatter plot: X = mean ASAR loss, Y = accuracy drop at CV=0.2
Each point = one (dataset, method, seed) combination.

Usage:
    python analysis/eval_asar_vs_robustness.py
"""
import copy
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def _find_project_root(start_file: str) -> str:
    cur = os.path.abspath(os.path.dirname(start_file))
    while True:
        if os.path.isdir(os.path.join(cur, "snn_robust_tau")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot locate project root")
        cur = parent


PROJECT_ROOT = _find_project_root(__file__)
sys.path.insert(0, PROJECT_ROOT)
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS, ckpt_dir  # noqa: E402,F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Dataset configs ──────────────────────────────────────────────────────────
DATASETS = {
    "SHD-FC": {
        "script": os.path.join(PROJECT_ROOT, "train", "shd_fc.py"),
        "results": str(RESULTS / "training" / "SHD-FC.json"),
        "ckpt_dir": str(ckpt_dir("SHD-FC")),
        "model_class": "SHDSNN",
        "build_loaders_fn": "build_loaders",
        "needs_input_size": True,
    },
    "SHD-RSNN": {
        "script": os.path.join(PROJECT_ROOT, "train", "shd_rsnn.py"),
        "results": str(RESULTS / "training" / "SHD-RSNN.json"),
        "ckpt_dir": str(ckpt_dir("SHD-RSNN")),
        "model_class": "SHDRSNN",
        "build_loaders_fn": "build_loaders",
        "needs_input_size": True,
    },
    "DVS": {
        "script": os.path.join(PROJECT_ROOT, "train", "dvs_gesture.py"),
        "results": str(RESULTS / "training" / "DVS-Gesture.json"),
        "ckpt_dir": str(ckpt_dir("DVS-Gesture")),
        "model_class": "DVSGestureSNN",
        "build_loaders_fn": "build_loaders",
        "needs_input_size": False,
    },
    "CIFAR10": {
        "script": os.path.join(PROJECT_ROOT, "train", "cifar10.py"),
        "results": str(RESULTS / "training" / "CIFAR-10.json"),
        "ckpt_dir": str(ckpt_dir("CIFAR-10")),
        "model_class": "SpikingResNet18",
        "build_loaders_fn": "build_loaders",
        "needs_input_size": False,
    },
}

SEEDS = [42, 52, 72, 92, 102]
MODES = ["clean", "hajr", "asar", "noise_aug"]
EVAL_CV = 0.2  # which CV to use for drop


def compute_asar_loss(model, loader, max_batches=50):
    """Run forward pass with reg_mode='asar' and return mean ASAR loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE).float()
            _, reg_loss = model(data, reg_mode="asar")
            total_loss += reg_loss.item()
            n_batches += 1
            if n_batches >= max_batches:
                break
    return total_loss / max(n_batches, 1)


def main():
    output_dir = str(OUTPUT_ROOT / "analysis" / "asar_vs_robustness")
    os.makedirs(output_dir, exist_ok=True)

    all_points = []  # list of dicts: dataset, mode, seed, asar_loss, acc_drop, clean_acc, robust_acc

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {ds_name}")
        print(f"{'='*60}")

        # Load results.json for robustness data
        with open(ds_cfg["results"]) as f:
            results = json.load(f)

        # Import the experiment module
        mod = import_module_from_path(f"exp_{ds_name}", ds_cfg["script"])
        cfg = mod.CONFIG
        from snntorch import surrogate
        surrogate_fn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))

        # Build data loaders (just need test set)
        mod.seed_everything(42)
        loader_result = mod.build_loaders(42, cfg)
        if ds_cfg["needs_input_size"]:
            _, _, test_loader, input_size = loader_result
        else:
            _, _, test_loader = loader_result
            input_size = None

        for seed in SEEDS:
            for mode in MODES:
                ckpt_path = os.path.join(ds_cfg["ckpt_dir"], f"best_s{seed}_{mode}.pth")
                if not os.path.exists(ckpt_path):
                    print(f"  SKIP {seed}/{mode}: checkpoint not found")
                    continue

                # Instantiate model
                ModelClass = getattr(mod, ds_cfg["model_class"])
                if ds_name in ("SHD-FC", "SHD-RSNN"):
                    # SHD models: (input_size, beta_init, threshold_init, surrogate_fn, ...)
                    model = ModelClass(
                        input_size=input_size,
                        beta_init=cfg["beta_init"],
                        threshold_init=cfg.get("threshold_init", 1.0),
                        surrogate_fn=surrogate_fn,
                        learn_beta=cfg.get("learn_beta", True),
                        learn_threshold=cfg.get("learn_threshold", False),
                    ).to(DEVICE)
                elif ds_name == "DVS":
                    # DVSGestureSNN: (num_classes, beta, threshold, surrogate_fn, ...)
                    model = ModelClass(
                        num_classes=cfg.get("num_classes", 11),
                        beta=cfg["beta_init"],
                        threshold=cfg.get("threshold_init", 1.0),
                        surrogate_fn=surrogate_fn,
                        learn_beta=cfg.get("learn_beta", True),
                        learn_threshold=cfg.get("learn_threshold", False),
                    ).to(DEVICE)
                elif ds_name == "CIFAR10":
                    # SpikingResNet18: (num_classes, beta, threshold, surrogate_fn, ..., num_steps)
                    model = ModelClass(
                        num_classes=cfg.get("num_classes", 10),
                        beta=cfg["beta_init"],
                        threshold=cfg.get("threshold_init", 1.0),
                        surrogate_fn=surrogate_fn,
                        learn_beta=cfg.get("learn_beta", True),
                        learn_threshold=cfg.get("learn_threshold", False),
                        num_steps=cfg.get("time_steps", 4),
                    ).to(DEVICE)

                # Load checkpoint
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                # Compute ASAR loss
                asar_loss = compute_asar_loss(model, test_loader, max_batches=50)

                # Get robustness data from results.json
                key = f"s{seed}_{mode}"
                per_seed = results["per_seed"]
                if key in per_seed:
                    rob = per_seed[key]["robustness"]
                    clean_acc = rob["0.0"]["mean"]
                    robust_acc = rob[str(EVAL_CV)]["mean"]
                    acc_drop = clean_acc - robust_acc
                else:
                    clean_acc = robust_acc = acc_drop = float("nan")

                point = {
                    "dataset": ds_name,
                    "mode": mode,
                    "seed": seed,
                    "asar_loss": asar_loss,
                    "clean_acc": clean_acc,
                    "robust_acc": robust_acc,
                    "acc_drop": acc_drop,
                }
                all_points.append(point)
                print(f"  s{seed}/{mode}: ASAR_loss={asar_loss:.6f}, drop@CV{EVAL_CV}={acc_drop:.2f}%")

    # Save raw data
    with open(os.path.join(output_dir, "asar_vs_robustness.json"), "w") as f:
        json.dump(all_points, f, indent=2)

    # ── Plot 1: per-dataset scatter ──────────────────────────────────────────
    from scipy.stats import pearsonr
    COLORS = {"clean": "#2196F3", "hajr": "#F44336", "asar": "#4CAF50", "noise_aug": "#FF9800"}
    MARKERS = {"clean": "o", "hajr": "s", "asar": "D", "noise_aug": "^"}

    datasets_present = sorted(set(p["dataset"] for p in all_points))
    fig, axes = plt.subplots(1, len(datasets_present), figsize=(5 * len(datasets_present), 4.5))
    if len(datasets_present) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets_present):
        pts = [p for p in all_points if p["dataset"] == ds]
        for mode in MODES:
            mode_pts = [p for p in pts if p["mode"] == mode]
            if not mode_pts:
                continue
            xs = [p["asar_loss"] for p in mode_pts]
            ys = [p["acc_drop"] for p in mode_pts]
            ax.scatter(xs, ys, c=COLORS[mode], marker=MARKERS[mode],
                       s=60, label=mode, alpha=0.8, edgecolors="k", linewidths=0.5)
        # Log scale on X axis for all datasets except CIFAR10
        if ds != "CIFAR10":
            ax.set_xscale("log")
        ax.set_xlabel("ASAR Loss (mean ||Ψ||²)")
        ax.set_ylabel(f"Accuracy Drop @ CV={EVAL_CV} (%)")
        ax.set_title(ds)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("ASAR Loss vs Robustness Drop (Theorem 1 Validation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "asar_vs_robustness_per_dataset.png"), dpi=200)
    plt.close(fig)

    # ── Plot 2: combined scatter (all datasets) ─────────────────────────────
    DS_COLORS = {"SHD-FC": "#1f77b4", "SHD-RSNN": "#ff7f0e", "DVS": "#2ca02c", "CIFAR10": "#d62728"}
    TEMPORAL_DATASETS = {"SHD-FC", "SHD-RSNN", "DVS"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in datasets_present:
        pts = [p for p in all_points if p["dataset"] == ds]
        # Average over seeds per (dataset, mode)
        for mode in MODES:
            mode_pts = [p for p in pts if p["mode"] == mode]
            if not mode_pts:
                continue
            x_mean = np.mean([p["asar_loss"] for p in mode_pts])
            y_mean = np.mean([p["acc_drop"] for p in mode_pts])
            x_std = np.std([p["asar_loss"] for p in mode_pts])
            y_std = np.std([p["acc_drop"] for p in mode_pts])
            ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std,
                        color=DS_COLORS[ds], marker=MARKERS[mode],
                        ms=8, capsize=3, alpha=0.8, linestyle="none")
            ax.annotate(f"{ds}\n{mode}", (x_mean, y_mean), fontsize=6,
                        textcoords="offset points", xytext=(5, 5))

    ax.set_xscale("log")

    # Compute overall correlation (excluding NaN)
    valid = [p for p in all_points if not np.isnan(p["acc_drop"]) and not np.isnan(p["asar_loss"])]
    if len(valid) > 2:
        xs = np.array([p["asar_loss"] for p in valid])
        ys = np.array([p["acc_drop"] for p in valid])
        corr = np.corrcoef(xs, ys)[0, 1]
        # Fit line (in log space for display)
        xs_log = np.log10(xs)
        z = np.polyfit(xs_log, ys, 1)
        xline_log = np.linspace(xs_log.min(), xs_log.max(), 100)
        ax.plot(10**xline_log, np.polyval(z, xline_log), "k--", alpha=0.4, lw=1)
        title_str = f"ASAR Loss vs Accuracy Drop (r = {corr:.3f})"
    else:
        corr = float("nan")
        title_str = "ASAR Loss vs Accuracy Drop"

    # Per-category Pearson r: temporal datasets
    temporal_pts = [p for p in valid if p["dataset"] in TEMPORAL_DATASETS]
    if len(temporal_pts) > 2:
        xs_t = np.array([p["asar_loss"] for p in temporal_pts])
        ys_t = np.array([p["acc_drop"] for p in temporal_pts])
        r_temporal, p_temporal = pearsonr(xs_t, ys_t)
    else:
        r_temporal, p_temporal = float("nan"), float("nan")

    # Per-dataset Pearson r
    r_per_ds = {}
    for ds in datasets_present:
        ds_pts = [p for p in valid if p["dataset"] == ds]
        if len(ds_pts) > 2:
            xs_d = np.array([p["asar_loss"] for p in ds_pts])
            ys_d = np.array([p["acc_drop"] for p in ds_pts])
            r_val, _ = pearsonr(xs_d, ys_d)
            r_per_ds[ds] = r_val
        else:
            r_per_ds[ds] = float("nan")

    # Build annotation text
    annot_lines = [f"Overall r = {corr:.3f}"]
    if not np.isnan(r_temporal):
        annot_lines.append(f"Temporal r = {r_temporal:.3f}")
    for ds in datasets_present:
        if not np.isnan(r_per_ds[ds]):
            annot_lines.append(f"  {ds}: r = {r_per_ds[ds]:.3f}")
    annot_text = "\n".join(annot_lines)
    ax.text(0.02, 0.98, annot_text, transform=ax.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    ax.set_title(title_str)
    ax.set_xlabel("ASAR Loss (mean ||Ψ||²)")
    ax.set_ylabel(f"Accuracy Drop @ CV={EVAL_CV} (%)")
    ax.grid(alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    for ds in datasets_present:
        handles.append(Line2D([0], [0], marker="o", color=DS_COLORS[ds], linestyle="none", ms=8, label=ds))
    for mode in MODES:
        handles.append(Line2D([0], [0], marker=MARKERS[mode], color="gray", linestyle="none", ms=8, label=mode))
    ax.legend(handles=handles, fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "asar_vs_robustness_combined.png"), dpi=200)
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}")
    if len(valid) > 2:
        print(f"Overall Pearson r = {corr:.3f}")
    if not np.isnan(r_temporal):
        print(f"Temporal Pearson r = {r_temporal:.3f} (p={p_temporal:.4f})")
    for ds in datasets_present:
        if not np.isnan(r_per_ds[ds]):
            print(f"  {ds} Pearson r = {r_per_ds[ds]:.3f}")


if __name__ == "__main__":
    main()
