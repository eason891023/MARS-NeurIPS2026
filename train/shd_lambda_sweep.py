"""
SHD Lambda Sweep for Online HAJR / Reset-aware ASAR
===================================================
Sweeps lambda on the current online SHD FC-SNN setup.

Methods:
  - hajr
  - asar

Each lambda is trained with 5 seeds. The script reports:
  - clean test accuracy
  - robustness AUC
  - quantization AUC
  - drop@cv=0.15 / 0.20
  - drop@4bit / 3bit
"""

import copy
import importlib.util
import json
import os
import random
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from snntorch import surrogate


def _find_project_root(start_file: str) -> str:
    cur = os.path.abspath(os.path.dirname(start_file))
    while True:
        if os.path.isdir(os.path.join(cur, "snn_robust_tau")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot locate project root containing 'snn_robust_tau'.")
        cur = parent


PROJECT_ROOT = _find_project_root(__file__)
sys.path.insert(0, PROJECT_ROOT)

from snn_robust_tau.paths import OUTPUT_ROOT  # noqa: E402


def _load_shd_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shd_fc.py")
    spec = importlib.util.spec_from_file_location("shd_p0_complete", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


shd = _load_shd_module()

CONFIG = {
    "methods": {
        "hajr": [0.001, 0.01, 0.1, 0.5],
        "asar": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    },
    "seeds": [42, 52, 62, 72, 82],
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def summarize_method_lambda(all_runs, method, lam, cfg):
    rows = [all_runs[(method, lam, seed)] for seed in cfg["seeds"]]
    test_accs = [row["test_acc"] for row in rows]
    robustness_auc = [row["robustness_auc"] for row in rows]
    robustness_lw_auc = [row["robustness_layerwise_auc"] for row in rows]
    robustness_gl_auc = [row["robustness_global_auc"] for row in rows]
    quantization_auc = [row["quantization_auc"] for row in rows]
    drop_cv_015 = [row["drop_at_cv_0.15"] for row in rows]
    drop_cv_020 = [row["drop_at_cv_0.20"] for row in rows]
    drop_4bit = [row["drop_at_4bit"] for row in rows]
    drop_3bit = [row["drop_at_3bit"] for row in rows]
    best_epochs = [row["best_epoch"] for row in rows]
    best_vals = [row["best_val"] for row in rows]

    return {
        "lambda": lam,
        "test_acc": {"mean": float(np.mean(test_accs)), "std": float(np.std(test_accs, ddof=1))},
        "robustness_auc": {"mean": float(np.mean(robustness_auc)), "std": float(np.std(robustness_auc, ddof=1))},
        "robustness_layerwise_auc": {"mean": float(np.mean(robustness_lw_auc)), "std": float(np.std(robustness_lw_auc, ddof=1))},
        "robustness_global_auc": {"mean": float(np.mean(robustness_gl_auc)), "std": float(np.std(robustness_gl_auc, ddof=1))},
        "quantization_auc": {"mean": float(np.mean(quantization_auc)), "std": float(np.std(quantization_auc, ddof=1))},
        "drop_at_cv_0.15": {"mean": float(np.mean(drop_cv_015)), "std": float(np.std(drop_cv_015, ddof=1))},
        "drop_at_cv_0.20": {"mean": float(np.mean(drop_cv_020)), "std": float(np.std(drop_cv_020, ddof=1))},
        "drop_at_4bit": {"mean": float(np.mean(drop_4bit)), "std": float(np.std(drop_4bit, ddof=1))},
        "drop_at_3bit": {"mean": float(np.mean(drop_3bit)), "std": float(np.std(drop_3bit, ddof=1))},
        "best_val": {"mean": float(np.mean(best_vals)), "std": float(np.std(best_vals, ddof=1))},
        "best_epoch": {"mean": float(np.mean(best_epochs)), "std": float(np.std(best_epochs, ddof=1))},
    }


def plot_metric_vs_lambda(summary_rows, method, metric_key, ylabel, out_path, higher_is_better=True):
    lambdas = [row["lambda"] for row in summary_rows]
    means = [row[metric_key]["mean"] for row in summary_rows]
    stds = [row[metric_key]["std"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lambdas, means, marker="o", lw=2)
    ax.fill_between(lambdas, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel(ylabel)
    suffix = "higher is better" if higher_is_better else "closer to 0 is better"
    ax.set_title(f"SHD {method} lambda sweep: {metric_key} ({suffix})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base_cfg = copy.deepcopy(shd.CONFIG)
    base_cfg["modes"] = ["clean", "hajr", "asar"]
    base_cfg["learn_beta"] = True
    base_cfg["learn_threshold"] = False
    base_cfg["seeds"] = CONFIG["seeds"]
    # The archived sweep (paper appendix, lambda table) ran with an earlier SHD-FC configuration
    # that had no input augmentation; force it off to reproduce that setting.
    base_cfg["use_augmentations"] = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(str(OUTPUT_ROOT), "train", f"shd_fc_lambda_sweep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump({"base_cfg": base_cfg, "sweep_cfg": CONFIG}, handle, indent=2, default=str)

    shd.seed_everything(CONFIG["seeds"][0])
    train_loader, val_loader, test_loader, input_size = shd.build_loaders(CONFIG["seeds"][0], base_cfg)
    surrogate_fn = surrogate.atan(alpha=base_cfg["surrogate_alpha"])

    all_runs = {}
    summary = {}

    for method, lambda_list in CONFIG["methods"].items():
        summary_rows = []
        print(f"\n{'='*80}\nMethod: {method}\n{'='*80}")
        for lam in lambda_list:
            print(f"\n--- lambda={lam} ---")
            for seed in CONFIG["seeds"]:
                shd.seed_everything(seed)
                cfg = copy.deepcopy(base_cfg)
                cfg["lambda_hajr"] = lam if method == "hajr" else base_cfg["lambda_hajr"]
                cfg["lambda_asar"] = lam if method == "asar" else base_cfg["lambda_asar"]

                model = shd.SHDSNN(
                    input_size=input_size,
                    beta_init=cfg["beta_init"],
                    threshold_init=cfg["threshold_init"],
                    surrogate_fn=surrogate_fn,
                    learn_beta=cfg["learn_beta"],
                    learn_threshold=cfg["learn_threshold"],
                ).to(shd.DEVICE)

                model, test_acc, best_val, best_epoch, history, beta_trace = shd.train_one_mode(
                    model, method, (train_loader, val_loader, test_loader), cfg, seed
                )
                robust = shd.evaluate_robustness(model, test_loader, cfg["eval_cvs"], cfg["eval_mc_runs"])
                robust_lw = shd.evaluate_robustness_layerwise(model, test_loader, cfg["eval_cvs"], cfg["eval_mc_runs"])
                robust_gl = shd.evaluate_robustness_global(model, test_loader, cfg["eval_cvs"], cfg["eval_mc_runs"])
                quant = shd.evaluate_quantization(model, test_loader, cfg["quant_bits"])

                robustness_means = [robust[cv]["mean"] for cv in cfg["eval_cvs"]]
                quant_means = [quant[bits] for bits in cfg["quant_bits"]]

                all_runs[(method, lam, seed)] = {
                    "seed": seed,
                    "lambda": lam,
                    "history": history,
                    "test_acc": test_acc,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "robustness": {str(cv): robust[cv] for cv in cfg["eval_cvs"]},
                    "robustness_layerwise": {str(cv): robust_lw[cv] for cv in cfg["eval_cvs"]},
                    "robustness_global": {str(cv): robust_gl[cv] for cv in cfg["eval_cvs"]},
                    "quantization": {str(bits): quant[bits] for bits in cfg["quant_bits"]},
                    "robustness_auc": shd.compute_robustness_auc(cfg["eval_cvs"], robustness_means),
                    "robustness_layerwise_auc": shd.compute_robustness_auc(cfg["eval_cvs"], [robust_lw[cv]["mean"] for cv in cfg["eval_cvs"]]),
                    "robustness_global_auc": shd.compute_robustness_auc(cfg["eval_cvs"], [robust_gl[cv]["mean"] for cv in cfg["eval_cvs"]]),
                    "quantization_auc": shd.compute_quantization_auc(cfg["quant_bits"], quant_means),
                    "drop_at_cv_0.15": robust[0.15]["mean"] - robust[0.0]["mean"],
                    "drop_at_cv_0.20": robust[0.2]["mean"] - robust[0.0]["mean"],
                    "drop_at_4bit": quant[4] - quant[32],
                    "drop_at_3bit": quant[3] - quant[32],
                    "beta_trace": beta_trace,
                }
                print(
                    f"seed={seed} lambda={lam} test={test_acc:.2f} "
                    f"rob_auc={all_runs[(method, lam, seed)]['robustness_auc']:.2f} "
                    f"quant_auc={all_runs[(method, lam, seed)]['quantization_auc']:.2f}"
                )

            summary_rows.append(summarize_method_lambda(all_runs, method, lam, CONFIG))

        summary[method] = summary_rows

        plot_metric_vs_lambda(summary_rows, method, "test_acc", "Test Accuracy (%)",
                              os.path.join(output_dir, "plots", f"{method}_test_acc_vs_lambda.png"))
        plot_metric_vs_lambda(summary_rows, method, "robustness_auc", "Robustness AUC (i.i.d.)",
                              os.path.join(output_dir, "plots", f"{method}_robustness_auc_vs_lambda.png"))
        plot_metric_vs_lambda(summary_rows, method, "robustness_layerwise_auc", "Robustness AUC (layer-wise)",
                              os.path.join(output_dir, "plots", f"{method}_robustness_lw_auc_vs_lambda.png"))
        plot_metric_vs_lambda(summary_rows, method, "robustness_global_auc", "Robustness AUC (global)",
                              os.path.join(output_dir, "plots", f"{method}_robustness_gl_auc_vs_lambda.png"))
        plot_metric_vs_lambda(summary_rows, method, "quantization_auc", "Quantization AUC",
                              os.path.join(output_dir, "plots", f"{method}_quantization_auc_vs_lambda.png"))
        plot_metric_vs_lambda(summary_rows, method, "drop_at_cv_0.15", "Drop @ CV=0.15",
                              os.path.join(output_dir, "plots", f"{method}_drop_cv015_vs_lambda.png"), higher_is_better=False)
        plot_metric_vs_lambda(summary_rows, method, "drop_at_cv_0.20", "Drop @ CV=0.20",
                              os.path.join(output_dir, "plots", f"{method}_drop_cv020_vs_lambda.png"), higher_is_better=False)
        plot_metric_vs_lambda(summary_rows, method, "drop_at_4bit", "Drop @ 4-bit",
                              os.path.join(output_dir, "plots", f"{method}_drop_4bit_vs_lambda.png"), higher_is_better=False)
        plot_metric_vs_lambda(summary_rows, method, "drop_at_3bit", "Drop @ 3-bit",
                              os.path.join(output_dir, "plots", f"{method}_drop_3bit_vs_lambda.png"), higher_is_better=False)

    payload = {
        "base_cfg": base_cfg,
        "sweep_cfg": CONFIG,
        "per_run": {
            f"{method}_lam{lam}_s{seed}": value
            for (method, lam, seed), value in all_runs.items()
        },
        "summary": summary,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as handle:
        json.dump(payload, handle, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
