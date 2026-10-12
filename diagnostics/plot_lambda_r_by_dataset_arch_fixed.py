"""
Worst-case vs typical-case lambda diagnostics by dataset/architecture (corrected arctan derivative).
  * arctan derivative -> snnTorch ATan (a/2)/(1+(pi a x/2)^2), L_sigma = a/2 = 1
    (was a/(pi(1+(a x)^2)), L_sigma = 2/pi); --legacy restores the old formula
    to reproduce the old numbers.
  * cached inputs: results/diagnostics/typical_case_shd_fc.json and
    results/diagnostics/assumptions_rsnn_ssc_fixed.json (SHD-FC, SHD-RSNN, SSC);
    DVS-Gesture / CIFAR-10 are computed from checkpoints/<setting>/best_s42_*.pth.
  * outputs: outputs/diagnostics/lambda_r_by_dataset_arch_{fixed,legacy}*
  * hook additionally records per-element |alpha_eff| (max and p99 over a random
    element subsample), so that hard-reset layers can be checked against a
    per-neuron worst case and not only the p99 of the neuron-averaged value.

Plot worst-case and typical-case lambda_r diagnostics by dataset/architecture.

This script reuses cached SHD-FC, SHD-RSNN, and SSC-DelRec diagnostics when
available, and can compute quick seed-42 diagnostics for DVS-Gesture and CIFAR-10
from checkpoints using a few test batches.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS, ckpt_dir  # noqa: E402

OUT_DIR = OUTPUT_ROOT / "diagnostics"
LEGACY = "--legacy" in sys.argv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODES = ["clean", "hajr", "noise_aug", "asar"]
MODE_LABELS = {
    "clean": "Clean",
    "hajr": "HAJR",
    "noise_aug": "Noise Aug",
    "asar": "MARS",
}
MODE_COLORS = {
    "clean": "#4477AA",
    "hajr": "#EE6677",
    "noise_aug": "#CCBB44",
    "asar": "#228833",
}
DATASET_ORDER = ["SHD-FC", "SHD-RSNN", "SSC-DelRec", "DVS-Gesture", "CIFAR-10"]
PLOT_CAP = 1e3


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def surrogate_deriv_atan(x, alpha=2.0):
    if LEGACY:
        return alpha / (math.pi * (1.0 + (alpha * x).pow(2)))
    return (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * x).pow(2))


def l_sigma_atan(alpha=2.0):
    return alpha / math.pi if LEGACY else alpha / 2.0


def flatten_weight(weight):
    return weight.detach().view(weight.shape[0], -1).float()


def power_iteration_op_norm(w, n_iter=40):
    w = w.to(DEVICE)
    v = torch.randn(w.shape[1], device=w.device)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(n_iter):
        u = w @ v
        u = u / u.norm().clamp_min(1e-12)
        v = w.t() @ u
        v = v / v.norm().clamp_min(1e-12)
    return float((w @ v).norm().item())


def weight_norms(weight):
    w = flatten_weight(weight)
    n_in = w.shape[1]
    frob = float(torch.norm(w, "fro").item())
    eff = frob / math.sqrt(n_in)
    if w.numel() > 2_000_000:
        op = power_iteration_op_norm(w)
        op_note = "power_iter"
    else:
        op = float(torch.linalg.svdvals(w.to(DEVICE))[0].item())
        op_note = "svd"
    return {
        "W_op_norm": op,
        "W_eff_norm": eff,
        "W_frob_norm": frob,
        "n_in": int(n_in),
        "kappa": (op / eff) ** 2 if eff > 0 else float("inf"),
        "op_norm_method": op_note,
    }


def beta_value(layer):
    beta = layer.beta.detach()
    return float(beta.view(-1)[0].item())


def threshold_value(layer, device, dtype):
    threshold = getattr(layer, "threshold", None)
    if threshold is None:
        return torch.tensor(1.0, device=device, dtype=dtype)
    if isinstance(threshold, nn.Parameter):
        threshold = threshold.detach()
    return threshold.to(device=device, dtype=dtype)


def make_hook_stats(model, n_batches, loader, alpha=2.0):
    lif_layers = [(name, layer) for name, layer in model._lif_layers]
    stats = {
        name: {"alpha": [], "eta": [], "n_neurons": None, "prev_mem": None, "prev_spk": None,
               "elem_max": 0.0, "elem_sample": [], "signed_min": float("inf"), "signed_max": -float("inf")}
        for name, _ in lif_layers
    }
    l_sigma = l_sigma_atan(alpha)
    handles = []

    def make_hook(name, layer):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                spk, mem = output[0].detach(), output[1].detach()
            else:
                return
            mem_flat = mem.reshape(mem.shape[0], -1)
            spk_flat = spk.reshape(spk.shape[0], -1)
            item = stats[name]
            item["n_neurons"] = int(mem_flat.shape[1])

            thr = threshold_value(layer, mem_flat.device, mem_flat.dtype)
            surr = surrogate_deriv_atan(mem_flat - thr, alpha)
            eta_step = (surr.pow(2).sum(dim=1).mean() / (mem_flat.shape[1] * (l_sigma ** 2))).item()
            item["eta"].append(float(eta_step))

            reset = getattr(layer, "reset_mechanism", "subtract")
            beta = layer.beta.detach().to(device=mem_flat.device, dtype=mem_flat.dtype)
            if beta.dim() == 0:
                beta = beta.view(1, 1)
            else:
                beta = beta.view(1, -1)

            if reset == "none":
                alpha_eff = beta.abs().mean().item()
                item["alpha"].append(float(alpha_eff))
            elif item["prev_mem"] is not None:
                prev_mem = item["prev_mem"]
                prev_spk = item["prev_spk"]
                if reset == "zero":
                    surr_prev = surrogate_deriv_atan(prev_mem - thr, alpha)
                    coeff = beta * ((1.0 - prev_spk) - prev_mem * surr_prev)
                else:
                    surr_prev = surrogate_deriv_atan(prev_mem - thr, alpha)
                    coeff = beta - thr * surr_prev
                cabs = coeff.abs()
                alpha_eff = cabs.mean().item()
                item["alpha"].append(float(alpha_eff))
                item["elem_max"] = max(item["elem_max"], float(cabs.max().item()))
                item["signed_min"] = min(item["signed_min"], float(coeff.min().item()))
                item["signed_max"] = max(item["signed_max"], float(coeff.max().item()))
                flat = cabs.flatten()
                k = min(flat.numel(), 20000)
                idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
                item["elem_sample"].append(flat[idx].float().cpu().numpy())

            item["prev_mem"] = mem_flat
            item["prev_spk"] = spk_flat

        return hook

    for name, layer in lif_layers:
        handles.append(layer.register_forward_hook(make_hook(name, layer)))

    model.eval()
    count = 0
    with torch.no_grad():
        for data, _targets in loader:
            for item in stats.values():
                item["prev_mem"] = None
                item["prev_spk"] = None
            data = data.to(DEVICE).float()
            model(data, reg_mode=None)
            count += 1
            if count >= n_batches:
                break

    for handle in handles:
        handle.remove()

    out = {}
    for name, item in stats.items():
        alpha_vals = np.asarray(item["alpha"], dtype=float)
        eta_vals = np.asarray(item["eta"], dtype=float)
        out[name] = {
            "n_neurons": item["n_neurons"],
            "alpha_tilde": float(alpha_vals.mean()) if alpha_vals.size else None,
            "alpha_bar_p99": float(np.percentile(alpha_vals, 99)) if alpha_vals.size else None,
            "eta": float(eta_vals.mean()) if eta_vals.size else None,
            "sqrt_eta": float(np.sqrt(eta_vals.mean())) if eta_vals.size else None,
            "alpha_elem_max": item["elem_max"] if item["elem_sample"] else None,
            "alpha_elem_p99": float(np.percentile(np.concatenate(item["elem_sample"]), 99)) if item["elem_sample"] else None,
            "alpha_elem_frac_ge1": float((np.concatenate(item["elem_sample"]) >= 1.0).mean()) if item["elem_sample"] else None,
            "alpha_signed_min": item["signed_min"] if item["elem_sample"] else None,
            "alpha_signed_max": item["signed_max"] if item["elem_sample"] else None,
        }
    return out


def rows_from_cached_shd_fc(path):
    data = json.loads(path.read_text())
    if "fixed" in data:
        data = data["legacy" if LEGACY else "fixed"]
    rows = []
    for mode, layers in data.items():
        for layer in layers:
            if layer.get("lambda_r") is None:
                continue
            row = {
                "dataset_arch": "SHD-FC",
                "mode": mode,
                **layer,
            }
            rows.append(row)
    return rows


def rows_from_cached_nested(path):
    data = json.loads(path.read_text())
    rows = []
    for dataset, modes in data.items():
        l_sigma = 1.0 if dataset == "SSC-DelRec" else l_sigma_atan(2.0)  # triangle: 1
        for mode, payload in modes.items():
            for layer_name, layer in payload["stability"].items():
                layer = dict(layer)
                if layer.get("rec_op_norm") is not None:
                    layer["lambda_r_feedforward_only"] = layer["lambda_r"]
                    layer["lambda_tilde_feedforward_only"] = layer["lambda_tilde"]
                    alpha_rec_bar = layer["alpha_bar_p99"] + l_sigma * layer["rec_op_norm"]
                    alpha_rec_tilde = (
                        layer["alpha_tilde"]
                        + layer["sqrt_eta"] * l_sigma * layer["rec_eff_norm"]
                    )
                    denom_wc = 1.0 - alpha_rec_bar
                    denom_tc = 1.0 - alpha_rec_tilde
                    layer["alpha_rec_bar_p99"] = alpha_rec_bar
                    layer["alpha_rec_tilde"] = alpha_rec_tilde
                    layer["lambda_r"] = (
                        l_sigma * layer["W_op_norm"] / denom_wc
                        if denom_wc > 0 else float("inf")
                    )
                    layer["lambda_tilde"] = (
                        layer["sqrt_eta"] * l_sigma * layer["W_eff_norm"] / denom_tc
                        if denom_tc > 0 else float("inf")
                    )
                    layer["stable_worstcase"] = layer["lambda_r"] < 1.0
                    layer["stable_typicalcase"] = layer["lambda_tilde"] < 1.0
                    layer["recurrent_corrected"] = True
                row = {
                    "dataset_arch": dataset,
                    "mode": mode,
                    "layer": layer_name,
                    **layer,
                }
                rows.append(row)
    return rows


def instantiate_dvs(seed, n_batches):
    from snntorch import surrogate

    mod = load_module(ROOT / "train" / "dvs_gesture.py", "dvs_p0")
    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0
    cfg["batch_size"] = 4
    mod.seed_everything(seed)
    _, _, test_loader = mod.build_loaders(seed, cfg)
    surrogate_fn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))

    ckpt_root = ckpt_dir("DVS-Gesture")
    def make_model():
        return mod.DVSGestureSNN(
            num_classes=cfg["num_classes"],
            beta=cfg["beta_init"],
            threshold=cfg["threshold_init"],
            surrogate_fn=surrogate_fn,
            learn_beta=True,
            learn_threshold=False,
        ).to(DEVICE)

    weight_map = lambda model: {
        "lif1": model.conv1,
        "lif2": model.conv2,
        "lif3": model.conv3,
        "lif4": model.conv4,
        "lif5": model.conv5,
        "lif6": model.fc1,
        "lif7": model.fc2,
    }
    return cfg, test_loader, ckpt_root, make_model, weight_map


def instantiate_cifar(seed, n_batches):
    from snntorch import surrogate

    mod = load_module(ROOT / "train" / "cifar10.py", "cifar_p0")
    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0
    cfg["batch_size"] = 16
    mod.seed_everything(seed)
    _, _, test_loader = mod.build_loaders(seed, cfg)
    surrogate_fn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))

    ckpt_root = ckpt_dir("CIFAR-10")
    def make_model():
        return mod.SpikingResNet18(
            num_classes=10,
            beta=cfg["beta_init"],
            threshold=cfg["threshold_init"],
            surrogate_fn=surrogate_fn,
            learn_beta=True,
            learn_threshold=False,
            num_steps=cfg["time_steps"],
        ).to(DEVICE)

    def weight_map(model):
        mapping = {"stem.lif1": model.conv1, "head.lif_out": model.fc}
        for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], start=1):
            for block_idx, block in enumerate(layer):
                mapping[f"layer{layer_idx}.{block_idx}.lif1"] = block.conv1
                mapping[f"layer{layer_idx}.{block_idx}.lif2"] = block.conv2
        return mapping

    return cfg, test_loader, ckpt_root, make_model, weight_map


def compute_dataset_rows(dataset_arch, instantiate_fn, seed, n_batches):
    cfg, test_loader, ckpt_dir, make_model, weight_map_fn = instantiate_fn(seed, n_batches)
    alpha = cfg.get("surrogate_alpha", 2.0)
    l_sigma = l_sigma_atan(alpha)
    rows = []
    for mode in MODES:
        ckpt_path = ckpt_dir / f"best_s{seed}_{mode}.pth"
        if not ckpt_path.exists():
            print(f"Skipping {dataset_arch}/{mode}: missing {ckpt_path}")
            continue
        model = make_model()
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        typical = make_hook_stats(model, n_batches=n_batches, loader=test_loader, alpha=alpha)
        wmap = weight_map_fn(model)

        for layer_name, layer in model._lif_layers:
            reset = getattr(layer, "reset_mechanism", "subtract")
            if reset == "none":
                continue
            wi = weight_norms(wmap[layer_name].weight)
            tc = typical[layer_name]
            denom_wc = 1.0 - tc["alpha_bar_p99"]
            lambda_r = l_sigma * wi["W_op_norm"] / denom_wc if denom_wc > 0 else float("inf")
            denom_tc = 1.0 - tc["alpha_tilde"]
            lambda_tilde = (
                tc["sqrt_eta"] * l_sigma * wi["W_eff_norm"] / denom_tc
                if denom_tc > 0 else float("inf")
            )
            rows.append({
                "dataset_arch": dataset_arch,
                "mode": mode,
                "layer": layer_name,
                "reset": reset,
                "seed": seed,
                "n_batches": n_batches,
                "beta": beta_value(layer),
                **wi,
                **tc,
                "lambda_r": float(lambda_r),
                "lambda_tilde": float(lambda_tilde),
                "stable_worstcase": bool(lambda_r < 1.0),
                "stable_typicalcase": bool(lambda_tilde < 1.0),
                "tightening_ratio": float(lambda_tilde / lambda_r) if lambda_r > 0 and np.isfinite(lambda_r) else None,
            })
    return rows


def aggregate_max(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset_arch"], row["mode"])
        slot = grouped.setdefault(key, {"lambda_r": [], "lambda_tilde": []})
        slot["lambda_r"].append(row["lambda_r"])
        slot["lambda_tilde"].append(row["lambda_tilde"])
    out = []
    for (dataset, mode), vals in grouped.items():
        out.append({
            "dataset_arch": dataset,
            "mode": mode,
            "max_lambda_r": float(np.nanmax(vals["lambda_r"])),
            "max_lambda_tilde": float(np.nanmax(vals["lambda_tilde"])),
            "n_layers": len(vals["lambda_r"]),
        })
    return out


def save_csv(rows, path):
    keys = [
        "dataset_arch", "mode", "layer", "reset", "seed", "n_batches", "beta",
        "W_op_norm", "W_eff_norm", "kappa", "eta", "sqrt_eta",
        "alpha_bar_p99", "alpha_tilde", "lambda_r", "lambda_tilde",
        "stable_worstcase", "stable_typicalcase", "op_norm_method",
        "recurrent_corrected", "alpha_rec_bar_p99", "alpha_rec_tilde",
        "lambda_r_feedforward_only", "lambda_tilde_feedforward_only",
    ]
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def make_plot(summary, path_base):
    by_dataset = {d: {} for d in DATASET_ORDER}
    for row in summary:
        by_dataset.setdefault(row["dataset_arch"], {})[row["mode"]] = row

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(14.5, 3.4), sharey=True)
    width = 0.34
    x = np.arange(len(MODES))
    for ax, dataset in zip(axes, DATASET_ORDER):
        worst = [by_dataset.get(dataset, {}).get(m, {}).get("max_lambda_r", np.nan) for m in MODES]
        typical = [by_dataset.get(dataset, {}).get(m, {}).get("max_lambda_tilde", np.nan) for m in MODES]
        worst_plot = [PLOT_CAP if np.isinf(v) else v for v in worst]
        typical_plot = [PLOT_CAP if np.isinf(v) else v for v in typical]
        ax.bar(x - width / 2, worst_plot, width, color="#D9D9D9", edgecolor="0.35", label=r"Worst $\lambda_r$")
        ax.bar(x + width / 2, typical_plot, width, color="#7AA974", edgecolor="#3A6B35", label=r"Typical $\tilde{\lambda}_r$")
        for idx, value in enumerate(worst):
            if np.isinf(value):
                ax.text(idx - width / 2, PLOT_CAP * 0.85, r"$\infty$", ha="center", va="top", fontsize=8)
        for idx, value in enumerate(typical):
            if np.isinf(value):
                ax.text(idx + width / 2, PLOT_CAP * 0.85, r"$\infty$", ha="center", va="top", fontsize=8)
        ax.axhline(1.0, color="black", ls="--", lw=0.8)
        if np.all(np.isnan(worst)) and np.all(np.isnan(typical)):
            ax.text(
                0.5, 0.5, "not computed\n(data unavailable)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="0.35",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95),
            )
        ax.set_title(dataset, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in MODES], rotation=35, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("max gain across reset layers")
    axes[0].set_ylim(1e-3, 1e3)
    axes[-1].legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{path_base}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy", action="store_true", help="old (wrong) arctan formula")
    parser.add_argument("--skip-dvs", action="store_true", help="DVS test set not available locally")
    parser.add_argument("--skip-conv-datasets", action="store_true",
                        help="Only plot cached SHD-FC/SHD-RSNN/SSC diagnostics.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    shd_fc = RESULTS / "diagnostics" / "typical_case_shd_fc.json"
    # legacy RSNN/SSC diagnostics are not shipped; --legacy therefore only covers SHD-FC + conv datasets
    nested = (OUTPUT_ROOT / "analysis" / "typical_case_assumptions" / "assumptions_rsnn_ssc.json") if LEGACY \
        else (RESULTS / "diagnostics" / "assumptions_rsnn_ssc_fixed.json")
    if shd_fc.exists():
        rows.extend(rows_from_cached_shd_fc(shd_fc))
    if nested.exists():
        rows.extend(rows_from_cached_nested(nested))

    if not args.skip_conv_datasets:
        print(f"Computing DVS-Gesture and CIFAR-10 from {args.n_batches} batches on {DEVICE}...")
        for dataset_arch, instantiate_fn in [
            ("DVS-Gesture", instantiate_dvs),
            ("CIFAR-10", instantiate_cifar),
        ]:
            if dataset_arch == "DVS-Gesture" and args.skip_dvs:
                print("Skipping DVS-Gesture (--skip-dvs)")
                continue
            try:
                rows.extend(compute_dataset_rows(dataset_arch, instantiate_fn, args.seed, args.n_batches))
            except Exception as exc:
                print(f"Skipping {dataset_arch}: {type(exc).__name__}: {exc}")

    summary = aggregate_max(rows)
    tag = "_legacy" if LEGACY else "_fixed"
    out_json = OUT_DIR / f"lambda_r_by_dataset_arch{tag}.json"
    out_csv = OUT_DIR / f"lambda_r_by_dataset_arch{tag}.csv"
    out_summary = OUT_DIR / f"lambda_r_by_dataset_arch_summary{tag}.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    save_csv(rows, out_csv)
    make_plot(summary, OUT_DIR / f"lambda_r_by_dataset_arch{tag}")

    print(f"Saved detailed rows: {out_json}")
    print(f"Saved CSV:           {out_csv}")
    print(f"Saved summary:       {out_summary}")
    print(f"Saved plot:          {OUT_DIR / ('lambda_r_by_dataset_arch' + tag + '.png')}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
