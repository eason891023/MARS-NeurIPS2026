"""
Uses the snnTorch ATan derivative sigma'(x)=(a/2)/(1+(pi a x/2)^2), L_sigma=a/2=1.
Inputs : checkpoints/SHD-FC/best_s42_*.pth, SHD test set
Output : outputs/diagnostics/stability_verification/results.json
         (shipped copy: results/diagnostics/stability_verification_fixed.json; seed-42 ||W||_op)

Stability verification: compute λ_r per layer for all trained checkpoints.
Validates Theorem 1 assumption (A3): λ_r^(l) < 1.

For FFSNN: λ_r = L_σ · ||W_ff||_op / (1 - α_eff)
For RSNN: α_R = α_eff + L_σ · ||W_rec||_op, then λ_R = L_σ · ||W_ff||_op / (1 - α_R)
"""
import importlib.util
import json
import os
import sys

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIXED: snnTorch ATan derivative sigma'(x) = (a/2)/(1+(pi*a*x/2)^2) <= a/2 = 1 (a=2)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from surrogate_atan import atan_deriv, l_sigma_atan  # noqa: E402
from snn_robust_tau.paths import OUTPUT_ROOT, ckpt_dir as _ckpt_dir  # noqa: E402
L_SIGMA = l_sigma_atan(2.0)  # = 1.0


def spectral_norm(weight_matrix):
    """Compute operator norm (largest singular value) of a weight matrix."""
    with torch.no_grad():
        if weight_matrix.dim() == 2:
            s = torch.linalg.svdvals(weight_matrix)
        else:
            # Reshape conv weight to 2D
            w = weight_matrix.view(weight_matrix.shape[0], -1)
            s = torch.linalg.svdvals(w)
        return float(s[0].item())


def compute_alpha_eff(model, loader, n_batches=20):
    """
    Compute effective decay α_eff for each LIF layer by running forward pass
    and tracking the actual α_eff(t) values.
    Returns dict: layer_name -> {mean, max, p95, p99}
    """
    model.eval()
    from snntorch import surrogate
    surr_fn = surrogate.atan(alpha=2.0)

    alpha_stats = {}
    for name, layer in model._lif_layers:
        alpha_stats[name] = []

    count = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE).float()
            bsz = data.shape[0]

            if data.dim() >= 3:
                steps = data.shape[1]
                x = data.view(bsz, steps, -1)
            else:
                continue

            # Manual forward pass to track α_eff
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()

            lif_layers_dict = dict(model._lif_layers)

            prev_spk1 = torch.zeros_like(mem1)
            prev_spk2 = torch.zeros_like(mem2)

            for t in range(steps):
                # Layer 1
                cur1 = model.fc1(x[:, t, :])
                spk1, mem1 = model.lif1(cur1, mem1)

                beta1 = model.lif1.beta.detach()
                reset1 = getattr(model.lif1, "reset_mechanism", "subtract")
                thr1 = getattr(model.lif1, "threshold", torch.tensor(1.0))
                if isinstance(thr1, nn.Parameter):
                    thr1 = thr1.detach()

                if reset1 == "none":
                    alpha1 = beta1.expand(bsz, -1) if beta1.dim() > 0 else beta1.unsqueeze(0).expand(bsz, 1)
                elif reset1 == "subtract":
                    surr_val = atan_deriv(mem1 - thr1, 2.0)
                    alpha1 = (beta1 - thr1 * surr_val).abs()
                else:  # zero/hard reset
                    pre_reset = cur1 + beta1 * mem1  # approximate
                    surr_val = atan_deriv(mem1 - thr1, 2.0)
                    alpha1 = (beta1 * (1 - prev_spk1) - pre_reset * surr_val).abs()

                alpha_stats["lif1"].append(alpha1.cpu().numpy())
                prev_spk1 = spk1

                # Layer 2
                cur2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur2, mem2)

                beta2 = model.lif2.beta.detach()
                reset2 = getattr(model.lif2, "reset_mechanism", "subtract")
                thr2 = getattr(model.lif2, "threshold", torch.tensor(1.0))
                if isinstance(thr2, nn.Parameter):
                    thr2 = thr2.detach()

                if reset2 == "none":
                    alpha2 = beta2.expand(bsz, -1) if beta2.dim() > 0 else beta2.unsqueeze(0).expand(bsz, 1)
                elif reset2 == "subtract":
                    surr_val = atan_deriv(mem2 - thr2, 2.0)
                    alpha2 = (beta2 - thr2 * surr_val).abs()
                else:
                    pre_reset = cur2 + beta2 * mem2
                    surr_val = atan_deriv(mem2 - thr2, 2.0)
                    alpha2 = (beta2 * (1 - prev_spk2) - pre_reset * surr_val).abs()

                alpha_stats["lif2"].append(alpha2.cpu().numpy())
                prev_spk2 = spk2

                # Layer 3 (readout, no-reset)
                cur3 = model.fc3(spk2)
                spk3, mem3 = model.lif3(cur3, mem3)
                beta3 = model.lif3.beta.detach()
                alpha3 = beta3.expand(bsz, -1) if beta3.dim() > 0 else beta3.unsqueeze(0).expand(bsz, 1)
                alpha_stats["lif3"].append(alpha3.cpu().numpy())

            count += 1
            if count >= n_batches:
                break

    # Aggregate
    result = {}
    for name in alpha_stats:
        all_vals = np.concatenate(alpha_stats[name], axis=0).flatten()
        result[name] = {
            "mean": float(np.mean(all_vals)),
            "max": float(np.max(all_vals)),
            "p95": float(np.percentile(all_vals, 95)),
            "p99": float(np.percentile(all_vals, 99)),
        }
    return result


def analyze_model(model, loader, model_name, n_batches=20):
    """Full stability analysis for one model."""
    lif_dict = dict(model._lif_layers)

    # Get weight spectral norms
    # For SHDSNN: fc1->lif1, fc2->lif2, fc3->lif3
    fc_layers = []
    for name in ["fc1", "fc2", "fc3"]:
        if hasattr(model, name):
            fc_layers.append((name, getattr(model, name)))

    weight_norms = {}
    for fc_name, fc_layer in fc_layers:
        weight_norms[fc_name] = spectral_norm(fc_layer.weight)

    # Get recurrent weight norms (for RSNN)
    rec_norms = {}
    for name, layer in model._lif_layers:
        if hasattr(layer, "recurrent"):
            rec_w = layer.recurrent.weight if hasattr(layer.recurrent, "weight") else None
            if rec_w is not None:
                rec_norms[name] = spectral_norm(rec_w)
        elif hasattr(layer, "V"):
            rec_norms[name] = spectral_norm(layer.V)

    # Get alpha_eff
    alpha_eff = compute_alpha_eff(model, loader, n_batches)

    # Compute lambda_r per hidden layer
    lif_names = [n for n, _ in model._lif_layers]
    fc_map = {lif_names[i]: fc_layers[i][0] for i in range(min(len(lif_names), len(fc_layers)))}

    results = []
    for lif_name, layer in model._lif_layers:
        beta_val = float(layer.beta.detach().view(-1)[0].item())
        reset = getattr(layer, "reset_mechanism", "subtract")
        fc_name = fc_map.get(lif_name, None)
        w_norm = weight_norms.get(fc_name, 0.0) if fc_name else 0.0
        rec_norm = rec_norms.get(lif_name, 0.0)
        alpha = alpha_eff[lif_name]

        # Use p99 of alpha_eff for stability (conservative)
        alpha_bar = alpha["p99"]

        if reset == "none":
            # Readout layer, skip stability analysis
            entry = {
                "layer": lif_name,
                "reset": reset,
                "beta": beta_val,
                "W_ff_norm": w_norm,
                "W_rec_norm": rec_norm,
                "alpha_bar": alpha_bar,
                "alpha_mean": alpha["mean"],
                "lambda_r": None,
                "stable": None,
                "note": "readout (no stability check)",
            }
        else:
            if rec_norm > 0:
                # RSNN
                alpha_R = alpha_bar + L_SIGMA * rec_norm
                denom = 1.0 - alpha_R
                if denom <= 0:
                    lambda_r = float("inf")
                else:
                    # Next layer's W_ff norm
                    next_idx = lif_names.index(lif_name) + 1
                    if next_idx < len(fc_layers):
                        w_next = weight_norms[fc_layers[next_idx][0]]
                    else:
                        w_next = w_norm
                    lambda_r = L_SIGMA * w_next / denom
                entry = {
                    "layer": lif_name,
                    "reset": reset,
                    "beta": beta_val,
                    "W_ff_norm": w_norm,
                    "W_rec_norm": rec_norm,
                    "alpha_bar": alpha_bar,
                    "alpha_R": alpha_R,
                    "alpha_mean": alpha["mean"],
                    "lambda_r": lambda_r,
                    "stable": lambda_r < 1.0,
                }
            else:
                # FFSNN
                # lambda_r uses the next layer's W_ff to propagate
                next_idx = lif_names.index(lif_name) + 1
                if next_idx < len(fc_layers):
                    w_next = weight_norms[fc_layers[next_idx][0]]
                else:
                    w_next = w_norm
                denom = 1.0 - alpha_bar
                if denom <= 0:
                    lambda_r = float("inf")
                else:
                    lambda_r = L_SIGMA * w_next / denom
                entry = {
                    "layer": lif_name,
                    "reset": reset,
                    "beta": beta_val,
                    "W_ff_norm": w_norm,
                    "W_rec_norm": 0.0,
                    "alpha_bar": alpha_bar,
                    "alpha_mean": alpha["mean"],
                    "lambda_r": lambda_r,
                    "stable": lambda_r < 1.0,
                }
        results.append(entry)

    return results


def main():
    output_dir = str(OUTPUT_ROOT / "diagnostics" / "stability_verification")
    os.makedirs(output_dir, exist_ok=True)

    # SHD-FC
    spec = importlib.util.spec_from_file_location(
        "shd", os.path.join(PROJECT_ROOT, "train", "shd_fc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0

    from snntorch import surrogate
    surrogate_fn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))

    mod.seed_everything(42)
    _, _, test_loader, input_size = mod.build_loaders(42, cfg)

    ckpt_dir = str(_ckpt_dir("SHD-FC"))

    all_results = {}
    for mode in ["clean", "asar", "hajr", "noise_aug"]:
        ckpt_path = os.path.join(ckpt_dir, f"best_s42_{mode}.pth")
        if not os.path.exists(ckpt_path):
            continue

        model = mod.SHDSNN(
            input_size=input_size,
            beta_init=cfg["beta_init"],
            threshold_init=cfg.get("threshold_init", 1.0),
            surrogate_fn=surrogate_fn,
            learn_beta=True,
            learn_threshold=False,
        ).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(f"\n=== SHD-FC / {mode} ===")
        results = analyze_model(model, test_loader, f"SHD-FC/{mode}")
        all_results[f"SHD-FC/{mode}"] = results

        print(f"{'Layer':>8s} {'Reset':>8s} {'β':>6s} {'||W||':>7s} {'ᾱ(p99)':>8s} {'λ_r':>7s} {'Stable':>7s}")
        print("-" * 60)
        for r in results:
            lr_str = f"{r['lambda_r']:.3f}" if r["lambda_r"] is not None else "—"
            st_str = "✓" if r.get("stable") else ("✗" if r.get("stable") is False else "—")
            print(f"{r['layer']:>8s} {r['reset']:>8s} {r['beta']:>6.3f} {r['W_ff_norm']:>7.2f} {r['alpha_bar']:>8.4f} {lr_str:>7s} {st_str:>7s}")

    # Compute propagation factors
    for key in all_results:
        layers = all_results[key]
        hidden = [l for l in layers if l["lambda_r"] is not None]
        for i, l in enumerate(hidden):
            p = 1.0
            for j in range(i + 1, len(hidden)):
                if hidden[j]["lambda_r"] is not None and hidden[j]["lambda_r"] != float("inf"):
                    p *= hidden[j]["lambda_r"]
            l["propagation_factor"] = p
        print(f"\n{key} propagation factors:")
        for l in hidden:
            print(f"  {l['layer']}: p = {l.get('propagation_factor', '?')}")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
