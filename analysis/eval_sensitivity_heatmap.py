"""
Sensitivity heatmap: visualize ||Ψ_t^(l)|| per neuron over time for clean vs ASAR.
Loads checkpoints, runs forward pass, extracts per-neuron Ψ_t.
"""
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


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

from snn_robust_tau.loss.hajr import OnlineHAJR


def collect_psi_per_neuron(model, loader, n_batches=10):
    """
    Run forward pass with ASAR tracking and collect per-neuron |Ψ_t| at each timestep.
    Returns dict: layer_name -> (T, N) array of mean |Ψ_t| across batches.
    """
    model.eval()

    lif_names = [name for name, _ in model._lif_layers]
    # Accumulators: layer -> list of (T, N) arrays
    psi_accum = {name: [] for name in lif_names}
    count = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE).float()
            bsz, steps = data.shape[0], data.shape[1] if data.dim() > 2 else model.lif3.beta.shape[0]

            # We need to manually run the forward pass to extract Ψ per neuron
            if data.dim() == 4:
                # SHD: (B, T, C, H) -> flatten
                steps = data.shape[1]
                x = data.view(bsz, steps, -1)
            elif data.dim() == 3:
                steps = data.shape[1]
                x = data.view(bsz, steps, -1)
            else:
                continue

            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()
            model.online_reg.reset()

            psi_per_t = {name: [] for name in lif_names}

            for t in range(steps):
                cur = model.fc1(x[:, t, :])
                spk1, mem1 = model.lif1(cur, mem1)

                # Step ASAR and capture Ψ
                model.online_reg.step_from_layer(
                    "lif1", model.lif1, mem1, timestep=t, spike=spk1, decision_gate=False
                )
                psi1 = model.online_reg.states["lif1"]["psi"].detach()
                psi_per_t["lif1"].append(psi1.abs().mean(dim=0).cpu().numpy())  # (N,)

                cur = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur, mem2)
                model.online_reg.step_from_layer(
                    "lif2", model.lif2, mem2, timestep=t, spike=spk2, decision_gate=False
                )
                psi2 = model.online_reg.states["lif2"]["psi"].detach()
                psi_per_t["lif2"].append(psi2.abs().mean(dim=0).cpu().numpy())

                cur = model.fc3(spk2)
                spk3, mem3 = model.lif3(cur, mem3)
                model.online_reg.step_from_layer(
                    "lif3", model.lif3, mem3, timestep=t, spike=spk3, decision_gate=False
                )
                psi3 = model.online_reg.states["lif3"]["psi"].detach()
                psi_per_t["lif3"].append(psi3.abs().mean(dim=0).cpu().numpy())

            for name in lif_names:
                arr = np.stack(psi_per_t[name], axis=0)  # (T, N)
                psi_accum[name].append(arr)

            count += 1
            if count >= n_batches:
                break

    # Average across batches
    result = {}
    for name in lif_names:
        result[name] = np.mean(psi_accum[name], axis=0)  # (T, N)
    return result


def plot_heatmaps(psi_clean, psi_asar, output_dir):
    """Plot side-by-side heatmaps for clean vs ASAR."""
    layers = list(psi_clean.keys())

    fig, axes = plt.subplots(len(layers), 2, figsize=(14, 3.5 * len(layers)))
    if len(layers) == 1:
        axes = axes.reshape(1, -1)

    for i, layer in enumerate(layers):
        # Find shared vmax
        vmax = max(psi_clean[layer].max(), psi_asar[layer].max())
        vmax = max(vmax, 1e-3)  # avoid 0

        # Clean
        im0 = axes[i, 0].imshow(
            psi_clean[layer].T, aspect="auto", cmap="hot",
            vmin=0, vmax=vmax, interpolation="nearest"
        )
        axes[i, 0].set_title(f"{layer} — Clean")
        axes[i, 0].set_xlabel("Timestep")
        axes[i, 0].set_ylabel("Neuron index")
        plt.colorbar(im0, ax=axes[i, 0], label="|Ψ|")

        # ASAR
        im1 = axes[i, 1].imshow(
            psi_asar[layer].T, aspect="auto", cmap="hot",
            vmin=0, vmax=vmax, interpolation="nearest"
        )
        axes[i, 1].set_title(f"{layer} — ASAR")
        axes[i, 1].set_xlabel("Timestep")
        axes[i, 1].set_ylabel("Neuron index")
        plt.colorbar(im1, ax=axes[i, 1], label="|Ψ|")

    fig.suptitle("Membrane Sensitivity |Ψ_t| per Neuron: Clean vs ASAR (SHD-FC)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sensitivity_heatmap.png"), dpi=200)
    plt.close(fig)

    # Also plot per-layer time traces (mean over neurons)
    fig2, axes2 = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
    if len(layers) == 1:
        axes2 = [axes2]
    for i, layer in enumerate(layers):
        t = np.arange(psi_clean[layer].shape[0])
        mean_clean = psi_clean[layer].mean(axis=1)
        mean_asar = psi_asar[layer].mean(axis=1)
        axes2[i].plot(t, mean_clean, "r-", lw=2, label="Clean")
        axes2[i].plot(t, mean_asar, "g-", lw=2, label="ASAR")
        axes2[i].set_xlabel("Timestep")
        axes2[i].set_ylabel("Mean |Ψ_t|")
        axes2[i].set_title(layer)
        axes2[i].legend()
        axes2[i].grid(alpha=0.3)
    fig2.suptitle("Mean Sensitivity Over Time: Clean vs ASAR", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "sensitivity_traces.png"), dpi=200)
    plt.close(fig2)


def main():
    output_dir = str(OUTPUT_ROOT / "analysis" / "sensitivity_heatmap")
    os.makedirs(output_dir, exist_ok=True)

    # Import SHD module
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

    ckpt_dir = str(ckpt_dir("SHD-FC"))

    for mode_label, ckpt_name in [("clean", "best_s42_clean.pth"), ("asar", "best_s42_asar.pth")]:
        model = mod.SHDSNN(
            input_size=input_size,
            beta_init=cfg["beta_init"],
            threshold_init=cfg.get("threshold_init", 1.0),
            surrogate_fn=surrogate_fn,
            learn_beta=True,
            learn_threshold=False,
        ).to(DEVICE)

        ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {mode_label}: {ckpt_name}")

        psi = collect_psi_per_neuron(model, test_loader, n_batches=20)
        if mode_label == "clean":
            psi_clean = psi
        else:
            psi_asar = psi

        # Print summary
        for layer in psi:
            print(f"  {layer}: mean|Ψ|={psi[layer].mean():.4f}, max|Ψ|={psi[layer].max():.4f}")

    # Save raw per-neuron data for figure regeneration
    np.savez(
        os.path.join(output_dir, "heatmap_data.npz"),
        **{f"clean_{k}": v for k, v in psi_clean.items()},
        **{f"asar_{k}": v for k, v in psi_asar.items()},
    )
    print(f"Saved heatmap_data.npz")

    plot_heatmaps(psi_clean, psi_asar, output_dir)

    # Save numeric data
    data = {}
    for name in psi_clean:
        data[name] = {
            "clean_mean": float(psi_clean[name].mean()),
            "clean_max": float(psi_clean[name].max()),
            "asar_mean": float(psi_asar[name].mean()),
            "asar_max": float(psi_asar[name].max()),
            "ratio": float(psi_clean[name].mean() / max(psi_asar[name].mean(), 1e-10)),
        }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
