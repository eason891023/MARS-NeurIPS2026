#!/usr/bin/env python3
"""
Paper Fig. 3 (fig3_heatmap_stability): (a) per-neuron |Psi_t| heat maps of the seed-42 SHD-FC
Clean and MARS checkpoints, (b) worst-case lambda_r vs typical-case lambda~ diagnostics.

Inputs : results/diagnostics/heatmap_data.npz (diagnostics/rerun_sensitivity_heatmap.py)
         results/diagnostics/results_fixed_shd_fc.json (diagnostics/eval_typical_case_stability_fixed.py;
         snnTorch ATan surrogate derivative, L_sigma = 1)
Output : outputs/figures/fig3_heatmap_stability.{pdf,png} (+ figure_font_check.json)

The figure is drawn at its final printed width (\\textwidth = 5.5in) with every font >= 8 pt;
the script checks the smallest font after the tight-bbox crop.
Usage  : python3 paper_artifacts/make_figures_fixed.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402

FIG_DIR = OUTPUT_ROOT / "figures"
HEATMAP_NPZ = RESULTS / "diagnostics" / "heatmap_data.npz"
STABILITY_PATH = RESULTS / "diagnostics" / "results_fixed_shd_fc.json"

TEXTWIDTH_IN = 5.5
MIN_PT = 8.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.4,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})

MODE_LABELS = {"clean": "Clean", "noise_aug": "Noise\nAug", "asar": "MARS"}


def _save_and_check(fig, name, final_width_in):
    """Save PDF+PNG and verify min font size at the final printed width."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    saved_w = bbox.width + 2 * plt.rcParams["savefig.pad_inches"]
    scale = final_width_in / saved_w
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    min_final = min(sizes) * scale
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)
    msg = (f"{name}: saved width {saved_w:.2f}in -> printed {final_width_in:.2f}in "
           f"(scale {scale:.3f}); min font {min(sizes):.1f}pt -> {min_final:.2f}pt printed")
    print(msg)
    assert min_final >= MIN_PT - 1e-6, msg
    return {"name": name, "saved_width_in": saved_w, "printed_width_in": final_width_in,
            "scale": scale, "min_font_pt_printed": min_final}


def make_fig3_heatmap_stability():
    npz = np.load(HEATMAP_NPZ)
    stability = json.loads(STABILITY_PATH.read_text())

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.75))
    gs_left = gridspec.GridSpec(2, 2, left=0.085, right=0.53, top=0.88, bottom=0.15,
                                wspace=0.06, hspace=0.22)
    gs_right = gridspec.GridSpec(1, 1, left=0.70, right=0.995, top=0.88, bottom=0.15)

    layers = ["lif1", "lif2"]
    layer_labels = ["Layer 1", "Layer 2"]
    for i, (layer, llabel) in enumerate(zip(layers, layer_labels)):
        clean = npz[f"clean_{layer}"].T  # (N, T)
        mars = npz[f"asar_{layer}"].T
        vmax = max(clean.max(), mars.max())
        n, T = clean.shape
        axes = []
        for j, (data, title) in enumerate(((clean, "Clean"), (mars, "MARS"))):
            ax = fig.add_subplot(gs_left[i, j])
            im = ax.imshow(data, aspect="auto", cmap="hot", vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.grid(False)
            ax.set_yticks([0, n - 1])
            if j == 0:
                ax.set_yticklabels(["1", str(n)])
                ax.set_ylabel(f"{llabel}\nneuron", labelpad=1)
            else:
                ax.set_yticklabels([])
            ax.set_xticks([0, T // 2, T - 1])
            if i == 1:
                ax.set_xticklabels(["1", str(T // 2 + 1), str(T)])
                ax.set_xlabel("Timestep", labelpad=1)
            else:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(title, fontweight="bold", pad=3)
            ax.tick_params(length=2, pad=1)
            axes.append(ax)
        cbar = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.03, aspect=10)
        cbar.locator = matplotlib.ticker.MaxNLocator(3)
        cbar.update_ticks()
        cbar.ax.tick_params(length=2, pad=1)
        cbar.set_label(r"$|\Psi_t|$", labelpad=1)

    fig.text(0.005, 0.95, "(a)", fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs_right[0, 0])
    order = ["clean", "noise_aug", "asar"]
    worst, typical = [], []
    for m in order:
        hidden = [r for r in stability[m] if r["layer"] in {"lif1", "lif2"}]
        worst.append(max(r["lambda_r"] for r in hidden))
        typical.append(max(r["lambda_tilde"] for r in hidden))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, worst, w, color="#D9D9D9", edgecolor="0.4", lw=0.6,
           label=r"Worst-case $\lambda_r$")
    ax.bar(x + w / 2, typical, w, color="#7AA974", edgecolor="#3A6B35", lw=0.6,
           label=r"Typical-case $\tilde{\lambda}_r$")
    ax.axhline(1.0, color="black", ls="--", lw=0.8, label=r"$\lambda=1$")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 1500)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in order], rotation=0, linespacing=0.9)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("Max gain over hidden layers", labelpad=1)
    ax.set_title("Diagnostics (SHD-FC)", fontweight="bold", pad=3)
    ax.legend(loc="upper right", framealpha=0.95, handlelength=1.2, borderpad=0.3,
              labelspacing=0.25, handletextpad=0.4)
    fig.text(0.615, 0.95, "(b)", fontsize=10, fontweight="bold")
    return _save_and_check(fig, "fig3_heatmap_stability", TEXTWIDTH_IN)


if __name__ == "__main__":
    out = [make_fig3_heatmap_stability()]
    (FIG_DIR / "figure_font_check.json").write_text(json.dumps(out, indent=2))
