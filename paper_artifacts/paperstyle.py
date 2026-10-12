"""Shared matplotlib style for the NeurIPS 2026 camera-ready MARS figures.

Usage:
    from paperstyle import apply_style, COLORS, LABELS, MARKERS, LINESTYLES, SINGLE_COL, FULL_W
    apply_style()

Palette = Okabe-Ito (colorblind-safe).  Clean=grey, NoiseAug=orange, MARS=blue.
Method keys follow the code base: "clean", "noise_aug", "asar" (= MARS checkpoints).
"""
from __future__ import annotations

import matplotlib as mpl

# NeurIPS text width is 5.5 in; single-column figures use the full text width.
FULL_W = 5.5
SINGLE_COL = 5.5

COLORS = {
    "clean": "#6E6E6E",      # grey
    "noise_aug": "#E69F00",  # Okabe-Ito orange
    "asar": "#0072B2",       # Okabe-Ito blue  (MARS)
}
# extra accents (for ablations / controls) from the same Okabe-Ito set
ACCENTS = {
    "vermillion": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
}
LABELS = {"clean": "Clean", "noise_aug": "NoiseAug", "asar": "MARS"}
MARKERS = {"clean": "o", "noise_aug": "s", "asar": "D"}
LINESTYLES = {"solid": "-", "dashed": "--"}
MODES = ["clean", "noise_aug", "asar"]


def apply_style(base_size: float = 8.0) -> None:
    """Set rcParams: all text >= 8 pt at print size (figure saved at its true size)."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "legend.fontsize": base_size,
            "figure.titlesize": base_size,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
            "errorbar.capsize": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.columnspacing": 1.0,
            "legend.handletextpad": 0.4,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # embed TrueType (NeurIPS requires no Type 3)
            "ps.fonttype": 42,
        }
    )
