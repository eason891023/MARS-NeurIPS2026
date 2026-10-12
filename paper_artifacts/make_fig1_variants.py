#!/usr/bin/env python3
"""Paper Fig. 1 (3 temporally dependent + 2 control panels) and its layer-wise counterpart.

  fig1_robustness_perneuron.{pdf,png} : per-neuron beta perturbation (paper Fig. 1),
                                        results/partial/<setting>.json (10 draws)
  fig1_robustness_layerwise.{pdf,png} : layer-wise beta perturbation (one draw per layer; not in the paper),
                                        results/partial/<setting>_lw50.json (SSC: _lw30.json)

Per checkpoint the MC draws are averaged first; markers = mean over 5 training seeds,
error bars = 95% bootstrap CI over seeds, faint dots = individual seeds.  CPU-only, reads inputs only.
Usage: python3 paper_artifacts/make_fig1_variants.py   (writes outputs/figures/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from paperstyle import COLORS, FULL_W, apply_style  # noqa: E402
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402

PARTIAL = RESULTS / "partial"
OUT = OUTPUT_ROOT / "figures"
METHODS = ["clean", "noise_aug", "mars"]
COLOR = {"clean": COLORS["clean"], "noise_aug": COLORS["noise_aug"], "mars": COLORS["asar"]}
LABEL = {"clean": "Clean", "noise_aug": "Noise Aug", "mars": "MARS"}
MARKER = {"clean": "o", "noise_aug": "s", "mars": "D"}
PANELS = [("SHD-FC", "a"), ("SHD-RSNN", "b"), ("SSC", "c"), ("DVS-Gesture", "d"), ("CIFAR-10", "e")]
VARIANTS = {
    "layerwise": ("layer_shared", lambda s: f"{s}_lw30.json" if s == "SSC" else f"{s}_lw50.json",
                  "Layer-wise $\\beta$ perturbation"),
    "perneuron": ("per_neuron", lambda s: f"{s}.json", "Per-neuron $\\beta$ perturbation"),
}
N_BOOT, MIN_FONT, MIN_SPAN = 10_000, 8.0, 10.0


def load(fname, mode):
    d = json.load(open(PARTIAL / fname))
    nominal = {m: {} for m in METHODS}
    acc = {m: {} for m in METHODS}
    for e in d["entries"].values():
        meth, seed = e["method"], int(e["seed"])
        nominal[meth][seed] = float(e["nominal_acc"])
        for cv, r in e["results"].get(mode, {}).items():
            acc[meth].setdefault(float(cv), {})[seed] = float(np.mean(r["runs"]))
    return nominal, acc, d.get("draws")


def boot_ci(vals, rng):
    v = np.asarray(vals, float)
    bm = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return v.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def panel(ax, letter, setting, nominal, acc, rng, first_col, bottom_row):
    ax.set_title(rf"$\bf{{({letter})}}$ {setting}", loc="left", pad=2)
    lo, hi = np.inf, -np.inf
    for k, meth in enumerate(METHODS):
        pts = [(0.0, list(nominal[meth].values()))] + [(cv, list(v.values())) for cv, v in sorted(acc[meth].items())]
        xs, ms, ls, hs = [], [], [], []
        for cv, vals in pts:
            m, l, h = boot_ci(vals, rng)
            x = cv + (k - 1) * 0.006
            xs.append(x); ms.append(m); ls.append(l); hs.append(h)
            # seed-level points (each = one training seed, MC draws averaged)
            ax.scatter([x] * len(vals), vals, s=2.5, color=COLOR[meth], alpha=0.35, lw=0, zorder=2)
            lo, hi = min(lo, min(vals)), max(hi, max(vals))
        ms, ls, hs = map(np.array, (ms, ls, hs))
        ax.errorbar(xs, ms, yerr=[ms - ls, hs - ms], color=COLOR[meth], marker=MARKER[meth], ms=2.6,
                    lw=0.9, elinewidth=0.7, capsize=1.3, capthick=0.7, zorder=3 + k)
        lo, hi = min(lo, ls.min()), max(hi, hs.max())
    # common minimum y-span so that sub-pp changes in the control panels are not visually exaggerated
    pad = 0.08 * max(hi - lo, MIN_SPAN)
    ax.set_ylim(min(lo - pad, hi + pad - MIN_SPAN), hi + pad)
    ax.set_xlim(-0.02, 0.32)
    ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_xticklabels(["0", ".1", ".2", ".3"])
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    if first_col:
        ax.set_ylabel("Accuracy (%)", labelpad=1)
    if bottom_row:
        ax.set_xlabel("Perturbation CV", labelpad=1)


def check_fonts(fig):
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(sizes) >= MIN_FONT, f"font too small: {min(sizes)}"


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, (mode, fname, title) in VARIANTS.items():
        rng = np.random.default_rng(0)
        fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.3))
        draws = {}
        for i, (setting, letter) in enumerate(PANELS):
            ax = axes.flat[i]
            nominal, acc, draws[setting] = load(fname(setting), mode)
            panel(ax, letter, setting, nominal, acc, rng, first_col=(i % 3 == 0), bottom_row=(i >= 2))
        leg = axes.flat[5]
        leg.axis("off")
        handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ms=3.5, lw=1.0, label=LABEL[m]) for m in METHODS]
        leg.legend(handles=handles, loc="center", frameon=False, title=title, title_fontsize=MIN_FONT)
        fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.6)
        check_fonts(fig)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"fig1_robustness_{tag}.{ext}", dpi=300)
        plt.close(fig)
        print(tag, "draws per checkpoint:", draws)


if __name__ == "__main__":
    main()
