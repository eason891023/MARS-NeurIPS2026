#!/usr/bin/env python3
"""Appendix Fig. 4: post-hoc sensitivity energy vs per-neuron drop at CV=0.2.
Psi energies: results/mechanism/asar_vs_robustness.json (+ ssc_scatter_points.json); drops: unified
evaluator results/partial/ (SHD 200 draws, SSC/DVS 10 draws). Only Clean / Noise Aug / MARS
checkpoints (15 per setting).
Output: outputs/figures/fig4_sensitivity_drop_perneuron.{pdf,png}
Usage : python3 paper_artifacts/make_fig4_new.py"""
import importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402
spec = importlib.util.spec_from_file_location("mf", HERE / "make_figures_fixed.py")
mf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mf)
P = RESULTS / "partial"
recs = json.load(open(RESULTS / "mechanism/asar_vs_robustness.json")) + json.load(open(RESULTS / "mechanism/ssc_scatter_points.json"))
MODE = {"clean": "clean", "noise_aug": "noise_aug", "asar": "mars"}
FILES = {"SHD-FC": "SHD-FC_mc200.json", "SHD-RSNN": "SHD-RSNN_mc200.json", "SSC": "SSC.json", "DVS": "DVS-Gesture.json"}
def drops(fname):
    d = json.load(open(P / fname)); return {(e["method"], e["seed"]): e["nominal_acc"] - float(np.mean(e["results"]["per_neuron"]["0.2"]["runs"])) for e in d["entries"].values()}
order = ["SHD-FC", "SHD-RSNN", "SSC", "DVS"]
labels = {"SHD-FC": "SHD-FC", "SHD-RSNN": "SHD (RSNN)", "SSC": "SSC", "DVS": "DVS-Gesture"}
colors = {"SHD-FC": "#4477AA", "SHD-RSNN": "#EE6677", "SSC": "#228833", "DVS": "#AA3377"}
markers = {"SHD-FC": "o", "SHD-RSNN": "s", "SSC": "^", "DVS": "D"}
mf.apply_style() if hasattr(mf, "apply_style") else None
final_w = 0.78 * mf.TEXTWIDTH_IN
fig, ax = plt.subplots(figsize=(final_w, 2.85))
all_lx, all_y, per = [], [], {}
for ds in order:
    dr = drops(FILES[ds]); xs, ys = [], []
    for e in recs:
        if e["dataset"] == ds and e["mode"] in MODE and (MODE[e["mode"]], e["seed"]) in dr:
            xs.append(e["asar_loss"]); ys.append(dr[MODE[e["mode"]], e["seed"]])
    xs, ys = np.array(xs), np.array(ys); assert len(xs) == 15, (ds, len(xs))
    per[ds] = float(np.corrcoef(np.log10(xs), ys)[0, 1])
    ax.scatter(xs, ys, color=colors[ds], marker=markers[ds], s=22, alpha=0.85, edgecolors="white", linewidths=0.4, label=f"{labels[ds]} ($r={per[ds]:.2f}$)")
    all_lx += np.log10(xs).tolist(); all_y += ys.tolist()
r_pool = float(np.corrcoef(all_lx, all_y)[0, 1])
ax.set_xscale("log"); ax.axhline(0, color="0.55", ls="--", lw=0.7)
ax.set_xlabel(r"Sensitivity energy $\frac{1}{T}\sum_t \|\Psi_t\|^2$ (log scale)")
ax.set_ylabel("Drop @ CV=0.2 (pp)")
ax.set_title(f"Per-neuron mismatch (pooled $r={r_pool:.2f}$)", pad=3)
ax.legend(loc="upper left", ncol=1, framealpha=0.95, title="Per-setting Pearson", borderpad=0.3, labelspacing=0.2, handletextpad=0.2)
fig.tight_layout(pad=0.2)
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
out = OUTPUT_ROOT / "figures" / "fig4_sensitivity_drop_perneuron"
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=300)
sizes = [t.get_fontsize() for t in fig.findobj(plt.Text) if t.get_text().strip()]
assert min(sizes) >= 8.0, min(sizes)
print("per-setting r:", {k: round(v, 3) for k, v in per.items()}, "pooled r:", round(r_pool, 3))
