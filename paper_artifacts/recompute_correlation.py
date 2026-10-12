#!/usr/bin/env python3
"""Mechanism correlation (appendix table tab:app_mechanism): log post-hoc Psi energy vs beta-CV0.2 drop.
Psi energies: results/mechanism/asar_vs_robustness.json (SSC: results/mechanism/ssc_scatter_points.json).
Drops: unified evaluator, results/partial/. Method replicates analysis/mechanism_triangulation.py:
log10 energy and drop are demeaned within dataset, then pooled Pearson/Spearman over the 3 sensitive
settings. Output: outputs/tables/correlation_results.json (+ stdout).
Usage: python3 paper_artifacts/recompute_correlation.py"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402
P = RESULTS / "partial"
recs = json.load(open(RESULTS / "mechanism/asar_vs_robustness.json"))
recs += json.load(open(RESULTS / "mechanism/ssc_scatter_points.json"))
SETS = ["SHD-FC", "SHD-RSNN", "SSC"]
MAP = {"clean": "clean", "noise_aug": "noise_aug", "asar": "mars"}
psi = {(r["dataset"], r["mode"], r["seed"]): r["asar_loss"] for r in recs if r["dataset"] in SETS and r["mode"] in MAP}
old = {(r["dataset"], r["mode"], r["seed"]): r["acc_drop"] for r in recs if r["dataset"] in SETS and r["mode"] in MAP}


def drops(fname, mode):
    d = json.load(open(P / fname)); out = {}
    for e in d["entries"].values():
        r = e["results"].get(mode, {}).get("0.2")
        if r:
            out[e["method"], e["seed"]] = e["nominal_acc"] - float(np.mean(r["runs"]))
    return out


new = {}
for s in SETS:
    pn = drops(f"{s}_mc200.json" if s.startswith("SHD") else f"{s}.json", "per_neuron")
    lw = drops(f"{s}_cv02d100.json", "layer_shared")
    for (m, seed), v in pn.items():
        new.setdefault("per_neuron", {})[(s, {v2: k2 for k2, v2 in MAP.items()}[m], seed)] = v
    for (m, seed), v in lw.items():
        new.setdefault("layer_wise_100", {})[(s, {v2: k2 for k2, v2 in MAP.items()}[m], seed)] = v
new["old_submitted_10draw"] = old


def corr(dropmap, modes):
    rx, ry = [], []
    for s in SETS:
        keys = [k for k in psi if k[0] == s and k[1] in modes and k in dropmap]
        x = np.log10([psi[k] for k in keys]); y = np.array([dropmap[k] for k in keys])
        rx += list(x - x.mean()); ry += list(y - y.mean())
    p, s_ = pearsonr(rx, ry), spearmanr(rx, ry)
    return len(rx), p.statistic, p.pvalue, s_.statistic, s_.pvalue


res = {}
for name, dm in new.items():
    for label, modes in (("all", ("clean", "noise_aug", "asar")), ("clean+noiseaug", ("clean", "noise_aug"))):
        n, r, pr, rho, prho = corr(dm, modes)
        res[f"{name}|{label}"] = dict(n=n, pearson=r, pearson_p=pr, spearman=rho, spearman_p=prho)
        print(f"{name:22s} {label:15s} n={n} r={r:.3f} (p={pr:.2g}) rho={rho:.3f} (p={prho:.2g})")
order = sum(psi[(s, "asar", sd)] < psi[(s, "noise_aug", sd)] < psi[(s, "clean", sd)] for s in SETS for sd in (42, 52, 72, 92, 102))
print("ordering MARS<NA<Clean in", order, "/ 15")
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUTPUT_ROOT / "tables" / "correlation_results.json", "w"), indent=1)
