#!/usr/bin/env python3
"""Appendix tables (output: outputs/tables/tab_{main,perseed,paired}_rows.tex): tab:main (nominal + per-neuron drop with 95% CI),
per-seed per-neuron drops, and paired differences for the three mismatch structures.
Sources: results/partial/{SHD-*_mc200, SSC, DVS-Gesture, CIFAR-10, *_cv02d100}.json"""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402
P = RESULTS / "partial"
OUT = OUTPUT_ROOT / "tables"; OUT.mkdir(parents=True, exist_ok=True)
S = ["SHD-FC", "SHD-RSNN", "SSC", "DVS-Gesture", "CIFAR-10"]
M = ["clean", "noise_aug", "mars"]; LAB = {"clean": "Clean", "noise_aug": "Noise Aug", "mars": "MARS"}
SEEDS = [42, 52, 72, 92, 102]
def f(x):
    t = f"{x:.1f}"
    return "0.0" if t in ("-0.0", "0.0") else t.replace("-", "$-$")
def load(fname, mode):
    d = json.load(open(P / fname)); nom, dr = {}, {}
    for e in d["entries"].values():
        r = e["results"].get(mode, {}).get("0.2")
        if r is None: continue
        nom[e["method"], e["seed"]] = e["nominal_acc"]; dr[e["method"], e["seed"]] = e["nominal_acc"] - float(np.mean(r["runs"]))
    return nom, dr
def src(s, mode):
    if mode == "per_neuron": return (f"{s}_mc200.json" if s.startswith("SHD") else f"{s}.json")
    return f"{s}_cv02d100.json"
def ci(x):
    x = np.asarray(x); rng = np.random.default_rng(0); bm = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return x.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)
def pval(x):
    x = np.asarray(x); return float(np.mean([abs((x*np.array(sg)).mean()) >= abs(x.mean())-1e-12 for sg in itertools.product([1,-1], repeat=len(x))]))
# ---- tab:main
L = []
for s in S:
    nom, dr = load(src(s, "per_neuron"), "per_neuron")
    n = [np.array([nom[m, sd] for sd in SEEDS]) for m in M]
    d = [ci([dr[m, sd] for sd in SEEDS]) for m in M]
    L.append(f"\\multirow{{2}}{{*}}{{{s}}} & Nominal acc.\\,(\\%) & " + " & ".join(f"{f(v.mean())}$\\pm${v.std(ddof=1):.1f}" for v in n) + r" \\")
    L.append(" & Drop @ CV$=0.2$ (pp) & " + " & ".join(f"{f(a)} [{f(b)}, {f(c)}]" for a, b, c in d) + r" \\")
    if s != S[-1]: L.append(r"\addlinespace[2pt]")
(OUT / "tab_main_rows.tex").write_text("\n".join(L) + "\n")
# ---- per-seed drops
L = []
for s in S:
    _, dr = load(src(s, "per_neuron"), "per_neuron")
    for k, m in enumerate(M):
        lead = f"\\multirow{{3}}{{*}}{{{s}}}" if k == 0 else ""
        L.append(f"{lead} & {LAB[m]} & " + " & ".join(f(dr[m, sd]) for sd in SEEDS) + r" \\")
    if s != S[-1]: L.append(r"\midrule")
(OUT / "tab_perseed_rows.tex").write_text("\n".join(L) + "\n")
# ---- paired differences
L = []
for s in S:
    for k, (mode, name) in enumerate((("per_neuron", "Per-neuron"), ("layer_shared", "Layer-wise"), ("global", "Global"))):
        _, dr = load(src(s, mode), mode)
        a = [dr["mars", sd] - dr["noise_aug", sd] for sd in SEEDS]; b = [dr["mars", sd] - dr["clean", sd] for sd in SEEDS]
        ma, la, ha = ci(a); mb, lb, hb = ci(b)
        lead = f"\\multirow{{3}}{{*}}{{{s}}}" if k == 0 else ""
        L.append(f"{lead} & {name} & {f(ma)} [{f(la)}, {f(ha)}] & {sum(x<0 for x in a)}/5 & {pval(a):.3f} & {f(mb)} [{f(lb)}, {f(hb)}] & {sum(x<0 for x in b)}/5 & {pval(b):.3f}" + r" \\")
    if s != S[-1]: L.append(r"\midrule")
(OUT / "tab_paired_rows.tex").write_text("\n".join(L) + "\n")
print(open(OUT / "tab_main_rows.tex").read()); print(open(OUT / "tab_paired_rows.tex").read()[:1500])
