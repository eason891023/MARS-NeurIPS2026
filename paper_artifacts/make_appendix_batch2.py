#!/usr/bin/env python3
"""Appendix tables (output: outputs/tables/tab_{joint,ptq}_rows.tex): joint beta/weight/threshold (incl. aux-only) and PTQ (8/6 bit) from
results/partial/SHD-*_jointptq.json (200 draws; beta draws shared with SHD-*_mc200.json)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402
P = RESULTS / "partial"
OUT = OUTPUT_ROOT / "tables"; OUT.mkdir(parents=True, exist_ok=True)
M = ["clean", "noise_aug", "mars"]; LAB = {"clean": "Clean", "noise_aug": "Noise Aug", "mars": "MARS"}; SEEDS = [42, 52, 72, 92, 102]
def f(x):
    t = f"{x:.2f}"; return "0.00" if t in ("-0.00", "0.00") else t.replace("-", "$-$")
def ms(v): v = np.asarray(v); return f"{f(v.mean())}$\\pm${v.std(ddof=1):.2f}"
J, Q = [], []
for a in ("SHD-FC", "SHD-RSNN"):
    d = json.load(open(P / f"{a}_jointptq.json")); b = json.load(open(P / f"{a}_mc200.json"))
    E = {(e["method"], e["seed"]): e for e in d["entries"].values()}; B = {(e["method"], e["seed"]): e for e in b["entries"].values()}
    for k, m in enumerate(M):
        beta = [B[m, s]["nominal_acc"] - np.mean(B[m, s]["results"]["per_neuron"]["0.2"]["runs"]) for s in SEEDS]
        aux = [E[m, s]["nominal_acc"] - E[m, s]["results"]["aux_only"]["mean"] for s in SEEDS]
        jnt = [E[m, s]["nominal_acc"] - E[m, s]["results"]["joint"]["mean"] for s in SEEDS]
        J.append(f"{a if k == 0 else ''} & {LAB[m]} & {ms(aux)} & {ms(beta)} & {ms(jnt)}" + r" \\")
        for bits in ("32", "8", "6"):
            if bits == "32":
                nom = [B[m, s]["nominal_acc"] for s in SEEDS]; dr = beta
            else:
                nom = [E[m, s]["ptq_nominal_acc"][bits] for s in SEEDS]
                dr = [E[m, s]["ptq_nominal_acc"][bits] - E[m, s]["results"][f"ptq{bits}"]["mean"] for s in SEEDS]
            Q.append((a, bits, m, nom, dr))
    if a == "SHD-FC": J.append(r"\midrule")
(OUT / "tab_joint_rows.tex").write_text("\n".join(J) + "\n")
L = []
for a in ("SHD-FC", "SHD-RSNN"):
    for bits in ("32", "8", "6"):
        for k, m in enumerate(M):
            row = [r for r in Q if r[0] == a and r[1] == bits and r[2] == m][0]
            nom = np.asarray(row[3])
            L.append(f"{a if (bits == '32' and k == 0) else ''} & {bits if k == 0 else ''} & {LAB[m]} & {nom.mean():.1f}$\\pm${nom.std(ddof=1):.1f} & {ms(row[4])}" + r" \\")
    if a == "SHD-FC": L.append(r"\midrule")
(OUT / "tab_ptq_rows.tex").write_text("\n".join(L) + "\n")
print(open(OUT / "tab_joint_rows.tex").read()); print(open(OUT / "tab_ptq_rows.tex").read())
