#!/usr/bin/env python3
"""LaTeX rows of the three main-text tables, computed from the shipped evaluation results.

  Table 1 (tab:structured): accuracy drop (pp) at CV=0.2 under per-neuron, layer-wise and global
           beta mismatch. Per-neuron: results/partial/SHD-*_mc200.json (200 draws) for SHD,
           results/partial/{SSC,DVS-Gesture,CIFAR-10}.json (10 draws) otherwise; layer-wise and
           global: results/partial/<setting>_cv02d100.json (100 draws).
  Table 2 (tab:joint):      beta mismatch combined with weight/threshold mismatch or 8/6-bit PTQ (SHD);
           results/partial/SHD-*_jointptq.json and SHD-*_mc200.json (beta-only row).
  Table 3 (tab:baselines):  simple baselines under per-neuron beta mismatch at CV=0.2 (SHD);
           results/partial/SHD-*_controls_mc200.json and SHD-*_mc200.json.

Statistics: the drop of a checkpoint is its nominal accuracy minus the accuracy averaged over all
mismatch draws; tables report mean +- SD (ddof=1) over the 5 training seeds 42, 52, 72, 92, 102.

Output: outputs/tables/table{1,2,3}_rows.tex (and stdout).
Usage : python3 paper_artifacts/make_main_tables.py [--check path/to/main.tex]
        --check verifies that every generated row occurs verbatim in the given LaTeX source.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402

P = RESULTS / "partial"
SEEDS = [42, 52, 72, 92, 102]
METHODS = ["clean", "noise_aug", "mars"]
SETTINGS = ["SHD-FC", "SHD-RSNN", "SSC", "DVS-Gesture", "CIFAR-10"]


def load(fname: str) -> dict:
    d = json.loads((P / fname).read_text())
    return {(e["method"], int(e["seed"])): e for e in d["entries"].values()}


def num(x: float) -> str:
    t = f"{x:.1f}"
    if t in ("-0.0", "0.0"):
        return "0.0"
    return t.replace("-", "$-$")


def cell(values) -> str:
    v = np.asarray(values, float)
    return f"{num(v.mean())}$\\pm${v.std(ddof=1):.1f}"


def drops(entries: dict, method: str, mode: str, cv: str = "0.2") -> list[float]:
    out = []
    for s in SEEDS:
        e = entries[method, s]
        out.append(e["nominal_acc"] - float(np.mean(e["results"][mode][cv]["runs"])))
    return out


def per_neuron_file(setting: str) -> str:
    return f"{setting}_mc200.json" if setting.startswith("SHD") else f"{setting}.json"


def table1() -> list[str]:
    rows = []
    for s in SETTINGS:
        pn = load(per_neuron_file(s))
        st = load(f"{s}_cv02d100.json")
        cells = [cell(drops(pn, m, "per_neuron")) for m in METHODS]
        cells += [cell(drops(st, m, "layer_shared")) for m in METHODS]
        cells += [cell(drops(st, m, "global")) for m in METHODS]
        rows.append(f"{s} & " + " & ".join(cells) + r" \\")
    return rows


def table2() -> list[str]:
    beta, joint, ptq8, ptq6 = [], [], [], []
    for s in ("SHD-FC", "SHD-RSNN"):
        b = load(f"{s}_mc200.json")
        j = load(f"{s}_jointptq.json")
        for m in METHODS:
            beta.append(cell(drops(b, m, "per_neuron")))
            joint.append(cell([j[m, sd]["nominal_acc"] - j[m, sd]["results"]["joint"]["mean"] for sd in SEEDS]))
            ptq8.append(cell([j[m, sd]["ptq_nominal_acc"]["8"] - j[m, sd]["results"]["ptq8"]["mean"] for sd in SEEDS]))
            ptq6.append(cell([j[m, sd]["ptq_nominal_acc"]["6"] - j[m, sd]["results"]["ptq6"]["mean"] for sd in SEEDS]))
    return [
        r"$\beta$ only (Table~\ref{tab:structured}) & " + " & ".join(beta) + r" \\",
        r"$\beta$ + weight + threshold & " + " & ".join(joint) + r" \\",
        r"8-bit PTQ + $\beta$ & " + " & ".join(ptq8) + r" \\",
        r"6-bit PTQ + $\beta$ & " + " & ".join(ptq6) + r" \\",
    ]


def table3() -> list[str]:
    main = {s: load(f"{s}_mc200.json") for s in ("SHD-FC", "SHD-RSNN")}
    ctrl = {s: load(f"{s}_controls_mc200.json") for s in ("SHD-FC", "SHD-RSNN")}
    spec = [  # (row label, source, method key)
        (r"Clean (learnable $\beta$)", "main", "clean"),
        (r"$\beta$-L2", "ctrl", "beta_l2"),
        (r"Hard $\beta$ range $[0.1,0.9]$", "ctrl", "beta_clip"),
        ("Weight decay", "ctrl", "weight_decay"),
        ("Spectral normalization", "ctrl", "spectral_norm"),
        ("Recurrent DropConnect", "ctrl", "dropconnect"),
        (r"Fixed $\beta$, homogeneous", "ctrl", "fixed_homogeneous"),
        (r"Fixed $\beta$, heterogeneous", "ctrl", "fixed_heterogeneous"),
        ("Noise Aug", "main", "noise_aug"),
        ("MARS", "main", "mars"),
    ]
    rows = []
    for label, src, key in spec:
        cells = []
        for s in ("SHD-FC", "SHD-RSNN"):
            entries = (main if src == "main" else ctrl)[s]
            if (key, SEEDS[0]) not in entries:
                cells += ["--", "--"]
                continue
            cells.append(cell([entries[key, sd]["nominal_acc"] for sd in SEEDS]))
            cells.append(cell(drops(entries, key, "per_neuron")))
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path, default=None, help="LaTeX source to compare against (row-by-row)")
    args = ap.parse_args()
    out = OUTPUT_ROOT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tables = {"table1": table1(), "table2": table2(), "table3": table3()}
    for name, rows in tables.items():
        (out / f"{name}_rows.tex").write_text("\n".join(rows) + "\n")
        print(f"% ---- {name} ({out / (name + '_rows.tex')})")
        print("\n".join(rows))
    if args.check is not None:
        tex = {" ".join(line.split()) for line in args.check.read_text().splitlines()}
        n_bad = 0
        for name, rows in tables.items():
            for r in rows:
                if " ".join(r.split()) not in tex:
                    n_bad += 1
                    print(f"MISMATCH {name}: {r}")
        n = sum(len(r) for r in tables.values())
        print(f"check: {n - n_bad}/{n} rows found verbatim in {args.check}")
        sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
