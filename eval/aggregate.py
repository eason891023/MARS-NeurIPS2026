"""Merge per-setting partial results -> outputs/eval/unified_beta.json + outputs/eval/UNIFIED_RESULTS.md.

Inputs: results/partial/<setting>.json (main 10-draw sweep), results/partial/SHD-*_mc200.json,
results/training/<setting>.json (training-time evaluator) and results/analysis/joint_noise_*.json
(earlier 10-draw per-neuron evaluation). Set MARS_RESULTS_ROOT to aggregate your own evaluation.

Statistics (training seed is the inference unit, n=5):
  * per checkpoint: accuracy averaged over MC draws first; drop = nominal - mean.
  * across seeds: mean, SD (ddof=1), 95% percentile bootstrap CI of the mean
    drop (10,000 resamples, fixed RNG).
  * paired differences MARS-NoiseAug and MARS-Clean on the per-seed drop:
    mean, bootstrap CI (10,000), exact two-sided sign-flip p over all 2^5
    sign patterns (minimum attainable p = 2/32 = 0.0625).
"""

from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402

PARTIAL = RESULTS / "partial"
OUT = OUTPUT_ROOT / "eval"
SETTINGS = ["SHD-FC", "SHD-RSNN", "SSC", "DVS-Gesture", "CIFAR-10"]
METHODS = ["clean", "noise_aug", "mars"]
LABEL = {"clean": "Clean", "noise_aug": "NoiseAug", "mars": "MARS"}
SEEDS = [42, 52, 72, 92, 102]
N_BOOT = 10_000

ORIG = {
    "SHD-FC": [RESULTS / "training/SHD-FC.json"],
    "SHD-RSNN": [RESULTS / "training/SHD-RSNN.json"],
    "DVS-Gesture": [RESULTS / "training/DVS-Gesture.json"],
    "CIFAR-10": [RESULTS / "training/CIFAR-10.json"],
    "SSC": [RESULTS / "training/SSC.json", RESULTS / "training/SSC-clean.json"],
}
REBUTTAL = {
    "SHD-FC": RESULTS / "analysis/joint_noise_ff.json",
    "SHD-RSNN": RESULTS / "analysis/joint_noise_rsnn.json",
}
CODE = {"clean": "clean", "noise_aug": "noise_aug", "mars": "asar"}


def _rel(path):
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sign_flip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    flips = [abs((d * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean([f >= obs - 1e-12 for f in flips]))


def describe(x, rng):
    x = np.asarray(x, float)
    return {"values": x.tolist(), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ci95": boot_ci(x, rng)}


def paired(a, b, rng):
    d = np.asarray(a, float) - np.asarray(b, float)
    out = describe(d, rng)
    out["p_signflip"] = sign_flip_p(d)
    out["n_mars_better"] = int((d < 0).sum())  # lower drop is better
    return out


def original_drops(setting):
    per = {}
    for path in ORIG[setting]:
        d = json.loads(path.read_text())
        per.update(d.get("per_seed", d))
    out = {}
    for m in METHODS:
        out[m] = {}
        for key in ("robustness", "robustness_layerwise", "robustness_global"):
            vals = []
            for s in SEEDS:
                r = per[f"s{s}_{CODE[m]}"]
                vals.append(r["robustness"]["0.0"]["mean"] - r[key]["0.2"]["mean"])
            out[m][key] = vals
    return out


def rebuttal_drops(setting):
    if setting not in REBUTTAL:
        return None
    d = json.loads(REBUTTAL[setting].read_text())["per_checkpoint_summary"]
    arch = "ff" if setting == "SHD-FC" else "rsnn"
    return {
        m: [d[f"{arch}|{CODE[m]}|{s}|nominal"]["mean"] - d[f"{arch}|{CODE[m]}|{s}|beta20"]["mean"] for s in SEEDS]
        for m in METHODS
    }


def main():
    rng = np.random.default_rng(20261004)
    unified = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "formula": "beta_tilde = clip(beta * (1 + CV * z), 0.01, 0.99), z ~ N(0,1)",
            "modes": {
                "per_neuron": "beta expanded per neuron (per channel for conv LIF layers); independent z per neuron/channel per layer",
                "layer_shared": "one z per LIF layer shared by its neurons (== original submitted 'independent' protocol, since all trained betas are per-layer scalars)",
                "global": "one z shared by all LIF layers",
            },
            "only_beta_perturbed": True,
            "ssc_delays_perturbed": False,
            "mc_draws_per_checkpoint_per_cv": None,
            "pairing": "z depends on (setting, training seed, mode, draw) only: identical draws across the 3 methods; same z rescaled across CVs (common random numbers)",
            "test_set": "full official test set",
            "methods": {"clean": "clean", "noise_aug": "noise_aug", "mars": "asar checkpoints"},
        },
        "settings": {},
        "summary": {},
    }
    for setting in SETTINGS:
        p = PARTIAL / f"{setting}.json"
        if not p.exists():
            continue
        part = json.loads(p.read_text())
        unified["protocol"]["mc_draws_per_checkpoint_per_cv"] = part["draws"]
        S = unified["settings"][setting] = {"n_test": part["n_test"], "methods": {}}
        complete = True
        for m in METHODS:
            S["methods"][m] = {}
            for s in SEEDS:
                e = part["entries"].get(f"{m}|{s}")
                if not e or not e.get("complete"):
                    complete = False
                    continue
                S["methods"][m][str(s)] = {
                    "checkpoint": _rel(e["checkpoint"]),
                    "checkpoint_test_acc": e["checkpoint_test_acc"],
                    "nominal_acc": e["nominal_acc"],
                    "restored_acc": e["restored_acc"],
                    "beta_layers": e["beta_layers"],
                    "modes": {
                        mode: {cv: {"acc": r["mean"], "sd_over_draws": r["sd_over_draws"], "runs": r["runs"],
                                    "beta_relative_rms_mean": r["beta_relative_rms_mean"]}
                               for cv, r in res.items()}
                        for mode, res in e["results"].items()
                    },
                }
        S["complete"] = complete
        if not complete:
            continue
        summ = unified["summary"][setting] = {"nominal": {}, "modes": {}}
        for m in METHODS:
            summ["nominal"][m] = describe([S["methods"][m][str(s)]["nominal_acc"] for s in SEEDS], rng)
        mode_cvs = {"per_neuron": part["per_neuron_cvs"], "layer_shared": part["struct_cvs"], "global": part["struct_cvs"]}
        for mode, cvs in mode_cvs.items():
            summ["modes"][mode] = {}
            for cv in cvs:
                cv = str(cv)
                block = {"acc": {}, "drop": {}, "paired_drop": {}}
                drops = {}
                for m in METHODS:
                    rows = [S["methods"][m][str(s)] for s in SEEDS]
                    acc = [r["modes"][mode][cv]["acc"] for r in rows]
                    drops[m] = [r["nominal_acc"] - a for r, a in zip(rows, acc)]
                    block["acc"][m] = describe(acc, rng)
                    block["drop"][m] = describe(drops[m], rng)
                block["paired_drop"]["mars_minus_noise_aug"] = paired(drops["mars"], drops["noise_aug"], rng)
                block["paired_drop"]["mars_minus_clean"] = paired(drops["mars"], drops["clean"], rng)
                summ["modes"][mode][cv] = block
        # comparisons with previously reported numbers
        od = original_drops(setting)
        summ["original_submitted"] = {
            m: {k: float(np.mean(v)) for k, v in od[m].items()} for m in METHODS
        }
        rb = rebuttal_drops(setting)
        if rb:
            summ["rebuttal_per_neuron_beta20"] = {m: float(np.mean(v)) for m, v in rb.items()}
    # supplemental MC-precision runs (200 draws at CV=0.2; draws 0-9 identical to the main run)
    unified["mc_precision_cv0.2"] = {}
    for setting in ("SHD-FC", "SHD-RSNN"):
        p = PARTIAL / f"{setting}_mc200.json"
        if not p.exists():
            continue
        part = json.loads(p.read_text())
        ent = part["entries"]
        if not all(ent.get(f"{m}|{s}", {}).get("complete") for m in METHODS for s in SEEDS):
            continue
        res = {"draws": part["draws"]}
        for mode in ("per_neuron", "layer_shared", "global"):
            res[mode] = {}
            for m in METHODS:
                drops, ses, drops10 = [], [], []
                for s in SEEDS:
                    e = ent[f"{m}|{s}"]
                    runs = np.asarray(e["results"][mode]["0.2"]["runs"])
                    drops.append(e["nominal_acc"] - runs.mean())
                    drops10.append(e["nominal_acc"] - runs[:10].mean())
                    ses.append(runs.std(ddof=1) / np.sqrt(len(runs)))
                # SE of the seed-mean drop due to MC sampling only, for 200 and for 10 draws
                se200 = float(np.sqrt(np.sum(np.square(ses))) / len(SEEDS))
                res[mode][m] = {"drop_mean": float(np.mean(drops)), "drop_per_seed": drops,
                                "drop_mean_first10": float(np.mean(drops10)),
                                "mc_se_200": se200, "mc_se_10": se200 * np.sqrt(part["draws"] / 10)}
        unified["mc_precision_cv0.2"][setting] = res
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "unified_beta.json"
    out.write_text(json.dumps(unified, indent=1))
    write_markdown(unified)
    print(f"wrote {out}")


def f(x, nd=2):
    return f"{x:.{nd}f}"


def fmt_drop(d):
    return f"{f(d['mean'])} ± {f(d['sd'])} [{f(d['ci95'][0])}, {f(d['ci95'][1])}]"


def fmt_paired(d):
    return f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}], p={d['p_signflip']:.4f}, {d['n_mars_better']}/5"


def write_markdown(U):
    L = []
    L.append("# Unified β-mismatch results (camera-ready)\n")
    L.append(f"Generated {U['generated']} by `eval/aggregate.py` from `results/partial/*.json`. Machine-readable: `outputs/eval/unified_beta.json`.\n")
    P = U["protocol"]
    L.append("**Protocol.** " + P["formula"] + f"; {P['mc_draws_per_checkpoint_per_cv']} MC draws per checkpoint per CV; full test set; only β perturbed (weights, thresholds, SSC delays untouched). "
             "Modes: per_neuron = independent z per neuron (per channel for conv LIF layers); layer_shared = one z per LIF layer; global = one z for all layers. "
             "Draws are paired across Clean/NoiseAug/MARS (same z for a given training seed and draw) and reused across CVs (common random numbers). "
             "MARS = `asar` checkpoints. Drop = nominal − perturbed accuracy (pp; lower is better), MC-averaged per checkpoint, then mean ± SD over 5 training seeds with 95% bootstrap CI (10k resamples). "
             "Paired Δ = MARS drop − baseline drop (negative favours MARS): mean [95% bootstrap CI], exact two-sided sign-flip p (n=5, min p = 0.0625), #seeds where MARS drop is lower.\n")
    S = U["summary"]
    done = [s for s in SETTINGS if s in S]
    missing = [s for s in SETTINGS if s not in S]
    if missing:
        L.append(f"**Incomplete settings (not tabulated):** {', '.join(missing)}\n")

    L.append("## Table 1. Nominal accuracy (%), mean ± SD over 5 seeds\n")
    L.append("| Setting | Clean | NoiseAug | MARS |\n|---|---:|---:|---:|")
    for s in done:
        L.append(f"| {s} | " + " | ".join(f"{f(S[s]['nominal'][m]['mean'])} ± {f(S[s]['nominal'][m]['sd'])}" for m in METHODS) + " |")
    L.append("")

    L.append("## Table 2. Drop at CV=0.2 (pp): mean ± SD [95% CI], three perturbation modes\n")
    for mode in ["per_neuron", "layer_shared", "global"]:
        L.append(f"### {mode}\n")
        L.append("| Setting | Clean | NoiseAug | MARS |\n|---|---:|---:|---:|")
        for s in done:
            b = S[s]["modes"][mode]["0.2"]["drop"]
            L.append(f"| {s} | " + " | ".join(fmt_drop(b[m]) for m in METHODS) + " |")
        L.append("")

    L.append("## Table 3. Paired drop differences at CV=0.2 (pp)\n")
    L.append("| Setting | Mode | MARS − NoiseAug | MARS − Clean |\n|---|---|---|---|")
    for s in done:
        for mode in ["per_neuron", "layer_shared", "global"]:
            pd = S[s]["modes"][mode]["0.2"]["paired_drop"]
            L.append(f"| {s} | {mode} | {fmt_paired(pd['mars_minus_noise_aug'])} | {fmt_paired(pd['mars_minus_clean'])} |")
    L.append("")

    L.append("## Table 4. Per-neuron CV sweep: accuracy (%) mean ± SD over seeds (CV=0 is nominal)\n")
    for s in done:
        cvs = list(S[s]["modes"]["per_neuron"].keys())
        L.append(f"### {s}\n")
        L.append("| Method | 0.0 | " + " | ".join(cvs) + " |\n|---|" + "---:|" * (len(cvs) + 1))
        for m in METHODS:
            row = [f"{f(S[s]['nominal'][m]['mean'])} ± {f(S[s]['nominal'][m]['sd'])}"]
            row += [f"{f(S[s]['modes']['per_neuron'][cv]['acc'][m]['mean'])} ± {f(S[s]['modes']['per_neuron'][cv]['acc'][m]['sd'])}" for cv in cvs]
            L.append(f"| {LABEL[m]} | " + " | ".join(row) + " |")
        L.append("")
        L.append("Drop (pp) mean [95% CI] and paired MARS − NoiseAug:\n")
        L.append("| CV | Clean | NoiseAug | MARS | MARS − NoiseAug |\n|---|---:|---:|---:|---|")
        for cv in cvs:
            b = S[s]["modes"]["per_neuron"][cv]
            L.append(f"| {cv} | " + " | ".join(f"{f(b['drop'][m]['mean'])} [{f(b['drop'][m]['ci95'][0])}, {f(b['drop'][m]['ci95'][1])}]" for m in METHODS)
                     + f" | {fmt_paired(b['paired_drop']['mars_minus_noise_aug'])} |")
        L.append("")

    L.append("## Table 5. Structured drift at CV=0.1 (pp): mean ± SD [95% CI]\n")
    L.append("| Setting | Mode | Clean | NoiseAug | MARS | MARS − NoiseAug |\n|---|---|---:|---:|---:|---|")
    for s in done:
        for mode in ["layer_shared", "global"]:
            b = S[s]["modes"][mode]["0.1"]
            L.append(f"| {s} | {mode} | " + " | ".join(fmt_drop(b["drop"][m]) for m in METHODS)
                     + f" | {fmt_paired(b['paired_drop']['mars_minus_noise_aug'])} |")
    L.append("")

    L.append("## Table 6. Sanity check vs previously reported drops at CV=0.2 (pp, mean over seeds)\n")
    L.append("Original submitted keys: `robustness` (paper Fig. 2 / Table 'Full main-result numbers', labelled 'independent' but layer-shared in effect because β is a per-layer scalar), "
             "`robustness_layerwise` (Fig. 3 middle) and `robustness_global` (Fig. 3 right). Earlier = per-neuron β20 evaluation (results/analysis/joint_noise_*.json, 10 draws).\n")
    L.append("| Setting | Method | new per_neuron | earlier per-neuron (10 draws) | new layer_shared | orig 'independent' | orig layerwise | new global | orig global |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in done:
        o = S[s]["original_submitted"]
        rb = S[s].get("rebuttal_per_neuron_beta20")
        for m in METHODS:
            md = S[s]["modes"]
            L.append(f"| {s} | {LABEL[m]} | {f(md['per_neuron']['0.2']['drop'][m]['mean'])} | {f(rb[m]) if rb else '—'} | "
                     f"{f(md['layer_shared']['0.2']['drop'][m]['mean'])} | {f(o[m]['robustness'])} | {f(o[m]['robustness_layerwise'])} | "
                     f"{f(md['global']['0.2']['drop'][m]['mean'])} | {f(o[m]['robustness_global'])} |")
    L.append("")
    MC = U.get("mc_precision_cv0.2", {})
    if MC:
        L.append("## Table 7. MC precision check at CV=0.2 (SHD; 200 draws per checkpoint)\n")
        L.append("Drop (pp) averaged over 5 seeds. `±` = MC-sampling SE of the 5-seed mean (from within-checkpoint draw SD). "
                 "The first 10 of the 200 draws are exactly the main-run draws.\n")
        L.append("| Setting | Mode | Method | 200-draw drop ± MC SE | main run (10 draws) ± MC SE(10) |\n|---|---|---|---:|---:|")
        for s, res in MC.items():
            for mode in ("per_neuron", "layer_shared", "global"):
                for m in METHODS:
                    r = res[mode][m]
                    L.append(f"| {s} | {mode} | {LABEL[m]} | {f(r['drop_mean'])} ± {f(r['mc_se_200'])} | {f(r['drop_mean_first10'])} ± {f(r['mc_se_10'])} |")
        L.append("")
    (OUT / "UNIFIED_RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
