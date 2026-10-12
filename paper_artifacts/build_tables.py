"""Build LaTeX table snippets + a markdown digest from the shipped analysis results (no training, CPU only).

All mean +- SD are across independent training seeds (n=5, ddof=1) unless stated; MC mismatch draws are
averaged within a checkpoint first.  Every table also appends a markdown block to digest_auto.md
with the exact numbers and the source files they were taken from.

Inputs : results/analysis/*.json
Outputs: outputs/tables/{tab_cost,tab_temporal,tab_app_joint,tab_app_distribution,
         tab_app_mechanism}.tex and outputs/tables/digest_auto.md
Usage  : python3 paper_artifacts/build_tables.py
Paper sources: tab_app_distribution (top block = appendix table tab:app_distribution); tab_app_joint SSC
rows (SSC joint beta+weight+threshold+delay drops); tab_temporal (horizons, original/reversed/shuffled);
tab_cost (ratios); digest (Psi fold reductions 4.67x / 78.2x / 16.7x). Table 3 (tab:baselines) is
produced by make_main_tables.py (200 draws).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS  # noqa: E402

SEEDS = [42, 52, 72, 92, 102]
A = RESULTS / "analysis"
OUT = OUTPUT_ROOT / "tables"


def J(p):
    return json.loads(Path(p).read_text())


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1))


def pm(v, d=2, bold=False):
    m, s = ms(v)
    t = f"{m:.{d}f}$\\pm${s:.{d}f}"
    return f"\\textbf{{{t}}}" if bold else t


def pm2(m, s, d=2):
    return f"{m:.{d}f}$\\pm${s:.{d}f}"


DIG = []   # digest lines


def dig(*lines):
    DIG.extend(l.replace("$\\pm$", "±") for l in lines)


def write_tex(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)
    print("wrote", name)


NAMES = {"clean": "Clean", "noise_aug": "NoiseAug", "asar": "MARS"}

# ---------------------------------------------------------------- loaders
JN = {a: J(A / f"joint_noise_{a}.json")["per_checkpoint_summary"] for a in ("ff", "rsnn")}
JN_SRC = {a: rel(A / f"joint_noise_{a}.json") for a in ("ff", "rsnn")}


def jn(arch, mode, cond):
    return np.array([JN[arch][f"{arch}|{mode}|{s}|{cond}"]["mean"] for s in SEEDS])


ARCH_NAME = {"ff": "SHD-FC", "rsnn": "SHD-RSNN"}


# ================================================================= tab_cost
def tab_cost():
    o = J(A / "overhead.json")["records"]
    dvs = J(A / "overhead_real_dvs.json")["records"]
    ep = J(A / "training_overhead_v2.json")
    recs = {(r["setting"], r["mode"]): r for r in o if r["setting"] != "dvs_gesture"}
    recs.update({(r["setting"], r["mode"]): r for r in dvs})
    sett = [("shd_ff", "SHD-FC", "64/50"), ("shd_rsnn", "SHD-RSNN", "64/50"), ("ssc_delrec", "SSC (DelRec)", "128/250"),
            ("dvs_gesture", "DVS-Gesture", "8/16"), ("cifar10", "CIFAR-10", None)]
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Training cost relative to Clean (model-side optimization step on one RTX 5090: forward, regularizer or noise augmentation, loss, backward, gradient clipping, Adam update; data loading excluded; 3 warm-up + 10 timed steps). "
         r"Memory is the \emph{total} peak allocated GPU memory. MARS adds no inference cost. "
         r"The end-to-end SHD-FC epoch wall-clock (standard multi-worker loader and triple augmentation) is $58.3\to80.1$\,s, i.e.\ $1.37\times$, "
         r"and should not be confused with the model-side step ratios.}",
         r"\label{tab:cost}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Setting & Batch/$T$ & Step: NoiseAug & Step: MARS & Peak mem.: NoiseAug & Peak mem.: MARS \\", r"\midrule"]
    dg = ["", "## tab_cost.tex",
          "Protocol: RTX 5090, real batch from the submitted training loader, 3 warm-up + 10 measured steps, data loading excluded. Ratios are MARS/Clean (and NoiseAug/Clean) of seconds_per_step_mean and of absolute_peak_gib_max (TOTAL peak allocated, not incremental). "
          f"Sources: `{rel(A/'overhead.json')}` for SHD-FC, SHD-RSNN, SSC, CIFAR-10; DVS-Gesture row from the real-data rerun `{rel(A/'overhead_real_dvs.json')}` (the five-setting file's DVS row used a shape-matched fallback batch and is NOT used). "
          f"Single measurement run per setting (no seed replication); timing SD across the 10 steps is in the json. End-to-end SHD-FC epoch (3 epochs each): Clean {ep['clean']['mean_s']:.1f} s, MARS {ep['asar']['mean_s']:.1f} s, ratio {ep['asar']['mean_s'] / ep['clean']['mean_s']:.3f}x (source: `{rel(A/'training_overhead_v2.json')}`).",
          "", "| Setting | Clean s/step | NoiseAug s/step | MARS s/step | NoiseAug/Clean | MARS/Clean | Clean peak GiB | NoiseAug peak GiB | MARS peak GiB | NoiseAug mem ratio | MARS mem ratio | incremental MARS/Clean (NOT used) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tr, mr = [], []
    for key, name, bt in sett:
        c, n, a = recs[(key, "clean")], recs[(key, "noise_aug")], recs[(key, "asar")]
        bt = bt or f"{c['batch_size']}/{c['time_steps']}"
        t_n, t_a = n["seconds_per_step_mean"] / c["seconds_per_step_mean"], a["seconds_per_step_mean"] / c["seconds_per_step_mean"]
        m_n, m_a = n["absolute_peak_gib_max"] / c["absolute_peak_gib_max"], a["absolute_peak_gib_max"] / c["absolute_peak_gib_max"]
        inc = a["incremental_peak_gib_mean"] / c["incremental_peak_gib_mean"]
        tr.append(t_a); mr.append(m_a)
        L.append(f"{name} & {bt} & {t_n:.2f}$\\times$ & {t_a:.2f}$\\times$ & {m_n:.2f}$\\times$ & {m_a:.2f}$\\times$ \\\\")
        dg.append(f"| {name} | {c['seconds_per_step_mean']:.5f} | {n['seconds_per_step_mean']:.5f} | {a['seconds_per_step_mean']:.5f} | {t_n:.3f} | {t_a:.3f} | {c['absolute_peak_gib_max']:.4f} | {n['absolute_peak_gib_max']:.4f} | {a['absolute_peak_gib_max']:.4f} | {m_n:.3f} | {m_a:.3f} | {inc:.2f} |")
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    dg.append(f"| geometric mean | | | | | {gm(tr):.3f} | | | | | {gm(mr):.3f} | |")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_cost.tex", "\n".join(L) + "\n")
    dig(*dg)
    return tr, mr


# ================================================================= tab_temporal
def tab_temporal():
    mc = J(A / "memory_capacity.json")
    ssc = J(A / "ssc_temporal.json")
    ti = {a: J(A / f"temporal_information_{a}.json")["per_checkpoint_summary"] for a in ("ff", "rsnn")}
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Temporal expressivity. Task-context horizon: smallest retained prefix/suffix of the input (in time bins) that reaches 95\% of full-input accuracy. "
         r"$R^2_{20}$: linear read-out of the raw event count 20 bins in the past from frozen hidden states (generic trace memory; a lower value is not evidence of less task-relevant temporal processing). "
         r"Accuracy (\%) on original, time-reversed and fully time-shuffled test inputs. Mean$\pm$SD over 5 training seeds.}",
         r"\label{tab:temporal}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Setting & Method & Prefix & Suffix & $R^2_{20}$ & Original & Reversed & Shuffled \\", r"\midrule"]
    dg = ["", "## tab_temporal.tex",
          f"SHD-FC/RSNN horizon (T=50 bins) and lag-20 R2: `{rel(A/'memory_capacity.json')}` (context[arch][mode][prefix|suffix].horizon95_bins mean/std; probe[arch][mode]['20'].r2 mean/std). "
          f"SHD original/reverse/shuffle: per-seed values from `{rel(A/'temporal_information_ff.json')}` and `..._rsnn.json` (per_checkpoint_summary conditions original, reverse_time, shuffle_time; shuffle = 10 draws). "
          f"SSC (T=250 bins): `{rel(A/'ssc_temporal.json')}` (summary accuracy_mean/std for original/reverse/shuffle; horizons_95pct_original). SSC has no raw-count-probe result, so R2 is '--'. "
          "Perturbation granularity: N/A (input-side transforms). SD = sample SD over 5 seeds.",
          "", "| Setting | Method | Prefix H | Suffix H | R2 lag20 | Original | Reverse | Shuffle |", "|---|---|---|---|---|---|---|---|"]
    for arch in ("ff", "rsnn"):
        for mi, mode in enumerate(("clean", "noise_aug", "asar")):
            ctx = mc["context"][arch][mode]
            pre, suf = ctx["prefix"]["horizon95_bins"], ctx["suffix"]["horizon95_bins"]
            r2 = mc["probe"][arch][mode]["20"]["r2"]
            sv = lambda c: np.array([ti[arch][f"{arch}|{mode}|{s}|{c}"]["mean"] for s in SEEDS])
            o_, r_, s_ = sv("original"), sv("reverse_time"), sv("shuffle_time")
            first = ARCH_NAME[arch] if mi == 0 else ""
            L.append(f"{first} & {NAMES[mode]} & {pm2(pre['mean'],pre['std'],1)} & {pm2(suf['mean'],suf['std'],1)} & {pm2(r2['mean'],r2['std'],2)} & {pm(o_,1)} & {pm(r_,1)} & {pm(s_,1)} \\\\")
            dg.append(f"| {ARCH_NAME[arch]} | {NAMES[mode]} | {pm2(pre['mean'],pre['std'],1)} | {pm2(suf['mean'],suf['std'],1)} | {pm2(r2['mean'],r2['std'],3)} | {pm2(*ms(o_),3)} | {pm2(*ms(r_),3)} | {pm2(*ms(s_),3)} |")
        L.append(r"\midrule")
    for mi, mode in enumerate(("clean", "noise_aug", "asar")):
        S = ssc["summary"]; H = ssc["horizons_95pct_original"]
        g = lambda c: (S[f"{mode}|{c}"]["accuracy_mean"], S[f"{mode}|{c}"]["accuracy_std"])
        first = "SSC" if mi == 0 else ""
        L.append(f"{first} & {NAMES[mode]} & {H[mode+'|prefix']['mean']:.0f} & {H[mode+'|suffix']['mean']:.0f} & -- & {pm2(*g('original'),1)} & {pm2(*g('reverse'),1)} & {pm2(*g('shuffle'),1)} \\\\")
        dg.append(f"| SSC | {NAMES[mode]} | {H[mode+'|prefix']['mean']:.0f}±{H[mode+'|prefix']['std']:.0f} | {H[mode+'|suffix']['mean']:.0f}±{H[mode+'|suffix']['std']:.0f} | -- | {pm2(*g('original'),3)} | {pm2(*g('reverse'),3)} | {pm2(*g('shuffle'),3)} |")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_temporal.tex", "\n".join(L) + "\n")
    dig(*dg)


# ================================================================= tab_app_joint
def tab_app_joint():
    cols = [("beta05", "$\\beta$5"), ("beta20", "$\\beta$20"), ("threshold05", "$\\vartheta$5"), ("joint05", "all5"), ("beta20_joint_aux05", "$\\beta$20+aux5")]
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Joint $\beta$/weight/threshold mismatch (per-neuron $\beta$ and threshold, per-weight weight noise; multiplicative Gaussian, CV in \%). "
         r"Entries: accuracy change from nominal (pp; mean$\pm$SD over 5 seeds). ``all5'' = 5\% on $\beta$, threshold and weights; ``$\beta$20+aux5'' = 20\% $\beta$ with 5\% weight and threshold. "
         r"SSC adds a learned recurrent delay (joint-all $c$ = CV $c$\% on $\beta$, threshold, weights and delay; threshold is per-layer scalar there).}",
         r"\label{tab:app_joint}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Setting & Method & Nominal (\%) & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    dg = ["", "## tab_app_joint.tex",
          f"SHD-FC/RSNN: `{JN_SRC['ff']}`, `{JN_SRC['rsnn']}` (10 draws; per-neuron beta & threshold, per-weight weights). Values = per-seed (condition - nominal) averaged over 5 seeds ± sample SD. "
          f"SSC: `{rel(A/'ssc_joint.json')}` (5 draws; beta per neuron, threshold per-layer scalar, weights, delay).",
          "", "| Setting | Method | Nominal | " + " | ".join(c[0] for c in cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for arch in ("ff", "rsnn"):
        for mi, m in enumerate(("clean", "noise_aug", "asar")):
            nom = jn(arch, m, "nominal")
            ch = [jn(arch, m, c) - nom for c, _ in cols]
            first = ARCH_NAME[arch] if mi == 0 else ""
            L.append(f"{first} & {NAMES[m]} & {pm(nom,1)} & " + " & ".join(pm(c) for c in ch) + " \\\\")
            dg.append(f"| {ARCH_NAME[arch]} | {NAMES[m]} | {pm2(*ms(nom),3)} | " + " | ".join(pm2(*ms(c), 3) for c in ch) + " |")
        L.append(r"\midrule")
    # SSC
    SS = J(A / "ssc_joint.json")["per_checkpoint_summary"]
    sc = [("joint_all01", "all1"), ("joint_all02", "all2"), ("joint_all05", "all5"), ("beta20_aux05_delay05", "$\\beta$20+aux5+delay5")]
    L += [r"\midrule" if False else r"\multicolumn{8}{@{}l}{\emph{SSC with learned delay}} \\",
          r"Setting & Method & Nominal (\%) & " + " & ".join(c[1] for c in sc) + r" & \\"]
    dg += ["", "SSC block:", "", "| Method | Nominal | " + " | ".join(c[0] for c in sc) + " |", "|---|---|" + "---|" * len(sc)]
    for mi, m in enumerate(("clean", "noise_aug", "asar")):
        g = lambda c: np.array([SS[f"{m}|{s}|{c}"]["mean"] for s in SEEDS])
        nom = g("nominal")
        ch = [g(c) - nom for c, _ in sc]
        first = "SSC" if mi == 0 else ""
        L.append(f"{first} & {NAMES[m]} & {pm(nom,1)} & " + " & ".join(pm(c) for c in ch) + r" & \\")
        dg.append(f"| {NAMES[m]} | {pm2(*ms(nom),3)} | " + " | ".join(pm2(*ms(c), 3) for c in ch) + " |")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_app_joint.tex", "\n".join(L) + "\n")
    dig(*dg)


# ================================================================= tab_app_distribution
def tab_app_distribution():
    conds = [("gaussian_iid", "Gaussian"), ("lognormal_iid", "Log-normal"), ("student_t3_iid", "Student-$t_3$"),
             ("gaussian_rho05", "Corr. $\\rho$=0.5"), ("gaussian_rho09", "Corr. $\\rho$=0.9")]
    DS = {a: J(A / f"distribution_shift_{a}.json")["per_checkpoint_summary"] for a in ("ff", "rsnn")}
    L = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Non-Gaussian and correlated $\beta$ mismatch (CV 20\%, per-neuron; log-normal factors are mean-preserving before clipping, $t_3$ is standardized to unit variance; $\rho$ = equicorrelation across neurons). "
         r"Accuracy change from nominal (pp, mean$\pm$SD over 5 seeds).}",
         r"\label{tab:app_distribution}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Setting & Method & " + " & ".join(c[1] for c in conds) + r" \\", r"\midrule"]
    dg = ["", "## tab_app_distribution.tex",
          f"`{rel(A/'distribution_shift_ff.json')}`, `{rel(A/'distribution_shift_rsnn.json')}` (per-neuron beta, CV=0.2, 10 draws, MARS and NoiseAug only - no Clean arm was run). Change = per-seed (condition - nominal) where nominal = checkpoint test accuracy (from joint_noise nominal, same checkpoints).",
          "", "| Setting | Method | " + " | ".join(c[0] for c in conds) + " |", "|---|---|" + "---|" * len(conds)]
    for arch in ("ff", "rsnn"):
        for mi, m in enumerate(("noise_aug", "asar")):
            nom = jn(arch, m, "nominal")
            ch = [np.array([DS[arch][f"{arch}|{m}|{s}|{c}"]["mean"] for s in SEEDS]) - nom for c, _ in conds]
            first = ARCH_NAME[arch] if mi == 0 else ""
            L.append(f"{first} & {NAMES[m]} & " + " & ".join(pm(c) for c in ch) + " \\\\")
            dg.append(f"| {ARCH_NAME[arch]} | {NAMES[m]} | " + " | ".join(pm2(*ms(c), 3) for c in ch) + " |")
        if arch == "ff":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_app_distribution.tex", "\n".join(L) + "\n")
    dig(*dg)


# ================================================================= tab_app_mechanism
def tab_app_mechanism():
    mp = A / "mechanism_triangulation.json"
    d = J(mp)
    names = [("SHD-FC", "SHD-FC"), ("SHD-RSNN", "SHD-RSNN"), ("SSC", "SSC"), ("DVS", "DVS-Gesture"), ("CIFAR10", "CIFAR-10")]

    def sci(x):
        if x >= 1e4:
            e = int(np.floor(np.log10(x)))
            return f"${x/10**e:.2f}\\times10^{{{e}}}$"
        return f"{x:.4g}" if x < 100 else f"{x:.0f}"
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Mechanism triangulation on the submitted checkpoints (5 seeds). $\Psi$ = post-hoc time-averaged sensitivity $\|\partial V/\partial\beta\|$ energy (geometric mean over seeds); "
         r"drop = accuracy drop (pp) at layer-scalar $\beta$ CV 20\% (mean$\pm$SD). "
         r"Pooled over the 15 dataset-seed pairs of SHD-FC, SHD-RSNN and SSC: MARS lowers $\Psi$ by $78.2\times$ relative to Clean and NoiseAug by $4.67\times$; "
         r"after removing dataset offsets, $\log\Psi$ correlates with the drop (Pearson $r=0.730$, Spearman $\rho=0.781$, $n=45$; $r=0.787$ using Clean and NoiseAug only, $n=30$). "
         r"On the low-sensitivity DVS/CIFAR-10 controls all drops are small, so $\Psi$ is not claimed as a task-agnostic predictor.}",
         r"\label{tab:app_mechanism}", r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
         r"Dataset & Method & $\Psi$ (geo.\ mean) & Drop @ CV 0.2 (pp) \\", r"\midrule"]
    dg = ["", "## tab_app_mechanism.tex",
          f"Source: `{rel(mp)}` (per_dataset[ds][mode].sensitivity_geometric_mean, drop_pp mean/std; paired_sensitive_task_effects; correlations), produced by analysis/mechanism_triangulation.py. "
          "Inputs: `results/mechanism/asar_vs_robustness.json` (SHD-FC, SHD-RSNN, DVS, CIFAR-10) and `results/mechanism/ssc_scatter_points.json` (SSC; max_batches=10, i.e. drop evaluated on a 10-batch test subset). "
          "**Granularity: submitted-protocol beta mismatch. Submitted beta tensors are per-layer scalars and the submitted evaluator draws randn_like(layer.beta), i.e. ONE draw per layer (layer-scalar) - NOT the per-neuron protocol of the camera-ready main text.** Therefore these drops are not comparable to per-neuron tables (e.g. SHD-FC Clean drop 19.13 here vs 28.14 per-neuron). SD = sample SD over 5 seeds.",
          "", "| Dataset | Method | Psi geo-mean | Drop mean | Drop SD |", "|---|---|---|---|---|"]
    for key, nm in names:
        for mi, m in enumerate(("clean", "noise_aug", "asar")):
            e = d["per_dataset"][key][m]
            g = e["sensitivity_geometric_mean"]; dm, dsd = e["drop_pp"]["mean"], e["drop_pp"]["std"]
            L.append(f"{nm if mi==0 else ''} & {NAMES[m]} & {sci(g)} & {pm2(dm,dsd,2)} \\\\")
            dg.append(f"| {nm} | {NAMES[m]} | {g:.4g} | {dm:.3f} | {dsd:.3f} |")
        L.append(r"\midrule" if key != "CIFAR10" else "")
    L = [x for x in L if x != ""]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_app_mechanism.tex", "\n".join(L) + "\n")
    pe = d["paired_sensitive_task_effects"]; mn = d["mars_vs_noise_aug"]; co = d["correlations"]
    dg += ["", "Pooled statistics quoted in the caption (all from the md/json): "
           f"MARS vs Clean Psi fold {pe['asar']['sensitivity_fold_reduction_geometric_mean']:.2f}x CI {pe['asar']['sensitivity_fold_reduction_95ci']}; drop improvement {pe['asar']['drop_improvement_pp_mean']:.2f} pp CI {pe['asar']['drop_improvement_pp_95ci']}; "
           f"NoiseAug vs Clean Psi fold {pe['noise_aug']['sensitivity_fold_reduction_geometric_mean']:.2f}x, drop improvement {pe['noise_aug']['drop_improvement_pp_mean']:.2f} pp; "
           f"MARS vs NoiseAug Psi fold {mn['mars_sensitivity_fold_reduction_geometric_mean']:.2f}x, drop advantage {mn['mars_drop_advantage_pp_mean']:.2f} pp CI {mn['mars_drop_advantage_pp_95ci']}. "
           "Pooled correlations of the submitted (layer-scalar, 10-draw) drops are in results.json['correlations']; the per-neuron correlations of the paper are computed by recompute_correlation.py."]
    dig(*dg)


if __name__ == "__main__":
    tr, mr = tab_cost()
    tab_temporal()
    tab_app_joint()
    tab_app_distribution()
    tab_app_mechanism()
    (OUT / "digest_auto.md").write_text("\n".join(DIG) + "\n")
    print("done; step ratios", np.round(tr, 3), "mem ratios", np.round(mr, 3))
