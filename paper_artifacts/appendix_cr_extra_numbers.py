"""Print the appendix numbers that are not produced by the table scripts.

Read-only: prints values derived from the shipped result files (no training, no GPU).

    python3 paper_artifacts/appendix_cr_extra_numbers.py

Sections: lambda sweep (tab:app_lambda), learned beta / firing rates / reset tolerance
(tab:app_beta_activity, tab:app_memory Reset@k), linear trace memory R^2_k (tab:app_memory),
SHD temporal controls incl. rate-repeat (tab:app_temporal), low-bit PTQ (tab:app_ptq (b)),
typical-case diagnostics (tab:stability-both, (B3)), seed-42 weight norms, Psi suppression of Fig. 3a,
absolute step time / peak memory (tab:app_cost) and end-to-end SHD-FC epoch time.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_robust_tau.paths import RESULTS  # noqa: E402

SEEDS = [42, 52, 72, 92, 102]
MODES = ["clean", "noise_aug", "asar"]  # asar = MARS


def J(rel):
    return json.loads((RESULTS / rel).read_text())


def ms(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f}+-{x.std(ddof=1):.3f}"


def lambda_sweep():
    """SHD-FC MARS lambda sweep (layer-scalar evaluator, own training config, no augmentation)."""
    d = J("lambda_sweep/shd_fc_lambda_sweep.json")
    pr = d["per_run"]
    print("== lambda sweep (SHD-FC, seeds", d["sweep_cfg"]["seeds"], ")")
    for lam in d["sweep_cfg"]["methods"]["asar"]:
        ks = sorted(k for k in pr if k.startswith(f"asar_lam{lam}_s"))
        nom = np.array([pr[k]["test_acc"] for k in ks])
        dep = np.array([pr[k]["robustness"]["0.2"]["mean"] for k in ks])
        print(f"lam={lam:g} n={len(ks)} nominal {ms(nom)} cv0.2 {ms(dep)} drop {ms(nom - dep)} "
              f"per-seed drops {np.round(nom - dep, 2).tolist()}")


def state_activity():
    """Learned beta, firing rates, reset-interval accuracy."""
    rec = J("analysis/state_activity.json")["records"]
    for arch in ["ff", "rsnn"]:
        for mode in MODES:
            xs = sorted((r for r in rec if r["architecture"] == arch and r["mode"] == mode), key=lambda r: r["seed"])
            layers = list(xs[0]["effective_tau"].keys())
            print(f"-- {arch} {mode} seeds {[r['seed'] for r in xs]}")
            for l in layers:
                b = np.array([r["effective_tau"][l]["beta_mean"] for r in xs])
                fr = [r["activity"]["mean_firing_rate"][l] for r in xs]
                print(f"   {l}: beta mean {b.mean():.3f} [{b.min():.3f},{b.max():.3f}] firing {ms(fr)}")
            for k in ["1", "5", "10", "25", "50"]:
                print(f"   reset every {k} bins: acc {ms([r['reset_interval_accuracy'][k] for r in xs])}")


def memory_r2():
    """Linear decodability R^2_k of the raw event count k bins in the past (tab:app_memory).

    Printed with 3 decimals; the paper table rounds these 3-decimal values to 2 decimals.
    """
    print("== trace memory R^2_k (mean+-std over seeds)")
    probe = J("analysis/memory_capacity.json")["probe"]
    for arch in ["ff", "rsnn"]:
        for mode in MODES:
            row = {k: f"{probe[arch][mode][k]['r2']['mean']:.3f}+-{probe[arch][mode][k]['r2']['std']:.3f}"
                   for k in ["1", "5", "10", "20"]}
            print(arch, mode, row)


def temporal_controls():
    """Original/reverse/shuffle/rate-repeat accuracies on SHD."""
    print("== SHD temporal controls")
    for arch in ["ff", "rsnn"]:
        pc = J(f"analysis/temporal_information_{arch}.json")["per_checkpoint_summary"]
        for mode in MODES:
            out = {c: ms([pc[f"{arch}|{mode}|{s}|{c}"]["mean"] for s in SEEDS])
                   for c in ["original", "reverse_time", "shuffle_time", "rate_repeat"]}
            print(arch, mode, out)


def ptq_low_bit():
    """Nominal accuracy under 4/3-bit weight PTQ (no beta mismatch), main checkpoints."""
    print("== low-bit PTQ (no beta mismatch)")
    for name in ["SHD-FC", "SHD-RSNN"]:
        ps = J(f"training/{name}.json")["per_seed"]
        for mode in MODES:
            ks = [f"s{s}_{mode}" for s in SEEDS]
            q = {b: np.array([ps[k]["quantization"][b] for k in ks]) for b in ["32", "8", "6", "4", "3"]}
            print(name, mode, {b: ms(v) for b, v in q.items()}, "drop4", ms(q["32"] - q["4"]))


def stability_table():
    """Worst-case lambda_r vs typical-case lambda~ on SHD-FC seed 42 (tab:stability-both) and (B3) coupling."""
    print("== typical-case diagnostics, SHD-FC seed 42 (snnTorch ATan, L_sigma = 1)")
    d = J("diagnostics/results_fixed_shd_fc.json")
    for mode in ["clean", "asar", "hajr", "noise_aug"]:
        for r in d[mode]:
            if r["layer"] not in ("lif1", "lif2"):
                continue
            print(f"{mode:9s} {r['layer']}: |W|op {r['W_op_norm']:.1f} w_eff {r['W_eff_norm']:.2f} kappa {r['kappa']:.0f} "
                  f"eta {r['eta']:.3f} alpha_eff {r['alpha_bar_p99']:.3f} alpha~ {r['alpha_tilde']:.3f} "
                  f"lambda_r {r['lambda_r']:.1f} lambda~ {r['lambda_tilde']:.2f} (lambda_r per-element sup {r['lambda_r_elem_max']:.1f})")
    t = J("diagnostics/typical_case_shd_fc.json")["fixed"]
    ct = [r["B3_c_t_mean"] for m in t for r in t[m] if r["layer"] in ("lif1", "lif2")]
    cp = [r["B3_c_t_perm_mean"] for m in t for r in t[m] if r["layer"] in ("lif1", "lif2")]
    eta = [r["eta"] for m in t for r in t[m] if r["layer"] in ("lif1", "lif2")]
    print(f"(B3) c_t mean range [{min(ct):.2f}, {max(ct):.2f}], permuted baseline [{min(cp):.2f}, {max(cp):.2f}], "
          f"eta range [{min(eta):.3f}, {max(eta):.3f}]")


def weight_norms():
    """Seed-42 SHD-FC weight operator norms (diagnostics rerun)."""
    d = J("diagnostics/stability_verification_fixed.json")
    print("== SHD-FC seed-42 ||W||_op")
    for k in ["SHD-FC/clean", "SHD-FC/noise_aug", "SHD-FC/asar"]:
        print(k, [(x["layer"], round(x["W_ff_norm"], 2)) for x in d[k]])


def psi_suppression():
    """Fig. 3a: mean |Psi| Clean / MARS per hidden layer (seed-42 SHD-FC)."""
    d = J("diagnostics/heatmap_summary.json")
    print("== Psi suppression (Clean mean |Psi| / MARS mean |Psi|):",
          {l: round(v["ratio"], 2) for l, v in d.items()})


def overhead_abs():
    print("== absolute step time / total peak memory")
    rec = J("analysis/overhead.json")["records"]
    rec = [r for r in rec if r["setting"] != "dvs_gesture"]
    rec += J("analysis/overhead_real_dvs.json")["records"]
    for r in rec:
        print(r["setting"], r["mode"], r["batch_size"], r["time_steps"],
              f"{1e3 * r['seconds_per_step_mean']:.2f}+-{1e3 * r['seconds_per_step_std']:.2f} ms",
              f"{r['absolute_peak_gib_max']:.4f} GiB")
    ep = J("analysis/training_overhead_v2.json")
    print(f"== end-to-end SHD-FC epoch: Clean {ep['clean']['mean_s']:.1f} s, MARS {ep['asar']['mean_s']:.1f} s, "
          f"ratio {ep['asar']['mean_s'] / ep['clean']['mean_s']:.2f}x")


if __name__ == "__main__":
    lambda_sweep()
    state_activity()
    memory_r2()
    temporal_controls()
    ptq_low_bit()
    stability_table()
    weight_norms()
    psi_suppression()
    overhead_abs()
