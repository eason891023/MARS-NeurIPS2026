"""
Typical-case / worst-case diagnostics on SHD-FC, seed 42 (paper Table tab:stability-both, Fig. 3b).

Fix: the original used sigma'(x) = alpha/(pi(1+(alpha x)^2)) and L_sigma = 2/pi,
which is NOT snnTorch's ATan derivative. snnTorch (and the training regularizer,
snn_robust_tau/loss/hajr.py) use sigma'(x) = (alpha/2)/(1+(pi alpha x/2)^2),
max alpha/2 = 1 for alpha = 2.

Everything else (checkpoints, test loader, n_batches=30, argument of sigma' =
current membrane V_t - theta, neuron-averaged |alpha_eff(t)| per step, p99 over
steps for the "worst-case" alpha) is kept identical to the original so that the
old->new difference isolates the derivative fix. The script evaluates both the
legacy and the fixed formula in the same pass; the fixed branch is compared with the
shipped results/diagnostics/results_fixed_shd_fc.json (sanity check).
Inputs : checkpoints/SHD-FC/best_s42_{clean,asar,hajr,noise_aug}.pth, SHD test set
Outputs: outputs/diagnostics/{typical_case_shd_fc.json, results_fixed_shd_fc.json}
Usage  : python3 diagnostics/eval_typical_case_stability_fixed.py

Extras (not in the original):
  * per-element |alpha_eff| statistics (max, p99, p99.9) and signed range,
    i.e. a genuine per-neuron worst case rather than p99 of the neuron mean;
  * (B3) coupling c_t = <|sigma_hat_t|,|Psi_t|>/(||sigma_hat_t|| ||Psi_t||) with a
    coordinate-permuted baseline, Psi_t from the training OnlineHAJR recursion.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from surrogate_atan import FORMULAS, PROJECT_ROOT, check_against_snntorch  # noqa: E402
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS, ckpt_dir  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = str(OUTPUT_ROOT / "diagnostics")
CKPT_DIR = str(ckpt_dir("SHD-FC"))
MODES = ["clean", "asar", "hajr", "noise_aug"]
N_BATCHES = 30
N_BATCHES_B3 = 10
ALPHA_SURR = 2.0


def compute_weight_norms(weight_matrix):
    with torch.no_grad():
        w = weight_matrix.view(weight_matrix.shape[0], -1) if weight_matrix.dim() > 2 else weight_matrix
        s = torch.linalg.svdvals(w)
        n_in = w.shape[1]
        op_norm = float(s[0].item())
        frob_norm = float(torch.norm(w, "fro").item())
        eff_norm = frob_norm / np.sqrt(n_in)
        return {"op_norm": op_norm, "frob_norm": frob_norm, "eff_norm": eff_norm,
                "n_in": int(n_in), "kappa": (op_norm / eff_norm) ** 2}


def _thr(layer):
    thr = getattr(layer, "threshold", torch.tensor(1.0))
    return thr.detach() if isinstance(thr, (nn.Parameter, torch.Tensor)) else torch.tensor(thr)


def collect(model, loader, n_batches=N_BATCHES):
    model.eval()
    hidden = ["lif1", "lif2"]
    acc = {f: {n: {"alpha_step": [], "surr_sq": [], "alpha_elem": [], "alpha_signed_min": [],
                   "alpha_signed_max": [], "c_t": [], "c_t_perm": []} for n in hidden}
           for f in FORMULAS}
    n_neurons = {}
    gen = torch.Generator(device="cpu").manual_seed(0)
    count = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(DEVICE).float()
            bsz, steps = data.shape[0], data.shape[1]
            x = data.view(bsz, steps, -1)
            mem1, mem2, mem3 = model.lif1.init_leaky(), model.lif2.init_leaky(), model.lif3.init_leaky()
            model.online_reg.reset()
            do_b3 = count < N_BATCHES_B3
            for t in range(steps):
                spk1, mem1 = model.lif1(model.fc1(x[:, t, :]), mem1)
                if do_b3:
                    model.online_reg.step_from_layer("lif1", model.lif1, mem1, timestep=t,
                                                     spike=spk1, decision_gate=False)
                spk2, mem2 = model.lif2(model.fc2(spk1), mem2)
                if do_b3:
                    model.online_reg.step_from_layer("lif2", model.lif2, mem2, timestep=t,
                                                     spike=spk2, decision_gate=False)
                spk3, mem3 = model.lif3(model.fc3(spk2), mem3)

                for name, layer, mem in (("lif1", model.lif1, mem1), ("lif2", model.lif2, mem2)):
                    n_neurons[name] = mem.shape[-1]
                    thr = _thr(layer).to(mem.device)
                    beta = layer.beta.detach()
                    psi = model.online_reg.states[name]["psi"].detach().view(bsz, -1) if do_b3 else None
                    for f, (deriv, _) in FORMULAS.items():
                        a = acc[f][name]
                        s = deriv(mem - thr, ALPHA_SURR)
                        a["surr_sq"].append((s ** 2).sum(dim=-1).mean().item())
                        alpha_signed = beta - thr * s
                        alpha_abs = alpha_signed.abs()
                        a["alpha_step"].append(alpha_abs.mean(dim=-1).mean().item())
                        a["alpha_elem"].append(alpha_abs.flatten().cpu().numpy().astype(np.float32))
                        a["alpha_signed_min"].append(alpha_signed.min().item())
                        a["alpha_signed_max"].append(alpha_signed.max().item())
                        if do_b3 and t > 0:
                            sa, pa = s.abs(), psi.abs()
                            denom = (sa.norm(dim=-1) * pa.norm(dim=-1)).clamp_min(1e-12)
                            a["c_t"].append(((sa * pa).sum(-1) / denom).mean().item())
                            perm = torch.randperm(sa.shape[-1], generator=gen).to(sa.device)
                            a["c_t_perm"].append(((sa[:, perm] * pa).sum(-1) / denom).mean().item())
            count += 1
            if count >= n_batches:
                break

    out = {}
    for f, (_, lsig_fn) in FORMULAS.items():
        L = lsig_fn(ALPHA_SURR)
        out[f] = {}
        for name in hidden:
            a = acc[f][name]
            n = n_neurons[name]
            av = np.array(a["alpha_step"])
            elem = np.concatenate(a["alpha_elem"])
            eta = float(np.mean(a["surr_sq"]) / (n * L ** 2))
            out[f][name] = {
                "n_neurons": n,
                "L_sigma": L,
                "alpha_tilde": float(av.mean()),
                "alpha_bar": float(av.max()),
                "alpha_p99": float(np.percentile(av, 99)),
                "alpha_elem_max": float(elem.max()),
                "alpha_elem_p99": float(np.percentile(elem, 99)),
                "alpha_elem_p999": float(np.percentile(elem, 99.9)),
                "alpha_signed_min": float(min(a["alpha_signed_min"])),
                "alpha_signed_max": float(max(a["alpha_signed_max"])),
                "eta": eta,
                "sqrt_eta": float(np.sqrt(eta)),
                "rms_sigma_hat": float(np.sqrt(np.mean(a["surr_sq"]) / n)),
                "B3_c_t_mean": float(np.mean(a["c_t"])) if a["c_t"] else None,
                "B3_c_t_perm_mean": float(np.mean(a["c_t_perm"])) if a["c_t_perm"] else None,
            }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("snnTorch check:", check_against_snntorch(ALPHA_SURR))
    spec = importlib.util.spec_from_file_location(
        "shd", os.path.join(PROJECT_ROOT, "train", "shd_fc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0
    from snntorch import surrogate
    surrogate_fn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))
    mod.seed_everything(42)
    _, _, test_loader, input_size = mod.build_loaders(42, cfg)

    results = {"fixed": {}, "legacy": {}}
    for mode in MODES:
        ckpt_path = os.path.join(CKPT_DIR, f"best_s42_{mode}.pth")
        model = mod.SHDSNN(input_size=input_size, beta_init=cfg["beta_init"],
                           threshold_init=cfg.get("threshold_init", 1.0), surrogate_fn=surrogate_fn,
                           learn_beta=True, learn_threshold=False).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        winfo = {fc: compute_weight_norms(getattr(model, fc).weight) for fc in ("fc1", "fc2", "fc3")}
        tc_all = collect(model, test_loader)
        for f in FORMULAS:
            rows = []
            for lif, fc in (("lif1", "fc1"), ("lif2", "fc2")):
                tc, wi = tc_all[f][lif], winfo[fc]
                L = tc["L_sigma"]
                lam_r = L * wi["op_norm"] / (1 - tc["alpha_p99"]) if tc["alpha_p99"] < 1 else float("inf")
                lam_t = (tc["sqrt_eta"] * L * wi["eff_norm"] / (1 - tc["alpha_tilde"])
                         if tc["alpha_tilde"] < 1 else float("inf"))
                # per-element worst case (true sup over observed elements)
                lam_r_elem = (L * wi["op_norm"] / (1 - tc["alpha_elem_max"])
                              if tc["alpha_elem_max"] < 1 else float("inf"))
                beta_val = float(getattr(model, lif).beta.detach().view(-1)[0].item())
                rows.append({"layer": lif, "reset": "subtract", "beta": beta_val,
                             "W_op_norm": wi["op_norm"], "W_eff_norm": wi["eff_norm"],
                             "W_frob_norm": wi["frob_norm"], "kappa": wi["kappa"],
                             **{k: v for k, v in tc.items()},
                             "alpha_bar_p99": tc["alpha_p99"],
                             "lambda_r": lam_r, "lambda_tilde": lam_t,
                             "lambda_r_elem_max": lam_r_elem,
                             "stable_worstcase": lam_r < 1, "stable_typicalcase": lam_t < 1})
            results[f][mode] = rows
            for r in rows:
                print(f"[{f:6s}] {mode:9s} {r['layer']} eta={r['eta']:.4f} a~={r['alpha_tilde']:.3f} "
                      f"a_p99={r['alpha_bar_p99']:.3f} a_elem_max={r['alpha_elem_max']:.3f} "
                      f"lam_r={r['lambda_r']:.1f} lam~={r['lambda_tilde']:.3f} "
                      f"c_t={r['B3_c_t_mean']:.3f}/{r['B3_c_t_perm_mean']:.3f}")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "typical_case_shd_fc.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    # figure input (read by paper_artifacts/make_figures_fixed.py)
    with open(os.path.join(OUT_DIR, "results_fixed_shd_fc.json"), "w") as fh:
        json.dump(results["fixed"], fh, indent=2, default=str)

    # sanity: fixed branch vs the shipped result file
    old = json.load(open(RESULTS / "diagnostics" / "results_fixed_shd_fc.json"))
    maxdiff = 0.0
    for mode in MODES:
        for r_old in old[mode]:
            if r_old.get("lambda_r") is None:
                continue
            r_new = next(r for r in results["fixed"][mode] if r["layer"] == r_old["layer"])
            for k in ("eta", "alpha_tilde", "alpha_bar_p99", "lambda_r", "lambda_tilde"):
                maxdiff = max(maxdiff, abs(r_new[k] - r_old[k]) / max(abs(r_old[k]), 1e-12))
    print(f"Max relative diff vs results/diagnostics/results_fixed_shd_fc.json: {maxdiff:.2e}")


if __name__ == "__main__":
    main()
