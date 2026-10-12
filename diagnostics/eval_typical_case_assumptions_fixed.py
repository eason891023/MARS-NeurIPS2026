"""
Corrected version of the original eval_typical_case_assumptions.py (not shipped; it used
the wrong ATan surrogate slope): the arctan branch uses the snnTorch ATan derivative
sigma'(x)=(a/2)/(1+(pi a x/2)^2), L_sigma=a/2.
The SSC (triangle surrogate) branch is unaffected; by default it is NOT rerun and
the SSC block is copied from results/diagnostics/assumptions_rsnn_ssc_fixed.json
(pass --with-ssc to rerun).
Inputs : checkpoints/SHD-RSNN/best_s42_*.pth (+ checkpoints/SSC/ with --with-ssc), test sets
Output : outputs/diagnostics/assumptions_rsnn_ssc_fixed.json

Auxiliary typical-case diagnostic checks for SHD-RSNN and SSC (DelRec).

Reports empirical checks related to the typical-case diagnostic:
  - approximate isotropy of perturbation directions
  - sub-Gaussian membrane-potential check (legacy auxiliary diagnostic)
  - surrogate sparsity eta << 1
  - temporal autocorrelation of perturbation directions

Also computes lambda_r (worst-case) and lambda_tilde (typical-case diagnostic).

Extends the SHD-FC validation to SHD-RSNN and SSC.
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

def _find_project_root(start_file: str) -> str:
    cur = os.path.abspath(os.path.dirname(start_file))
    while True:
        if os.path.isdir(os.path.join(cur, "snn_robust_tau")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot locate project root")
        cur = parent

PROJECT_ROOT = _find_project_root(__file__)
sys.path.insert(0, PROJECT_ROOT)
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS, ckpt_dir  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Surrogate derivative helpers ──

def surrogate_deriv_atan(v_minus_thr, alpha=2.0):
    """FIXED: snnTorch ATan derivative (alpha/2) / (1 + (pi*alpha*x/2)^2)"""
    return (alpha / 2.0) / (1 + (np.pi / 2.0 * alpha * v_minus_thr)**2)

def surrogate_deriv_triangle(v_minus_thr, gamma=1.0):
    """triangular surrogate derivative: max(0, gamma - |x|) / gamma^2"""
    return torch.clamp(gamma - v_minus_thr.abs(), min=0.0) / (gamma * gamma)

def get_L_sigma(surrogate_type, surrogate_alpha):
    """Max value of surrogate derivative."""
    if surrogate_type in ("atan",):
        return surrogate_alpha / 2.0  # FIXED (was alpha/pi)
    elif surrogate_type in ("triangle", "triangular"):
        return 1.0 / surrogate_alpha
    else:
        raise ValueError(f"Unknown surrogate: {surrogate_type}")


# ── (B1) Isotropy: pairwise cosine similarity of perturbation directions ──

def check_B1_isotropy(perturbation_dirs):
    """
    perturbation_dirs: list of tensors [n_neurons] (normalized directions per timestep).
    Returns mean pairwise cosine similarity (0 = isotropic).
    """
    dirs = torch.stack(perturbation_dirs)  # [T, N]
    norms = dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    dirs_normed = dirs / norms
    # Sample pairs to avoid O(T^2) for long sequences
    T = dirs_normed.shape[0]
    if T > 200:
        idx = torch.randperm(T)[:200]
        dirs_normed = dirs_normed[idx]
        T = 200
    cos_sim = dirs_normed @ dirs_normed.T  # [T, T]
    # Upper triangle only (exclude diagonal)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    pairwise = cos_sim[mask]
    return float(pairwise.mean().item()), float(pairwise.std().item())


# ── (B2) Sub-Gaussian: Shapiro-Wilk on per-neuron membrane potentials ──

def check_B2_subgaussian(mem_traces):
    """
    mem_traces: tensor [T, N] of membrane potentials across timesteps.
    Tests sub-Gaussian tail behavior: for each neuron, check if the
    distribution of V across time has light tails (Shapiro-Wilk p > 0.05).

    Uses individual samples (not batch-averaged) when available.
    Returns fraction of neurons passing.
    """
    if isinstance(mem_traces, list):
        mem_traces = torch.stack(mem_traces)
    T, N = mem_traces.shape
    if T < 8:
        return 1.0, N
    # Shapiro-Wilk max samples
    max_sw = min(T, 5000)
    if T > max_sw:
        idx = np.random.choice(T, max_sw, replace=False)
        mem_traces = mem_traces[idx]
    pass_count = 0
    for i in range(N):
        vals = mem_traces[:, i].cpu().numpy()
        if np.std(vals) < 1e-10:
            pass_count += 1
            continue
        _, p = stats.shapiro(vals)
        if p > 0.05:
            pass_count += 1
    return pass_count / N, N


# ── (B4) Temporal independence: autocorrelation decay ──

def check_B4_temporal_independence(perturbation_dirs, max_lag=10):
    """
    perturbation_dirs: list of tensors [n_neurons].
    Returns dict of lag -> mean correlation.
    """
    dirs = torch.stack(perturbation_dirs).cpu().numpy()  # [T, N]
    T, N = dirs.shape
    if T < max_lag + 2:
        max_lag = max(T - 2, 1)

    autocorr = {}
    for k in range(1, max_lag + 1):
        corrs = []
        for i in range(N):
            x = dirs[:T-k, i]
            y = dirs[k:, i]
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                continue
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
        autocorr[k] = float(np.mean(corrs)) if corrs else 0.0
    return autocorr


# ══════════════════════════════════════════════════════════════
# SHD-RSNN evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_shd_rsnn(n_batches=30):
    """Evaluate assumptions on SHD-RSNN models."""
    import snntorch as snn
    from snntorch import surrogate

    spec = importlib.util.spec_from_file_location(
        "shd_rsnn", os.path.join(PROJECT_ROOT, "train", "shd_rsnn.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0

    surrogate_fn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))
    L_SIGMA = get_L_sigma("atan", cfg.get("surrogate_alpha", 2.0))
    alpha_surr = cfg.get("surrogate_alpha", 2.0)

    mod.seed_everything(42)
    _, _, test_loader, input_size = mod.build_loaders(42, cfg)

    ckpt_root = str(ckpt_dir("SHD-RSNN"))

    all_results = {}

    for mode in ["clean", "asar", "hajr", "noise_aug"]:
        ckpt_path = os.path.join(ckpt_root, f"best_s42_{mode}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping SHD-RSNN/{mode}: checkpoint not found at {ckpt_path}")
            continue

        model = mod.SHDRSNN(
            input_size=input_size,
            beta_init=cfg["beta_init"],
            threshold_init=cfg.get("threshold_init", 1.0),
            surrogate_fn=surrogate_fn,
            learn_beta=True,
            learn_threshold=False,
            hidden=cfg.get("hidden", 256),
        ).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Layer info: rlif1 (RLeaky), rlif2 (RLeaky), lif3 (readout, no reset)
        layer_names = ["rlif1", "rlif2"]
        lif_layers = {"rlif1": model.rlif1, "rlif2": model.rlif2, "lif3": model.lif3}
        fc_layers = {"rlif1": model.fc1, "rlif2": model.fc2, "lif3": model.fc3}

        # --- Weight norms ---
        weight_info = {}
        for name in ["rlif1", "rlif2", "lif3"]:
            w = fc_layers[name].weight
            with torch.no_grad():
                wmat = w.view(w.shape[0], -1)
                s = torch.linalg.svdvals(wmat)
                n_in = wmat.shape[1]
                op_norm = float(s[0].item())
                frob_norm = float(torch.norm(wmat, 'fro').item())
                eff_norm = frob_norm / np.sqrt(n_in)
            weight_info[name] = {
                "op_norm": op_norm, "eff_norm": eff_norm, "frob_norm": frob_norm,
                "n_in": int(n_in), "kappa": (op_norm / eff_norm)**2 if eff_norm > 0 else float('inf'),
            }
            # Also get recurrent weight norms for RLeaky layers
            if name in ("rlif1", "rlif2"):
                rec_w = lif_layers[name].recurrent.weight
                with torch.no_grad():
                    rmat = rec_w.view(rec_w.shape[0], -1)
                    rs = torch.linalg.svdvals(rmat)
                    rn_in = rmat.shape[1]
                weight_info[name]["rec_op_norm"] = float(rs[0].item())
                weight_info[name]["rec_frob_norm"] = float(torch.norm(rmat, 'fro').item())
                weight_info[name]["rec_eff_norm"] = float(torch.norm(rmat, 'fro').item()) / np.sqrt(rn_in)

        # --- Forward pass to collect per-layer quantities ---
        alpha_all = {n: [] for n in layer_names}
        surr_sq_all = {n: [] for n in layer_names}
        n_neurons = {}

        # B1/B4: Ψ directions from single-sample forward (not batch-averaged)
        psi_dirs = {n: [] for n in layer_names}
        # B2: per-neuron V from single sample across time
        mem_traces_b2 = {n: [] for n in layer_names}

        count = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(DEVICE).float()
                bsz = data.shape[0]
                if data.dim() < 3:
                    continue
                steps = data.shape[1]
                x = data.view(bsz, steps, -1)

                hidden_dim = cfg.get("hidden", 256)

                spk1, mem1 = model.rlif1.init_rleaky()
                spk2, mem2 = model.rlif2.init_rleaky()
                if not isinstance(mem1, torch.Tensor) or mem1.dim() == 0:
                    mem1 = torch.zeros(bsz, hidden_dim, device=DEVICE)
                if not isinstance(mem2, torch.Tensor) or mem2.dim() == 0:
                    mem2 = torch.zeros(bsz, hidden_dim, device=DEVICE)

                psi1 = torch.zeros(bsz, hidden_dim, device=DEVICE)
                psi2 = torch.zeros(bsz, hidden_dim, device=DEVICE)
                mem1_prev = torch.zeros(bsz, hidden_dim, device=DEVICE)
                mem2_prev = torch.zeros(bsz, hidden_dim, device=DEVICE)

                for t in range(steps):
                    cur1 = model.fc1(x[:, t, :])
                    spk1, mem1 = model.rlif1(cur1, spk1, mem1)

                    thr1 = getattr(model.rlif1, 'threshold', torch.tensor(1.0))
                    if isinstance(thr1, nn.Parameter):
                        thr1 = thr1.detach()
                    beta1 = model.rlif1.beta.detach()
                    surr_val1 = surrogate_deriv_atan(mem1 - thr1, alpha_surr)
                    surr_sq_all["rlif1"].append((surr_val1 ** 2).sum(dim=-1).mean().item())
                    alpha_all["rlif1"].append((beta1 - thr1 * surr_val1).abs().mean(dim=-1).mean().item())

                    # Track Ψ per sample
                    psi1 = mem1_prev + (beta1 - thr1 * surr_val1) * psi1
                    # B1/B4: use first sample's Ψ direction
                    if count < 5:
                        psi_dirs["rlif1"].append(psi1[0].cpu())
                    # B2: per-neuron V from first sample
                    if count == 0:
                        mem_traces_b2["rlif1"].append(mem1[0].cpu())

                    if t == 0:
                        n_neurons["rlif1"] = mem1.shape[-1]

                    cur2 = model.fc2(spk1)
                    spk2, mem2 = model.rlif2(cur2, spk2, mem2)
                    thr2 = getattr(model.rlif2, 'threshold', torch.tensor(1.0))
                    if isinstance(thr2, nn.Parameter):
                        thr2 = thr2.detach()
                    beta2 = model.rlif2.beta.detach()
                    surr_val2 = surrogate_deriv_atan(mem2 - thr2, alpha_surr)
                    surr_sq_all["rlif2"].append((surr_val2 ** 2).sum(dim=-1).mean().item())
                    alpha_all["rlif2"].append((beta2 - thr2 * surr_val2).abs().mean(dim=-1).mean().item())

                    psi2 = mem2_prev + (beta2 - thr2 * surr_val2) * psi2
                    if count < 5:
                        psi_dirs["rlif2"].append(psi2[0].cpu())
                    if count == 0:
                        mem_traces_b2["rlif2"].append(mem2[0].cpu())

                    if t == 0:
                        n_neurons["rlif2"] = mem2.shape[-1]

                    mem1_prev = mem1.clone()
                    mem2_prev = mem2.clone()

                count += 1
                if count >= n_batches:
                    break

        # --- Aggregate ---
        mode_results = {"assumptions": {}, "stability": {}}
        for name in layer_names:
            n = n_neurons[name]
            alpha_vals = np.array(alpha_all[name])
            surr_vals = np.array(surr_sq_all[name])

            # B1: Isotropy of Ψ directions (per-sample, not batch-averaged)
            b1_mean, b1_std = check_B1_isotropy(psi_dirs[name])

            # B2: Sub-Gaussian on per-neuron V from single sample
            mem_stack = torch.stack(mem_traces_b2[name])  # [T, N]
            b2_pass_frac, b2_n = check_B2_subgaussian(mem_stack)

            # B3: Surrogate sparsity
            eta = float(np.mean(surr_vals) / (n * L_SIGMA**2))

            # B4: Temporal independence of Ψ directions
            autocorr = check_B4_temporal_independence(psi_dirs[name])

            mode_results["assumptions"][name] = {
                "B1_cosine_mean": b1_mean,
                "B1_cosine_std": b1_std,
                "B1_pass": abs(b1_mean) < 0.15,
                "B2_shapiro_pass_frac": b2_pass_frac,
                "B2_n_neurons": b2_n,
                "B2_pass": b2_pass_frac > 0.80,
                "B3_eta": eta,
                "B3_sqrt_eta": float(np.sqrt(eta)) if eta > 0 else 0.0,
                "B3_pass": eta < 0.1,
                "B4_autocorr": autocorr,
                "B4_decay_to_01_by_lag": next((k for k, v in autocorr.items() if abs(v) < 0.1), None),
                "B4_pass": any(abs(v) < 0.1 for k, v in autocorr.items() if k <= 5),
            }

            # Stability quantities
            alpha_tilde = float(np.mean(alpha_vals))
            alpha_bar_p99 = float(np.percentile(alpha_vals, 99))
            wi = weight_info[name]

            denom_wc = 1.0 - alpha_bar_p99
            lambda_r = L_SIGMA * wi["op_norm"] / denom_wc if denom_wc > 0 else float("inf")
            denom_tc = 1.0 - alpha_tilde
            sqrt_eta = float(np.sqrt(eta)) if eta > 0 else 0.0
            lambda_tilde = sqrt_eta * L_SIGMA * wi["eff_norm"] / denom_tc if denom_tc > 0 else float("inf")

            mode_results["stability"][name] = {
                "beta": float(lif_layers[name].beta.detach().view(-1)[0].item()),
                "n_neurons": n,
                "W_op_norm": wi["op_norm"],
                "W_eff_norm": wi["eff_norm"],
                "kappa": wi["kappa"],
                "rec_op_norm": wi.get("rec_op_norm"),
                "rec_eff_norm": wi.get("rec_eff_norm"),
                "eta": eta,
                "sqrt_eta": sqrt_eta,
                "alpha_bar_p99": alpha_bar_p99,
                "alpha_tilde": alpha_tilde,
                "lambda_r": lambda_r,
                "lambda_tilde": lambda_tilde,
                "stable_worstcase": lambda_r < 1.0,
                "stable_typicalcase": lambda_tilde < 1.0,
                "tightening_ratio": lambda_tilde / lambda_r if lambda_r > 0 and lambda_r != float("inf") else None,
            }

        all_results[mode] = mode_results
        print(f"  ✓ SHD-RSNN/{mode} done")

    return all_results


# ══════════════════════════════════════════════════════════════
# SSC (DelRec) evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_ssc_delrec(n_batches=30):
    """Evaluate assumptions on SSC DelRec models (spikingjelly backend)."""
    delrec_dir = os.path.join(PROJECT_ROOT, "third_party", "DelRec")
    sys.path.insert(0, delrec_dir)

    import h5py
    from src.SSC.snn import SNN_recurrent_delays
    from src.recurrent_neurons import axonal_recdel

    # Load config from checkpoint to ensure consistency
    ckpt_ref = torch.load(
        os.path.join(str(ckpt_dir("SSC")), "best_s42_clean.pth"),
        map_location="cpu", weights_only=False)
    saved_cfg = ckpt_ref["config"]

    # Reconstruct config object
    # Use the actual config from the experiment
    sys.path.insert(0, os.path.join(delrec_dir, "configs"))
    from configs.perf_SSC_p0_spikingjelly import Config as SSCConfigClass
    config = SSCConfigClass()
    config.store_v_seq = True  # Need voltage traces

    surr_alpha_val = config.reg_surrogate_alpha  # 1.0

    L_SIGMA = get_L_sigma("triangle", surr_alpha_val)

    # Build test data iterator
    from src.SSC.snntorch_p0 import SSCSpikeIterator
    from snn_robust_tau.paths import SSC_ROOT
    datasets_path = str(SSC_ROOT)
    test_h5 = h5py.File(os.path.join(datasets_path, "ssc_test.h5"), "r")
    test_loader = SSCSpikeIterator(
        test_h5, batch_size=config.batch_size, nb_steps=config.time_window,
        nb_units=700, max_time=1.0, n_bins=config.n_bins, shuffle=False,
    )

    ckpt_dirs = {m: str(ckpt_dir("SSC")) for m in ("clean", "asar", "hajr", "noise_aug")}

    all_results = {}

    for mode in ["clean", "asar", "hajr", "noise_aug"]:
        ckpt_path = os.path.join(ckpt_dirs[mode], f"best_s42_{mode}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping SSC/{mode}: checkpoint not found at {ckpt_path}")
            continue

        model = SNN_recurrent_delays(config).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Set sigma to 0 (evaluation mode - no delay smoothing)
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.sigma = 0.0

        # Parse model structure: layers is Sequential
        # Structure: Linear, Dropout, axonal_recdel, spike_registrator, [batchnorm], ... , Linear
        hidden_names = []
        linear_layers = []
        recdel_layers = []
        for i, layer in enumerate(model.layers):
            if isinstance(layer, torch.nn.Linear):
                linear_layers.append((i, layer))
            if isinstance(layer, axonal_recdel):
                recdel_layers.append((i, layer))
                hidden_names.append(f"hidden{len(recdel_layers)}")

        # Weight norms
        weight_info = {}
        for idx, (name, (li, recdel)) in enumerate(zip(hidden_names, recdel_layers)):
            # Feedforward weight is the Linear layer before this recdel
            ff_layer = linear_layers[idx][1]
            w = ff_layer.weight
            with torch.no_grad():
                wmat = w.view(w.shape[0], -1)
                s = torch.linalg.svdvals(wmat)
                n_in = wmat.shape[1]
                op_norm = float(s[0].item())
                frob_norm = float(torch.norm(wmat, 'fro').item())
                eff_norm = frob_norm / np.sqrt(n_in)
            weight_info[name] = {
                "op_norm": op_norm, "eff_norm": eff_norm, "frob_norm": frob_norm,
                "n_in": int(n_in), "kappa": (op_norm / eff_norm)**2 if eff_norm > 0 else float('inf'),
            }
            # Recurrent weights
            rec_w = recdel.recurrent_weights
            with torch.no_grad():
                rmat = rec_w.view(rec_w.shape[0], -1)
                rs = torch.linalg.svdvals(rmat)
                rn_in = rmat.shape[1]
            weight_info[name]["rec_op_norm"] = float(rs[0].item())
            weight_info[name]["rec_frob_norm"] = float(torch.norm(rmat, 'fro').item())
            weight_info[name]["rec_eff_norm"] = float(torch.norm(rmat, 'fro').item()) / np.sqrt(rn_in)

        # Run forward pass with store_v_seq to get membrane traces
        # The spikingjelly model uses multi_step_forward: input is (T, B, N)
        alpha_all = {n: [] for n in hidden_names}
        surr_sq_all = {n: [] for n in hidden_names}
        psi_dirs = {n: [] for n in hidden_names}
        mem_traces_b2 = {n: [] for n in hidden_names}
        n_neurons = {}

        count = 0
        with torch.no_grad():
            test_loader.reset()
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(DEVICE).float()
                bsz, steps, feat = batch_data.shape

                x_seq = batch_data.permute(1, 0, 2)  # (T, B, N)

                from spikingjelly.activation_based import functional
                functional.reset_net(model)

                cur = x_seq
                for i, layer_module in enumerate(model.layers):
                    if isinstance(layer_module, axonal_recdel):
                        rd_idx = [j for j, (li, rd) in enumerate(recdel_layers) if rd is layer_module][0]
                        name = hidden_names[rd_idx]

                        cur = layer_module(cur)

                        v_seq = layer_module.v_seq  # (T, B, N)
                        T_out = v_seq.shape[0]
                        n_neurons[name] = v_seq.shape[2]

                        w_param = layer_module.neuron_module.w.detach()
                        beta_val = 1.0 - torch.sigmoid(w_param)  # ParametricLIF: β = 1 - σ(w)

                        thr_val = config.v_threshold

                        psi = torch.zeros(bsz, v_seq.shape[2], device=DEVICE)
                        mem_prev = torch.zeros(bsz, v_seq.shape[2], device=DEVICE)

                        for t in range(T_out):
                            mem_t = v_seq[t]
                            surr_val = surrogate_deriv_triangle(mem_t - thr_val, surr_alpha_val)
                            surr_sq = (surr_val ** 2).sum(dim=-1).mean().item()
                            surr_sq_all[name].append(surr_sq)

                            alpha_eff = (beta_val - thr_val * surr_val).abs().mean(dim=-1).mean().item()
                            alpha_all[name].append(alpha_eff)

                            psi = mem_prev + (beta_val - thr_val * surr_val) * psi
                            # B1/B4: single-sample Ψ direction
                            if count < 5:
                                psi_dirs[name].append(psi[0].detach().cpu())
                            # B2: single-sample V
                            if count == 0:
                                mem_traces_b2[name].append(mem_t[0].detach().cpu())

                            mem_prev = mem_t
                    else:
                        cur = layer_module(cur)

                count += 1
                if count >= n_batches:
                    break

        # Aggregate
        mode_results = {"assumptions": {}, "stability": {}}
        for name in hidden_names:
            n = n_neurons[name]
            alpha_vals = np.array(alpha_all[name])
            surr_vals = np.array(surr_sq_all[name])

            b1_mean, b1_std = check_B1_isotropy(psi_dirs[name])
            mem_stack = torch.stack(mem_traces_b2[name])
            b2_pass_frac, b2_n = check_B2_subgaussian(mem_stack)
            eta = float(np.mean(surr_vals) / (n * L_SIGMA**2))
            autocorr = check_B4_temporal_independence(psi_dirs[name])

            mode_results["assumptions"][name] = {
                "B1_cosine_mean": b1_mean,
                "B1_cosine_std": b1_std,
                "B1_pass": abs(b1_mean) < 0.15,
                "B2_shapiro_pass_frac": b2_pass_frac,
                "B2_n_neurons": b2_n,
                "B2_pass": b2_pass_frac > 0.80,
                "B3_eta": eta,
                "B3_sqrt_eta": float(np.sqrt(eta)) if eta > 0 else 0.0,
                "B3_pass": eta < 0.1,
                "B4_autocorr": autocorr,
                "B4_decay_to_01_by_lag": next((k for k, v in autocorr.items() if abs(v) < 0.1), None),
                "B4_pass": any(abs(v) < 0.1 for k, v in autocorr.items() if k <= 5),
            }

            alpha_tilde = float(np.mean(alpha_vals))
            alpha_bar_p99 = float(np.percentile(alpha_vals, 99))
            wi = weight_info[name]
            sqrt_eta = float(np.sqrt(eta)) if eta > 0 else 0.0

            denom_wc = 1.0 - alpha_bar_p99
            lambda_r = L_SIGMA * wi["op_norm"] / denom_wc if denom_wc > 0 else float("inf")
            denom_tc = 1.0 - alpha_tilde
            lambda_tilde = sqrt_eta * L_SIGMA * wi["eff_norm"] / denom_tc if denom_tc > 0 else float("inf")

            rd_idx = hidden_names.index(name)
            recdel = recdel_layers[rd_idx][1]
            beta_scalar = float((1.0 - torch.sigmoid(recdel.neuron_module.w.detach())).item())

            mode_results["stability"][name] = {
                "beta": beta_scalar,
                "n_neurons": n,
                "W_op_norm": wi["op_norm"],
                "W_eff_norm": wi["eff_norm"],
                "kappa": wi["kappa"],
                "rec_op_norm": wi.get("rec_op_norm"),
                "rec_eff_norm": wi.get("rec_eff_norm"),
                "eta": eta,
                "sqrt_eta": sqrt_eta,
                "alpha_bar_p99": alpha_bar_p99,
                "alpha_tilde": alpha_tilde,
                "lambda_r": lambda_r,
                "lambda_tilde": lambda_tilde,
                "stable_worstcase": lambda_r < 1.0,
                "stable_typicalcase": lambda_tilde < 1.0,
                "tightening_ratio": lambda_tilde / lambda_r if lambda_r > 0 and lambda_r != float("inf") else None,
            }

        all_results[mode] = mode_results
        print(f"  ✓ SSC/{mode} done")

    test_h5.close()
    return all_results


# ══════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════

def print_assumptions_table(dataset_name, results):
    print(f"\n{'='*90}")
    print(f"  {dataset_name} — Assumption Validation (B1)–(B4)")
    print(f"{'='*90}")

    for mode, data in results.items():
        print(f"\n  [{mode}]")
        assumptions = data["assumptions"]
        for layer_name, a in assumptions.items():
            print(f"    {layer_name}:")
            b1_ok = "✓" if a["B1_pass"] else "✗"
            print(f"      (B1) Isotropy:    cos_sim = {a['B1_cosine_mean']:.3f} ± {a['B1_cosine_std']:.3f}  {b1_ok}")
            b2_ok = "✓" if a["B2_pass"] else "✗"
            print(f"      (B2) Sub-Gaussian: {a['B2_shapiro_pass_frac']*100:.0f}% neurons pass Shapiro-Wilk  {b2_ok}")
            b3_ok = "✓" if a["B3_pass"] else "✗"
            print(f"      (B3) Sparsity:    η = {a['B3_eta']:.4f}, √η = {a['B3_sqrt_eta']:.4f}  {b3_ok}")
            b4_lag = a.get("B4_decay_to_01_by_lag")
            b4_ok = "✓" if a["B4_pass"] else "✗"
            b4_str = f"lag {b4_lag}" if b4_lag else "> max_lag"
            print(f"      (B4) Temporal ind: autocorr < 0.1 by {b4_str}  {b4_ok}")


def print_stability_table(dataset_name, results):
    print(f"\n{'='*90}")
    print(f"  {dataset_name} — Stability: Worst-case (λ_r) vs Typical-case (λ̃_r)")
    print(f"{'='*90}")
    print(f"  {'Mode':>10s} {'Layer':>8s} {'β':>6s} {'||W||op':>8s} {'||W||eff':>8s} "
          f"{'η':>6s} {'√η':>6s} {'ᾱ(p99)':>7s} {'α̃':>7s} "
          f"{'λ_r':>8s} {'λ̃_r':>8s} {'WC':>3s} {'TC':>3s}")
    print("  " + "-"*102)
    for mode, data in results.items():
        for layer_name, s in data["stability"].items():
            wc = "✓" if s["stable_worstcase"] else "✗"
            tc = "✓" if s["stable_typicalcase"] else "✗"
            lr = f"{s['lambda_r']:.2f}" if s["lambda_r"] != float("inf") else "∞"
            lt = f"{s['lambda_tilde']:.4f}" if s["lambda_tilde"] != float("inf") else "∞"
            print(f"  {mode:>10s} {layer_name:>8s} {s['beta']:>6.3f} {s['W_op_norm']:>8.2f} "
                  f"{s['W_eff_norm']:>8.2f} {s['eta']:>6.3f} {s['sqrt_eta']:>6.3f} "
                  f"{s['alpha_bar_p99']:>7.4f} {s['alpha_tilde']:>7.4f} "
                  f"{lr:>8s} {lt:>8s} {wc:>3s} {tc:>3s}")


def main():
    output_dir = str(OUTPUT_ROOT / "diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    # Load existing SHD-FC results for reference
    shd_fc_path = os.path.join(output_dir, "results.json")
    if os.path.exists(shd_fc_path):
        print("SHD-FC results already exist (reference).")

    print("\n" + "="*90)
    print("  Running SHD-RSNN assumption validation...")
    print("="*90)
    rsnn_results = evaluate_shd_rsnn(n_batches=30)
    print_assumptions_table("SHD-RSNN", rsnn_results)
    print_stability_table("SHD-RSNN", rsnn_results)

    print("\n" + "="*90)
    print("  Running SSC (DelRec) assumption validation...")
    print("="*90)
    if "--with-ssc" in sys.argv:
        ssc_results = evaluate_ssc_delrec(n_batches=30)
    else:
        # triangle surrogate: unaffected by the arctan fix -> reuse original numbers
        ssc_results = json.load(open(RESULTS / "diagnostics" / "assumptions_rsnn_ssc_fixed.json"))["SSC-DelRec"]
    print_assumptions_table("SSC (DelRec)", ssc_results)
    print_stability_table("SSC (DelRec)", ssc_results)

    # Save all
    combined = {
        "SHD-RSNN": rsnn_results,
        "SSC-DelRec": ssc_results,
    }
    out_path = os.path.join(output_dir, "assumptions_rsnn_ssc_fixed.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
