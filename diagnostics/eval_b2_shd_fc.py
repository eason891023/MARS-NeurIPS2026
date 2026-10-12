"""
(B2)/(B4) coherence diagnostics on SHD-FC (paper App. "Typical-case diagnostic", paragraph (B2)).

Two constructions of a per-timestep "perturbation direction" in hidden layer l:
  * psi  : first-order sensitivity Psi_t = dV_t/dbeta from the recursion used in
           diagnostics/eval_typical_case_assumptions_fixed.py
           (Psi_t = V_{t-1} + (beta - theta * sigma'(V_t - theta)) Psi_{t-1}, snnTorch ATan sigma').
  * dV   : actual deviation dV_t = V_t(beta_tilde) - V_t(beta) under per-neuron beta mismatch
           beta_tilde = clip(beta (1 + CV z), 0.01, 0.99), z ~ N(0,1) per neuron, one draw per batch.

Statistics (all on the first N_BATCHES test batches, timesteps with a zero vector excluded):
  * code_def      : exact replica of check_B1_isotropy / check_B4_temporal_independence from
                    eval_typical_case_assumptions_fixed.py (Psi of the first sample of the first
                    5 batches, concatenated in time; B1 on a random subset of 200 timesteps;
                    B4 = per-neuron temporal autocorrelation, averaged over neurons).
  * text_def      : the paper sentence: cosine between normalized vectors dV_t/||dV_t|| for all
                    timestep pairs (t < t') within each sample, pooled over every sample of the
                    N_BATCHES batches (mean +- SD over pairs; SD of the batch means also given);
                    "cross_batch_cos" pairs dV_t of sample 0 of batch b with dV_t' of sample 0 of
                    batch b' != b (independent mismatch draws), for comparison only;
                    corr(dV_t, dV_{t+k}) = Pearson correlation across neurons of the two vectors,
                    averaged over t and samples; plus the per-neuron temporal autocorrelation.
"decay_lag" is the first lag k with |corr| < 0.1 (None if not reached by MAX_LAG).

Inputs : checkpoints/SHD-FC/best_s{seed}_{clean,asar,hajr,noise_aug}.pth, SHD test set
Outputs: outputs/diagnostics/b2_shd_fc.json  (shipped copy: results/diagnostics/b2_shd_fc.json)
Usage  : python3 diagnostics/eval_b2_shd_fc.py [--seeds 42,52,72,92,102] [--cvs 0.1,0.2]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from surrogate_atan import PROJECT_ROOT, atan_deriv  # noqa: E402
from snn_robust_tau.paths import OUTPUT_ROOT, ckpt_dir  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = str(OUTPUT_ROOT / "diagnostics")
CKPT_DIR = str(ckpt_dir("SHD-FC"))
MODES = ["clean", "asar", "hajr", "noise_aug"]
HIDDEN = ["lif1", "lif2"]
N_BATCHES = 30
N_CODE_SAMPLES = 5  # code_def: first sample of the first 5 batches
MAX_LAG = 10
ALPHA_SURR = 2.0
EPS = 1e-8


# ---------------------------------------------------------------- code_def (replica)
def check_B1_isotropy(perturbation_dirs):
    dirs = torch.stack(perturbation_dirs)
    dirs_normed = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    T = dirs_normed.shape[0]
    if T > 200:
        dirs_normed = dirs_normed[torch.randperm(T)[:200]]
        T = 200
    cos_sim = dirs_normed @ dirs_normed.T
    pairwise = cos_sim[torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)]
    return float(pairwise.mean()), float(pairwise.std())


def check_B4_temporal_independence(perturbation_dirs, max_lag=MAX_LAG):
    dirs = torch.stack(perturbation_dirs).cpu().numpy()
    T, N = dirs.shape
    out = {}
    for k in range(1, max_lag + 1):
        corrs = []
        for i in range(N):
            x, y = dirs[:T - k, i], dirs[k:, i]
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                continue
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
        out[k] = float(np.mean(corrs)) if corrs else 0.0
    return out


# ---------------------------------------------------------------- text_def accumulators
class Acc:
    """Pooled pairwise-cosine and lagged-correlation statistics for X of shape [B, T, N]."""

    def __init__(self):
        self.s = self.ss = self.n = 0.0
        self.batch_means = []
        self.vec = {k: [0.0, 0.0] for k in range(1, MAX_LAG + 1)}   # sum, count
        self.neu = {k: [0.0, 0.0] for k in range(1, MAX_LAG + 1)}

    def update(self, X):
        X = X.double()
        B, T, _ = X.shape
        nrm = X.norm(dim=-1)
        valid = nrm > EPS
        U = X / nrm.clamp_min(EPS).unsqueeze(-1)
        C = U @ U.transpose(1, 2)                                     # [B, T, T]
        pm = valid.unsqueeze(2) & valid.unsqueeze(1)
        pm &= torch.triu(torch.ones(T, T, dtype=torch.bool, device=X.device), 1)
        v = C[pm]
        self.s += float(v.sum()); self.ss += float((v ** 2).sum()); self.n += v.numel()
        self.batch_means.append(float(v.mean()))
        # Pearson correlation across neurons between X_t and X_{t+k}
        Xc = X - X.mean(-1, keepdim=True)
        sd = Xc.norm(dim=-1)
        Z = Xc / sd.clamp_min(EPS).unsqueeze(-1)
        okv = sd > EPS
        for k in range(1, min(MAX_LAG, T - 2) + 1):
            r = (Z[:, :-k] * Z[:, k:]).sum(-1)
            m = okv[:, :-k] & okv[:, k:]
            self.vec[k][0] += float(r[m].sum()); self.vec[k][1] += float(m.sum())
            # per-neuron temporal autocorrelation within each sample (code_def statistic)
            a, b = X[:, :-k], X[:, k:]
            a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
            na, nb = a.norm(dim=1), b.norm(dim=1)
            m2 = (na > EPS) & (nb > EPS)
            rr = (a * b).sum(1) / (na * nb).clamp_min(EPS)
            self.neu[k][0] += float(rr[m2].sum()); self.neu[k][1] += float(m2.sum())

    def result(self):
        mean = self.s / self.n
        sd = float(np.sqrt(max(self.ss / self.n - mean ** 2, 0.0)))
        vec = {k: (s / c if c else None) for k, (s, c) in self.vec.items()}
        neu = {k: (s / c if c else None) for k, (s, c) in self.neu.items()}
        return {"cos_mean": mean, "cos_sd_pairs": sd, "cos_sd_batch_means": float(np.std(self.batch_means)),
                "n_pairs": int(self.n),
                "corr_vec_by_lag": vec, "corr_vec_decay_lag": decay_lag(vec),
                "corr_neuron_by_lag": neu, "corr_neuron_decay_lag": decay_lag(neu)}


def cross_batch_cos(seqs):
    """Mean +- SD cosine between normalized dV rows taken from different batches."""
    rows, owner = [], []
    for b, X in enumerate(seqs):
        nrm = X.norm(dim=-1)
        keep = nrm > EPS
        rows.append(X[keep] / nrm[keep].unsqueeze(-1))
        owner.append(torch.full((int(keep.sum()),), b))
    U, o = torch.cat(rows), torch.cat(owner)
    C = U @ U.T
    v = C[(o.unsqueeze(0) != o.unsqueeze(1)) & torch.triu(torch.ones_like(C, dtype=torch.bool), 1)]
    return [float(v.mean()), float(v.std())]


def decay_lag(d):
    return next((k for k, v in d.items() if v is not None and abs(v) < 0.1), None)


# ---------------------------------------------------------------- model helpers
def _thr(layer):
    thr = getattr(layer, "threshold", torch.tensor(1.0))
    return thr.detach() if isinstance(thr, torch.Tensor) else torch.tensor(thr)


def set_beta(layer, value):
    if "beta" in layer._parameters:
        layer._parameters["beta"] = nn.Parameter(value.detach(), requires_grad=False)
    else:
        layer._buffers["beta"] = value.detach()


def run(model, x, track_psi):
    """Forward pass; returns {layer: V [B,T,N]} and optionally {layer: Psi [B,T,N]}."""
    bsz, steps = x.shape[0], x.shape[1]
    mem = {"lif1": model.lif1.init_leaky(), "lif2": model.lif2.init_leaky()}
    mem3 = model.lif3.init_leaky()
    V = {n: [] for n in HIDDEN}
    P = {n: [] for n in HIDDEN}
    psi, prev = {}, {}
    h = None
    for t in range(steps):
        inp = x[:, t, :]
        for name, fc in (("lif1", model.fc1), ("lif2", model.fc2)):
            layer = getattr(model, name)
            h, mem[name] = layer(fc(inp), mem[name])
            m = mem[name]
            V[name].append(m)
            if track_psi:
                if t == 0:
                    psi[name] = torch.zeros_like(m); prev[name] = torch.zeros_like(m)
                s = atan_deriv(m - _thr(layer).to(m.device), ALPHA_SURR)
                psi[name] = prev[name] + (layer.beta.detach() - _thr(layer).to(m.device) * s) * psi[name]
                prev[name] = m.clone()
                P[name].append(psi[name])
            inp = h
        _, mem3 = model.lif3(model.fc3(h), mem3)
    V = {n: torch.stack(v, 1) for n, v in V.items()}
    P = {n: torch.stack(p, 1) for n, p in P.items()} if track_psi else None
    return V, P


def evaluate(mod, cfg, loader, input_size, seed, mode, cvs):
    from snntorch import surrogate
    model = mod.SHDSNN(input_size=input_size, beta_init=cfg["beta_init"],
                       threshold_init=cfg.get("threshold_init", 1.0),
                       surrogate_fn=surrogate.atan(alpha=ALPHA_SURR),
                       learn_beta=True, learn_threshold=False).to(DEVICE)
    ckpt = torch.load(os.path.join(CKPT_DIR, f"best_s{seed}_{mode}.pth"), map_location=DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    base = {n: getattr(model, n).beta.detach().clone() for n in HIDDEN}
    acc_psi = {n: Acc() for n in HIDDEN}
    acc_dv = {cv: {n: Acc() for n in HIDDEN} for cv in cvs}
    code_dirs = {n: [] for n in HIDDEN}
    xb = {cv: {n: [] for n in HIDDEN} for cv in cvs}  # sample-0 dV per batch (cross-batch pairs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for b, (data, _) in enumerate(loader):
            if b >= N_BATCHES:
                break
            x = data.to(DEVICE).float()
            x = x.view(x.shape[0], x.shape[1], -1)
            V0, P = run(model, x, track_psi=True)
            for n in HIDDEN:
                acc_psi[n].update(P[n])
                if b < N_CODE_SAMPLES:
                    code_dirs[n].extend(list(P[n][0].cpu()))
            z = {n: torch.randn(V0[n].shape[-1], generator=gen) for n in HIDDEN}
            for cv in cvs:
                for n in HIDDEN:
                    set_beta(getattr(model, n),
                             (base[n].cpu() * (1 + cv * z[n])).clamp(0.01, 0.99).to(DEVICE))
                V1, _ = run(model, x, track_psi=False)
                for n in HIDDEN:
                    set_beta(getattr(model, n), base[n])
                    acc_dv[cv][n].update(V1[n] - V0[n])
                    xb[cv][n].append((V1[n] - V0[n])[0].double().cpu())
    torch.manual_seed(0)
    out = {"beta": {n: float(base[n].view(-1)[0]) for n in HIDDEN}, "code_def": {}, "text_def_psi": {},
           "text_def_dV": {str(cv): {} for cv in cvs}}
    for n in HIDDEN:
        m, s = check_B1_isotropy(code_dirs[n])
        ac = check_B4_temporal_independence(code_dirs[n])
        out["code_def"][n] = {"B1_cosine_mean": m, "B1_cosine_std": s, "B4_autocorr": ac,
                              "B4_decay_to_01_by_lag": decay_lag(ac)}
        out["text_def_psi"][n] = acc_psi[n].result()
        for cv in cvs:
            out["text_def_dV"][str(cv)][n] = acc_dv[cv][n].result()
            out["text_def_dV"][str(cv)][n]["cross_batch_cos"] = cross_batch_cos(xb[cv][n])
    del model
    torch.cuda.empty_cache()
    return out


def summarize(per_seed, cvs):
    """Mean +- SD across seeds of the headline numbers."""
    def stat(vals):
        vals = [v for v in vals if v is not None]
        return [float(np.mean(vals)), float(np.std(vals))] if vals else None
    summ = {}
    seeds = list(per_seed)
    for mode in MODES:
        summ[mode] = {}
        for n in HIDDEN:
            rows = [per_seed[s][mode] for s in seeds]
            d = {"code_def_cos": stat([r["code_def"][n]["B1_cosine_mean"] for r in rows]),
                 "code_def_lags": [r["code_def"][n]["B4_decay_to_01_by_lag"] for r in rows],
                 "psi_cos": stat([r["text_def_psi"][n]["cos_mean"] for r in rows]),
                 "psi_vec_lags": [r["text_def_psi"][n]["corr_vec_decay_lag"] for r in rows],
                 "psi_neuron_lags": [r["text_def_psi"][n]["corr_neuron_decay_lag"] for r in rows]}
            for cv in cvs:
                rr = [r["text_def_dV"][str(cv)][n] for r in rows]
                d[f"dV_cv{cv}_cos"] = stat([x["cos_mean"] for x in rr])
                d[f"dV_cv{cv}_cross_batch_cos"] = stat([x["cross_batch_cos"][0] for x in rr])
                d[f"dV_cv{cv}_vec_lags"] = [x["corr_vec_decay_lag"] for x in rr]
                d[f"dV_cv{cv}_neuron_lags"] = [x["corr_neuron_decay_lag"] for x in rr]
            summ[mode][n] = d
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,52,72,92,102")
    ap.add_argument("--cvs", default="0.1,0.2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    cvs = [float(c) for c in args.cvs.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    spec = importlib.util.spec_from_file_location("shd", os.path.join(PROJECT_ROOT, "train", "shd_fc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0
    mod.seed_everything(42)
    _, _, test_loader, input_size = mod.build_loaders(42, cfg)

    per_seed = {}
    for seed in seeds:
        per_seed[seed] = {}
        for mode in MODES:
            t0 = time.time()
            r = evaluate(mod, cfg, test_loader, input_size, seed, mode, cvs)
            per_seed[seed][mode] = r
            for n in HIDDEN:
                c, p = r["code_def"][n], r["text_def_psi"][n]
                dv = r["text_def_dV"][str(cvs[0])][n]
                print(f"s{seed} {mode:9s} {n} code cos={c['B1_cosine_mean']:.3f}+-{c['B1_cosine_std']:.3f} "
                      f"lag={c['B4_decay_to_01_by_lag']} | psi cos={p['cos_mean']:.3f}+-{p['cos_sd_pairs']:.3f} "
                      f"lag(vec/neu)={p['corr_vec_decay_lag']}/{p['corr_neuron_decay_lag']} | "
                      f"dV(cv{cvs[0]}) cos={dv['cos_mean']:.3f}+-{dv['cos_sd_pairs']:.3f} "
                      f"lag(vec/neu)={dv['corr_vec_decay_lag']}/{dv['corr_neuron_decay_lag']} "
                      f"xbatch={dv['cross_batch_cos'][0]:.3f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    result = {"setting": "SHD-FC", "n_batches": N_BATCHES, "batch_size": cfg["batch_size"],
              "max_lag": MAX_LAG, "cvs": cvs, "seeds": seeds,
              "per_seed": {str(s): v for s, v in per_seed.items()},
              "across_seeds": summarize(per_seed, cvs)}
    path = os.path.join(OUT_DIR, "b2_shd_fc.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", os.path.relpath(path, PROJECT_ROOT))


if __name__ == "__main__":
    main()
