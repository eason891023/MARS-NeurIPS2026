"""
Re-run of analysis/eval_sensitivity_heatmap.py (panel (a) of fig3_heatmap_stability).
Psi_t is produced by the training regularizer OnlineHAJR (snn_robust_tau/loss/hajr.py),
which auto-detects snnTorch ATan(alpha=2) and already uses the CORRECT derivative
(alpha/2)/(1+(pi alpha x/2)^2). No formula change is needed; this script only
regenerates heatmap_data.npz (outputs/diagnostics/) from the seed-42 SHD-FC checkpoints
(checkpoints/SHD-FC/best_s42_{clean,asar}.pth) and compares it with the shipped
results/diagnostics/heatmap_data.npz.
Usage: python3 diagnostics/rerun_sensitivity_heatmap.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from surrogate_atan import PROJECT_ROOT  # noqa: E402
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS, ckpt_dir  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "heat", os.path.join(PROJECT_ROOT, "analysis", "eval_sensitivity_heatmap.py"))
heat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heat)
DEVICE = heat.DEVICE


def main():
    out = str(OUTPUT_ROOT / "diagnostics")
    os.makedirs(out, exist_ok=True)
    spec2 = importlib.util.spec_from_file_location(
        "shd", os.path.join(PROJECT_ROOT, "train", "shd_fc.py"))
    mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)
    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0
    from snntorch import surrogate
    sfn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))
    mod.seed_everything(42)
    _, _, test_loader, input_size = mod.build_loaders(42, cfg)
    ckpt_root = str(ckpt_dir("SHD-FC"))
    psi = {}
    for mode in ("clean", "asar"):
        model = mod.SHDSNN(input_size=input_size, beta_init=cfg["beta_init"],
                           threshold_init=cfg.get("threshold_init", 1.0), surrogate_fn=sfn,
                           learn_beta=True, learn_threshold=False).to(DEVICE)
        ck = torch.load(os.path.join(ckpt_root, f"best_s42_{mode}.pth"), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        st = model.online_reg._resolve_surrogate_derivative("lif1", layer=model.lif1)
        x0 = torch.tensor([0.0, 0.3], device=DEVICE)
        print(mode, "OnlineHAJR sigma'(0), sigma'(0.3) =", st(x0).tolist())
        psi[mode] = heat.collect_psi_per_neuron(model, test_loader, n_batches=20)
    np.savez(os.path.join(out, "heatmap_data.npz"),
             **{f"clean_{k}": v for k, v in psi["clean"].items()},
             **{f"asar_{k}": v for k, v in psi["asar"].items()})
    old = np.load(RESULTS / "diagnostics" / "heatmap_data.npz")
    summary = {}
    for k in ("lif1", "lif2", "lif3"):
        c, a = psi["clean"][k], psi["asar"][k]
        summary[k] = {"clean_mean": float(c.mean()), "clean_max": float(c.max()),
                      "asar_mean": float(a.mean()), "asar_max": float(a.max()),
                      "ratio": float(c.mean() / max(a.mean(), 1e-10)),
                      "max_rel_diff_vs_old_npz": float(max(
                          np.abs(c - old[f"clean_{k}"]).max() / max(np.abs(old[f"clean_{k}"]).max(), 1e-12),
                          np.abs(a - old[f"asar_{k}"]).max() / max(np.abs(old[f"asar_{k}"]).max(), 1e-12)))}
    json.dump(summary, open(os.path.join(out, "heatmap_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
