#!/usr/bin/env python3
"""Camera-ready Table 2: beta mismatch combined with weight/threshold mismatch or low-bit weights (SHD only).

Conditions (per checkpoint, `--draws` draws each):
  joint     : per-neuron beta CV 0.2 + per-weight CV 0.05 + per-neuron threshold CV 0.05
  aux_only  : per-weight CV 0.05 + per-neuron threshold CV 0.05 (beta unperturbed)
  ptq8/ptq6 : symmetric per-layer (per-tensor) max-abs PTQ of every nn.Linear, then per-neuron beta CV 0.2.
              Drop is measured from the quantized nominal; the quantized nominal itself is stored.

The beta draw z for draw d is perturb.draw_z(..., "per_neuron", d), i.e. exactly the draws used for
the per-neuron results in Table 1 / Fig. 1 (results/partial/SHD-*_mc200.json), so the beta-only
row of Table 2 is paired with every other row. Weight/threshold draws use separate deterministic keys.
Usage: python3 eval/run_joint_ptq.py --setting SHD-FC --draws 200
Output: outputs/eval/partial/<setting>_jointptq.json (same format as results/partial/SHD-*_jointptq.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import perturb  # noqa: E402
from settings import DEVICE, _replace_registered_tensor, make_adapter  # noqa: E402
from snn_robust_tau.paths import OUTPUT_ROOT  # noqa: E402

SEEDS = [42, 52, 72, 92, 102]
METHODS = ["clean", "noise_aug", "mars"]
BETA_CV, W_CV, TH_CV = 0.2, 0.05, 0.05


def gen(name, seed, tag, d):
    return torch.Generator(device="cpu").manual_seed(perturb.draw_seed(name, seed, tag, d))


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def quantize_(model, bits):
    qmax = (2 ** (bits - 1)) - 1
    with torch.no_grad():
        for m in linears(model):
            w = m.weight
            max_abs = float(w.abs().max())
            if max_abs > 0:
                scale = max_abs / qmax
                w.copy_(torch.round(w / scale).clamp(-qmax, qmax) * scale)


def apply_aux(model, layers, name, seed, d, orig_w, orig_th):
    g = gen(name, seed, "aux_weight_threshold", d)
    with torch.no_grad():
        for m, w0 in zip(linears(model), orig_w):
            z = torch.randn(w0.shape, generator=g).to(w0.device)
            m.weight.copy_(w0 * (1.0 + W_CV * z))
        for layer, th0 in zip(layers, orig_th):
            base = th0.detach().float().cpu()
            base = base.expand(layer.neuron_shape) if base.numel() == 1 else base.reshape(layer.neuron_shape)
            z = torch.randn(layer.neuron_shape, generator=g)
            _replace_registered_tensor(layer.module, "threshold", (base * (1.0 + TH_CV * z)).clamp_min(0.01).to(DEVICE))


def restore_aux(model, layers, orig_w, orig_th):
    with torch.no_grad():
        for m, w0 in zip(linears(model), orig_w):
            m.weight.copy_(w0)
    for layer, th0 in zip(layers, orig_th):
        _replace_registered_tensor(layer.module, "threshold", th0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["SHD-FC", "SHD-RSNN"])
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()
    ad = make_adapter(args.setting)
    out = OUTPUT_ROOT / "eval" / "partial" / f"{args.setting}_jointptq.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {
        "setting": args.setting, "draws": args.draws, "beta_cv": BETA_CV, "weight_cv": W_CV,
        "threshold_cv": TH_CV, "entries": {}}
    for seed in SEEDS:
        for meth in METHODS:
            key = f"{meth}|{seed}"
            if data["entries"].get(key, {}).get("complete"):
                continue
            t0 = time.time()
            model = ad.load(seed, meth)
            layers = ad.beta_layers(model)
            orig_w = [m.weight.detach().clone() for m in linears(model)]
            orig_th = [l.module.threshold.detach().clone() for l in layers]
            nominal = ad.accuracy(model)
            res = {k: [] for k in ("joint", "aux_only", "ptq8", "ptq6")}
            for d in range(args.draws):
                zb = perturb.draw_z(layers, args.setting, seed, "per_neuron", d)
                apply_aux(model, layers, args.setting, seed, d, orig_w, orig_th)
                res["aux_only"].append(ad.accuracy(model))
                perturb.apply(layers, zb, BETA_CV)
                res["joint"].append(ad.accuracy(model))
                perturb.restore(layers)
                restore_aux(model, layers, orig_w, orig_th)
            q_nominal = {}
            for bits in (8, 6):
                quantize_(model, bits)
                q_nominal[bits] = ad.accuracy(model)
                for d in range(args.draws):
                    zb = perturb.draw_z(layers, args.setting, seed, "per_neuron", d)
                    perturb.apply(layers, zb, BETA_CV)
                    res[f"ptq{bits}"].append(ad.accuracy(model))
                    perturb.restore(layers)
                restore_aux(model, layers, orig_w, orig_th)
            assert abs(ad.accuracy(model) - nominal) < 1e-9, "restore failed"
            data["entries"][key] = {
                "method": meth, "seed": seed, "nominal_acc": nominal,
                "ptq_nominal_acc": {str(b): v for b, v in q_nominal.items()},
                "results": {k: {"runs": v, "mean": float(np.mean(v))} for k, v in res.items()},
                "complete": True}
            out.write_text(json.dumps(data))
            print(f"[{args.setting}] {key}: nominal={nominal:.2f} joint={np.mean(res['joint']):.2f} "
                  f"ptq8={q_nominal[8]:.2f}->{np.mean(res['ptq8']):.2f} ptq6={q_nominal[6]:.2f}->{np.mean(res['ptq6']):.2f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    print(f"[{args.setting}] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
