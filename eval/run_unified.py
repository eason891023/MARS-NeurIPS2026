"""Unified beta-mismatch evaluation (inference only) for one setting.

Example:
  python eval/run_unified.py --setting SHD-FC
  python eval/run_unified.py --setting SSC --smoke   # timing smoke test, no output file
Results are written incrementally (resumable) to outputs/eval/partial/<setting><tag>.json
(same format as the shipped results/partial/*.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import perturb
from settings import METHODS, SEEDS, make_adapter, make_control_adapter
from snn_robust_tau.paths import OUTPUT_ROOT, REPO_ROOT


def _rel(path) -> str:
    """Repository-relative path when possible (keeps result files free of local paths)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

PER_NEURON_CVS = [0.05, 0.1, 0.15, 0.2, 0.3]  # original Fig. 2 sweep (plus CV=0 nominal)
STRUCT_CVS = [0.1, 0.2]
N_DRAWS = 10  # original protocol: eval_mc_runs = 10


def plan(only_cv=None, modes=None, cvs_override=None):
    for mode, cvs in (("per_neuron", PER_NEURON_CVS), ("layer_shared", STRUCT_CVS), ("global", STRUCT_CVS)):
        if modes is not None and mode not in modes:
            continue
        if cvs_override is not None:
            yield mode, cvs_override
        else:
            yield mode, ([only_cv] if only_cv is not None else cvs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--only-cv", type=float, default=None, help="supplemental runs: evaluate a single CV in all modes")
    ap.add_argument("--tag", default="", help="output suffix for supplemental runs (results/partial/<setting><tag>.json)")
    ap.add_argument("--controls", action="store_true", help="SHD simple-baseline control checkpoints (same draws as main run)")
    ap.add_argument("--modes", default=None, help="comma list of modes to run (default: all)")
    ap.add_argument("--cvs", default=None, help="comma list of CVs for the selected modes (overrides defaults)")
    ap.add_argument("--smoke", action="store_true", help="time 1 nominal + 2 perturbed passes and exit")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    adapter = make_control_adapter(args.setting) if args.controls else make_adapter(args.setting)
    draw_name = getattr(adapter, "draw_name", args.setting)
    if args.controls:
        args.tag = "_controls" + args.tag
        if args.methods == ",".join(METHODS):
            args.methods = ",".join(adapter.methods)
    t0 = time.time()
    adapter.test_batches()
    print(f"[{args.setting}] test cache built: n={adapter.n_test} in {time.time() - t0:.1f}s", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")

    if args.smoke:
        model = adapter.load(seeds[0], methods[0])
        layers = adapter.beta_layers(model)
        print("beta layers:", [(l.name, tuple(l.neuron_shape), round(float(l.base_beta.float().mean()), 4)) for l in layers])
        torch.cuda.synchronize()
        t = time.time(); acc = adapter.accuracy(model); torch.cuda.synchronize()
        dt = time.time() - t
        print(f"nominal acc={acc:.3f} (ckpt test_acc={adapter.ckpt_meta.get('test_acc')}) pass={dt:.2f}s")
        for mode in perturb.MODES:
            zs = perturb.draw_z(layers, draw_name, seeds[0], mode, 0)
            info = perturb.apply(layers, zs, 0.2)
            t = time.time(); acc_p = adapter.accuracy(model); dt = time.time() - t
            perturb.restore(layers)
            print(f"{mode} cv0.2 draw0 acc={acc_p:.3f} rel_rms={np.round(info['beta_relative_rms'], 3).tolist()} pass={dt:.2f}s")
        print(f"restored acc={adapter.accuracy(model):.3f}")
        n_pass = 1 + args.draws * (len(PER_NEURON_CVS) + 2 * len(STRUCT_CVS))
        print(f"passes/ckpt={n_pass}; est total={n_pass * 15 * dt / 3600:.2f} h (15 ckpts)")
        return

    _modes = args.modes.split(",") if args.modes else None
    _cvs = [float(c) for c in args.cvs.split(",")] if args.cvs else None
    out_dir = OUTPUT_ROOT / "eval" / "partial"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.setting}{args.tag}.json"
    data = json.loads(out_path.read_text()) if out_path.exists() else {
        "setting": args.setting,
        "n_test": adapter.n_test,
        "draws": args.draws,
        "per_neuron_cvs": [args.only_cv] if args.only_cv is not None else PER_NEURON_CVS,
        "struct_cvs": [args.only_cv] if args.only_cv is not None else STRUCT_CVS,
        "modes_override": _modes,
        "cvs_override": _cvs,
        "entries": {},
    }
    for seed in seeds:
        for method in methods:
            key = f"{method}|{seed}"
            if key in data["entries"] and data["entries"][key].get("complete"):
                continue
            t_ck = time.time()
            model = adapter.load(seed, method)
            layers = adapter.beta_layers(model)
            nominal = adapter.accuracy(model)
            entry = {
                "method": method,
                "seed": seed,
                "checkpoint": _rel(adapter.checkpoint(seed, method)),
                "checkpoint_test_acc": float(adapter.ckpt_meta.get("test_acc", float("nan"))),
                "nominal_acc": nominal,
                "beta_layers": [
                    {"name": l.name, "base_beta": float(l.base_beta.float().mean()), "base_beta_numel": int(l.base_beta.numel()), "neuron_shape": list(l.neuron_shape)}
                    for l in layers
                ],
                "results": {},
            }
            for mode, cvs in plan(args.only_cv, _modes, _cvs):
                runs = {str(cv): [] for cv in cvs}
                rms = {str(cv): [] for cv in cvs}
                for d in range(args.draws):
                    zs = perturb.draw_z(layers, draw_name, seed, mode, d)
                    for cv in cvs:
                        info = perturb.apply(layers, zs, cv)
                        runs[str(cv)].append(adapter.accuracy(model))
                        rms[str(cv)].append(float(np.mean(info["beta_relative_rms"])))
                        perturb.restore(layers)
                entry["results"][mode] = {
                    str(cv): {
                        "runs": runs[str(cv)],
                        "mean": float(np.mean(runs[str(cv)])),
                        "sd_over_draws": float(np.std(runs[str(cv)], ddof=1)),
                        "beta_relative_rms_mean": float(np.mean(rms[str(cv)])),
                    }
                    for cv in cvs
                }
            check = adapter.accuracy(model)
            entry["restored_acc"] = check
            entry["complete"] = True
            entry["seconds"] = time.time() - t_ck
            data["entries"][key] = entry
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, out_path)
            pn = entry["results"].get("per_neuron", {}).get("0.2", {}).get("mean", float("nan"))
            ls = entry["results"].get("layer_shared", {}).get("0.2", {}).get("mean", float("nan"))
            gl = entry["results"].get("global", {}).get("0.2", {}).get("mean", float("nan"))
            print(
                f"[{args.setting}] {key}: nominal={nominal:.2f} (ckpt {entry['checkpoint_test_acc']:.2f}) "
                f"cv0.2 pn={pn:.2f} ls={ls:.2f} gl={gl:.2f} restored={check:.2f} t={entry['seconds']:.0f}s",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
    print(f"[{args.setting}{args.tag}] done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
