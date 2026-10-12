#!/usr/bin/env python3
"""Formal common-random-number evaluation of exact-matched simple controls."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from snntorch import surrogate


ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = ROOT / "train" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from submitted_eval_common import (  # noqa: E402
    DEVICE,
    _replace_registered_tensor,
    apply_device_mismatch,
    build_test_loader,
    evaluate_accuracy,
    load_source_module,
    set_all_seeds,
)


CONDITIONS = {
    "nominal": {"beta_cv": 0.0, "threshold_cv": 0.0, "weight_cv": 0.0},
    "beta01": {"beta_cv": 0.01, "threshold_cv": 0.0, "weight_cv": 0.0},
    "beta02": {"beta_cv": 0.02, "threshold_cv": 0.0, "weight_cv": 0.0},
    "beta05": {"beta_cv": 0.05, "threshold_cv": 0.0, "weight_cv": 0.0},
    "beta20": {"beta_cv": 0.20, "threshold_cv": 0.0, "weight_cv": 0.0},
    "beta20_aux05": {"beta_cv": 0.20, "threshold_cv": 0.05, "weight_cv": 0.05},
}


def load_control_model(path: Path, input_size: int) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    cfg = checkpoint["config"]
    source = load_source_module(architecture)
    mode = checkpoint["mode"]
    learn_beta = mode not in {"fixed_homogeneous", "fixed_heterogeneous"}
    kwargs = {
        "input_size": input_size,
        "beta_init": cfg["beta_init"],
        "threshold_init": cfg["threshold_init"],
        "surrogate_fn": surrogate.atan(alpha=cfg["surrogate_alpha"]),
        "learn_beta": learn_beta,
        "learn_threshold": cfg["learn_threshold"],
    }
    if architecture == "rsnn":
        kwargs["hidden"] = cfg["hidden"]
        model = source.SHDRSNN(**kwargs)
    else:
        model = source.SHDSNN(**kwargs)
    state = checkpoint["model_state_dict"]
    for name, layer in model._lif_layers:
        value = state[f"{name}.beta"]
        if value.numel() > 1:
            _replace_registered_tensor(layer, "beta", value)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval(), checkpoint


def write_state(path: Path, metadata: dict[str, Any], records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[f"{row['architecture']}|{row['mode']}|{row['seed']}|{row['condition']}"].append(row["accuracy"])
    summary = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "runs": len(values),
        }
        for key, values in grouped.items()
    }
    path.write_text(json.dumps({"metadata": metadata, "records": records, "per_checkpoint_summary": summary}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mc-runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-specs", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    records = json.loads(output_path.read_text()).get("records", []) if output_path.exists() else []
    completed = {
        (row["architecture"], row["mode"], row["seed"], row["condition"], row["trial"])
        for row in records
    }
    checkpoints = sorted(args.controls_dir.glob("formal_*/checkpoints/best_s*_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No formal control checkpoints under {args.controls_dir}")
    loader, input_size = build_test_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    metadata = {
        "device": str(DEVICE),
        "conditions": CONDITIONS,
        "mc_runs": args.mc_runs,
        "controls_dir": str(args.controls_dir),
        "common_random_numbers": "seed depends only on architecture, checkpoint seed, condition, and trial; identical across control modes",
    }
    specs_done = 0
    for checkpoint_path in checkpoints:
        base_model, checkpoint = load_control_model(checkpoint_path, input_size)
        architecture = checkpoint["architecture"]
        mode = checkpoint["mode"]
        seed = int(checkpoint["seed"])
        for condition_name, condition in CONDITIONS.items():
            runs = 1 if condition_name == "nominal" else int(args.mc_runs)
            for trial in range(runs):
                key = (architecture, mode, seed, condition_name, trial)
                if key in completed:
                    continue
                if args.max_specs is not None and specs_done >= args.max_specs:
                    write_state(output_path, metadata, records)
                    return
                random_seed = (
                    60_260_724
                    + (0 if architecture == "ff" else 1) * 1_000_000
                    + seed * 10_000
                    + list(CONDITIONS).index(condition_name) * 100
                    + trial
                )
                set_all_seeds(random_seed)
                model = copy.deepcopy(base_model).to(DEVICE).eval()
                realized = apply_device_mismatch(model, **condition)
                started = time.time()
                accuracy = evaluate_accuracy(model, loader, max_batches=args.max_eval_batches)
                records.append(
                    {
                        "architecture": architecture,
                        "mode": mode,
                        "seed": seed,
                        "condition": condition_name,
                        "trial": trial,
                        "accuracy": accuracy,
                        "random_seed": random_seed,
                        "elapsed_seconds": time.time() - started,
                        "checkpoint_test_acc": float(checkpoint["test_acc"]),
                        "realized": realized,
                    }
                )
                specs_done += 1
                print(f"[control] {architecture} {mode} s{seed} {condition_name} trial={trial} acc={accuracy:.3f}", flush=True)
                write_state(output_path, metadata, records)


if __name__ == "__main__":
    main()
