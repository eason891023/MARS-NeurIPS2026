#!/usr/bin/env python3
"""Joint mismatch evaluation on exact submitted SSC delayed-RSNN checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
DELREC = ROOT / "third_party" / "DelRec"
if str(DELREC) not in sys.path:
    sys.path.insert(0, str(DELREC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import SSC_ROOT, ckpt_dir  # noqa: E402

from configs.perf_SSC_p0_spikingjelly import Config  # noqa: E402
from src.SSC.snn import SNN_recurrent_delays  # noqa: E402
from src.SSC.trainer_p0_spikingjelly import (  # noqa: E402
    evaluate,
    get_plif_layers,
    load_checkpoint_snapshot,
)
from src.datasets import load_dataset  # noqa: E402
from src.recurrent_neurons import axonal_recdel  # noqa: E402
from src.spikingjelly_compat import beta_from_plif  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 52, 72, 92, 102]
CHECKPOINT_ROOTS = {mode: ckpt_dir("SSC") for mode in ("clean", "asar", "noise_aug")}
CONDITIONS = {
    "nominal": {"beta_cv": 0.0, "threshold_cv": 0.0, "weight_cv": 0.0, "delay_cv": 0.0},
    "joint_all01": {"beta_cv": 0.01, "threshold_cv": 0.01, "weight_cv": 0.01, "delay_cv": 0.01},
    "joint_all02": {"beta_cv": 0.02, "threshold_cv": 0.02, "weight_cv": 0.02, "delay_cv": 0.02},
    "joint_all05": {"beta_cv": 0.05, "threshold_cv": 0.05, "weight_cv": 0.05, "delay_cv": 0.05},
    "beta20_aux05_delay05": {"beta_cv": 0.20, "threshold_cv": 0.05, "weight_cv": 0.05, "delay_cv": 0.05},
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(mode: str, seed: int) -> Path:
    return CHECKPOINT_ROOTS[mode] / f"best_s{seed}_{mode}.pth"


def make_config(batch_size: int) -> Config:
    cfg = Config()
    cfg.datasets_path = str(SSC_ROOT)
    cfg.batch_size = int(batch_size)
    cfg.seed = 42
    return cfg


def load_model(mode: str, seed: int, cfg: Config) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path(mode, seed), map_location="cpu", weights_only=False)
    model = SNN_recurrent_delays(cfg)
    load_checkpoint_snapshot(model, payload)
    return model.to(DEVICE).eval(), payload


def _replace_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    old = getattr(module, name)
    requires_grad = bool(getattr(old, "requires_grad", False))
    module._parameters[name] = nn.Parameter(value.detach(), requires_grad=requires_grad)


def apply_joint_mismatch(model: nn.Module, condition: dict[str, float]) -> dict[str, Any]:
    realized: dict[str, Any] = {
        "beta_relative_rms": [],
        "threshold_relative_rms": [],
        "weight_relative_rms": [],
        "delay_relative_rms": [],
        "granularity": {
            "beta": "per neuron",
            "threshold": "per layer scalar (required by exact SpikingJelly reset kernel)",
            "weight": "per weight",
            "learned_delay": "per delayed recurrent neuron",
        },
    }
    with torch.no_grad():
        for _, node in get_plif_layers(model):
            beta = beta_from_plif(node).detach()
            width = next(
                layer.neurons
                for layer in model.layers
                if isinstance(layer, axonal_recdel) and layer.neuron_module is node
            )
            if beta.numel() == 1:
                beta = beta.reshape(1).expand(width).clone()
            factor = 1.0 + torch.randn_like(beta) * condition["beta_cv"]
            beta_new = (beta * factor).clamp(0.01, 0.99)
            beta_delta = (beta_new - beta) / beta.abs().clamp_min(1e-12)
            realized["beta_relative_rms"].append(float(beta_delta.square().mean().sqrt()))
            w_new = torch.log((1.0 - beta_new) / beta_new)
            _replace_parameter(node, "w", w_new)

            threshold = float(node.v_threshold)
            threshold_factor = 1.0 + float(torch.randn(1).item()) * condition["threshold_cv"]
            threshold_new = max(0.01, threshold * threshold_factor)
            realized["threshold_relative_rms"].append(abs(threshold_new - threshold) / max(abs(threshold), 1e-12))
            node.v_threshold = float(threshold_new)

        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                base = layer.weight.detach().clone()
                perturbed = base * (1.0 + torch.randn_like(base) * condition["weight_cv"])
                realized["weight_relative_rms"].append(
                    float(torch.linalg.vector_norm(perturbed - base) / torch.linalg.vector_norm(base).clamp_min(1e-12))
                )
                layer.weight.copy_(perturbed)
            elif isinstance(layer, axonal_recdel):
                base_w = layer.recurrent_weights.detach().clone()
                perturbed_w = base_w * (1.0 + torch.randn_like(base_w) * condition["weight_cv"])
                realized["weight_relative_rms"].append(
                    float(torch.linalg.vector_norm(perturbed_w - base_w) / torch.linalg.vector_norm(base_w).clamp_min(1e-12))
                )
                layer.recurrent_weights.copy_(perturbed_w)

                base_d = layer.recurrent_delays.detach().clone()
                perturbed_d = (base_d * (1.0 + torch.randn_like(base_d) * condition["delay_cv"])).clamp_min(0.0)
                positive = base_d.abs() > 1e-6
                if bool(positive.any()):
                    delay_rel = (perturbed_d[positive] - base_d[positive]) / base_d[positive].abs()
                    realized["delay_relative_rms"].append(float(delay_rel.square().mean().sqrt()))
                else:
                    realized["delay_relative_rms"].append(0.0)
                layer.recurrent_delays.copy_(perturbed_d)
    return realized


def write_state(path: Path, metadata: dict[str, Any], records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[f"{row['mode']}|{row['seed']}|{row['condition']}"].append(row["accuracy"])
    summary = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "runs": len(values),
        }
        for key, values in grouped.items()
    }
    path.write_text(json.dumps({"metadata": metadata, "records": records, "per_checkpoint_summary": summary}, indent=2))


def evaluate_maybe_limited(loader, model, cfg: Config, max_batches: int | None) -> float:
    if max_batches is None:
        return evaluate(loader, model, DEVICE, cfg, show_progress=False)
    original_batches = loader.number_of_batches
    try:
        loader.number_of_batches = min(int(original_batches), int(max_batches))
        return evaluate(loader, model, DEVICE, cfg, show_progress=False)
    finally:
        loader.number_of_batches = original_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["clean", "asar", "noise_aug"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--mc-runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-specs", type=int)
    args = parser.parse_args()
    if set(args.modes) - set(CHECKPOINT_ROOTS):
        raise ValueError(f"Unknown modes: {sorted(set(args.modes) - set(CHECKPOINT_ROOTS))}")
    if set(args.conditions) - set(CONDITIONS):
        raise ValueError(f"Unknown conditions: {sorted(set(args.conditions) - set(CONDITIONS))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    records = json.loads(output_path.read_text()).get("records", []) if output_path.exists() else []
    completed = {(row["mode"], row["seed"], row["condition"], row["trial"]) for row in records}
    metadata = {
        "device": str(DEVICE),
        "modes": args.modes,
        "seeds": args.seeds,
        "conditions": {name: CONDITIONS[name] for name in args.conditions},
        "mc_runs": args.mc_runs,
        "checkpoint_roots": {key: str(value) for key, value in CHECKPOINT_ROOTS.items()},
        "common_random_numbers": "random seed depends only on checkpoint seed, condition, and trial; identical across modes",
    }
    cfg = make_config(args.batch_size)
    cfg.seed = int(args.seeds[0])
    _, _, test_loader = load_dataset(cfg)
    specs_done = 0
    for seed in args.seeds:
        for mode in args.modes:
            base_model, payload = load_model(mode, seed, cfg)
            for condition_name in args.conditions:
                runs = 1 if condition_name == "nominal" else int(args.mc_runs)
                for trial in range(runs):
                    key = (mode, seed, condition_name, trial)
                    if key in completed:
                        continue
                    if args.max_specs is not None and specs_done >= args.max_specs:
                        write_state(output_path, metadata, records)
                        return
                    random_seed = 50_260_724 + seed * 10_000 + list(CONDITIONS).index(condition_name) * 100 + trial
                    set_seeds(random_seed)
                    model = copy.deepcopy(base_model).to(DEVICE).eval()
                    realized = apply_joint_mismatch(model, CONDITIONS[condition_name])
                    started = time.time()
                    accuracy = evaluate_maybe_limited(
                        test_loader, model, cfg, args.max_eval_batches
                    )
                    records.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "condition": condition_name,
                            "trial": trial,
                            "random_seed": random_seed,
                            "accuracy": accuracy,
                            "elapsed_seconds": time.time() - started,
                            "checkpoint_test_acc": float(payload["test_acc"]),
                            "realized": realized,
                        }
                    )
                    specs_done += 1
                    print(f"[ssc] {mode} s{seed} {condition_name} trial={trial} acc={accuracy:.3f}", flush=True)
                    write_state(output_path, metadata, records)
    print(f"Completed {len(records)} records -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
