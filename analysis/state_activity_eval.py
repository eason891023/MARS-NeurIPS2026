#!/usr/bin/env python3
"""Exact-checkpoint state-reset horizon and firing/activity diagnostics for SHD."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "train" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from submitted_eval_common import (  # noqa: E402
    DEVICE,
    build_test_loader,
    load_submitted_model,
)


@torch.inference_mode()
def evaluate_reset_interval(model, loader, interval: int, max_batches: int | None = None) -> float:
    model.eval()
    correct = total = 0
    for batch_idx, (data, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data = data.to(DEVICE, non_blocking=True).float()
        targets = targets.to(DEVICE, non_blocking=True)
        logits_sum = None
        for start in range(0, data.shape[1], interval):
            logits_t, _ = model(data[:, start : start + interval], reg_mode=None)
            chunk_sum = logits_t.sum(0)
            logits_sum = chunk_sum if logits_sum is None else logits_sum + chunk_sum
        correct += int((logits_sum.argmax(1) == targets).sum())
        total += int(targets.numel())
    return 100.0 * correct / max(total, 1)


@torch.inference_mode()
def collect_activity(model, loader, max_batches: int) -> dict[str, Any]:
    per_layer_time: dict[str, list[np.ndarray]] = defaultdict(list)
    raster: dict[str, np.ndarray] = {}
    current: dict[str, list[torch.Tensor]] = defaultdict(list)
    hooks = []

    def hook_for(name: str):
        def hook(_module, _inputs, output):
            spike = output[0] if isinstance(output, tuple) else output
            current[name].append(spike.detach())

        return hook

    for name, layer in model._lif_layers:
        hooks.append(layer.register_forward_hook(hook_for(name)))

    batches = 0
    for batch_idx, (data, _targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        data = data.to(DEVICE, non_blocking=True).float()
        current.clear()
        model(data, reg_mode=None)
        for name, spikes in current.items():
            stacked = torch.stack(spikes, dim=0)
            per_layer_time[name].append(stacked.float().mean(dim=(1, 2)).cpu().numpy())
            if batch_idx == 0:
                raster[name] = stacked[:, 0, : min(64, stacked.shape[-1])].float().cpu().numpy()
        batches += 1

    for hook in hooks:
        hook.remove()
    return {
        "batches": batches,
        "per_layer_time": {
            name: np.mean(np.stack(values, axis=0), axis=0).tolist()
            for name, values in per_layer_time.items()
        },
        "mean_firing_rate": {
            name: float(np.mean(np.stack(values, axis=0)))
            for name, values in per_layer_time.items()
        },
        "raster": {name: value.tolist() for name, value in raster.items()},
    }


def effective_tau(model) -> dict[str, Any]:
    out = {}
    for name, layer in model._lif_layers:
        beta = layer.beta.detach().float().cpu().numpy().reshape(-1)
        tau = -1.0 / np.log(np.clip(beta, 0.01, 0.99))
        out[name] = {
            "beta_mean": float(beta.mean()),
            "tau_over_dt_mean": float(tau.mean()),
            "tau_over_dt_min": float(tau.min()),
            "tau_over_dt_max": float(tau.max()),
        }
    return out


def plot_activity(records: list[dict[str, Any]], output_dir: Path) -> None:
    for row in records:
        if row["seed"] != 42:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for name, values in row["activity"]["per_layer_time"].items():
            ax.plot(values, label=name, linewidth=1.8)
        ax.set_xlabel("Time bin")
        ax.set_ylabel("Mean spike probability")
        ax.set_title(f"{row['architecture'].upper()} {row['mode']} activity")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"activity_{row['architecture']}_{row['mode']}_s42.png", dpi=180)
        plt.close(fig)

        for name, raster in row["activity"]["raster"].items():
            arr = np.asarray(raster)
            fig, ax = plt.subplots(figsize=(7.4, 3.8))
            positions = np.argwhere(arr > 0)
            if positions.size:
                ax.scatter(positions[:, 0], positions[:, 1], s=3, color="black")
            ax.set_xlabel("Time bin")
            ax.set_ylabel("Neuron (first 64)")
            ax.set_title(f"{row['architecture'].upper()} {row['mode']} {name} raster")
            fig.tight_layout()
            fig.savefig(output_dir / f"raster_{row['architecture']}_{row['mode']}_{name}_s42.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", default=["ff", "rsnn"])
    parser.add_argument("--modes", nargs="+", default=["clean", "asar", "noise_aug"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 72, 92, 102])
    parser.add_argument("--reset-intervals", nargs="+", type=int, default=[1, 2, 5, 10, 25, 50])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--activity-batches", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-checkpoints", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    payload = json.loads(output_path.read_text()) if output_path.exists() else {"records": []}
    records: list[dict[str, Any]] = payload["records"]
    completed = {(r["architecture"], r["mode"], r["seed"]) for r in records}
    loader, input_size = build_test_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    done = 0
    for architecture in args.architectures:
        for seed in args.seeds:
            for mode in args.modes:
                key = (architecture, mode, seed)
                if key in completed:
                    continue
                if args.max_checkpoints is not None and done >= args.max_checkpoints:
                    plot_activity(records, args.output_dir)
                    return
                model, checkpoint = load_submitted_model(architecture, seed, mode, input_size)
                horizon = {
                    str(interval): evaluate_reset_interval(
                        model, loader, interval, max_batches=args.max_eval_batches
                    )
                    for interval in args.reset_intervals
                }
                activity = collect_activity(model, loader, max_batches=args.activity_batches)
                row = {
                    "architecture": architecture,
                    "mode": mode,
                    "seed": seed,
                    "checkpoint_test_acc": float(checkpoint["test_acc"]),
                    "reset_interval_accuracy": horizon,
                    "activity": activity,
                    "effective_tau": effective_tau(model),
                }
                records.append(row)
                done += 1
                output_path.write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "device": str(DEVICE),
                                "reset_definition": "The model state is reset at every non-overlapping input chunk of the stated length.",
                                "activity_definition": "Mean spike probability over examples and neurons; rasters use the first test sample and first 64 neurons.",
                            },
                            "records": records,
                        },
                        indent=2,
                    )
                )
                print(f"[activity] {architecture} {mode} s{seed} reset={horizon}", flush=True)
    plot_activity(records, args.output_dir)


if __name__ == "__main__":
    main()
