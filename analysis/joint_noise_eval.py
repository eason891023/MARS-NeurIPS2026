#!/usr/bin/env python3
"""Joint device-mismatch evaluation of exact submitted SHD checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "common"))

from submitted_eval_common import (  # noqa: E402
    DEVICE,
    apply_device_mismatch,
    build_test_loader,
    clone_model,
    evaluate_accuracy,
    load_submitted_model,
    set_all_seeds,
)


DEFAULT_CONDITIONS = {
    "nominal": {"beta_cv": 0.0, "threshold_cv": 0.0, "weight_cv": 0.0},
    "beta05": {"beta_cv": 0.05, "threshold_cv": 0.0, "weight_cv": 0.0},
    "beta20": {"beta_cv": 0.20, "threshold_cv": 0.0, "weight_cv": 0.0},
    "threshold05": {"beta_cv": 0.0, "threshold_cv": 0.05, "weight_cv": 0.0},
    "joint05": {"beta_cv": 0.05, "threshold_cv": 0.05, "weight_cv": 0.05},
    "beta20_joint_aux05": {
        "beta_cv": 0.20,
        "threshold_cv": 0.05,
        "weight_cv": 0.05,
    },
}


def write_state(path: Path, metadata: dict, records: list[dict]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        key = "|".join(
            [
                row["architecture"],
                row["mode"],
                str(row["seed"]),
                row["condition"],
            ]
        )
        grouped[key].append(row["accuracy"])
    summary = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "runs": len(values),
        }
        for key, values in grouped.items()
    }
    path.write_text(
        json.dumps(
            {"metadata": metadata, "records": records, "per_checkpoint_summary": summary},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["ff", "rsnn"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["asar", "noise_aug", "clean"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 72, 92, 102])
    parser.add_argument(
        "--conditions", nargs="+", default=list(DEFAULT_CONDITIONS)
    )
    parser.add_argument("--mc-runs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-specs", type=int)
    args = parser.parse_args()
    unknown = sorted(set(args.conditions) - set(DEFAULT_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    records: list[dict] = []
    if output_path.exists():
        records = json.loads(output_path.read_text()).get("records", [])
    completed = {
        (r["mode"], r["seed"], r["condition"], r["trial"]) for r in records
    }
    metadata = {
        "architecture": args.architecture,
        "device": str(DEVICE),
        "modes": args.modes,
        "seeds": args.seeds,
        "conditions": {k: DEFAULT_CONDITIONS[k] for k in args.conditions},
        "mc_runs": args.mc_runs,
        "batch_size": args.batch_size,
        "max_eval_batches": args.max_eval_batches,
        "common_random_numbers": (
            "The same seed/condition/trial realization is used across methods."
        ),
        "inference_granularity": (
            "Independent training seeds; mismatch MC runs are nested within seed."
        ),
    }
    loader, input_size = build_test_loader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    specs_done = 0
    for seed in args.seeds:
        for mode in args.modes:
            base_model, checkpoint = load_submitted_model(
                args.architecture, seed, mode, input_size
            )
            for condition_name in args.conditions:
                condition = DEFAULT_CONDITIONS[condition_name]
                runs = 1 if condition_name == "nominal" else args.mc_runs
                for trial in range(runs):
                    spec_key = (mode, seed, condition_name, trial)
                    if spec_key in completed:
                        continue
                    if args.max_specs is not None and specs_done >= args.max_specs:
                        write_state(output_path, metadata, records)
                        return
                    random_seed = (
                        20_260_724
                        + seed * 10_000
                        + list(DEFAULT_CONDITIONS).index(condition_name) * 100
                        + trial
                    )
                    set_all_seeds(random_seed)
                    model = clone_model(base_model)
                    realized = apply_device_mismatch(model, **condition)
                    started = time.time()
                    accuracy = evaluate_accuracy(
                        model, loader, max_batches=args.max_eval_batches
                    )
                    row = {
                        "architecture": args.architecture,
                        "mode": mode,
                        "seed": seed,
                        "condition": condition_name,
                        "trial": trial,
                        "random_seed": random_seed,
                        "accuracy": accuracy,
                        "elapsed_seconds": time.time() - started,
                        "checkpoint_test_acc": float(checkpoint["test_acc"]),
                        "requested_cv": condition,
                        "realized": realized,
                    }
                    records.append(row)
                    specs_done += 1
                    print(
                        f"[{args.architecture}] {mode} s{seed} {condition_name} "
                        f"trial={trial} acc={accuracy:.3f}",
                        flush=True,
                    )
                    write_state(output_path, metadata, records)
                    del model
    print(f"Completed {len(records)} records -> {output_path}", flush=True)


if __name__ == "__main__":
    main()

