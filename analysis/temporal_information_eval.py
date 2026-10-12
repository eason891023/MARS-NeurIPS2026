#!/usr/bin/env python3
"""Test whether submitted models use temporal order beyond total event rate."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "common"))

from submitted_eval_common import (  # noqa: E402
    DEVICE,
    build_test_loader,
    evaluate_accuracy,
    load_submitted_model,
)


def original(data: torch.Tensor, _: int) -> torch.Tensor:
    return data


def reverse_time(data: torch.Tensor, _: int) -> torch.Tensor:
    return torch.flip(data, dims=[1])


def shuffle_time(data: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=data.device).manual_seed(seed)
    permutation = torch.randperm(data.shape[1], generator=generator, device=data.device)
    return data[:, permutation]


def rate_repeat(data: torch.Tensor, _: int) -> torch.Tensor:
    return data.mean(dim=1, keepdim=True).expand_as(data)


TRANSFORMS = {
    "original": original,
    "reverse_time": reverse_time,
    "shuffle_time": shuffle_time,
    "rate_repeat": rate_repeat,
}


def write_state(path: Path, metadata: dict, records: list[dict]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        key = "|".join(
            [row["architecture"], row["mode"], str(row["seed"]), row["transform"]]
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
    parser.add_argument("--transforms", nargs="+", default=list(TRANSFORMS))
    parser.add_argument("--shuffle-runs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-specs", type=int)
    args = parser.parse_args()
    unknown = sorted(set(args.transforms) - set(TRANSFORMS))
    if unknown:
        raise ValueError(f"Unknown transforms: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    records: list[dict] = []
    if output_path.exists():
        records = json.loads(output_path.read_text()).get("records", [])
    completed = {(r["mode"], r["seed"], r["transform"], r["trial"]) for r in records}
    metadata = {
        "architecture": args.architecture,
        "device": str(DEVICE),
        "modes": args.modes,
        "seeds": args.seeds,
        "transforms": args.transforms,
        "shuffle_runs": args.shuffle_runs,
        "preservation_notes": {
            "reverse_time": "Preserves every event count and reverses temporal order.",
            "shuffle_time": (
                "Uses a common global time permutation per batch; preserves every "
                "sample's per-channel total event count."
            ),
            "rate_repeat": (
                "Repeats the temporal mean at every step, exactly preserving total "
                "input mass but producing fractional frames."
            ),
        },
    }
    loader, input_size = build_test_loader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    specs_done = 0
    for seed in args.seeds:
        for mode in args.modes:
            model, checkpoint = load_submitted_model(
                args.architecture, seed, mode, input_size
            )
            for transform_name in args.transforms:
                runs = args.shuffle_runs if transform_name == "shuffle_time" else 1
                for trial in range(runs):
                    spec_key = (mode, seed, transform_name, trial)
                    if spec_key in completed:
                        continue
                    if args.max_specs is not None and specs_done >= args.max_specs:
                        write_state(output_path, metadata, records)
                        return
                    transform_seed = (
                        20_260_724
                        + seed * 10_000
                        + list(TRANSFORMS).index(transform_name) * 100
                        + trial
                    )
                    started = time.time()
                    accuracy = evaluate_accuracy(
                        model,
                        loader,
                        input_transform=TRANSFORMS[transform_name],
                        transform_seed=transform_seed,
                        max_batches=args.max_eval_batches,
                    )
                    row = {
                        "architecture": args.architecture,
                        "mode": mode,
                        "seed": seed,
                        "transform": transform_name,
                        "trial": trial,
                        "transform_seed": transform_seed,
                        "accuracy": accuracy,
                        "elapsed_seconds": time.time() - started,
                        "checkpoint_test_acc": float(checkpoint["test_acc"]),
                    }
                    records.append(row)
                    specs_done += 1
                    print(
                        f"[{args.architecture}] {mode} s{seed} {transform_name} "
                        f"trial={trial} acc={accuracy:.3f}",
                        flush=True,
                    )
                    write_state(output_path, metadata, records)
    print(f"Completed {len(records)} records -> {output_path}", flush=True)


if __name__ == "__main__":
    main()

