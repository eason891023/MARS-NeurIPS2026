#!/usr/bin/env python3
"""Temporal-resolution and context tests on submitted SSC checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DELREC = ROOT / "third_party" / "DelRec"
if str(DELREC) not in sys.path:
    sys.path.insert(0, str(DELREC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import SSC_ROOT, ckpt_dir  # noqa: E402

from configs.perf_SSC_p0_spikingjelly import Config  # noqa: E402
from src.SSC.snn import SNN_recurrent_delays  # noqa: E402
from src.SSC.trainer_p0_spikingjelly import load_checkpoint_snapshot  # noqa: E402
from src.datasets import load_dataset  # noqa: E402
from src.utils import calc_metric_SSC, reset_states  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 52, 72, 92, 102]
CHECKPOINT_ROOTS = {mode: ckpt_dir("SSC") for mode in ("clean", "asar", "noise_aug")}


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


def load_model(mode: str, seed: int, cfg: Config) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(
        checkpoint_path(mode, seed), map_location="cpu", weights_only=False
    )
    model = SNN_recurrent_delays(cfg)
    load_checkpoint_snapshot(model, payload)
    return model.to(DEVICE).eval(), payload


def make_transform(condition: str) -> Callable[[torch.Tensor, int], torch.Tensor]:
    if condition == "original":
        return lambda data, _seed: data
    if condition == "reverse":
        return lambda data, _seed: torch.flip(data, dims=[1])
    if condition == "rate_repeat":
        return lambda data, _seed: data.mean(dim=1, keepdim=True).expand_as(data)
    if condition == "shuffle":
        def shuffle(data: torch.Tensor, seed: int) -> torch.Tensor:
            generator = torch.Generator(device=data.device).manual_seed(seed)
            permutation = torch.randperm(
                data.shape[1], device=data.device, generator=generator
            )
            return data[:, permutation, :]

        return shuffle
    if condition.startswith("prefix"):
        length = int(condition.removeprefix("prefix"))

        def prefix(data: torch.Tensor, _seed: int) -> torch.Tensor:
            output = torch.zeros_like(data)
            output[:, :length] = data[:, :length]
            return output

        return prefix
    if condition.startswith("suffix"):
        length = int(condition.removeprefix("suffix"))

        def suffix(data: torch.Tensor, _seed: int) -> torch.Tensor:
            output = torch.zeros_like(data)
            output[:, -length:] = data[:, -length:]
            return output

        return suffix
    if condition.startswith("jitter"):
        radius = int(condition.removeprefix("jitter"))

        def jitter(data: torch.Tensor, seed: int) -> torch.Tensor:
            generator = torch.Generator(device=data.device).manual_seed(seed)
            shifts = torch.randint(
                -radius,
                radius + 1,
                data.shape,
                device=data.device,
                generator=generator,
            )
            output = torch.zeros_like(data)
            time_steps = data.shape[1]
            for offset in range(-radius, radius + 1):
                if offset >= 0:
                    source = data[:, : time_steps - offset]
                    mask = shifts[:, : time_steps - offset] == offset
                    output[:, offset:] += source * mask
                else:
                    source = data[:, -offset:]
                    mask = shifts[:, -offset:] == offset
                    output[:, : time_steps + offset] += source * mask
            return output

        return jitter
    if condition.startswith("local_shuffle"):
        window = int(condition.removeprefix("local_shuffle"))

        def local_shuffle(data: torch.Tensor, seed: int) -> torch.Tensor:
            generator = torch.Generator(device=data.device).manual_seed(seed)
            output = data.clone()
            for start in range(0, data.shape[1], window):
                stop = min(start + window, data.shape[1])
                permutation = torch.randperm(
                    stop - start, device=data.device, generator=generator
                )
                output[:, start:stop] = data[:, start:stop][:, permutation]
            return output

        return local_shuffle
    raise ValueError(f"Unknown condition: {condition}")


@torch.inference_mode()
def evaluate_transformed(
    loader,
    model: torch.nn.Module,
    *,
    transform: Callable[[torch.Tensor, int], torch.Tensor],
    transform_seed: int,
    max_batches: int | None,
) -> float:
    model.eval()
    if hasattr(loader, "reset"):
        loader.reset()
    correct = 0
    total = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.float().to(DEVICE)
        targets = targets.to(DEVICE)
        inputs = transform(inputs, transform_seed + batch_index)
        reset_states(model=model)
        outputs = model(inputs.permute(1, 0, 2))
        correct += calc_metric_SSC(outputs, targets)
        total += int(targets.size(0))
    return 100.0 * correct / max(total, 1)


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def write_state(path: Path, metadata: dict, records: list[dict]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[f"{row['mode']}|{row['condition']}"].append(float(row["accuracy"]))
    summary = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "seeds": len(values),
        }
        for key, values in grouped.items()
    }
    atomic_write(
        path,
        {"metadata": metadata, "records": records, "condition_summary": summary},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modes", nargs="+", default=["clean", "asar", "noise_aug"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[50, 100, 150, 200, 250],
    )
    parser.add_argument(
        "--jitter-radii", nargs="+", type=int, default=[1, 2, 5, 10]
    )
    parser.add_argument(
        "--local-shuffle-windows", nargs="+", type=int, default=[2, 5, 10, 25]
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Optional explicit subset of generated conditions",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-models", type=int)
    args = parser.parse_args()

    if set(args.modes) - set(CHECKPOINT_ROOTS):
        raise ValueError(f"Unknown modes: {set(args.modes) - set(CHECKPOINT_ROOTS)}")
    if any(length <= 0 or length > 250 for length in args.context_lengths):
        raise ValueError("SSC context lengths must be in [1, 250]")
    generated_conditions = ["original", "reverse", "shuffle", "rate_repeat"]
    generated_conditions += [f"prefix{value}" for value in args.context_lengths]
    generated_conditions += [f"suffix{value}" for value in args.context_lengths]
    generated_conditions += [f"jitter{value}" for value in args.jitter_radii]
    generated_conditions += [
        f"local_shuffle{value}" for value in args.local_shuffle_windows
    ]
    conditions = args.conditions or generated_conditions
    unknown_conditions = sorted(set(conditions) - set(generated_conditions))
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    records: list[dict] = []
    if output_path.exists():
        records = json.loads(output_path.read_text()).get("records", [])
    completed = {
        (row["mode"], int(row["seed"]), row["condition"]) for row in records
    }
    metadata = {
        "device": str(DEVICE),
        "modes": args.modes,
        "seeds": args.seeds,
        "conditions": conditions,
        "time_steps": 250,
        "dataset": "SSC, 35-class spoken command recognition",
        "jitter": (
            "each occupied time-bin/channel count is independently reassigned "
            "within the requested radius; out-of-range assignments are dropped"
        ),
        "local_shuffle": (
            "random permutation within non-overlapping temporal windows, "
            "preserving coarse window order"
        ),
        "common_transform": (
            "condition and batch determine transform RNG; identical across "
            "training methods and checkpoint seeds"
        ),
        "inference_unit": "independent training seed",
    }

    cfg = make_config(args.batch_size)
    _, _, test_loader = load_dataset(cfg)
    models_done = 0
    for seed in args.seeds:
        for mode in args.modes:
            if args.max_models is not None and models_done >= args.max_models:
                write_state(output_path, metadata, records)
                return
            set_seeds(33_072_500 + seed)
            model, payload = load_model(mode, seed, cfg)
            for condition_index, condition in enumerate(conditions):
                key = (mode, seed, condition)
                if key in completed:
                    continue
                transform_seed = 33_500_000 + condition_index * 10_000
                started = time.time()
                accuracy = evaluate_transformed(
                    test_loader,
                    model,
                    transform=make_transform(condition),
                    transform_seed=transform_seed,
                    max_batches=args.max_eval_batches,
                )
                records.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "condition": condition,
                        "accuracy": accuracy,
                        "elapsed_seconds": time.time() - started,
                        "checkpoint_test_acc": float(payload["test_acc"]),
                        "transform_seed": transform_seed,
                    }
                )
                print(
                    f"[ssc-temporal] {mode} s{seed} {condition} "
                    f"acc={accuracy:.3f}",
                    flush=True,
                )
                write_state(output_path, metadata, records)
            models_done += 1
            del model
    print(f"Completed {len(records)} records -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
