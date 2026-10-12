#!/usr/bin/env python3
"""Temporal-expressivity diagnostics on exact submitted SHD checkpoints.

Two complementary tests are run:
1) prefix-only and suffix-only context-length accuracy curves;
2) a frozen-state lagged linear probe that decodes past input event rate from
   the second hidden layer's membrane state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
COMMON_DIR = HERE.parent / "train" / "common"
sys.path.insert(0, str(COMMON_DIR))

from submitted_eval_common import (  # noqa: E402
    DATA_ROOT,
    DEVICE,
    build_test_loader,
    evaluate_accuracy,
    load_submitted_model,
    set_all_seeds,
)


def context_transform(kind: str, length: int):
    def transform(data: torch.Tensor, _seed: int) -> torch.Tensor:
        transformed = torch.zeros_like(data)
        if kind == "prefix":
            transformed[:, :length] = data[:, :length]
        elif kind == "suffix":
            transformed[:, -length:] = data[:, -length:]
        else:
            raise ValueError(kind)
        return transformed

    return transform


def build_probe_loaders(
    *,
    time_steps: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    sensor_size = tonic.datasets.SHD.sensor_size
    transform = transforms.Compose(
        [transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps)]
    )
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    train_ds = tonic.datasets.SHD(
        save_to=str(DATA_ROOT), train=True, transform=transform
    )
    test_ds = tonic.datasets.SHD(
        save_to=str(DATA_ROOT), train=False, transform=transform
    )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(train_ds, **kwargs), DataLoader(test_ds, **kwargs)


@torch.inference_mode()
def collect_lag_pairs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    architecture: str,
    lags: list[int],
    max_batches: int,
    max_rows_per_lag: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    layer = model.lif2 if architecture == "ff" else model.rlif2
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        # snn.Leaky/RLeaky return (spike, membrane).
        captured.append(output[-1].detach())

    handle = layer.register_forward_hook(hook)
    features: dict[int, list[np.ndarray]] = {lag: [] for lag in lags}
    targets: dict[int, list[np.ndarray]] = {lag: [] for lag in lags}
    row_counts = {lag: 0 for lag in lags}
    try:
        model.eval()
        for batch_index, (data, _labels) in enumerate(loader):
            if batch_index >= max_batches:
                break
            if all(row_counts[lag] >= max_rows_per_lag for lag in lags):
                break
            data = data.to(DEVICE, non_blocking=True).float()
            captured.clear()
            model(data, reg_mode=None)
            state = torch.stack(captured, dim=1)  # [B, T, H]
            counts = data.flatten(2).sum(2)
            for lag in lags:
                if row_counts[lag] >= max_rows_per_lag:
                    continue
                x = state[:, lag:, :].reshape(-1, state.shape[-1])
                # log1p stabilizes the strongly skewed event-count target.
                y = torch.log1p(counts[:, :-lag]).reshape(-1)
                remaining = max_rows_per_lag - row_counts[lag]
                x = x[:remaining].float().cpu().numpy()
                y = y[:remaining].float().cpu().numpy()
                features[lag].append(x)
                targets[lag].append(y)
                row_counts[lag] += len(y)
    finally:
        handle.remove()
    return {
        lag: (np.concatenate(features[lag]), np.concatenate(targets[lag]))
        for lag in lags
    }


def ridge_probe(
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    ridge: float,
) -> dict[str, float]:
    x_train, y_train = train
    x_test, y_test = test
    x_mean = x_train.mean(0, keepdims=True)
    x_std = x_train.std(0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(y_train.mean())
    x_train = ((x_train - x_mean) / x_std).astype(np.float64)
    x_test = ((x_test - x_mean) / x_std).astype(np.float64)
    y_center = y_train.astype(np.float64) - y_mean
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += ridge
    coef = np.linalg.solve(gram, x_train.T @ y_center)
    pred = x_test @ coef + y_mean
    residual = float(np.sum((y_test - pred) ** 2))
    total = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - residual / max(total, 1e-12)
    corr = float(np.corrcoef(y_test, pred)[0, 1])
    return {
        "r2": r2,
        "pearson_r": corr,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
    }


def write_state(path: Path, metadata: dict, records: list[dict]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row["test"] == "context":
            key = "|".join(
                [
                    row["architecture"],
                    row["mode"],
                    row["kind"],
                    str(row["length"]),
                ]
            )
            grouped[key].append(row["accuracy"])
    summary = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "seeds": len(values),
        }
        for key, values in grouped.items()
    }
    path.write_text(
        json.dumps(
            {"metadata": metadata, "records": records, "context_summary": summary},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", default=["ff", "rsnn"])
    parser.add_argument("--modes", nargs="+", default=["clean", "asar", "noise_aug"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 72, 92, 102])
    parser.add_argument("--lengths", nargs="+", type=int, default=[5, 10, 20, 30, 40, 50])
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 5, 10, 20])
    parser.add_argument("--time-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--probe-train-batches", type=int, default=8)
    parser.add_argument("--probe-test-batches", type=int, default=4)
    parser.add_argument("--probe-max-rows", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--max-models", type=int)
    args = parser.parse_args()

    if any(length <= 0 or length > args.time_steps for length in args.lengths):
        raise ValueError("Context lengths must be in [1, time_steps]")
    if any(lag <= 0 or lag >= args.time_steps for lag in args.lags):
        raise ValueError("Probe lags must be in [1, time_steps-1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    records: list[dict] = []
    if output_path.exists():
        records = json.loads(output_path.read_text()).get("records", [])
    completed_context = {
        (r["architecture"], r["mode"], r["seed"], r["kind"], r["length"])
        for r in records
        if r["test"] == "context"
    }
    completed_probe = {
        (r["architecture"], r["mode"], r["seed"], r["lag"])
        for r in records
        if r["test"] == "lag_probe"
    }
    metadata = {
        "device": str(DEVICE),
        "architectures": args.architectures,
        "modes": args.modes,
        "seeds": args.seeds,
        "context_lengths": args.lengths,
        "probe_lags": args.lags,
        "context_test": "zero all bins outside the retained prefix or suffix",
        "probe": (
            "ridge decoder of log1p past input event count from the frozen second "
            "hidden-layer membrane state; probe weights only are fitted"
        ),
        "ridge": args.ridge,
    }
    test_loader, input_size = build_test_loader(
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_probe_loader, test_probe_loader = build_probe_loaders(
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    models_done = 0
    for architecture in args.architectures:
        for seed in args.seeds:
            for mode in args.modes:
                if args.max_models is not None and models_done >= args.max_models:
                    write_state(output_path, metadata, records)
                    return
                set_all_seeds(25_072_500 + seed)
                model, checkpoint = load_submitted_model(
                    architecture, seed, mode, input_size
                )
                for kind in ["prefix", "suffix"]:
                    for length in args.lengths:
                        key = (architecture, mode, seed, kind, length)
                        if key in completed_context:
                            continue
                        started = time.time()
                        accuracy = evaluate_accuracy(
                            model,
                            test_loader,
                            input_transform=context_transform(kind, length),
                            max_batches=args.max_eval_batches,
                        )
                        records.append(
                            {
                                "test": "context",
                                "architecture": architecture,
                                "mode": mode,
                                "seed": seed,
                                "kind": kind,
                                "length": length,
                                "accuracy": accuracy,
                                "elapsed_seconds": time.time() - started,
                                "checkpoint_test_acc": float(checkpoint["test_acc"]),
                            }
                        )
                        print(
                            f"[context] {architecture} {mode} s{seed} "
                            f"{kind}{length} acc={accuracy:.3f}",
                            flush=True,
                        )
                        write_state(output_path, metadata, records)

                pending_lags = [
                    lag
                    for lag in args.lags
                    if (architecture, mode, seed, lag) not in completed_probe
                ]
                if pending_lags:
                    started = time.time()
                    train_pairs = collect_lag_pairs(
                        model,
                        train_probe_loader,
                        architecture=architecture,
                        lags=pending_lags,
                        max_batches=args.probe_train_batches,
                        max_rows_per_lag=args.probe_max_rows,
                    )
                    test_pairs = collect_lag_pairs(
                        model,
                        test_probe_loader,
                        architecture=architecture,
                        lags=pending_lags,
                        max_batches=args.probe_test_batches,
                        max_rows_per_lag=args.probe_max_rows,
                    )
                    for lag in pending_lags:
                        metrics = ridge_probe(
                            train_pairs[lag], test_pairs[lag], ridge=args.ridge
                        )
                        records.append(
                            {
                                "test": "lag_probe",
                                "architecture": architecture,
                                "mode": mode,
                                "seed": seed,
                                "lag": lag,
                                **metrics,
                                "elapsed_seconds_group": time.time() - started,
                            }
                        )
                        print(
                            f"[probe] {architecture} {mode} s{seed} lag={lag} "
                            f"R2={metrics['r2']:.4f} r={metrics['pearson_r']:.4f}",
                            flush=True,
                        )
                        write_state(output_path, metadata, records)
                models_done += 1
    print(f"Completed {len(records)} records -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
