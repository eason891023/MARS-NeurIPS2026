#!/usr/bin/env python3
"""Benchmark exact-setting training-step overhead for five submitted models."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from snntorch import surrogate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import SSC_ROOT  # noqa: E402
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SETTING_SOURCES = {
    "shd_ff": ROOT / "train" / "shd_fc.py",
    "shd_rsnn": ROOT / "train" / "shd_rsnn.py",
    "dvs_gesture": ROOT / "train" / "dvs_gesture.py",
    "cifar10": ROOT / "train" / "cifar10.py",
}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_real_dvs_training_batch(source, cfg: dict, seed: int = 42):
    """Load one real DVS-Gesture batch with the submitted Tonic preprocessing.

    The overhead benchmark only consumes a training batch, so this reproduces
    the submitted training transform, split, and shuffle without constructing
    the unused test loader (and therefore without requiring the test archive).
    """
    sensor_size = source.tonic.datasets.DVSGesture.sensor_size
    transform = source.transforms.Compose(
        [
            source.transforms.ToFrame(
                sensor_size=sensor_size,
                n_time_bins=cfg["time_steps"],
            ),
        ]
    )
    os.makedirs(cfg["data_root"], exist_ok=True)
    train_ds = source.tonic.datasets.DVSGesture(
        save_to=cfg["data_root"],
        train=True,
        transform=transform,
    )
    total = len(train_ds)
    val_size = max(1, int(round(total * cfg["val_ratio"])))
    split_gen = torch.Generator().manual_seed(seed)
    train_sub, _val_sub = source.random_split(
        train_ds,
        [total - val_size, val_size],
        generator=split_gen,
    )
    train_gen = torch.Generator().manual_seed(seed)
    train_loader = source.DataLoader(
        train_sub,
        shuffle=True,
        drop_last=True,
        **source._loader_kwargs(cfg, generator=train_gen),
    )
    return next(iter(train_loader))


def prepare_snntorch(setting: str, mode: str):
    source = import_file(f"overhead_{setting}_{mode}", SETTING_SOURCES[setting])
    cfg = dict(source.CONFIG)
    # Loader worker persistence is undesirable for a one-batch benchmark.
    cfg["num_workers"] = 0
    cfg["show_progress"] = False
    batch_source = "real_submitted_training_loader"
    if setting == "dvs_gesture":
        data, targets = make_real_dvs_training_batch(source, cfg)
        loader_outputs = None
    else:
        loader_outputs = source.build_loaders(42, cfg)
        train_loader, _val_loader, _test_loader = loader_outputs[:3]
        data, targets = next(iter(train_loader))
    data = data.to(DEVICE).float()
    targets = targets.to(DEVICE)
    spike_grad = surrogate.atan(alpha=cfg["surrogate_alpha"])

    if setting == "shd_ff":
        assert loader_outputs is not None
        input_size = int(loader_outputs[3])
        model = source.SHDSNN(
            input_size=input_size,
            beta_init=cfg["beta_init"],
            threshold_init=cfg["threshold_init"],
            surrogate_fn=spike_grad,
            learn_beta=cfg["learn_beta"],
            learn_threshold=cfg["learn_threshold"],
        )
    elif setting == "shd_rsnn":
        assert loader_outputs is not None
        input_size = int(loader_outputs[3])
        model = source.SHDRSNN(
            input_size=input_size,
            beta_init=cfg["beta_init"],
            threshold_init=cfg["threshold_init"],
            surrogate_fn=spike_grad,
            learn_beta=cfg["learn_beta"],
            learn_threshold=cfg["learn_threshold"],
            hidden=cfg["hidden"],
        )
    elif setting == "dvs_gesture":
        model = source.DVSGestureSNN(
            num_classes=cfg["num_classes"],
            beta=cfg["beta_init"],
            threshold=cfg["threshold_init"],
            surrogate_fn=spike_grad,
            learn_beta=cfg["learn_beta"],
            learn_threshold=cfg["learn_threshold"],
        )
    elif setting == "cifar10":
        model = source.SpikingResNet18(
            num_classes=10,
            beta=cfg["beta_init"],
            threshold=cfg["threshold_init"],
            surrogate_fn=spike_grad,
            learn_beta=cfg["learn_beta"],
            learn_threshold=cfg["learn_threshold"],
            num_steps=cfg["time_steps"],
        )
    else:
        raise ValueError(setting)
    model = model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    def step() -> float:
        beta_originals = None
        if mode == "noise_aug":
            beta_originals = source._perturb_betas_for_training(
                model, cfg["noise_aug_cv"]
            )
        reg_mode = "asar" if mode == "asar" else None
        logits_t, loss_reg = model(data, reg_mode=reg_mode)
        logits = logits_t.sum(0)
        loss_task = criterion(logits, targets)
        lam = cfg["lambda_asar"] if mode == "asar" else 0.0
        loss = loss_task + lam * loss_reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if beta_originals is not None:
            source._restore_betas(model, beta_originals)
        optimizer.step()
        with torch.no_grad():
            for _, lif in model._lif_layers:
                lif.beta.clamp_(0.01, 0.99)
        return float(loss.detach())

    meta = {
        "batch_size": int(targets.shape[0]),
        "time_steps": int(cfg["time_steps"]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "batch_source": batch_source,
    }
    return model, step, meta


def prepare_ssc(mode: str):
    delrec = ROOT / "third_party" / "DelRec"
    sys.path.insert(0, str(delrec))
    from configs.perf_SSC_p0_spikingjelly import Config
    from src.SSC.snn import SNN_recurrent_delays
    from src.SSC.trainer_p0_spikingjelly import (
        clamp_plif_betas,
        compute_regularizer_loss,
        get_regularizer,
        init_optim_sche,
        perturb_betas_for_training,
        restore_betas,
    )
    from src.datasets import load_dataset
    from src.utils import calc_loss_SSC, reset_states

    cfg = Config()
    cfg.datasets_path = str(SSC_ROOT)
    cfg.seed = 42
    train_loader, _valid_loader, _test_loader = load_dataset(cfg)
    inputs, targets = next(iter(train_loader))
    inputs = inputs.permute(1, 0, 2).float().to(DEVICE)
    targets = targets.to(DEVICE)
    model = SNN_recurrent_delays(cfg).to(DEVICE).train()
    optimizers, _schedulers = init_optim_sche(model, cfg)
    regularizer = get_regularizer(cfg, mode)

    def step() -> float:
        beta_originals = None
        if mode == "noise_aug":
            beta_originals = perturb_betas_for_training(model, cfg.noise_aug_cv)
        reset_states(model=model)
        outputs = model(inputs)
        loss_task = calc_loss_SSC(outputs, targets)
        loss_reg = compute_regularizer_loss(model, cfg, mode, regularizer)
        lam = cfg.lambda_asar if mode == "asar" else 0.0
        loss = loss_task + lam * loss_reg
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if beta_originals is not None:
            restore_betas(model, beta_originals)
        for optimizer in optimizers:
            optimizer.step()
        clamp_plif_betas(model)
        model.clamp_delays()
        return float(loss.detach())

    meta = {
        "batch_size": int(targets.shape[0]),
        "time_steps": int(inputs.shape[0]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }
    return model, step, meta


def measure(
    step: Callable[[], float],
    *,
    warmup: int,
    repeats: int,
) -> dict:
    for _ in range(warmup):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = []
    incremental_peaks = []
    absolute_peaks = []
    losses = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
        else:
            before = 0
        started = time.perf_counter()
        losses.append(step())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
        else:
            peak = 0
        seconds.append(time.perf_counter() - started)
        incremental_peaks.append(peak - before)
        absolute_peaks.append(peak)
    return {
        "seconds_per_step_mean": float(np.mean(seconds)),
        "seconds_per_step_std": float(np.std(seconds, ddof=1))
        if len(seconds) > 1
        else 0.0,
        "steps_per_second": float(1.0 / np.mean(seconds)),
        "incremental_peak_gib_mean": float(np.mean(incremental_peaks) / 2**30),
        "absolute_peak_gib_max": float(np.max(absolute_peaks) / 2**30),
        "loss_last": losses[-1],
        "repeats": repeats,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["shd_ff", "shd_rsnn", "ssc_delrec", "dvs_gesture", "cifar10"],
    )
    parser.add_argument("--modes", nargs="+", default=["clean", "asar", "noise_aug"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    valid = {*SETTING_SOURCES, "ssc_delrec"}
    unknown = sorted(set(args.settings) - valid)
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    state = {
        "metadata": {
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "protocol": (
                "exact submitted architecture, real training batch, forward + "
                "regularizer/noise augmentation + backward + gradient handling + "
                "Adam update; excludes data loading"
            ),
            "settings": args.settings,
            "modes": args.modes,
        },
        "records": [],
    }
    if output_path.exists():
        state = json.loads(output_path.read_text())
    completed = {(r["setting"], r["mode"]) for r in state["records"]}

    for setting in args.settings:
        for mode in args.modes:
            if (setting, mode) in completed:
                continue
            seed_all(42)
            print(f"[overhead] preparing {setting} {mode}", flush=True)
            try:
                if setting == "ssc_delrec":
                    model, step, meta = prepare_ssc(mode)
                else:
                    model, step, meta = prepare_snntorch(setting, mode)
                metrics = measure(step, warmup=args.warmup, repeats=args.repeats)
                row = {
                    "setting": setting,
                    "mode": mode,
                    "status": "ok",
                    **meta,
                    **metrics,
                }
            except torch.cuda.OutOfMemoryError as exc:
                row = {
                    "setting": setting,
                    "mode": mode,
                    "status": "cuda_oom_at_exact_batch_size",
                    "error": str(exc),
                }
            state["records"].append(row)
            output_path.write_text(json.dumps(state, indent=2))
            print(f"[overhead] {json.dumps(row)}", flush=True)
            if "model" in locals():
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Completed -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
