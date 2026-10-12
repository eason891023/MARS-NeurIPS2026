#!/usr/bin/env python3
"""Train simple beta controls under the exact submitted SHD protocol."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
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
    load_source_module,
)


VALID_MODES = {
    "beta_l2",
    "beta_clip",
    "fixed_homogeneous",
    "fixed_heterogeneous",
    "dropconnect",
}


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def parse_strings(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def widths_for(model: nn.Module) -> list[int]:
    return [model.fc1.out_features, model.fc2.out_features, model.fc3.out_features]


def make_model(source, architecture: str, cfg: dict[str, Any], mode: str, seed: int) -> nn.Module:
    learn_beta = mode not in {"fixed_homogeneous", "fixed_heterogeneous"}
    kwargs = {
        "input_size": cfg["input_size"],
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
    model = model.to(DEVICE)

    if mode == "fixed_heterogeneous":
        generator = torch.Generator(device=DEVICE).manual_seed(70_000 + seed)
        for (_, layer), width in zip(model._lif_layers, widths_for(model)):
            base = torch.full((width,), float(cfg["beta_init"]), device=DEVICE)
            factor = 1.0 + torch.randn(width, generator=generator, device=DEVICE) * float(cfg["fixed_hetero_cv"])
            value = (base * factor).clamp(float(cfg["clip_low"]), float(cfg["clip_high"]))
            _replace_registered_tensor(layer, "beta", value)
    return model


@torch.inference_mode()
def evaluate(model: nn.Module, loader, max_batches: int = 0) -> float:
    model.eval()
    correct = total = 0
    for batch_idx, (data, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        data = data.to(DEVICE, non_blocking=True).float()
        targets = targets.to(DEVICE, non_blocking=True)
        logits_t, _ = model(data, reg_mode=None)
        correct += int((logits_t.sum(0).argmax(1) == targets).sum())
        total += int(targets.numel())
    return 100.0 * correct / max(total, 1)


def beta_l2(model: nn.Module) -> torch.Tensor:
    return torch.cat([layer.beta.reshape(-1) for _, layer in model._lif_layers]).square().mean()


def apply_dropconnect(model: nn.Module, probability: float) -> list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]]:
    saved = []
    keep = 1.0 - probability
    with torch.no_grad():
        for name, layer in model._lif_layers:
            if not name.startswith("rlif"):
                continue
            parameter = layer.recurrent.weight
            original = parameter.detach().clone()
            mask = torch.bernoulli(torch.full_like(parameter, keep)) / keep
            parameter.mul_(mask)
            saved.append((parameter, original, mask))
    return saved


def restore_dropconnect(saved: list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]]) -> None:
    with torch.no_grad():
        for parameter, original, mask in saved:
            if parameter.grad is not None:
                parameter.grad.mul_(mask)
            parameter.copy_(original)


def train_one(
    model: nn.Module,
    loaders,
    cfg: dict[str, Any],
    mode: str,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        correct = total = 0
        task_sum = regularizer_sum = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            if int(cfg["max_train_batches"]) > 0 and batch_idx >= int(cfg["max_train_batches"]):
                break
            data = data.to(DEVICE, non_blocking=True).float()
            targets = targets.to(DEVICE, non_blocking=True)
            drop_state = apply_dropconnect(model, float(cfg["dropconnect_p"])) if mode == "dropconnect" else []
            logits_t, _ = model(data, reg_mode=None)
            logits = logits_t.sum(0)
            task_loss = criterion(logits, targets)
            regularizer = beta_l2(model) if mode == "beta_l2" else data.new_zeros(())
            loss = task_loss + float(cfg["lambda_beta_l2"]) * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if drop_state:
                restore_dropconnect(drop_state)
            optimizer.step()
            with torch.no_grad():
                for _, layer in model._lif_layers:
                    if mode == "beta_clip":
                        layer.beta.clamp_(float(cfg["clip_low"]), float(cfg["clip_high"]))
                    else:
                        layer.beta.clamp_(0.01, 0.99)

            count = targets.numel()
            total += count
            correct += int((logits.argmax(1) == targets).sum())
            task_sum += float(task_loss) * count
            regularizer_sum += float(regularizer) * count

        val_acc = evaluate(model, val_loader, int(cfg["max_eval_batches"]))
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch + 1,
            "train_acc": 100.0 * correct / max(total, 1),
            "val_acc": val_acc,
            "task_loss": task_sum / max(total, 1),
            "regularizer": regularizer_sum / max(total, 1),
            "beta": [float(layer.beta.detach().mean()) for _, layer in model._lif_layers],
        }
        history.append(row)
        print(
            f"[{cfg['architecture']}] {mode} s{seed} e{epoch + 1:02d}/{cfg['epochs']} "
            f"train={row['train_acc']:.2f} val={val_acc:.2f} beta={row['beta']}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("No best state was produced.")
    model.load_state_dict(best_state, strict=True)
    test_acc = evaluate(model, test_loader, int(cfg["max_eval_batches"]))
    return model, {
        "test_acc": test_acc,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["ff", "rsnn"], required=True)
    parser.add_argument("--modes", type=parse_strings, required=True)
    parser.add_argument("--seeds", type=parse_ints, default=[42, 52, 72, 92, 102])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda-beta-l2", type=float, default=1e-2)
    parser.add_argument("--clip-low", type=float, default=0.10)
    parser.add_argument("--clip-high", type=float, default=0.90)
    parser.add_argument("--fixed-hetero-cv", type=float, default=0.20)
    parser.add_argument("--dropconnect-p", type=float, default=0.10)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    invalid = set(args.modes) - VALID_MODES
    if invalid:
        raise ValueError(f"Unknown modes: {sorted(invalid)}")
    if args.architecture == "ff" and "dropconnect" in args.modes:
        raise ValueError("DropConnect is restricted to recurrent paths and is only defined for RSNN.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    payload = json.loads(results_path.read_text()) if results_path.exists() else {"runs": {}}

    source = load_source_module(args.architecture)
    cfg = copy.deepcopy(source.CONFIG)
    cfg.update(
        {
            "architecture": args.architecture,
            "epochs": int(args.epochs),
            "lambda_beta_l2": float(args.lambda_beta_l2),
            "clip_low": float(args.clip_low),
            "clip_high": float(args.clip_high),
            "fixed_hetero_cv": float(args.fixed_hetero_cv),
            "dropconnect_p": float(args.dropconnect_p),
            "max_train_batches": int(args.max_train_batches),
            "max_eval_batches": int(args.max_eval_batches),
            "num_workers": int(args.num_workers),
        }
    )
    payload["config"] = cfg
    # The exact submitted scripts construct the SHD train/validation/test loaders
    # once with seed 42 and reuse those same splits/loaders for every run.
    seed_all(args.seeds[0])
    loader_bundle = source.build_loaders(args.seeds[0], cfg)
    cfg["input_size"] = int(loader_bundle[3])
    loaders = loader_bundle[:3]
    for seed in args.seeds:
        for mode in args.modes:
            run_key = f"s{seed}_{mode}"
            if run_key in payload["runs"]:
                print(f"skip completed {run_key}", flush=True)
                continue
            seed_all(seed)
            model = make_model(source, args.architecture, cfg, mode, seed)
            model, result = train_one(model, loaders, cfg, mode, seed)
            checkpoint_path = checkpoints_dir / f"best_s{seed}_{mode}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "architecture": args.architecture,
                    "mode": mode,
                    "seed": seed,
                    **{key: result[key] for key in ["test_acc", "best_val", "best_epoch"]},
                },
                checkpoint_path,
            )
            payload["runs"][run_key] = {
                **result,
                "checkpoint": str(checkpoint_path),
                "beta": [float(layer.beta.detach().mean()) for _, layer in model._lif_layers],
            }
            results_path.write_text(json.dumps(payload, indent=2))
            print(
                f"DONE [{args.architecture}] {mode} s{seed} "
                f"val={result['best_val']:.3f} test={result['test_acc']:.3f}",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
