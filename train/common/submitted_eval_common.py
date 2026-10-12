"""Shared utilities for post-hoc evaluations of the trained SHD-FC / SHD-RSNN checkpoints."""

from __future__ import annotations

import copy
import importlib.util
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import snntorch as snn
import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
from snntorch import surrogate
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from snn_robust_tau.paths import TONIC_ROOT, ckpt_dir  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = TONIC_ROOT

ARCH_SPECS = {
    "ff": {
        "source": PROJECT_ROOT / "train" / "shd_fc.py",
        "class_name": "SHDSNN",
        "result_dir": ckpt_dir("SHD-FC"),
    },
    "rsnn": {
        "source": PROJECT_ROOT / "train" / "shd_rsnn.py",
        "class_name": "SHDRSNN",
        "result_dir": ckpt_dir("SHD-RSNN"),
    },
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=2)
def load_source_module(architecture: str):
    spec = ARCH_SPECS[architecture]
    module_name = f"submitted_shd_{architecture}_source"
    module_spec = importlib.util.spec_from_file_location(module_name, spec["source"])
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not import {spec['source']}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def checkpoint_path(architecture: str, seed: int, mode: str) -> Path:
    return ARCH_SPECS[architecture]["result_dir"] / f"best_s{seed}_{mode}.pth"


def build_test_loader(
    *,
    time_steps: int = 50,
    batch_size: int = 256,
    num_workers: int = 2,
) -> tuple[DataLoader, int]:
    sensor_size = tonic.datasets.SHD.sensor_size
    transform = transforms.Compose(
        [transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps)]
    )
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    test_ds = tonic.datasets.SHD(
        save_to=str(DATA_ROOT), train=False, transform=transform
    )
    sample_x, _ = test_ds[0]
    input_size = int(np.prod(sample_x.shape[1:]))
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, input_size


def load_submitted_model(
    architecture: str, seed: int, mode: str, input_size: int
) -> tuple[nn.Module, dict[str, Any]]:
    path = checkpoint_path(architecture, seed, mode)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    source = load_source_module(architecture)
    spike_grad = surrogate.atan(alpha=cfg["surrogate_alpha"])
    kwargs = {
        "input_size": input_size,
        "beta_init": cfg["beta_init"],
        "threshold_init": cfg["threshold_init"],
        "surrogate_fn": spike_grad,
        "learn_beta": cfg["learn_beta"],
        "learn_threshold": cfg["learn_threshold"],
    }
    if architecture == "rsnn":
        kwargs["hidden"] = cfg["hidden"]
    model = getattr(source, ARCH_SPECS[architecture]["class_name"])(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model, checkpoint


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).to(DEVICE).eval()


def _replace_registered_tensor(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """Replace a scalar parameter/buffer by a vector while preserving registration."""
    value = value.detach()
    if name in module._parameters:
        old = module._parameters[name]
        requires_grad = bool(old.requires_grad) if old is not None else False
        module._parameters[name] = nn.Parameter(value, requires_grad=requires_grad)
    elif name in module._buffers:
        module._buffers[name] = value
    else:
        setattr(module, name, value)


def _lif_widths(model: nn.Module) -> list[int]:
    return [model.fc1.out_features, model.fc2.out_features, model.fc3.out_features]


def apply_device_mismatch(
    model: nn.Module,
    *,
    beta_cv: float,
    threshold_cv: float,
    weight_cv: float,
) -> dict[str, Any]:
    """Apply independent Gaussian device mismatch to an evaluation-only model.

    Submitted beta/threshold tensors are scalar per layer. For device-level
    evaluation, each scalar is expanded to one independently perturbed value per
    neuron. Linear weights are independently perturbed element-wise.
    """
    realized: dict[str, Any] = {
        "beta_relative_rms": [],
        "threshold_relative_rms": [],
        "weight_relative_rms": [],
        "granularity": {
            "beta": "per_neuron (expanded from trained per-layer scalar)",
            "threshold": "per_neuron (expanded from configured per-layer scalar)",
            "weight": "per_weight",
        },
    }
    widths = _lif_widths(model)
    with torch.no_grad():
        for (_, layer), width in zip(model._lif_layers, widths):
            if beta_cv > 0:
                base = layer.beta.detach()
                if base.numel() == 1:
                    base = base.reshape(1).expand(width).clone()
                noise = torch.randn_like(base) * beta_cv
                perturbed = (base * (1.0 + noise)).clamp(0.01, 0.99)
                denom = base.abs().clamp_min(1e-12)
                realized["beta_relative_rms"].append(
                    float(torch.sqrt(torch.mean(((perturbed - base) / denom) ** 2)))
                )
                _replace_registered_tensor(layer, "beta", perturbed)

            if threshold_cv > 0:
                base = layer.threshold.detach()
                if base.numel() == 1:
                    base = base.reshape(1).expand(width).clone()
                noise = torch.randn_like(base) * threshold_cv
                perturbed = (base * (1.0 + noise)).clamp_min(0.01)
                denom = base.abs().clamp_min(1e-12)
                realized["threshold_relative_rms"].append(
                    float(torch.sqrt(torch.mean(((perturbed - base) / denom) ** 2)))
                )
                _replace_registered_tensor(layer, "threshold", perturbed)

        if weight_cv > 0:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    base = module.weight.detach().clone()
                    noise = torch.randn_like(base) * weight_cv
                    perturbed = base * (1.0 + noise)
                    denom = torch.linalg.vector_norm(base).clamp_min(1e-12)
                    rel = torch.linalg.vector_norm(perturbed - base) / denom
                    realized["weight_relative_rms"].append(float(rel))
                    module.weight.copy_(perturbed)
    return realized


def apply_beta_distribution_mismatch(
    model: nn.Module,
    *,
    cv: float,
    distribution: str,
    correlation: float = 0.0,
) -> dict[str, Any]:
    """Apply per-neuron beta mismatch under non-Gaussian/correlated models."""
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [0, 1]")
    if distribution not in {"gaussian", "lognormal", "student_t3"}:
        raise ValueError(f"Unsupported distribution: {distribution}")
    relative_rms = []
    realized_mean_factor = []
    widths = _lif_widths(model)
    with torch.no_grad():
        for (_, layer), width in zip(model._lif_layers, widths):
            base = layer.beta.detach()
            if base.numel() == 1:
                base = base.reshape(1).expand(width).clone()
            if distribution == "lognormal":
                sigma = float(np.sqrt(np.log1p(cv**2)))
                global_z = torch.randn(1, device=base.device, dtype=base.dtype)
                local_z = torch.randn_like(base)
                z = (
                    np.sqrt(correlation) * global_z
                    + np.sqrt(1.0 - correlation) * local_z
                )
                factor = torch.exp(sigma * z - 0.5 * sigma**2)
            else:
                if distribution == "student_t3":
                    dist = torch.distributions.StudentT(df=3.0)
                    global_z = dist.sample((1,)).to(base)
                    local_z = dist.sample(base.shape).to(base)
                    # A t_3 variable has standard deviation sqrt(3).
                    global_z = global_z / np.sqrt(3.0)
                    local_z = local_z / np.sqrt(3.0)
                else:
                    global_z = torch.randn(1, device=base.device, dtype=base.dtype)
                    local_z = torch.randn_like(base)
                z = (
                    np.sqrt(correlation) * global_z
                    + np.sqrt(1.0 - correlation) * local_z
                )
                factor = 1.0 + cv * z
            perturbed = (base * factor).clamp(0.01, 0.99)
            denom = base.abs().clamp_min(1e-12)
            relative_rms.append(
                float(torch.sqrt(torch.mean(((perturbed - base) / denom) ** 2)))
            )
            realized_mean_factor.append(float(torch.mean(perturbed / denom)))
            _replace_registered_tensor(layer, "beta", perturbed)
    return {
        "requested_cv": cv,
        "distribution": distribution,
        "equicorrelation": correlation,
        "beta_relative_rms": relative_rms,
        "beta_mean_factor": realized_mean_factor,
        "granularity": "per_neuron (expanded from trained per-layer scalar)",
    }


@torch.inference_mode()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    *,
    input_transform: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    transform_seed: int = 0,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_index, (data, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        data = data.to(DEVICE, non_blocking=True).float()
        targets = targets.to(DEVICE, non_blocking=True)
        if input_transform is not None:
            data = input_transform(data, transform_seed + batch_index)
        logits_t, _ = model(data, reg_mode=None)
        prediction = logits_t.sum(0).argmax(1)
        correct += int((prediction == targets).sum())
        total += int(targets.numel())
    if total == 0:
        raise RuntimeError("No evaluation examples were processed.")
    return 100.0 * correct / total
