"""
DVS-Gesture P0 Complete: Clean vs HAJR vs ASAR vs NoiseAug
===========================================================
All modes use the same VGG-style SNN and a single training entrypoint.

Modes:
  - clean:     no explicit regularization
  - hajr:      HAJR on hidden reset layers + ASAR on output no-reset layer
  - asar:      raw sensitivity regularization on all layers
  - noise_aug: per-batch beta perturbation without explicit regularizer

Regularization is computed online inside the timestep loop using snnTorch-
aligned reset_delay=True dynamics.
"""

import copy
import gc
import json
import os
import random
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
from snntorch import surrogate
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def _find_project_root(start_file: str) -> str:
    cur = os.path.abspath(os.path.dirname(start_file))
    while True:
        if os.path.isdir(os.path.join(cur, "snn_robust_tau")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot locate project root containing 'snn_robust_tau'.")
        cur = parent


PROJECT_ROOT = _find_project_root(__file__)
sys.path.insert(0, PROJECT_ROOT)

from snn_robust_tau.paths import OUTPUT_ROOT, TONIC_ROOT  # noqa: E402

from snn_robust_tau.loss.hajr import OnlineHAJR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "modes": ["hajr", "asar", "noise_aug", "clean"],
    "seeds": [42, 52, 72, 92, 102],
    "epochs": 50,
    "batch_size": 8,
    "num_workers": 4,
    "lr": 1e-3,
    "beta_init": 0.5,
    "threshold_init": 1.0,
    "surrogate_alpha": 2.0,
    "learn_beta": True,
    "learn_threshold": False,
    "lambda_hajr": 0.1,
    "lambda_asar": 5e-4,
    "rampup_epochs": 1,
    "noise_aug_cv": 0.15,
    "time_steps": 16,
    "num_classes": 11,
    "val_ratio": 0.1,
    "data_root": str(TONIC_ROOT),
    "eval_cvs": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    "eval_mc_runs": 10,
    "quant_bits": [32, 16, 8, 6, 4, 3, 2],
    "show_progress": False,
    "progress_update_interval": 20,
}

COLORS = {"clean": "#2196F3", "hajr": "#F44336", "asar": "#4CAF50", "noise_aug": "#FF9800"}


def compute_robustness_auc(eval_cvs, means):
    return float(np.trapz(np.asarray(means, dtype=float), np.asarray(eval_cvs, dtype=float)))


def compute_quantization_auc(quant_bits, means):
    bits = np.asarray(quant_bits, dtype=float)
    vals = np.asarray(means, dtype=float)
    order = np.argsort(bits)
    return float(np.trapz(vals[order], bits[order]))



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(cfg, generator=None):
    kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    if generator is not None:
        kwargs["generator"] = generator
    if cfg["num_workers"] > 0:
        kwargs["worker_init_fn"] = seed_worker
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


class DVSGestureSNN(nn.Module):
    def __init__(self, num_classes=11, beta=0.5, threshold=1.0,
                 surrogate_fn=None, learn_beta=False, learn_threshold=False):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate_fn,
            learn_beta=learn_beta, learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate_fn,
            learn_beta=learn_beta, learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lif3 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate_fn,
            learn_beta=learn_beta, learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )
        self.mp3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate_fn,
            learn_beta=learn_beta, learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )
        self.mp4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.lif5 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate_fn,
            learn_beta=learn_beta, learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )
        self.mp5 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.lif6 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate_fn,
            learn_beta=learn_beta, learn_threshold=learn_threshold,
            reset_mechanism="zero",
        )

        self.fc2 = nn.Linear(1024, num_classes)
        self.lif7 = snn.Leaky(
            beta=beta, spike_grad=surrogate_fn,
            learn_beta=learn_beta, reset_mechanism="none",
        )

        self._lif_layers = [
            ("lif1", self.lif1),
            ("lif2", self.lif2),
            ("lif3", self.lif3),
            ("lif4", self.lif4),
            ("lif5", self.lif5),
            ("lif6", self.lif6),
            ("lif7", self.lif7),
        ]
        self.online_reg = OnlineHAJR(surrogate_type="auto", detach_state=False)

    def forward(self, x, reg_mode=None):
        _, steps = x.shape[0], x.shape[1]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()

        reg_loss = torch.tensor(0.0, device=x.device)
        layer_loss_sums = {}
        active_layers = []
        if reg_mode is not None:
            self.online_reg.reset()

        mem7_rec = []
        for t in range(steps):
            out = self.bn1(self.conv1(x[:, t]))
            spk1, mem1 = self.lif1(out, mem1)
            if reg_mode == "hajr":
                layer_loss_sums["lif1"] = layer_loss_sums.get("lif1", 0.0) + self.online_reg.step_from_layer(
                    "lif1", self.lif1, mem1, timestep=t, spike=spk1, decision_gate=True
                )
                if "lif1" not in active_layers:
                    active_layers.append("lif1")
            elif reg_mode == "asar":
                layer_loss_sums["lif1"] = layer_loss_sums.get("lif1", 0.0) + self.online_reg.step_from_layer(
                    "lif1", self.lif1, mem1, timestep=t, spike=spk1, decision_gate=False
                )
                if "lif1" not in active_layers:
                    active_layers.append("lif1")
            out = self.mp1(spk1)

            out = self.bn2(self.conv2(out))
            spk2, mem2 = self.lif2(out, mem2)
            if reg_mode == "hajr":
                layer_loss_sums["lif2"] = layer_loss_sums.get("lif2", 0.0) + self.online_reg.step_from_layer(
                    "lif2", self.lif2, mem2, timestep=t, spike=spk2, decision_gate=True
                )
                if "lif2" not in active_layers:
                    active_layers.append("lif2")
            elif reg_mode == "asar":
                layer_loss_sums["lif2"] = layer_loss_sums.get("lif2", 0.0) + self.online_reg.step_from_layer(
                    "lif2", self.lif2, mem2, timestep=t, spike=spk2, decision_gate=False
                )
                if "lif2" not in active_layers:
                    active_layers.append("lif2")
            out = self.mp2(spk2)

            out = self.bn3(self.conv3(out))
            spk3, mem3 = self.lif3(out, mem3)
            if reg_mode == "hajr":
                layer_loss_sums["lif3"] = layer_loss_sums.get("lif3", 0.0) + self.online_reg.step_from_layer(
                    "lif3", self.lif3, mem3, timestep=t, spike=spk3, decision_gate=True
                )
                if "lif3" not in active_layers:
                    active_layers.append("lif3")
            elif reg_mode == "asar":
                layer_loss_sums["lif3"] = layer_loss_sums.get("lif3", 0.0) + self.online_reg.step_from_layer(
                    "lif3", self.lif3, mem3, timestep=t, spike=spk3, decision_gate=False
                )
                if "lif3" not in active_layers:
                    active_layers.append("lif3")
            out = self.mp3(spk3)

            out = self.bn4(self.conv4(out))
            spk4, mem4 = self.lif4(out, mem4)
            if reg_mode == "hajr":
                layer_loss_sums["lif4"] = layer_loss_sums.get("lif4", 0.0) + self.online_reg.step_from_layer(
                    "lif4", self.lif4, mem4, timestep=t, spike=spk4, decision_gate=True
                )
                if "lif4" not in active_layers:
                    active_layers.append("lif4")
            elif reg_mode == "asar":
                layer_loss_sums["lif4"] = layer_loss_sums.get("lif4", 0.0) + self.online_reg.step_from_layer(
                    "lif4", self.lif4, mem4, timestep=t, spike=spk4, decision_gate=False
                )
                if "lif4" not in active_layers:
                    active_layers.append("lif4")
            out = self.mp4(spk4)

            out = self.bn5(self.conv5(out))
            spk5, mem5 = self.lif5(out, mem5)
            if reg_mode == "hajr":
                layer_loss_sums["lif5"] = layer_loss_sums.get("lif5", 0.0) + self.online_reg.step_from_layer(
                    "lif5", self.lif5, mem5, timestep=t, spike=spk5, decision_gate=True
                )
                if "lif5" not in active_layers:
                    active_layers.append("lif5")
            elif reg_mode == "asar":
                layer_loss_sums["lif5"] = layer_loss_sums.get("lif5", 0.0) + self.online_reg.step_from_layer(
                    "lif5", self.lif5, mem5, timestep=t, spike=spk5, decision_gate=False
                )
                if "lif5" not in active_layers:
                    active_layers.append("lif5")
            out = self.mp5(spk5)

            out = self.flatten(out)
            out = self.fc1(out)
            spk6, mem6 = self.lif6(out, mem6)
            if reg_mode == "hajr":
                layer_loss_sums["lif6"] = layer_loss_sums.get("lif6", 0.0) + self.online_reg.step_from_layer(
                    "lif6", self.lif6, mem6, timestep=t, spike=spk6, decision_gate=True
                )
                if "lif6" not in active_layers:
                    active_layers.append("lif6")
            elif reg_mode == "asar":
                layer_loss_sums["lif6"] = layer_loss_sums.get("lif6", 0.0) + self.online_reg.step_from_layer(
                    "lif6", self.lif6, mem6, timestep=t, spike=spk6, decision_gate=False
                )
                if "lif6" not in active_layers:
                    active_layers.append("lif6")

            out = self.fc2(spk6)
            spk7, mem7 = self.lif7(out, mem7)
            if reg_mode in {"hajr", "asar"}:
                layer_loss_sums["lif7"] = layer_loss_sums.get("lif7", 0.0) + self.online_reg.step_from_layer(
                    "lif7", self.lif7, mem7, timestep=t, spike=spk7, decision_gate=False
                )
                if "lif7" not in active_layers:
                    active_layers.append("lif7")
            mem7_rec.append(mem7)

        if active_layers:
            layer_map = dict(self._lif_layers)
            per_layer_losses = []
            for layer_name in active_layers:
                layer = layer_map[layer_name]
                denom = steps if getattr(layer, "reset_mechanism", "subtract") == "none" else max(steps - 1, 1)
                per_layer_losses.append(layer_loss_sums[layer_name] / denom)
            reg_loss = torch.stack(per_layer_losses).mean()

        return torch.stack(mem7_rec, dim=0), reg_loss

    def get_beta_values(self):
        values = []
        for _, lif in self._lif_layers:
            beta = lif.beta
            values.append(beta.item() if isinstance(beta, torch.Tensor) else float(beta))
        return values


def build_loaders(seed, cfg):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=cfg["time_steps"]),
    ])
    os.makedirs(cfg["data_root"], exist_ok=True)
    train_ds = tonic.datasets.DVSGesture(save_to=cfg["data_root"], train=True, transform=transform)
    test_ds = tonic.datasets.DVSGesture(save_to=cfg["data_root"], train=False, transform=transform)

    total = len(train_ds)
    val_size = max(1, int(round(total * cfg["val_ratio"])))
    gen = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(train_ds, [total - val_size, val_size], generator=gen)

    train_gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_sub, shuffle=True, drop_last=True, **_loader_kwargs(cfg, generator=train_gen)),
        DataLoader(val_sub, shuffle=False, **_loader_kwargs(cfg)),
        DataLoader(test_ds, shuffle=False, **_loader_kwargs(cfg)),
    )


def _perturb_betas_for_training(model, cv):
    originals = {}
    with torch.no_grad():
        for idx, (_, lif) in enumerate(model._lif_layers):
            originals[idx] = lif.beta.data.clone()
            noise = torch.randn_like(lif.beta) * cv
            lif.beta.data.copy_((lif.beta.data * (1.0 + noise)).clamp(0.01, 0.99))
    return originals


def _restore_betas(model, originals):
    with torch.no_grad():
        for idx, (_, lif) in enumerate(model._lif_layers):
            lif.beta.data.copy_(originals[idx])


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for data, targets in loader:
        data, targets = data.to(DEVICE).float(), targets.to(DEVICE)
        logits_t, _ = model(data, reg_mode=None)
        correct += (logits_t.sum(0).argmax(1) == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def _apply_beta_noise(model, cv):
    originals = {}
    with torch.no_grad():
        for idx, (_, lif) in enumerate(model._lif_layers):
            originals[idx] = lif.beta.clone()
            noise = torch.randn_like(lif.beta) * cv
            lif.beta.copy_((lif.beta * (1.0 + noise)).clamp(0.01, 0.99))
    return originals


def _restore_beta(model, originals):
    with torch.no_grad():
        for idx, (_, lif) in enumerate(model._lif_layers):
            lif.beta.copy_(originals[idx])


def evaluate_robustness(model, loader, cv_list, mc_runs):
    results = {}
    for cv in cv_list:
        if cv == 0.0:
            acc = evaluate(model, loader)
            results[cv] = {"mean": acc, "std": 0.0, "runs": [acc]}
            continue
        runs = []
        for _ in range(mc_runs):
            original = _apply_beta_noise(model, cv)
            runs.append(evaluate(model, loader))
            _restore_beta(model, original)
        results[cv] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs, ddof=1)),
            "runs": runs,
        }
    return results


def _apply_beta_noise_layerwise(model, cv):
    """Layer-wise correlated: each layer gets its own shared delta ~ N(0, cv²)."""
    originals = {}
    with torch.no_grad():
        for idx, (_, lif) in enumerate(model._lif_layers):
            originals[idx] = lif.beta.clone()
            delta = float(torch.randn(1).item()) * cv
            lif.beta.copy_((lif.beta * (1.0 + delta)).clamp(0.01, 0.99))
    return originals


def _apply_beta_noise_global(model, cv):
    """Global correlated: all layers share one delta ~ N(0, cv²)."""
    originals = {}
    delta = float(torch.randn(1).item()) * cv
    with torch.no_grad():
        for idx, (_, lif) in enumerate(model._lif_layers):
            originals[idx] = lif.beta.clone()
            lif.beta.copy_((lif.beta * (1.0 + delta)).clamp(0.01, 0.99))
    return originals


def evaluate_robustness_layerwise(model, loader, cv_list, mc_runs):
    """Robustness under layer-wise correlated beta perturbation."""
    results = {}
    for cv in cv_list:
        if cv == 0.0:
            acc = evaluate(model, loader)
            results[cv] = {"mean": acc, "std": 0.0, "runs": [acc]}
            continue
        runs = []
        for _ in range(mc_runs):
            original = _apply_beta_noise_layerwise(model, cv)
            runs.append(evaluate(model, loader))
            _restore_beta(model, original)
        results[cv] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs, ddof=1)),
            "runs": runs,
        }
    return results


def evaluate_robustness_global(model, loader, cv_list, mc_runs):
    """Robustness under global (chip-wide) beta perturbation."""
    results = {}
    for cv in cv_list:
        if cv == 0.0:
            acc = evaluate(model, loader)
            results[cv] = {"mean": acc, "std": 0.0, "runs": [acc]}
            continue
        runs = []
        for _ in range(mc_runs):
            original = _apply_beta_noise_global(model, cv)
            runs.append(evaluate(model, loader))
            _restore_beta(model, original)
        results[cv] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs, ddof=1)),
            "runs": runs,
        }
    return results


def _quantize_weights(model, bits):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight
                max_abs = weight.abs().max()
                if max_abs.item() == 0.0:
                    continue
                qmax = (2 ** (bits - 1)) - 1
                scale = max_abs / qmax
                module.weight.copy_(torch.round(weight / scale).clamp(-qmax, qmax) * scale)


def evaluate_quantization(model, loader, bits_list):
    base_state = copy.deepcopy(model.state_dict())
    results = {}
    for bits in bits_list:
        _quantize_weights(model, bits)
        results[bits] = evaluate(model, loader)
        model.load_state_dict(base_state)
    return results


def collect_lif_snapshot(model):
    snapshot = []
    for name, lif in model._lif_layers:
        item = {
            "name": name,
            "reset_mechanism": getattr(lif, "reset_mechanism", "subtract"),
            "beta": float(lif.beta.detach().view(-1)[0].item()),
        }
        threshold = getattr(lif, "threshold", None)
        item["threshold"] = None if threshold is None else float(threshold.detach().view(-1)[0].item())
        snapshot.append(item)
    return snapshot


def save_checkpoint(path, model, cfg, seed, mode, history, test_acc, best_val, best_epoch):
    torch.save({
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "config": copy.deepcopy(cfg),
        "seed": seed,
        "mode": mode,
        "history": copy.deepcopy(history),
        "test_acc": test_acc,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "lif_parameters": collect_lif_snapshot(model),
    }, path)


def train_one_mode(model, mode, loaders, cfg, seed):
    train_loader, val_loader, test_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(cfg["epochs"]):
        model.train()
        samples = correct = 0
        task_sum = reg_sum = 0.0

        ramp = min(1.0, (epoch + 1) / cfg["rampup_epochs"])
        if mode == "hajr":
            lam = cfg["lambda_hajr"] * ramp
        elif mode == "asar":
            lam = cfg["lambda_asar"] * ramp
        else:
            lam = 0.0

        progress = tqdm(train_loader, desc=f"s{seed} {mode} e{epoch+1}", leave=False) if cfg["show_progress"] else train_loader
        for batch_idx, (data, targets) in enumerate(progress):
            data, targets = data.to(DEVICE).float(), targets.to(DEVICE)

            beta_originals = None
            if mode == "noise_aug":
                beta_originals = _perturb_betas_for_training(model, cfg["noise_aug_cv"])

            reg_mode = mode if mode in {"hajr", "asar"} else None
            logits_t, loss_reg = model(data, reg_mode=reg_mode)
            logits = logits_t.sum(dim=0)
            loss_task = criterion(logits, targets)
            if reg_mode is None:
                loss_reg = torch.tensor(0.0, device=DEVICE)

            loss = loss_task + lam * loss_reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if beta_originals is not None:
                _restore_betas(model, beta_originals)

            optimizer.step()
            with torch.no_grad():
                for _, lif in model._lif_layers:
                    lif.beta.clamp_(0.01, 0.99)
                    if hasattr(lif, "threshold") and lif.reset_mechanism != "none":
                        lif.threshold.clamp_(0.1, 3.0)

            correct += (logits.argmax(1) == targets).sum().item()
            batch = targets.size(0)
            samples += batch
            task_sum += loss_task.item() * batch
            reg_sum += loss_reg.item() * batch
            if cfg["show_progress"] and (
                batch_idx == 0
                or (batch_idx + 1) % cfg["progress_update_interval"] == 0
                or (batch_idx + 1) == len(train_loader)
            ):
                progress.set_postfix(acc=f"{100*correct/samples:.1f}", reg=f"{loss_reg.item():.4f}")

        scheduler.step()
        train_acc = 100.0 * correct / samples
        val_acc = evaluate(model, val_loader)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        betas = model.get_beta_values()
        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "task_loss": task_sum / samples,
            "reg_loss": reg_sum / samples,
            "beta_mean": float(np.mean(betas)),
            "lambda": lam,
        })
        print(f"  [s{seed}][{mode}] e{epoch+1:02d} train={train_acc:.2f} val={val_acc:.2f} beta_mean={np.mean(betas):.3f} lambda={lam:.5f}")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)
    print(f"  [s{seed}][{mode}] BEST val={best_val:.2f}@e{best_epoch} -> test={test_acc:.2f}")
    return model, test_acc, best_val, best_epoch, history


def plot_training_curves(history, seed, mode, output_dir):
    epochs = [row["epoch"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    task_loss = [row["task_loss"] for row in history]
    reg_loss = [row["reg_loss"] for row in history]
    color = COLORS.get(mode, "gray")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_acc, color=color, lw=2, label="train")
    axes[0].plot(epochs, val_acc, color=color, lw=2, ls="--", label="val")
    axes[0].set_title(f"Accuracy (seed={seed}, {mode})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Acc (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, task_loss, color=color, lw=2, label="task")
    axes[1].plot(epochs, reg_loss, color="orange", lw=2, label="reg")
    axes[1].set_title(f"Loss (seed={seed}, {mode})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", f"training_s{seed}_{mode}.png"), dpi=150)
    plt.close(fig)


def plot_beta_trajectory(all_histories, modes, seeds, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode in modes:
        epochs_ref = [row["epoch"] for row in all_histories[(seeds[0], mode)]]
        mat = np.array([[row["beta_mean"] for row in all_histories[(seed, mode)]] for seed in seeds])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
        color = COLORS.get(mode, "gray")
        ax.plot(epochs_ref, mean, color=color, lw=2, label=mode)
        ax.fill_between(epochs_ref, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title("Mean beta trajectory")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("beta mean")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", "beta_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_comparison(summary, modes, eval_cvs, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in modes:
        means = [summary[mode]["robustness"][str(cv)]["mean"] for cv in eval_cvs]
        stds = [summary[mode]["robustness"][str(cv)]["std"] for cv in eval_cvs]
        color = COLORS.get(mode, "gray")
        ax.plot(eval_cvs, means, color=color, lw=2, marker="o", ms=5, label=mode)
        ax.fill_between(eval_cvs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=color, alpha=0.15)
    ax.set_title("beta variation robustness (i.i.d.)")
    ax.set_xlabel("CV")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", "robustness_cv.png"), dpi=150)
    plt.close(fig)


def plot_robustness_structured_comparison(summary, modes, eval_cvs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for noise_key, title, ax in [
        ("robustness_layerwise", "Layer-wise Correlated Noise", axes[0]),
        ("robustness_global", "Global (Chip-wide) Noise", axes[1]),
    ]:
        for mode in modes:
            if noise_key not in summary.get(mode, {}):
                continue
            means = [summary[mode][noise_key][str(cv)]["mean"] for cv in eval_cvs]
            stds = [summary[mode][noise_key][str(cv)]["std"] for cv in eval_cvs]
            color = COLORS.get(mode, "gray")
            ax.plot(eval_cvs, means, color=color, lw=2, marker="o", ms=5, label=mode)
            ax.fill_between(eval_cvs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("CV (noise level)")
        ax.set_ylabel("Test Accuracy (%)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", "robustness_structured.png"), dpi=150)
    plt.close(fig)


def plot_robustness_comparison_per_seed(all_results, modes, eval_cvs, seeds, output_dir):
    for seed in seeds:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in modes:
            payload = all_results[f"s{seed}_{mode}"]["robustness"]
            means = [payload[str(cv)]["mean"] for cv in eval_cvs]
            stds = [payload[str(cv)]["std"] for cv in eval_cvs]
            color = COLORS.get(mode, "gray")
            ax.plot(eval_cvs, means, color=color, lw=2, marker="o", ms=5, label=mode)
            ax.fill_between(eval_cvs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=color, alpha=0.15)
        ax.set_title(f"beta variation robustness (seed={seed})")
        ax.set_xlabel("CV")
        ax.set_ylabel("Test Accuracy (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "plots", f"robustness_cv_seed{seed}.png"), dpi=150)
        plt.close(fig)


def plot_quantization_comparison(summary, modes, quant_bits, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in modes:
        means = [summary[mode]["quantization"][str(bits)]["mean"] for bits in quant_bits]
        stds = [summary[mode]["quantization"][str(bits)]["std"] for bits in quant_bits]
        color = COLORS.get(mode, "gray")
        ax.plot(quant_bits, means, color=color, lw=2, marker="s", ms=5, label=mode)
        ax.fill_between(quant_bits, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=color, alpha=0.15)
    ax.set_title("weight quantization robustness")
    ax.set_xlabel("Weight Bit-width")
    ax.set_ylabel("Test Accuracy (%)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plots", "quantization.png"), dpi=150)
    plt.close(fig)


def plot_quantization_comparison_per_seed(all_results, modes, quant_bits, seeds, output_dir):
    for seed in seeds:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in modes:
            payload = all_results[f"s{seed}_{mode}"]["quantization"]
            means = [payload[str(bits)] for bits in quant_bits]
            color = COLORS.get(mode, "gray")
            ax.plot(quant_bits, means, color=color, lw=2, marker="s", ms=5, label=mode)
        ax.set_title(f"weight quantization robustness (seed={seed})")
        ax.set_xlabel("Weight Bit-width")
        ax.set_ylabel("Test Accuracy (%)")
        ax.invert_xaxis()
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "plots", f"quantization_seed{seed}.png"), dpi=150)
        plt.close(fig)


def main():
    cfg = CONFIG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(str(OUTPUT_ROOT), "train", f"dvs_gesture_{timestamp}")
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Output : {output_dir}")
    print(f"Config : {json.dumps(cfg, indent=2, default=str)}")
    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump(cfg, handle, indent=2, default=str)

    surrogate_fn = surrogate.atan(alpha=cfg["surrogate_alpha"])
    seed_everything(cfg["seeds"][0])
    train_loader, val_loader, test_loader = build_loaders(cfg["seeds"][0], cfg)

    all_results = {}
    all_histories = {}
    for seed in cfg["seeds"]:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        for mode in cfg["modes"]:
            print(f"\n--- Training: {mode} ---")
            seed_everything(seed)
            model = DVSGestureSNN(
                num_classes=cfg["num_classes"],
                beta=cfg["beta_init"],
                threshold=cfg["threshold_init"],
                surrogate_fn=surrogate_fn,
                learn_beta=cfg["learn_beta"],
                learn_threshold=cfg["learn_threshold"],
            ).to(DEVICE)

            model, test_acc, best_val, best_epoch, history = train_one_mode(
                model, mode, (train_loader, val_loader, test_loader), cfg, seed
            )
            all_histories[(seed, mode)] = history
            plot_training_curves(history, seed, mode, output_dir)
            checkpoint_path = os.path.join(output_dir, "checkpoints", f"best_s{seed}_{mode}.pth")
            save_checkpoint(checkpoint_path, model, cfg, seed, mode, history, test_acc, best_val, best_epoch)

            robust = evaluate_robustness(model, test_loader, cfg["eval_cvs"], cfg["eval_mc_runs"])
            robust_lw = evaluate_robustness_layerwise(model, test_loader, cfg["eval_cvs"], cfg["eval_mc_runs"])
            robust_gl = evaluate_robustness_global(model, test_loader, cfg["eval_cvs"], cfg["eval_mc_runs"])
            quant = evaluate_quantization(model, test_loader, cfg["quant_bits"])
            all_results[f"s{seed}_{mode}"] = {
                "seed": seed,
                "mode": mode,
                "test_acc": test_acc,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "checkpoint": checkpoint_path,
                "lif_parameters": collect_lif_snapshot(model),
                "robustness": {str(cv): value for cv, value in robust.items()},
                "robustness_layerwise": {str(cv): value for cv, value in robust_lw.items()},
                "robustness_global": {str(cv): value for cv, value in robust_gl.items()},
                "quantization": {str(bits): value for bits, value in quant.items()},
            }
            with open(os.path.join(output_dir, "results.json"), "w") as handle:
                json.dump(all_results, handle, indent=2, default=str)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {}
    for mode in cfg["modes"]:
        test_accs = [all_results[f"s{seed}_{mode}"]["test_acc"] for seed in cfg["seeds"]]
        robustness = {}
        for cv in cfg["eval_cvs"]:
            vals = [all_results[f"s{seed}_{mode}"]["robustness"][str(cv)]["mean"] for seed in cfg["seeds"]]
            robustness[str(cv)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        quantization = {}
        for bits in cfg["quant_bits"]:
            vals = [all_results[f"s{seed}_{mode}"]["quantization"][str(bits)] for seed in cfg["seeds"]]
            quantization[str(bits)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        summary[mode] = {
            "test_acc": {
                "mean": float(np.mean(test_accs)),
                "std": float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
            },
            "robustness": robustness,
            "quantization": quantization,
            "robustness_auc": {
                "mean": compute_robustness_auc(cfg["eval_cvs"], [robustness[str(cv)]["mean"] for cv in cfg["eval_cvs"]]),
            },
            "quantization_auc": {
                "mean": compute_quantization_auc(cfg["quant_bits"], [quantization[str(bits)]["mean"] for bits in cfg["quant_bits"]]),
            },
            "drop_at_cv_0.15": float(robustness["0.15"]["mean"] - robustness["0.0"]["mean"]) if "0.15" in robustness else float("nan"),
            "drop_at_cv_0.20": float(robustness["0.2"]["mean"] - robustness["0.0"]["mean"]) if "0.2" in robustness else float("nan"),
            "drop_at_4bit": float(quantization["4"]["mean"] - quantization["32"]["mean"]) if "4" in quantization and "32" in quantization else float("nan"),
            "drop_at_3bit": float(quantization["3"]["mean"] - quantization["32"]["mean"]) if "3" in quantization and "32" in quantization else float("nan"),
        }
        for struct_key in ["robustness_layerwise", "robustness_global"]:
            rob_s = {}
            for cv in cfg["eval_cvs"]:
                vals = [all_results[f"s{seed}_{mode}"][struct_key][str(cv)]["mean"] for seed in cfg["seeds"]]
                rob_s[str(cv)] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
            summary[mode][struct_key] = rob_s
            summary[mode][f"{struct_key}_auc"] = {
                "mean": compute_robustness_auc(cfg["eval_cvs"], [rob_s[str(cv)]["mean"] for cv in cfg["eval_cvs"]]),
            }

    plot_beta_trajectory(all_histories, cfg["modes"], cfg["seeds"], output_dir)
    plot_robustness_comparison(summary, cfg["modes"], cfg["eval_cvs"], output_dir)
    plot_robustness_structured_comparison(summary, cfg["modes"], cfg["eval_cvs"], output_dir)
    plot_quantization_comparison(summary, cfg["modes"], cfg["quant_bits"], output_dir)
    plot_robustness_comparison_per_seed(all_results, cfg["modes"], cfg["eval_cvs"], cfg["seeds"], output_dir)
    plot_quantization_comparison_per_seed(all_results, cfg["modes"], cfg["quant_bits"], cfg["seeds"], output_dir)

    final = {"config": cfg, "per_seed": all_results, "summary": summary}
    with open(os.path.join(output_dir, "results.json"), "w") as handle:
        json.dump(final, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
