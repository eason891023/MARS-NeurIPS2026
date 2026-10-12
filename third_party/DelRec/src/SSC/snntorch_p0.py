import copy
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError as exc:
    raise ImportError(
        "The snnTorch DelRec-SSC port requires `snntorch` to be installed."
    ) from exc

from src.SSC.delay_aware_regularizer import DelRecDelayAwareRegularizer
from snn_robust_tau.loss.hajr import (
    surrogate_derivative_atan,
    surrogate_derivative_fast_sigmoid,
    surrogate_derivative_triangle,
)


COLORS = {"clean": "#2196F3", "hajr": "#F44336", "asar": "#4CAF50", "noise_aug": "#FF9800"}


class TriangleSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, gamma: float):
        ctx.gamma = float(gamma)
        ctx.save_for_backward(input_tensor)
        return input_tensor.ge(0.0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_tensor,) = ctx.saved_tensors
        gamma = ctx.gamma
        surrogate_grad = (gamma - input_tensor.abs()).clamp(min=0.0) / (gamma * gamma)
        return grad_output * surrogate_grad, None


def triangle_surrogate(gamma: float = 1.0):
    def inner(x: torch.Tensor) -> torch.Tensor:
        return TriangleSurrogate.apply(x, gamma)

    return inner


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_surrogate_pair(surrogate_type: str, surrogate_alpha: float):
    surrogate_type = surrogate_type.lower()
    if surrogate_type == "atan":
        return surrogate.atan(alpha=surrogate_alpha), lambda x: surrogate_derivative_atan(x, surrogate_alpha)
    if surrogate_type in {"fast_sigmoid", "sigmoid"}:
        return surrogate.fast_sigmoid(slope=surrogate_alpha), lambda x: surrogate_derivative_fast_sigmoid(x, surrogate_alpha)
    if surrogate_type in {"triangle", "triangular"}:
        return triangle_surrogate(gamma=surrogate_alpha), lambda x: surrogate_derivative_triangle(x, surrogate_alpha)
    raise ValueError(f"Unsupported surrogate_type={surrogate_type!r}")


def map_snntorch_reset(reset_mechanism: str) -> str:
    mapping = {
        "subtract": "soft",
        "zero": "hard",
        "none": "none",
    }
    if reset_mechanism not in mapping:
        raise ValueError(f"Unsupported snntorch reset_mechanism={reset_mechanism!r}")
    return mapping[reset_mechanism]


class SSCSpikeIterator:
    def __init__(
        self,
        h5_file: h5py.File,
        batch_size: int,
        nb_steps: int,
        nb_units: int,
        max_time: float,
        n_bins: int = 1,
        shuffle: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.nb_steps = int(nb_steps)
        self.nb_units = int(nb_units)
        self.shuffle = bool(shuffle)
        self.n_bins = int(n_bins)

        self.firing_times = h5_file["spikes"]["times"]
        self.units_fired = h5_file["spikes"]["units"]
        self.labels = np.array(h5_file["labels"], dtype=np.int64)
        self.num_samples = len(self.labels)
        self.number_of_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.sample_index = np.arange(self.num_samples)
        self.time_bins = np.linspace(0.0, float(max_time), num=self.nb_steps)
        self.reset()

    def reset(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self.number_of_batches

    def __next__(self):
        if self.counter >= self.number_of_batches:
            raise StopIteration

        start = self.batch_size * self.counter
        end = min(self.batch_size * (self.counter + 1), self.num_samples)
        batch_index = self.sample_index[start:end]

        coo = [[] for _ in range(3)]
        for batch_pos, idx in enumerate(batch_index):
            times = np.digitize(self.firing_times[idx], self.time_bins)
            units = self.units_fired[idx]
            coo[0].extend([batch_pos] * len(times))
            coo[1].extend(times.tolist())
            coo[2].extend(units.tolist())

        indices = torch.tensor(coo, dtype=torch.long)
        values = torch.ones(len(coo[0]), dtype=torch.float32)
        dense = torch.sparse_coo_tensor(
            indices,
            values,
            size=(len(batch_index), self.nb_steps, self.nb_units),
        ).to_dense()

        binned_len = dense.shape[-1] // self.n_bins
        binned = torch.zeros((len(batch_index), self.nb_steps, binned_len), dtype=torch.float32)
        for bin_idx in range(binned_len):
            lo = self.n_bins * bin_idx
            hi = self.n_bins * (bin_idx + 1)
            binned[:, :, bin_idx] = dense[:, :, lo:hi].sum(dim=-1)

        targets = torch.tensor(self.labels[batch_index], dtype=torch.long)
        self.counter += 1
        return binned, targets


def load_ssc_iterators(config) -> Tuple[SSCSpikeIterator, SSCSpikeIterator, SSCSpikeIterator]:
    root_path = config.datasets_path
    dataset = config.dataset.lower()
    train_path = os.path.join(root_path, f"{dataset}_train.h5")
    valid_path = os.path.join(root_path, f"{dataset}_valid.h5")
    test_path = os.path.join(root_path, f"{dataset}_test.h5")

    missing = [path for path in [train_path, valid_path, test_path] if not os.path.exists(path)]
    if missing:
        missing_str = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "SSC dataset files were not found.\n"
            f"Expected these HDF5 files under `datasets_path={root_path}`:\n"
            f"{missing_str}\n"
            "Use `--datasets-path /abs/path/to/SSC` or update the config."
        )

    train_file = h5py.File(train_path, "r")
    valid_file = h5py.File(valid_path, "r")
    test_file = h5py.File(test_path, "r")

    train_loader = SSCSpikeIterator(
        train_file,
        batch_size=config.batch_size,
        nb_steps=config.time_window,
        nb_units=700,
        max_time=1.4,
        n_bins=config.n_bins,
        shuffle=True,
    )
    valid_loader = SSCSpikeIterator(
        valid_file,
        batch_size=config.batch_size,
        nb_steps=config.time_window,
        nb_units=700,
        max_time=1.4,
        n_bins=config.n_bins,
        shuffle=False,
    )
    test_loader = SSCSpikeIterator(
        test_file,
        batch_size=config.batch_size,
        nb_steps=config.time_window,
        nb_units=700,
        max_time=1.4,
        n_bins=config.n_bins,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader


class DelayedRecBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        config,
        spike_grad,
        surrogate_deriv,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = config
        self.surrogate_deriv = surrogate_deriv
        self.ff = nn.Linear(input_dim, hidden_dim, bias=config.bias)
        self.ff_dropout = nn.Dropout(config.feedforward_dropout_rate)
        self.rec_dropout_rate = float(config.recurrent_dropout_rate)

        beta_init = math.exp(-1.0 / max(float(config.tau_init), 1e-3))
        self.lif = snn.Leaky(
            beta=beta_init,
            threshold=config.threshold_init,
            spike_grad=spike_grad,
            learn_beta=config.learn_beta,
            learn_threshold=config.learn_threshold,
            reset_mechanism=config.reset_mechanism,
            reset_delay=True,
        )

        self.recurrent_weights = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.recurrent_delays = nn.Parameter(torch.zeros(hidden_dim))
        self.use_sig_p = bool(config.use_sig_p)
        if self.use_sig_p:
            self.p_spread = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter("p_spread", None)

        self.sigma = float(config.sigma_init)
        self.init_recurrent_weights()
        self.init_recurrent_delays()

    def init_recurrent_weights(self) -> None:
        nn.init.orthogonal_(self.recurrent_weights, gain=float(self.config.rec_delay_init_gain))

    def init_recurrent_delays(self) -> None:
        with torch.no_grad():
            if self.config.init_rec_delay == "half_normal":
                delays = torch.abs(torch.randn_like(self.recurrent_delays) * float(self.config.delay_std_init))
                self.recurrent_delays.copy_(delays)
            elif self.config.init_rec_delay == "uniform":
                nn.init.uniform_(self.recurrent_delays, a=0.0, b=float(self.config.delay_std_init))
            else:
                raise ValueError(f"Unsupported init_rec_delay={self.config.init_rec_delay!r}")
            self.recurrent_delays.clamp_(min=0.0)

    def update_sigma(self, current_epoch: int) -> None:
        decay_per_epoch = float(self.config.sigma_decay) ** (100.0 / max(int(self.config.epochs), 1))
        self.sigma = float(self.config.sigma_init) * (decay_per_epoch ** current_epoch)

    def clamp_recurrent_delays(self) -> None:
        with torch.no_grad():
            self.recurrent_delays.clamp_(min=0.0)

    def _build_delay_mask(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, int]:
        delays = self.recurrent_delays.to(device=device, dtype=dtype)
        if self.use_sig_p:
            spread = 1.0 + 2.0 * self.sigma * torch.sigmoid(self.p_spread).to(device=device, dtype=dtype)
            spread_max = spread.max()
            spread = spread.unsqueeze(1)
        else:
            spread = torch.tensor(1.0 + float(self.sigma), device=device, dtype=dtype)
            spread_max = spread

        max_delay = torch.max(delays)
        length = int(torch.ceil(1.0 + max_delay + spread_max).item()) + 1
        support = torch.arange(length, device=device, dtype=dtype)
        mask = torch.clamp(spread - torch.abs(support.unsqueeze(0) - (1.0 + delays).unsqueeze(1)), min=0.0) / (spread * spread)
        return mask, length

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        mask, buffer_len = self._build_delay_mask(device=device, dtype=dtype)
        return {
            "mem": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "buffer": torch.zeros(batch_size, self.hidden_dim, buffer_len, device=device, dtype=dtype),
            "pointer": torch.tensor(0, device=device, dtype=torch.long),
            "mask": mask,
        }

    def step(
        self,
        x_t: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        current = self.ff_dropout(self.ff(x_t))
        current_pointer = int(state["pointer"].item())
        rec_now = state["buffer"][:, :, current_pointer]

        if self.training and self.rec_dropout_rate > 0.0:
            keep_prob = 1.0 - self.rec_dropout_rate
            rec_mask = torch.bernoulli(torch.full_like(rec_now, keep_prob)) / keep_prob
        else:
            rec_mask = torch.ones_like(rec_now)

        rec_now_dropped = rec_now * rec_mask
        spk, mem = self.lif(current + rec_now_dropped, state["mem"])

        y_masked = spk.unsqueeze(2) * state["mask"].unsqueeze(0)
        delayed_rec = torch.einsum("io,bol->bil", self.recurrent_weights, y_masked)

        buffer = state["buffer"]
        buffer[:, :, current_pointer] = 0.0
        next_pointer = (current_pointer + 1) % buffer.shape[2]

        first_chunk = min(buffer.shape[2] - 1, buffer.shape[2] - next_pointer)
        if first_chunk > 0:
            buffer[:, :, next_pointer:next_pointer + first_chunk].add_(delayed_rec[:, :, 1:1 + first_chunk])

        remaining = buffer.shape[2] - 1 - first_chunk
        if remaining > 0:
            buffer[:, :, 0:remaining].add_(delayed_rec[:, :, 1 + first_chunk:1 + first_chunk + remaining])

        state["mem"] = mem
        state["buffer"] = buffer
        state["pointer"] = torch.tensor(next_pointer, device=x_t.device, dtype=torch.long)
        reg_ctx = {
            "buffer_pointer": torch.tensor(current_pointer, device=x_t.device, dtype=torch.long),
            "rec_mask": rec_mask,
            "delay_mask": state["mask"],
        }
        return spk, mem, state, reg_ctx


class DelRecSNNtorchSSC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spike_grad, self.surrogate_deriv = build_surrogate_pair(config.surrogate_type, config.surrogate_alpha)
        dims = [config.input_size] + list(config.hidden_layers)
        self.hidden_blocks = nn.ModuleList(
            [
                DelayedRecBlock(dims[idx], dims[idx + 1], config, self.spike_grad, self.surrogate_deriv)
                for idx in range(len(config.hidden_layers))
            ]
        )
        beta_init = math.exp(-1.0 / max(float(config.tau_init), 1e-3))
        self.fc_out = nn.Linear(dims[-1], config.output_size, bias=config.bias)
        self.lif_out = snn.Leaky(
            beta=beta_init,
            threshold=config.threshold_init,
            spike_grad=self.spike_grad,
            learn_beta=config.learn_beta,
            learn_threshold=False,
            reset_mechanism="none",
            reset_delay=True,
        )
        self.delay_aware_reg = DelRecDelayAwareRegularizer(
            threshold=config.threshold_init,
            surrogate_type=config.surrogate_type,
            surrogate_alpha=config.surrogate_alpha,
            detach_state=False,
        )

        self._lif_layers: List[Tuple[str, nn.Module]] = [
            (f"hidden{idx + 1}", block.lif) for idx, block in enumerate(self.hidden_blocks)
        ] + [("output", self.lif_out)]

    def forward(self, x: torch.Tensor, reg_mode: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, T, N], got {tuple(x.shape)}")

        batch_size, steps, _ = x.shape
        dtype = x.dtype
        device = x.device
        block_states = [block.init_state(batch_size=batch_size, device=device, dtype=dtype) for block in self.hidden_blocks]
        mem_out = torch.zeros(batch_size, self.config.output_size, device=device, dtype=dtype)

        layer_loss_sums: Dict[str, torch.Tensor] = {}
        active_layers: List[str] = []
        if reg_mode is not None:
            self.delay_aware_reg.reset()

        mem_out_rec = []
        for t in range(steps):
            cur = x[:, t, :]
            for idx, block in enumerate(self.hidden_blocks):
                spk, mem, block_states[idx], reg_ctx = block.step(cur, block_states[idx])
                if reg_mode in {"hajr", "asar"}:
                    layer_name = f"hidden{idx + 1}"
                    layer_loss_sums[layer_name] = layer_loss_sums.get(
                        layer_name,
                        torch.tensor(0.0, device=device, dtype=dtype),
                    ) + self.delay_aware_reg.step(
                        layer_id=layer_name,
                        mem=mem,
                        spike=spk,
                        beta=block.lif.beta,
                        timestep=t,
                        reset_mechanism=map_snntorch_reset(block.lif.reset_mechanism),
                        threshold=block.lif.threshold,
                        surrogate_deriv=block.surrogate_deriv,
                        decision_gate=(reg_mode == "hajr"),
                        recurrent_weight=block.recurrent_weights,
                        delay_mask=reg_ctx["delay_mask"],
                        buffer_pointer=int(reg_ctx["buffer_pointer"].item()),
                        rec_input_mask=reg_ctx["rec_mask"],
                    )
                    if layer_name not in active_layers:
                        active_layers.append(layer_name)
                cur = spk

            logits = self.fc_out(cur)
            spk_out, mem_out = self.lif_out(logits, mem_out)
            if reg_mode in {"hajr", "asar"}:
                layer_loss_sums["output"] = layer_loss_sums.get(
                    "output",
                    torch.tensor(0.0, device=device, dtype=dtype),
                ) + self.delay_aware_reg.step(
                    layer_id="output",
                    mem=mem_out,
                    spike=spk_out,
                    beta=self.lif_out.beta,
                    timestep=t,
                    reset_mechanism="none",
                    threshold=self.lif_out.threshold,
                    surrogate_deriv=self.surrogate_deriv,
                    decision_gate=False,
                )
                if "output" not in active_layers:
                    active_layers.append("output")
            mem_out_rec.append(mem_out)

        reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if active_layers:
            per_layer_losses = []
            for layer_name in active_layers:
                denom = steps if layer_name == "output" else max(steps - 1, 1)
                per_layer_losses.append(layer_loss_sums[layer_name] / denom)
            reg_loss = torch.stack(per_layer_losses).mean()

        return torch.stack(mem_out_rec, dim=0), reg_loss

    def get_beta_values(self) -> List[float]:
        return [float(layer.beta.detach().view(-1)[0].item()) for _, layer in self._lif_layers]

    def update_sigma(self, current_epoch: int) -> None:
        for block in self.hidden_blocks:
            block.update_sigma(current_epoch)

    def clamp_after_step(self) -> None:
        with torch.no_grad():
            for _, layer in self._lif_layers:
                layer.beta.clamp_(0.01, 0.99)
                threshold = getattr(layer, "threshold", None)
                if threshold is not None and getattr(layer, "reset_mechanism", "subtract") != "none":
                    threshold.clamp_(0.1, 3.0)
            for block in self.hidden_blocks:
                block.clamp_recurrent_delays()

    def collect_snapshot(self) -> List[Dict[str, Any]]:
        snapshot = []
        for idx, block in enumerate(self.hidden_blocks):
            snapshot.append(
                {
                    "name": f"hidden{idx + 1}",
                    "reset_mechanism": getattr(block.lif, "reset_mechanism", "subtract"),
                    "beta": float(block.lif.beta.detach().view(-1)[0].item()),
                    "threshold": float(block.lif.threshold.detach().view(-1)[0].item()),
                    "recurrent_delay_mean": float(block.recurrent_delays.detach().mean().item()),
                    "recurrent_delay_std": float(block.recurrent_delays.detach().std().item()),
                    "sigma": float(block.sigma),
                }
            )
        snapshot.append(
            {
                "name": "output",
                "reset_mechanism": getattr(self.lif_out, "reset_mechanism", "none"),
                "beta": float(self.lif_out.beta.detach().view(-1)[0].item()),
                "threshold": float(self.lif_out.threshold.detach().view(-1)[0].item()),
            }
        )
        return snapshot


def _perturb_betas_for_training(model: DelRecSNNtorchSSC, cv: float) -> Dict[int, torch.Tensor]:
    originals: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for idx, (_, layer) in enumerate(model._lif_layers):
            originals[idx] = layer.beta.data.clone()
            noise = torch.randn_like(layer.beta) * cv
            layer.beta.data.copy_((layer.beta.data * (1.0 + noise)).clamp(0.01, 0.99))
    return originals


def _restore_betas(model: DelRecSNNtorchSSC, originals: Dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for idx, (_, layer) in enumerate(model._lif_layers):
            layer.beta.data.copy_(originals[idx])


def _apply_beta_noise(model: DelRecSNNtorchSSC, cv: float) -> Dict[int, torch.Tensor]:
    originals: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for idx, (_, layer) in enumerate(model._lif_layers):
            originals[idx] = layer.beta.detach().clone()
            noise = torch.randn_like(layer.beta) * cv
            layer.beta.copy_((layer.beta * (1.0 + noise)).clamp(0.01, 0.99))
    return originals


def _restore_beta_noise(model: DelRecSNNtorchSSC, originals: Dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for idx, (_, layer) in enumerate(model._lif_layers):
            layer.beta.copy_(originals[idx])


def _quantize_tensor_inplace(param: torch.Tensor, bits: int) -> None:
    max_abs = param.abs().max()
    if max_abs.item() == 0.0:
        return
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    param.copy_(torch.round(param / scale).clamp(-qmax, qmax) * scale)


def evaluate(model: DelRecSNNtorchSSC, loader: SSCSpikeIterator, device: torch.device) -> float:
    model.eval()
    loader.reset()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device).float()
            targets = targets.to(device)
            logits_t, _ = model(data, reg_mode=None)
            logits = logits_t.sum(dim=0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def evaluate_robustness(
    model: DelRecSNNtorchSSC,
    loader: SSCSpikeIterator,
    device: torch.device,
    cv_list: Iterable[float],
    mc_runs: int,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for cv in cv_list:
        cv = float(cv)
        if cv == 0.0:
            acc = evaluate(model, loader, device)
            results[str(cv)] = {"mean": acc, "std": 0.0, "runs": [acc]}
            continue

        runs = []
        for _ in range(int(mc_runs)):
            original = _apply_beta_noise(model, cv)
            runs.append(evaluate(model, loader, device))
            _restore_beta_noise(model, original)
        results[str(cv)] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0,
            "runs": runs,
        }
    return results


def evaluate_quantization(
    model: DelRecSNNtorchSSC,
    loader: SSCSpikeIterator,
    device: torch.device,
    bits_list: Iterable[int],
) -> Dict[str, float]:
    base_state = copy.deepcopy(model.state_dict())
    results: Dict[str, float] = {}
    with torch.no_grad():
        for bits in bits_list:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    _quantize_tensor_inplace(module.weight, int(bits))
                    if module.bias is not None:
                        _quantize_tensor_inplace(module.bias, int(bits))
                elif isinstance(module, DelayedRecBlock):
                    _quantize_tensor_inplace(module.recurrent_weights, int(bits))
            results[str(bits)] = evaluate(model, loader, device)
            model.load_state_dict(base_state)
    return results


def save_checkpoint(
    path: str,
    model: DelRecSNNtorchSSC,
    config,
    seed: int,
    mode: str,
    history: List[Dict[str, Any]],
    test_acc: float,
    best_val: float,
    best_epoch: int,
    beta_trace: Dict[str, List[float]],
) -> None:
    payload = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "config": asdict(config),
        "seed": seed,
        "mode": mode,
        "history": copy.deepcopy(history),
        "test_acc": float(test_acc),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "lif_parameters": model.collect_snapshot(),
        "beta_trace": copy.deepcopy(beta_trace),
    }
    torch.save(payload, path)


def train_one_mode(
    model: DelRecSNNtorchSSC,
    mode: str,
    loaders: Tuple[SSCSpikeIterator, SSCSpikeIterator, SSCSpikeIterator],
    config,
    seed: int,
    device: torch.device,
) -> Tuple[DelRecSNNtorchSSC, float, float, int, List[Dict[str, Any]], Dict[str, List[float]]]:
    train_loader, val_loader, test_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    history: List[Dict[str, Any]] = []
    beta_trace: Dict[str, List[float]] = {"batch": []}
    for idx in range(len(model._lif_layers)):
        beta_trace[f"layer{idx + 1}"] = []
    global_batch = 0

    for epoch in range(config.epochs):
        model.train()
        model.update_sigma(epoch)
        train_loader.reset()
        samples = 0
        correct = 0
        task_sum = 0.0
        reg_sum = 0.0

        ramp = min(1.0, float(epoch + 1) / max(int(config.rampup_epochs), 1))
        if mode == "hajr":
            lam = float(config.lambda_hajr) * ramp
        elif mode == "asar":
            lam = float(config.lambda_asar) * ramp
        else:
            lam = 0.0

        iterator = tqdm(train_loader, desc=f"s{seed} {mode} e{epoch + 1}", leave=False) if config.show_progress else train_loader
        for batch_idx, (data, targets) in enumerate(iterator):
            data = data.to(device).float()
            targets = targets.to(device)

            beta_originals = None
            if mode == "noise_aug":
                beta_originals = _perturb_betas_for_training(model, float(config.noise_aug_cv))

            reg_mode = mode if mode in {"hajr", "asar"} else None
            logits_t, loss_reg = model(data, reg_mode=reg_mode)
            logits = logits_t.sum(dim=0)
            loss_task = criterion(logits, targets)
            if reg_mode is None:
                loss_reg = torch.tensor(0.0, device=device, dtype=loss_task.dtype)

            loss = loss_task + lam * loss_reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))

            if beta_originals is not None:
                _restore_betas(model, beta_originals)

            optimizer.step()
            model.clamp_after_step()

            beta_values = model.get_beta_values()
            beta_trace["batch"].append(global_batch)
            for idx, beta_val in enumerate(beta_values):
                beta_trace[f"layer{idx + 1}"].append(beta_val)
            global_batch += 1

            batch_size = targets.size(0)
            samples += batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            task_sum += loss_task.item() * batch_size
            reg_sum += loss_reg.item() * batch_size

            if config.show_progress and hasattr(iterator, "set_postfix"):
                if (
                    batch_idx == 0
                    or (batch_idx + 1) % int(config.progress_update_interval) == 0
                    or (batch_idx + 1) == len(train_loader)
                ):
                    iterator.set_postfix(acc=f"{100.0 * correct / max(samples, 1):.1f}", reg=f"{loss_reg.item():.4f}")

        train_acc = 100.0 * correct / max(samples, 1)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        beta_values = model.get_beta_values()
        history.append(
            {
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "task_loss": task_sum / max(samples, 1),
                "reg_loss": reg_sum / max(samples, 1),
                "beta_mean": float(np.mean(beta_values)),
                "lambda": lam,
            }
        )
        print(
            f"  [s{seed}][{mode}] e{epoch + 1:03d} "
            f"train={train_acc:.2f} val={val_acc:.2f} beta_mean={np.mean(beta_values):.3f} lambda={lam:.5f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without producing a best checkpoint state.")

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"  [s{seed}][{mode}] BEST val={best_val:.2f}@e{best_epoch} -> test={test_acc:.2f}")
    return model, test_acc, best_val, best_epoch, history, beta_trace


def run_p0_experiment(config) -> Dict[str, Any]:
    config.ensure_derived()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_root = Path(config.results_dir) / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    with open(run_root / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    summary: Dict[str, Any] = {
        "device": str(device),
        "results_dir": str(run_root),
        "modes": {},
    }

    for seed in config.seeds:
        for mode in config.modes:
            config.seed = int(seed)
            seed_everything(int(seed))
            loaders = load_ssc_iterators(config)
            print(f"\n=== Seed {seed} | Mode: {mode} ===")
            model = DelRecSNNtorchSSC(config).to(device)
            model, test_acc, best_val, best_epoch, history, beta_trace = train_one_mode(
                model=model,
                mode=mode,
                loaders=loaders,
                config=config,
                seed=int(seed),
                device=device,
            )
            robustness = evaluate_robustness(
                model=model,
                loader=loaders[2],
                device=device,
                cv_list=config.eval_cvs,
                mc_runs=config.eval_mc_runs,
            )
            quantization = evaluate_quantization(
                model=model,
                loader=loaders[2],
                device=device,
                bits_list=config.quant_bits,
            )

            mode_bucket = summary["modes"].setdefault(mode, {"per_seed": []})
            mode_bucket["per_seed"].append(
                {
                    "seed": int(seed),
                    "test_acc": float(test_acc),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "history": history,
                    "robustness": robustness,
                    "quantization": quantization,
                    "lif_parameters": model.collect_snapshot(),
                }
            )

            ckpt_path = run_root / "checkpoints" / f"ssc_delrec_snntorch_s{seed}_{mode}.pth"
            save_checkpoint(
                path=str(ckpt_path),
                model=model,
                config=config,
                seed=int(seed),
                mode=mode,
                history=history,
                test_acc=test_acc,
                best_val=best_val,
                best_epoch=best_epoch,
                beta_trace=beta_trace,
            )

            with open(run_root / "summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

    for mode, mode_summary in summary["modes"].items():
        test_accs = [item["test_acc"] for item in mode_summary["per_seed"]]
        mode_summary["mean_test_acc"] = float(np.mean(test_accs)) if test_accs else 0.0
        mode_summary["std_test_acc"] = float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0

    with open(run_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
