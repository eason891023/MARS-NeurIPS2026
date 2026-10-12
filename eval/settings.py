"""Setting adapters for the unified beta-mismatch evaluation (inference only).

Each adapter exposes:
  * checkpoint(seed, method) -> Path
  * load(seed, method) -> nn.Module (eval mode, on DEVICE)
  * beta_layers(model) -> list[BetaLayer]
  * test_batches() -> list[(x, y)] cached on DEVICE (built once, full test set)
  * predict(model, x) -> logits (B, classes)

Model classes are imported from the training scripts in train/ (and
third_party/DelRec for SSC) so that the architecture is identical to the one
used for training. Checkpoints are read from CKPT_ROOT (env MARS_CKPT_ROOT,
default checkpoints/), one sub-directory per setting:
checkpoints/<setting>/best_s{seed}_{mode}.pth with <setting> in
{SHD-FC, SHD-RSNN, SSC, DVS-Gesture, CIFAR-10, SHD-FC-controls, SHD-RSNN-controls}
(see snn_robust_tau/paths.py).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from snn_robust_tau.paths import CIFAR_ROOT, CKPT_ROOT, SSC_ROOT, TONIC_ROOT, ckpt_dir  # noqa: E402

TRAIN = PROJECT_ROOT / "train"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 52, 72, 92, 102]
# Paper label -> checkpoint mode string. MARS == "asar" in code.
METHODS = {"clean": "clean", "noise_aug": "noise_aug", "mars": "asar"}

# Checkpoint directory per setting: CKPT_ROOT / <setting> / best_s{seed}_{mode}.pth
RESULT_DIRS = {s: ckpt_dir(s) for s in ("SHD-FC", "SHD-RSNN", "DVS-Gesture", "CIFAR-10", "SSC")}


def _import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_registered_tensor(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """Replace a (scalar) parameter/buffer by a tensor of another shape."""
    value = value.detach()
    if name in module._parameters:
        module._parameters[name] = nn.Parameter(value, requires_grad=False)
    elif name in module._buffers:
        module._buffers[name] = value
    else:
        setattr(module, name, value)


@dataclass
class BetaLayer:
    name: str
    module: nn.Module
    base_beta: torch.Tensor  # trained value (scalar tensor for all submitted checkpoints)
    neuron_shape: tuple  # shape used for per-neuron / per-channel expansion
    kind: str  # "snntorch" or "plif"

    def __post_init__(self) -> None:
        self._pname = "beta" if self.kind == "snntorch" else "w"
        self._orig = self.module._parameters.get(self._pname, None)
        if self._orig is None:
            self._orig = getattr(self.module, self._pname)

    def set_beta(self, beta: torch.Tensor) -> None:
        beta = beta.to(DEVICE, torch.float32)
        if self.kind == "snntorch":
            _replace_registered_tensor(self.module, "beta", beta)
        else:  # spikingjelly ParametricLIFNode: beta = 1 - sigmoid(w)
            beta = beta.clamp(0.01, 0.99)
            w = torch.log((1.0 - beta) / beta)
            _replace_registered_tensor(self.module, "w", w)

    def restore(self) -> None:
        """Put back the exact original (trained) tensor object."""
        if self._pname in self.module._parameters:
            self.module._parameters[self._pname] = self._orig
        else:
            setattr(self.module, self._pname, self._orig)


def _neuron_shape_from_input(shape: torch.Size) -> tuple:
    feat = tuple(shape[1:])
    if len(feat) == 1:  # dense layer: one beta per neuron
        return (feat[0],)
    if len(feat) == 3:  # conv layer (C,H,W): one beta per channel
        return (feat[0], 1, 1)
    raise ValueError(f"Unexpected LIF input shape {tuple(shape)}")


def _probe_shapes(adapter: "Adapter", model: nn.Module, modules: list[nn.Module]) -> list[tuple]:
    shapes: dict[int, tuple] = {}
    hooks = []
    for i, m in enumerate(modules):
        def hook(mod, args, i=i):
            if i not in shapes:
                shapes[i] = _neuron_shape_from_input(args[0].shape)
        hooks.append(m.register_forward_pre_hook(hook))
    x, _ = adapter.test_batches()[0]
    with torch.inference_mode():
        adapter.predict(model, x[:2])
    for h in hooks:
        h.remove()
    return [shapes[i] for i in range(len(modules))]


class Adapter:
    name: str
    eval_batch_size: int = 256

    def __init__(self) -> None:
        self._batches: list | None = None
        self._shapes: list | None = None

    # -- data -------------------------------------------------------------
    def _iterate_test(self):
        raise NotImplementedError

    def test_batches(self):
        if self._batches is None:
            xs, ys = [], []
            for x, y in self._iterate_test():
                xs.append(x.cpu())
                ys.append(torch.as_tensor(y).long().cpu())
            X = torch.cat(xs)
            Y = torch.cat(ys)
            store_dtype = torch.float16 if X.numel() > 3e8 else torch.float32
            if store_dtype == torch.float16:
                assert torch.equal(X.half().float(), X.float()), "fp16 cache not exact"
            X = X.to(store_dtype)
            bs = self.eval_batch_size
            self._batches = [
                (X[i:i + bs].to(DEVICE), Y[i:i + bs].to(DEVICE)) for i in range(0, len(Y), bs)
            ]
            self.n_test = int(len(Y))
        return self._batches

    # -- model ------------------------------------------------------------
    def checkpoint(self, seed: int, method: str) -> Path:
        return RESULT_DIRS[self.name] / f"best_s{seed}_{METHODS[method]}.pth"

    def load(self, seed: int, method: str) -> nn.Module:
        raise NotImplementedError

    def lif_modules(self, model) -> list[tuple[str, nn.Module]]:
        return list(model._lif_layers)

    def beta_layers(self, model) -> list[BetaLayer]:
        named = self.lif_modules(model)
        if self._shapes is None:
            self._shapes = _probe_shapes(self, model, [m for _, m in named])
        return [
            BetaLayer(n, m, m.beta.detach().clone(), s, "snntorch")
            for (n, m), s in zip(named, self._shapes)
        ]

    def predict(self, model, x):
        logits_t, _ = model(x.float(), reg_mode=None)
        return logits_t.sum(0)

    @torch.inference_mode()
    def accuracy(self, model) -> float:
        correct = 0
        total = 0
        for x, y in self.test_batches():
            pred = self.predict(model, x).argmax(1)
            correct += int((pred == y).sum())
            total += int(y.numel())
        return 100.0 * correct / total


# ---------------------------------------------------------------- SHD FC/RSNN
class SHDAdapter(Adapter):
    eval_batch_size = 512

    def __init__(self, arch: str) -> None:
        super().__init__()
        self.arch = arch
        self.name = "SHD-FC" if arch == "ff" else "SHD-RSNN"
        sys.path.insert(0, str(TRAIN / "common"))
        import submitted_eval_common as sec  # shared SHD loaders (train/common)
        self.sec = sec
        self._loader, self.input_size = sec.build_test_loader(time_steps=50, batch_size=256)

    def _iterate_test(self):
        for x, y in self._loader:
            yield x.float(), y

    def load(self, seed, method):
        model, ckpt = self.sec.load_submitted_model(self.arch, seed, METHODS[method], self.input_size)
        self.ckpt_meta = ckpt
        return model.eval()


# ---------------------------------------------------------------- DVS-Gesture
class DVSAdapter(Adapter):
    name = "DVS-Gesture"
    eval_batch_size = 8  # original test batch size

    def __init__(self) -> None:
        super().__init__()
        self.src = _import_file(TRAIN / "dvs_gesture.py", "cr_dvs_src")

    def _iterate_test(self):
        import tonic
        import tonic.transforms as T
        from torch.utils.data import DataLoader

        transform = T.Compose([T.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_time_bins=16)])
        ds = tonic.datasets.DVSGesture(save_to=str(TONIC_ROOT), train=False, transform=transform)
        for x, y in DataLoader(ds, batch_size=8, shuffle=False, num_workers=4):
            yield x.float(), y

    def load(self, seed, method):
        from snntorch import surrogate

        ckpt = torch.load(self.checkpoint(seed, method), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = self.src.DVSGestureSNN(
            num_classes=cfg["num_classes"], beta=cfg["beta_init"], threshold=cfg["threshold_init"],
            surrogate_fn=surrogate.atan(alpha=cfg["surrogate_alpha"]),
            learn_beta=cfg["learn_beta"], learn_threshold=cfg["learn_threshold"],
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.ckpt_meta = ckpt
        return model.to(DEVICE).eval()


# ---------------------------------------------------------------- CIFAR-10
class CIFARAdapter(Adapter):
    name = "CIFAR-10"
    eval_batch_size = 64  # original test batch; reproduces stored test_acc exactly (cuDNN TF32 algo is batch-dependent)

    def __init__(self) -> None:
        super().__init__()
        self.src = _import_file(TRAIN / "cifar10.py", "cr_cifar_src")

    def _iterate_test(self):
        import torchvision
        import torchvision.transforms as tvt
        from torch.utils.data import DataLoader

        tf = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ds = torchvision.datasets.CIFAR10(root=str(CIFAR_ROOT), train=False, download=False, transform=tf)
        yield from DataLoader(ds, batch_size=500, shuffle=False, num_workers=4)

    def load(self, seed, method):
        from snntorch import surrogate

        ckpt = torch.load(self.checkpoint(seed, method), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = self.src.SpikingResNet18(
            num_classes=10, beta=cfg["beta_init"], threshold=cfg["threshold_init"],
            surrogate_fn=surrogate.atan(alpha=cfg["surrogate_alpha"]),
            learn_beta=cfg["learn_beta"], learn_threshold=cfg["learn_threshold"],
            num_steps=cfg["time_steps"],
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.ckpt_meta = ckpt
        return model.to(DEVICE).eval()


# ---------------------------------------------------------------- SSC (DelRec)
class SSCAdapter(Adapter):
    name = "SSC"
    eval_batch_size = 256

    def __init__(self) -> None:
        super().__init__()
        delrec = PROJECT_ROOT / "third_party" / "DelRec"
        sys.path.insert(0, str(delrec))
        sys.path.insert(0, str(PROJECT_ROOT))
        from configs.perf_SSC_p0_spikingjelly import Config
        from src.SSC.snn import SNN_recurrent_delays
        from src.SSC import trainer_p0_spikingjelly as tr
        from src import datasets as ds
        from src.utils import reset_states

        self.Config, self.Model, self.tr, self.ds, self.reset_states = Config, SNN_recurrent_delays, tr, ds, reset_states

    def _config(self):
        cfg = self.Config()
        cfg.datasets_path = str(SSC_ROOT)
        cfg.batch_size = 128
        cfg.seed = 0
        return cfg

    def _iterate_test(self):
        cfg = self._config()
        # Only the test iterator is needed; replicate SSC_dataloaders test branch.
        import h5py, os
        f = h5py.File(os.path.join(cfg.datasets_path, "ssc_test.h5"), "r")
        it = self.ds.SSC_SpikeIterator(f["spikes"], f["labels"], cfg.batch_size, cfg.time_window, 700, 1.4,
                                       n_bins=cfg.n_bins, shuffle=False)
        for x, y in it:
            yield x.float(), y

    def checkpoint(self, seed, method):
        return RESULT_DIRS["SSC"] / f"best_s{seed}_{METHODS[method]}.pth"

    def load(self, seed, method):
        ckpt = torch.load(self.checkpoint(seed, method), map_location="cpu", weights_only=False)
        cfg = self._config()
        model = self.Model(cfg).to(DEVICE)
        self.tr.load_checkpoint_snapshot(model, ckpt)
        self.ckpt_meta = ckpt
        return model.eval()

    def lif_modules(self, model):
        return self.tr.get_plif_layers(model)

    def beta_layers(self, model):
        named = self.lif_modules(model)
        # PLIF nodes are driven via node.single_step_forward (no hooks fire);
        # the neuron count is the parent axonal_recdel width.
        self._shapes = [(int(model.layers[int(n.replace("recdel", ""))].neurons),) for n, _ in named]
        from src.spikingjelly_compat import beta_from_plif
        return [
            BetaLayer(n, m, beta_from_plif(m).detach().clone(), s, "plif")
            for (n, m), s in zip(named, self._shapes)
        ]

    def predict(self, model, x):
        self.reset_states(model)
        out = model(x.float().permute(1, 0, 2))
        return out.sum(0)


def make_adapter(setting: str) -> Adapter:
    if setting == "SHD-FC":
        return SHDAdapter("ff")
    if setting == "SHD-RSNN":
        return SHDAdapter("rsnn")
    if setting == "DVS-Gesture":
        return DVSAdapter()
    if setting == "CIFAR-10":
        return CIFARAdapter()
    if setting == "SSC":
        return SSCAdapter()
    raise KeyError(setting)


# ---------------------------------------------------------------- SHD simple-baseline controls
CONTROL_SOURCES = {
    # method -> (checkpoint dir, file mode string)
    "ff": {
        "beta_l2": (ckpt_dir("SHD-FC-controls"), "beta_l2"),
        "beta_clip": (ckpt_dir("SHD-FC-controls"), "beta_clip"),
        "fixed_homogeneous": (ckpt_dir("SHD-FC-controls"), "fixed_homogeneous"),
        "fixed_heterogeneous": (ckpt_dir("SHD-FC-controls"), "fixed_heterogeneous"),
        "weight_decay": (ckpt_dir("SHD-FC-controls"), "weight_decay"),
        "spectral_norm": (ckpt_dir("SHD-FC-controls"), "spectral_norm"),
    },
    "rsnn": {
        "beta_l2": (ckpt_dir("SHD-RSNN-controls"), "beta_l2"),
        "beta_clip": (ckpt_dir("SHD-RSNN-controls"), "beta_clip"),
        "fixed_homogeneous": (ckpt_dir("SHD-RSNN-controls"), "fixed_homogeneous"),
        "fixed_heterogeneous": (ckpt_dir("SHD-RSNN-controls"), "fixed_heterogeneous"),
        "dropconnect": (ckpt_dir("SHD-RSNN-controls"), "dropconnect"),
    },
}


class SHDControlsAdapter(SHDAdapter):
    """Simple-baseline controls; draws use the SAME seed key as the main SHD run."""

    def __init__(self, arch: str) -> None:
        super().__init__(arch)
        self.draw_name = self.name  # "SHD-FC" / "SHD-RSNN" -> identical z as Clean/NoiseAug/MARS
        self.methods = list(CONTROL_SOURCES[arch])
        self._ctrl_eval = _import_file(TRAIN / "baselines" / "control_eval.py", "cr_control_eval")

    def checkpoint(self, seed, method):
        root, mode = CONTROL_SOURCES[self.arch][method]
        return root / f"best_s{seed}_{mode}.pth"

    def load(self, seed, method):
        path = self.checkpoint(seed, method)
        if method in ("weight_decay", "spectral_norm"):
            from snntorch import surrogate
            src = _import_file(TRAIN / "baselines" / "shd_extra_baselines.py", "cr_shd_extra_src")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            model = src.SHDSNN(
                self.input_size, cfg["beta_init"], cfg["threshold_init"],
                surrogate.atan(alpha=cfg["surrogate_alpha"]),
                learn_beta=cfg["learn_beta"], learn_threshold=cfg["learn_threshold"],
                use_spectral_norm=(method == "spectral_norm"),
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model = model.to(DEVICE).eval()
        else:
            model, ckpt = self._ctrl_eval.load_control_model(path, self.input_size)
        self.ckpt_meta = ckpt
        return model.eval()


def make_control_adapter(setting: str) -> Adapter:
    return SHDControlsAdapter("ff" if setting == "SHD-FC" else "rsnn")
