"""Beta-mismatch draws: beta_tilde = clip(beta * (1 + CV * z), 0.01, 0.99).

Modes
  per_neuron   : beta expanded to one value per neuron (per channel for conv
                 layers); independent z ~ N(0,1) per neuron/channel per layer.
  layer_shared : one z ~ N(0,1) per layer, shared by all neurons of the layer.
  global       : one z ~ N(0,1) shared by every layer (chip-wide drift).

Pairing / common random numbers: z depends only on (setting, training seed,
mode, draw index) - NOT on the method (Clean / NoiseAug / MARS) and NOT on CV.
Hence the three methods of a given training seed see identical z draws, and
the CV sweep is a deterministic rescaling of the same z (smooth curves).
"""

from __future__ import annotations

import hashlib

import torch

MODES = ("per_neuron", "layer_shared", "global")


def draw_seed(setting: str, train_seed: int, mode: str, draw: int) -> int:
    key = f"{setting}|{train_seed}|{mode}|{draw}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1)


def draw_z(layers, setting: str, train_seed: int, mode: str, draw: int) -> list[torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(draw_seed(setting, train_seed, mode, draw))
    if mode == "per_neuron":
        return [torch.randn(layer.neuron_shape, generator=gen) for layer in layers]
    if mode == "layer_shared":
        return [torch.randn((), generator=gen) for _ in layers]
    if mode == "global":
        z = torch.randn((), generator=gen)
        return [z.clone() for _ in layers]
    raise ValueError(mode)


def apply(layers, zs, cv: float) -> dict:
    """Set perturbed betas on the model; returns realized statistics."""
    rel = []
    for layer, z in zip(layers, zs):
        base = layer.base_beta.detach().cpu().float()
        if z.dim() > 0:
            base = base.expand(z.shape) if base.numel() == 1 else base.reshape(z.shape)
        pert = (base * (1.0 + cv * z)).clamp(0.01, 0.99)
        rel.append(float(torch.sqrt(torch.mean(((pert - base) / base.clamp_min(1e-12)) ** 2))))
        layer.set_beta(pert)
    return {"beta_relative_rms": rel}


def restore(layers) -> None:
    for layer in layers:
        layer.restore()
