"""Repository-relative locations used by the training, evaluation and artifact scripts.

All paths can be overridden with environment variables (relative values are resolved
against the repository root):

  MARS_DATA_ROOT    datasets            (default: data/)
  MARS_CKPT_ROOT    trained checkpoints (default: checkpoints/)
  MARS_OUTPUT_ROOT  generated files     (default: outputs/)
  MARS_RESULTS_ROOT result files read by paper_artifacts/ (default: results/, the shipped files)

Checkpoint layout expected by eval/, analysis/ and diagnostics/ (one file per training
seed and mode; mode strings: clean, noise_aug, asar (= MARS), hajr (surrogate-gated ablation)):

  checkpoints/SHD-FC/best_s{seed}_{mode}.pth
  checkpoints/SHD-RSNN/best_s{seed}_{mode}.pth
  checkpoints/SSC/best_s{seed}_{mode}.pth
  checkpoints/DVS-Gesture/best_s{seed}_{mode}.pth
  checkpoints/CIFAR-10/best_s{seed}_{mode}.pth
  checkpoints/SHD-FC-controls/best_s{seed}_{control}.pth    (beta_l2, beta_clip, fixed_homogeneous,
                                                             fixed_heterogeneous, weight_decay, spectral_norm)
  checkpoints/SHD-RSNN-controls/best_s{seed}_{control}.pth  (beta_l2, beta_clip, fixed_homogeneous,
                                                             fixed_heterogeneous, dropconnect)

Data layout:

  data/tonic/SHD/shd_{train,test}.h5          (tonic.datasets.SHD)
  data/tonic/DVSGesture/ibmGesture{Train,Test} (tonic.datasets.DVSGesture)
  data/cifar10/cifar-10-batches-py             (torchvision.datasets.CIFAR10)
  data/SSC/ssc_{train,valid,test}.h5           (Heidelberg SSC release)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else REPO_ROOT / p


DATA_ROOT = _env_path("MARS_DATA_ROOT", "data")
CKPT_ROOT = _env_path("MARS_CKPT_ROOT", "checkpoints")
OUTPUT_ROOT = _env_path("MARS_OUTPUT_ROOT", "outputs")
RESULTS = _env_path("MARS_RESULTS_ROOT", "results")

TONIC_ROOT = DATA_ROOT / "tonic"
CIFAR_ROOT = DATA_ROOT / "cifar10"
SSC_ROOT = DATA_ROOT / "SSC"


def repo_root() -> Path:
    return REPO_ROOT


def ckpt_dir(setting: str) -> Path:
    """Directory holding best_s{seed}_{mode}.pth for a setting (e.g. 'SHD-FC', 'SSC')."""
    return CKPT_ROOT / setting


def output_dir(*parts: str) -> Path:
    """Create and return outputs/<parts...>."""
    p = OUTPUT_ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
