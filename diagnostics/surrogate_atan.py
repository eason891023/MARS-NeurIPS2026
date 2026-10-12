"""
Correct snnTorch ATan surrogate derivative + the legacy (wrong) formula used by
the pre-camera-ready diagnostic scripts, kept only for old->new comparison.

snnTorch (snntorch.surrogate.ATan.backward, v0.9.4):
    sigma'(x) = (alpha/2) / (1 + (pi*alpha*x/2)^2),   max = alpha/2  (=1 for alpha=2)
Legacy formula in eval_* scripts (WRONG for snnTorch):
    sigma'(x) = alpha / (pi * (1 + (alpha*x)^2)),       max = alpha/pi (=2/pi)
"""
import math
import os
import sys

import torch


def _find_project_root(start_file: str) -> str:
    cur = os.path.abspath(os.path.dirname(start_file))
    while True:
        if os.path.isdir(os.path.join(cur, "snn_robust_tau")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot locate project root")
        cur = parent


PROJECT_ROOT = _find_project_root(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from snn_robust_tau.loss.hajr import surrogate_derivative_atan  # noqa: E402  (training formula)


def atan_deriv(x, alpha=2.0):
    """Correct snnTorch ATan derivative (identical to training regularizer)."""
    return surrogate_derivative_atan(x, alpha)


def atan_deriv_legacy(x, alpha=2.0):
    """Legacy WRONG derivative used by old diagnostics (for comparison only)."""
    return alpha / (math.pi * (1.0 + (alpha * x) ** 2))


def l_sigma_atan(alpha=2.0):
    return alpha / 2.0


def l_sigma_atan_legacy(alpha=2.0):
    return alpha / math.pi


FORMULAS = {
    "fixed": (atan_deriv, l_sigma_atan),
    "legacy": (atan_deriv_legacy, l_sigma_atan_legacy),
}


def check_against_snntorch(alpha=2.0):
    """Return max abs error between atan_deriv and snnTorch autograd."""
    from snntorch import surrogate
    fn = surrogate.atan(alpha=alpha)
    x = torch.linspace(-3, 3, 2001, requires_grad=True)
    fn(x).sum().backward()
    ref = x.grad.detach()
    with torch.no_grad():
        err_fixed = (atan_deriv(x.detach(), alpha) - ref).abs().max().item()
        err_legacy = (atan_deriv_legacy(x.detach(), alpha) - ref).abs().max().item()
    return {"max_err_fixed": err_fixed, "max_err_legacy": err_legacy,
            "snntorch_peak": float(ref.max()), "L_sigma_fixed": l_sigma_atan(alpha),
            "L_sigma_legacy": l_sigma_atan_legacy(alpha)}


if __name__ == "__main__":
    print(check_against_snntorch())
