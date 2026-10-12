"""
HAJR: Hardware-Aware Jacobian Regularization Loss
==================================================
A Decision-Aware robustness loss for Spiking Neural Networks.

Unlike ASAR (which minimizes ||Ψ||²), HAJR minimizes ||σ' ⊙ Ψ||²,
where σ' is the surrogate gradient derivative and Ψ is the state sensitivity.

This is "Decision-Aware" because:
- When V is far from threshold: σ' ≈ 0 → Loss ≈ 0 (safe zone, no penalty)
- When V ≈ threshold: σ' is large → Loss forces Ψ to be small (danger zone)

Mixed HAJR/ASAR:
- Hidden layers (with reset): Use HAJR (soft-reset Ψ + σ' gate)
- Output layer (no reset): Use ASAR (no-reset Ψ, no gate)
This is correct because non-reset neurons have no threshold firing decision
to protect, so we directly minimize membrane sensitivity ||Ψ||².

Mathematical Derivation:
------------------------
Soft Reset LIF:
    V[t] = β·V[t-1] + I[t] - V_th·S[t-1]
    Ψ[t] = V[t-1] + (β - V_th·σ'(V[t-1] - V_th))·Ψ[t-1]
    J[t] = σ'(V[t] - V_th) ⊙ Ψ[t]
    HAJR Loss = (1/T) Σ ||J[t]||²

No-Reset LIF (output layer):
    V[t] = β·V[t-1] + I[t]
    Ψ[t] = V[t-1] + β·Ψ[t-1]
    ASAR Loss = (1/T) Σ ||Ψ[t]||²

Naming in this code base: the paper's MARS objective is mode "asar" (OnlineHAJR with
decision_gate=False, i.e. the ungated ||Psi_t||^2 penalty); mode "hajr" (gated by sigma') is the
surrogate-gated ablation reported in the paper's diagnostics.
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Dict, Any, Tuple


# =============================================================================
# Surrogate Gradient Derivatives
# =============================================================================
def surrogate_derivative_atan(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    Derivative of the ATan surrogate gradient function.
    Matches snnTorch's ATan surrogate (snntorch.surrogate.ATan):
        S(x) ≈ (1/π)·arctan(π·α/2·x) + 0.5
        S'(x) = (α/2) / (1 + (π·α/2·x)²)

    Verified numerically against snntorch.surrogate.atan(alpha).
    Peak at x=0: S'(0) = α/2  (e.g., α=2.0 → S'(0) = 1.0)
    """
    return (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * x).pow(2))


def surrogate_derivative_fast_sigmoid(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    """
    Derivative of the Fast Sigmoid surrogate gradient function.
        S(x) = x / (1 + slope·|x|) * 0.5 + 0.5
        S'(x) = slope / (1 + slope·|x|)²

    Peak at x=0: S'(0) = slope  (e.g., slope=25 → S'(0) = 25.0)
    """
    denom = 1.0 + slope * torch.abs(x)
    return slope / (denom ** 2)


def surrogate_derivative_triangle(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Derivative of the Triangle/ZIF surrogate gradient function.
        S'(x) = (1/γ²) · max(0, γ - |x|)

    Peak at x=0: S'(0) = 1.0  (for γ=1.0)
    """
    return (1.0 / gamma) * (1.0 / gamma) * torch.clamp(gamma - torch.abs(x), min=0.0)


# =============================================================================
# Surrogate Consistency Checking
# =============================================================================
def detect_surrogate_from_snntorch(leaky_layer) -> Tuple[str, float]:
    """
    Detect the surrogate gradient type and parameter from an snnTorch Leaky layer.

    Handles both:
    - snnTorch built-in surrogates (closures with __qualname__ like 'atan.<locals>.inner')
    - Custom nn.Module surrogates (e.g., ZIFSurrogate with gamma attribute)

    Args:
        leaky_layer: An snn.Leaky (or snn.RLeaky) instance

    Returns:
        (surrogate_type, surrogate_param) tuple, e.g., ('atan', 2.0)

    Raises:
        ValueError: If the surrogate type cannot be detected
    """
    import snntorch as snn

    if not isinstance(leaky_layer, (snn.Leaky, snn.RLeaky)):
        raise TypeError(f"Expected snn.Leaky or snn.RLeaky, got {type(leaky_layer)}")

    spike_grad = leaky_layer.spike_grad

    # --- Strategy 1: snnTorch closure-based surrogates ---
    # snnTorch surrogates are closures: surr.atan(alpha=2.0) returns a function
    # with __qualname__ = 'atan.<locals>.inner' and closure var 'alpha'
    qualname = getattr(spike_grad, '__qualname__', '')
    if qualname:
        qualname_lower = qualname.lower()

        # Extract closure variables
        closure_vars = {}
        if hasattr(spike_grad, '__closure__') and spike_grad.__closure__:
            for name, cell in zip(
                spike_grad.__code__.co_freevars, spike_grad.__closure__
            ):
                try:
                    closure_vars[name] = cell.cell_contents
                except ValueError:
                    pass

        if 'atan' in qualname_lower:
            alpha = closure_vars.get('alpha', 2.0)
            return ('atan', float(alpha))

        if 'fast_sigmoid' in qualname_lower:
            slope = closure_vars.get('slope', 25.0)
            return ('fast_sigmoid', float(slope))

        if 'sigmoid' in qualname_lower and 'fast' not in qualname_lower:
            slope = closure_vars.get('slope', 25.0)
            return ('fast_sigmoid', float(slope))

    # --- Strategy 2: nn.Module-based custom surrogates ---
    # e.g., ZIFSurrogate(gamma=1.0) which is an nn.Module with .gamma
    cls_name = type(spike_grad).__name__.lower()

    if 'triangle' in cls_name or 'zif' in cls_name:
        gamma = getattr(spike_grad, 'gamma', 1.0)
        return ('triangle', float(gamma))

    if 'atan' in cls_name:
        alpha = getattr(spike_grad, 'alpha', 2.0)
        return ('atan', float(alpha))

    if 'sigmoid' in cls_name:
        slope = getattr(spike_grad, 'slope', 25.0)
        return ('fast_sigmoid', float(slope))

    raise ValueError(
        f"Cannot auto-detect surrogate type from spike_grad "
        f"(type={type(spike_grad).__name__}, qualname='{qualname}'). "
        f"Please manually specify surrogate_type and surrogate_alpha. "
        f"Supported types: 'atan', 'fast_sigmoid', 'triangle'"
    )


def validate_surrogate_consistency(
    model: nn.Module,
    surrogate_type: str,
    surrogate_alpha: float,
    strict: bool = True
) -> List[str]:
    """
    Validate that the model's surrogate gradient matches the HAJR loss configuration.

    This check is CRITICAL for correctness: if the model trains with ATan(α=2.0)
    but HAJR computes sensitivity with Triangle(γ=1.0), the criticality gate σ'
    will be wrong, leading to incorrect gradient signals.

    Args:
        model: The SNN model (must contain snn.Leaky layers)
        surrogate_type: The surrogate type configured in HAJRLoss
        surrogate_alpha: The surrogate parameter configured in HAJRLoss
        strict: If True, raise ValueError on mismatch. If False, return warnings.

    Returns:
        List of warning/error messages (empty if all consistent)
    """
    import snntorch as snn

    issues = []

    for name, module in model.named_modules():
        if not isinstance(module, (snn.Leaky, snn.RLeaky)):
            continue

        try:
            detected_type, detected_param = detect_surrogate_from_snntorch(module)
        except (ValueError, TypeError) as e:
            issues.append(f"[{name}] Cannot detect surrogate: {e}")
            continue

        # Check type match
        if detected_type != surrogate_type:
            msg = (f"[{name}] Surrogate MISMATCH: model uses '{detected_type}' "
                   f"but HAJR configured with '{surrogate_type}'. "
                   f"This will cause incorrect sensitivity computation!")
            issues.append(msg)
        # Check parameter match
        elif not math.isclose(detected_param, surrogate_alpha, rel_tol=0.01):
            msg = (f"[{name}] Surrogate parameter MISMATCH: model uses "
                   f"{detected_type}({detected_param}) but HAJR configured "
                   f"with {surrogate_type}({surrogate_alpha})")
            issues.append(msg)

    if issues and strict:
        raise ValueError(
            "Surrogate gradient inconsistency detected!\n" +
            "\n".join(issues) +
            "\nFix: ensure HAJRLoss surrogate_type and surrogate_alpha match "
            "the model's spike_grad exactly."
        )

    return issues


# =============================================================================
# HAJR Loss (Batch Version - Post Forward Pass)
# =============================================================================
class HAJRLoss(nn.Module):
    """
    Hardware-Aware Jacobian Regularization Loss with mixed HAJR/ASAR support.

    For layers with reset mechanism (soft/hard reset):
        Uses HAJR: ||σ'(V-θ) ⊙ Ψ||² with soft-reset Ψ recurrence

    For layers without reset (output layer, reset_mechanism='none'):
        Uses ASAR: ||Ψ||² with no-reset Ψ recurrence

    This mixed approach is correct because:
    - Reset layers have threshold decisions to protect → use criticality gate σ'
    - Non-reset layers accumulate membrane freely → minimize raw sensitivity Ψ

    IMPORTANT - Surrogate Consistency:
        The surrogate_type and surrogate_alpha MUST match the model's spike_grad
        exactly. Use validate_surrogate() to check before training.

    Args:
        threshold: Spike firing threshold (default: 1.0)
        mem_is_post_reset: True if input mem is post-reset (SpikingJelly),
                          False if pre-reset (SNNTorch)
        surrogate_type: 'atan', 'fast_sigmoid', or 'triangle'
        surrogate_alpha: Parameter for surrogate gradient function
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self,
                 threshold: float = 1.0,
                 mem_is_post_reset: bool = False,
                 surrogate_type: str = 'atan',
                 surrogate_alpha: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.threshold = threshold
        self.mem_is_post_reset = mem_is_post_reset
        self.surrogate_type = surrogate_type
        self.surrogate_alpha = surrogate_alpha
        self.reduction = reduction

        # Select surrogate derivative function
        if surrogate_type == 'atan':
            self.surrogate_deriv = lambda x: surrogate_derivative_atan(x, surrogate_alpha)
        elif surrogate_type == 'fast_sigmoid':
            self.surrogate_deriv = lambda x: surrogate_derivative_fast_sigmoid(x, surrogate_alpha)
        elif surrogate_type == 'triangle':
            self.surrogate_deriv = lambda x: surrogate_derivative_triangle(x, surrogate_alpha)
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}. "
                             f"Supported: 'atan', 'fast_sigmoid', 'triangle'")

    def validate_surrogate(self, model: nn.Module, strict: bool = True) -> List[str]:
        """
        Validate that the model's surrogate matches this loss's configuration.
        Call this ONCE before training starts.

        Args:
            model: The SNN model
            strict: If True, raise ValueError on mismatch

        Returns:
            List of issues found (empty if consistent)
        """
        return validate_surrogate_consistency(
            model, self.surrogate_type, self.surrogate_alpha, strict=strict
        )

    def forward(self,
                mem_rec_list: List[torch.Tensor],
                beta_list: List[Union[torch.Tensor, float]],
                spike_rec_list: Optional[List[torch.Tensor]] = None,
                threshold_list: Optional[List[Union[torch.Tensor, float]]] = None,
                reset_mechanism_list: Optional[List[str]] = None,
                recurrent_weight_list: Optional[List[Optional[torch.Tensor]]] = None,
                decision_gate_list: Optional[List[bool]] = None) -> torch.Tensor:
        """
        Compute mixed HAJR/ASAR loss across layers.

        Args:
            mem_rec_list: List of membrane potential tensors [T, B, N] per layer
            beta_list: List of beta (decay) values for each layer
            spike_rec_list: List of spike tensors [T, B, N].
                           REQUIRED if mem_is_post_reset=True
            threshold_list: Optional per-layer thresholds
            reset_mechanism_list: Per-layer reset mechanism.
                                 'soft' or 'hard' → HAJR (with σ' gate)
                                 'none' → ASAR (no gate, no-reset Ψ)
                                 If None, all layers use HAJR (soft reset).

        Returns:
            Loss value (scalar)
        """
        total_loss = 0.0
        count = 0

        device = mem_rec_list[0].device if len(mem_rec_list) > 0 else 'cpu'

        if self.mem_is_post_reset and spike_rec_list is None:
            raise ValueError("spike_rec_list is required when mem_is_post_reset=True")

        for layer_idx, (mem_rec, beta) in enumerate(zip(mem_rec_list, beta_list)):
            T = mem_rec.shape[0]
            if T < 2:
                continue

            B = mem_rec.shape[1]
            mem_flat = mem_rec.view(T, B, -1)
            N = mem_flat.shape[2]

            # Determine reset mechanism for this layer
            reset_mech = 'soft'  # default
            if reset_mechanism_list is not None and layer_idx < len(reset_mechanism_list):
                reset_mech = reset_mechanism_list[layer_idx]

            is_no_reset = (reset_mech == 'none')
            decision_gate = not is_no_reset
            if decision_gate_list is not None and layer_idx < len(decision_gate_list):
                decision_gate = decision_gate_list[layer_idx]
            recurrent_weight = None
            if recurrent_weight_list is not None and layer_idx < len(recurrent_weight_list):
                recurrent_weight = recurrent_weight_list[layer_idx]

            # Resolve threshold (not needed for no-reset layers)
            threshold = self.threshold
            if threshold_list is not None and layer_idx < len(threshold_list):
                thr_entry = threshold_list[layer_idx]
                if thr_entry is not None:
                    threshold = thr_entry

            if is_no_reset:
                # No-reset layers don't use threshold, set dummy
                thr_val = torch.tensor(1.0, device=device, dtype=mem_flat.dtype)
            elif isinstance(threshold, torch.Tensor):
                thr_val = threshold.to(device=device, dtype=mem_flat.dtype)
                if thr_val.dim() == 1:
                    if thr_val.shape[0] != N:
                        thr_val = thr_val.mean()
                    else:
                        thr_val = thr_val.unsqueeze(0)
            else:
                thr_val = torch.tensor(float(threshold), device=device, dtype=mem_flat.dtype)

            # Reconstruct V_pre if needed (SpikingJelly gives post-reset)
            if self.mem_is_post_reset and not is_no_reset:
                spike_flat = spike_rec_list[layer_idx].view(T, B, -1)
                v_pre_rec = mem_flat + thr_val * spike_flat
            else:
                v_pre_rec = mem_flat

            # Handle beta
            if isinstance(beta, torch.Tensor):
                beta_val = beta.clamp(0.01, 0.99)
                if beta_val.dim() == 0:
                    pass
                elif beta_val.dim() == 1:
                    if beta_val.shape[0] != N:
                        beta_val = beta_val.mean()
                    else:
                        beta_val = beta_val.unsqueeze(0)
            else:
                beta_val = torch.tensor(max(0.01, min(0.99, beta)),
                                        device=device, dtype=mem_flat.dtype)

            # Initialize Ψ[-1] = 0
            psi = torch.zeros(B, N, device=device, dtype=mem_flat.dtype)
            layer_loss = 0.0

            if is_no_reset:
                # === Non-reset layer: ASAR formula ===
                # Ψ[t] = V[t-1] + β·Ψ[t-1]   (no reset correction)
                # Loss = ||Ψ[t]||²              (no σ' gate)
                v_prev = torch.zeros(B, N, device=device, dtype=mem_flat.dtype)
                for t in range(T):
                    psi_prev = psi
                    psi = v_prev + beta_val * psi_prev
                    if recurrent_weight is not None:
                        psi = psi + F.linear(
                            self.surrogate_deriv(v_prev - thr_val) * psi_prev,
                            recurrent_weight,
                            bias=None,
                        )
                    layer_loss = layer_loss + psi.pow(2).mean()
                    v_prev = v_pre_rec[t]
                layer_loss = layer_loss / T
            else:
                # === Reset layer: HAJR formula ===
                # Ψ[t] = V[t-1] + (β - θ·σ'(V[t-1] - θ))·Ψ[t-1]
                # J[t] = σ'(V[t] - θ) ⊙ Ψ[t]
                # Loss = ||J[t]||²
                for t in range(1, T):
                    v_prev = v_pre_rec[t - 1]
                    v_curr = v_pre_rec[t]

                    # Ψ update: α = β - θ·σ'(V_prev - θ)
                    surr_deriv_prev = self.surrogate_deriv(v_prev - thr_val)
                    alpha = beta_val - thr_val * surr_deriv_prev
                    psi_prev = psi
                    psi = v_prev + alpha * psi_prev
                    if recurrent_weight is not None:
                        psi = psi + F.linear(
                            surr_deriv_prev * psi_prev,
                            recurrent_weight,
                            bias=None,
                        )

                    if decision_gate:
                        surr_deriv_curr = self.surrogate_deriv(v_curr - thr_val)
                        jacobian = surr_deriv_curr * psi
                        layer_loss = layer_loss + jacobian.pow(2).mean()
                    else:
                        layer_loss = layer_loss + psi.pow(2).mean()

                layer_loss = layer_loss / (T - 1)

            total_loss = total_loss + layer_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if self.reduction == 'mean':
            return total_loss / count
        elif self.reduction == 'sum':
            return total_loss
        else:
            return total_loss / count


# =============================================================================
# Online HAJR (Memory Efficient - Compute During Forward Pass)
# =============================================================================
class OnlineHAJR:
    """
    Online HAJR computation for memory-efficient training.
    Supports mixed HAJR/ASAR per layer via reset_mechanism parameter in step().

    Usage:
        hajr = OnlineHAJR(threshold=1.0, surrogate_type='triangle', surrogate_alpha=1.0)
        hajr.reset()

        for t in range(T):
            # Hidden layers (with reset) → HAJR
            loss += hajr.step('layer1', mem1, beta1, t, reset_mechanism='soft')
            loss += hajr.step('layer2', mem2, beta2, t, reset_mechanism='soft')
            # Output layer (no reset) → ASAR
            loss += hajr.step('output', mem_out, beta_out, t, reset_mechanism='none')

        total_loss = hajr.get_total_loss()

    Args:
        threshold: Spike firing threshold
        surrogate_type: 'atan', 'fast_sigmoid', or 'triangle'
        surrogate_alpha: Parameter for surrogate gradient
    """

    def __init__(self,
                 threshold: float = 1.0,
                 surrogate_type: str = 'auto',
                 surrogate_alpha: float = 2.0,
                 detach_state: bool = False):
        self.threshold = threshold
        self.surrogate_type = surrogate_type
        self.surrogate_alpha = surrogate_alpha
        self.detach_state = detach_state

        if surrogate_type == 'atan':
            self.surrogate_deriv = lambda x: surrogate_derivative_atan(x, surrogate_alpha)
        elif surrogate_type == 'fast_sigmoid':
            self.surrogate_deriv = lambda x: surrogate_derivative_fast_sigmoid(x, surrogate_alpha)
        elif surrogate_type == 'triangle':
            self.surrogate_deriv = lambda x: surrogate_derivative_triangle(x, surrogate_alpha)
        elif surrogate_type == 'auto':
            self.surrogate_deriv = None
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")

        self.states: Dict[str, Dict[str, Any]] = {}
        self.layer_surrogate_cfg: Dict[str, Tuple[str, float]] = {}
        self.total_loss = 0.0
        self.total_count = 0

    def reset(self):
        """Reset state at the start of each forward pass."""
        self.states.clear()
        self.total_loss = 0.0
        self.total_count = 0

    def _store_state(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach() if self.detach_state else tensor

    def _format_param(
        self,
        value: Union[torch.Tensor, float],
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        clamp: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            out = value.to(device=device, dtype=dtype)
        else:
            out = torch.tensor(float(value), device=device, dtype=dtype)

        if clamp is not None:
            out = out.clamp(clamp[0], clamp[1])

        if out.dim() == 0:
            return out
        if out.dim() == 1:
            if out.shape[0] != feature_dim:
                return out.mean()
            return out.unsqueeze(0)
        return out

    def _resolve_surrogate_derivative(self, layer_id: str, layer: Optional[nn.Module] = None):
        if self.surrogate_deriv is not None:
            return self.surrogate_deriv

        if layer is None:
            raise ValueError(
                "OnlineHAJR was configured with surrogate_type='auto', "
                "but no snntorch layer was provided."
            )

        if layer_id not in self.layer_surrogate_cfg:
            self.layer_surrogate_cfg[layer_id] = detect_surrogate_from_snntorch(layer)

        surrogate_type, surrogate_alpha = self.layer_surrogate_cfg[layer_id]
        if surrogate_type == 'atan':
            return lambda x: surrogate_derivative_atan(x, surrogate_alpha)
        if surrogate_type == 'fast_sigmoid':
            return lambda x: surrogate_derivative_fast_sigmoid(x, surrogate_alpha)
        if surrogate_type == 'triangle':
            return lambda x: surrogate_derivative_triangle(x, surrogate_alpha)
        raise ValueError(f"Unsupported surrogate type detected from layer: {surrogate_type}")

    @staticmethod
    def _map_snntorch_reset(reset_mechanism: str) -> str:
        mapping = {
            'subtract': 'soft',
            'zero': 'hard',
            'none': 'none',
        }
        if reset_mechanism not in mapping:
            raise ValueError(
                f"Unsupported snntorch reset_mechanism={reset_mechanism!r}. "
                "Expected 'subtract', 'zero', or 'none'."
            )
        return mapping[reset_mechanism]

    @staticmethod
    def _extract_recurrent_weight(layer: nn.Module) -> Optional[torch.Tensor]:
        recurrent = getattr(layer, 'recurrent', None)
        if recurrent is None:
            return None
        weight = getattr(recurrent, 'weight', None)
        return weight

    @staticmethod
    def _apply_recurrent(local_signal: torch.Tensor, recurrent_weight: torch.Tensor) -> torch.Tensor:
        if recurrent_weight.dim() != 2:
            raise NotImplementedError(
                "Only all-to-all recurrent weights with shape [N, N] are currently supported."
            )
        if recurrent_weight.shape[0] != local_signal.shape[1]:
            raise ValueError(
                f"Recurrent weight shape {tuple(recurrent_weight.shape)} is incompatible with "
                f"flattened membrane shape {tuple(local_signal.shape)}."
            )
        return F.linear(local_signal, recurrent_weight, bias=None)

    def validate_surrogate(self, model: nn.Module, strict: bool = True) -> List[str]:
        """Validate model's surrogate matches this loss configuration."""
        if self.surrogate_type == 'auto':
            return []
        return validate_surrogate_consistency(
            model, self.surrogate_type, self.surrogate_alpha, strict=strict
        )

    def step(self,
             layer_id: str,
             mem: torch.Tensor,
             beta: Union[torch.Tensor, float],
             timestep: int,
             reset_mechanism: str = 'soft',
             threshold: Optional[Union[torch.Tensor, float]] = None,
             spike: Optional[torch.Tensor] = None,
             recurrent_weight: Optional[torch.Tensor] = None,
             surrogate_deriv=None,
             decision_gate: bool = True) -> torch.Tensor:
        """
        Update loss for one layer at one timestep.

        Args:
            layer_id: Unique layer identifier
            mem: Current membrane potential [B, ...] (PRE-reset for SNNTorch)
            beta: Decay factor
            timestep: Current timestep (0-indexed)
            reset_mechanism: 'soft'/'hard' for HAJR, 'none' for ASAR

        Returns:
            step_loss: Loss contribution for this step (with gradient)
        """
        B = mem.shape[0]
        mem_flat = mem.reshape(B, -1)
        N = mem_flat.shape[1]
        device = mem_flat.device
        dtype = mem_flat.dtype

        beta_val = self._format_param(beta, N, device, dtype, clamp=(0.01, 0.99))
        thr_val = self._format_param(
            self.threshold if threshold is None else threshold,
            N,
            device,
            dtype,
        )
        layer_surrogate_deriv = surrogate_deriv or self.surrogate_deriv

        is_no_reset = (reset_mechanism == 'none')

        # Initialize state for new layer
        if layer_id not in self.states:
            self.states[layer_id] = {
                'psi': torch.zeros(B, N, device=device, dtype=dtype),
                'v_prev': torch.zeros(B, N, device=device, dtype=dtype),
                'spk_prev': torch.zeros(B, N, device=device, dtype=dtype),
                'reset_mechanism': reset_mechanism,
            }

        state = self.states[layer_id]
        psi = state['psi']
        v_prev = state['v_prev']
        spk_prev = state['spk_prev']

        if layer_surrogate_deriv is None and not is_no_reset:
            raise ValueError(
                "A surrogate derivative is required for reset layers. "
                "Pass surrogate_deriv explicitly or use step_from_layer(...)."
            )

        if timestep == 0:
            state['psi'] = self._store_state(psi)
            state['v_prev'] = self._store_state(mem_flat)
            if spike is not None:
                state['spk_prev'] = spike.reshape(B, -1).detach()
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        if is_no_reset:
            # === ASAR: no-reset Ψ, no gate ===
            # Ψ[t] = V[t-1] + β·Ψ[t-1]
            psi_new = v_prev + beta_val * psi
            if recurrent_weight is not None and layer_surrogate_deriv is not None:
                surr_prev = layer_surrogate_deriv(v_prev - thr_val)
                psi_new = psi_new + self._apply_recurrent(surr_prev * psi, recurrent_weight)
            step_loss = psi_new.pow(2).mean()
        else:
            surr_deriv_prev = layer_surrogate_deriv(v_prev - thr_val)

            if reset_mechanism == 'soft':
                # snnTorch subtract-reset (default reset_delay=True):
                # V_t = beta * V_{t-1} + I_t - theta * S_{t-1}
                alpha = beta_val - thr_val * surr_deriv_prev
                psi_new = v_prev + alpha * psi
            elif reset_mechanism == 'hard':
                gate = 1.0 - spk_prev
                coeff = beta_val * (gate - v_prev * surr_deriv_prev)
                psi_new = v_prev * gate + coeff * psi
            else:
                raise ValueError(
                    f"Unknown reset_mechanism={reset_mechanism!r}. "
                    "Expected 'soft', 'hard', or 'none'."
                )

            if recurrent_weight is not None:
                psi_new = psi_new + self._apply_recurrent(
                    surr_deriv_prev * psi, recurrent_weight
                )

            if decision_gate:
                # HAJR: J = σ'(V_curr - θ) ⊙ Ψ
                surr_deriv_curr = layer_surrogate_deriv(mem_flat - thr_val)
                jacobian = surr_deriv_curr * psi_new
                step_loss = jacobian.pow(2).mean()
            else:
                # Reset-aware ASAR: penalize raw sensitivity energy.
                step_loss = psi_new.pow(2).mean()

        # Update state. spk_prev is treated as a value state; its derivative path
        # is already accounted for analytically via the surrogate term in psi_new.
        state['psi'] = self._store_state(psi_new)
        state['v_prev'] = self._store_state(mem_flat)
        if spike is not None:
            state['spk_prev'] = spike.reshape(B, -1).detach()

        self.total_loss += step_loss.detach().item()
        self.total_count += 1

        return step_loss

    def step_from_layer(
        self,
        layer_id: str,
        layer: nn.Module,
        mem: torch.Tensor,
        timestep: int,
        spike: Optional[torch.Tensor] = None,
        reset_mechanism: Optional[str] = None,
        decision_gate: bool = True,
    ) -> torch.Tensor:
        """
        SNNtorch-compatible online HAJR/ASAR update.

        Call this immediately after `spk, mem = layer(...)` inside the timestep
        loop. For `snn.RLeaky`, the recurrent sensitivity term from the local
        recursion is included automatically when an all-to-all recurrent weight
        matrix is available.
        """
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError("step_from_layer requires snntorch to be installed") from exc

        if not isinstance(layer, (snn.Leaky, snn.RLeaky)):
            raise TypeError(f"Expected snntorch Leaky/RLeaky layer, got {type(layer)}")

        if not getattr(layer, 'reset_delay', True):
            raise NotImplementedError(
                "OnlineHAJR.step_from_layer currently assumes snntorch reset_delay=True."
            )

        inferred_reset = self._map_snntorch_reset(getattr(layer, 'reset_mechanism', 'subtract'))
        effective_reset = inferred_reset if reset_mechanism is None else reset_mechanism
        recurrent_weight = self._extract_recurrent_weight(layer)
        layer_surrogate_deriv = self._resolve_surrogate_derivative(layer_id, layer=layer)

        if spike is None and hasattr(layer, 'spk'):
            spike = layer.spk

        return self.step(
            layer_id=layer_id,
            mem=mem,
            beta=layer.beta,
            timestep=timestep,
            reset_mechanism=effective_reset,
            threshold=layer.threshold,
            spike=spike,
            recurrent_weight=recurrent_weight,
            surrogate_deriv=layer_surrogate_deriv,
            decision_gate=decision_gate,
        )

    def get_total_loss(self) -> float:
        """Get the average accumulated loss (for logging, not gradient)."""
        if self.total_count == 0:
            return 0.0
        return self.total_loss / self.total_count


# =============================================================================
# Factory Function
# =============================================================================
def get_hajr_loss(mode: str = 'batch', **kwargs) -> Union[HAJRLoss, OnlineHAJR]:
    """
    Factory function to create HAJR loss instances.

    Args:
        mode: 'batch' for post-forward computation, 'online' for incremental
        **kwargs: Arguments passed to the loss constructor

    Returns:
        HAJRLoss or OnlineHAJR instance
    """
    if mode == 'batch':
        return HAJRLoss(**kwargs)
    elif mode == 'online':
        return OnlineHAJR(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'batch' or 'online'.")
