from typing import Any, Dict, Optional, Tuple, Union

import torch


def surrogate_derivative_atan(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    import math

    return (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * x).pow(2))


def surrogate_derivative_fast_sigmoid(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    denom = 1.0 + slope * torch.abs(x)
    return slope / (denom ** 2)


def surrogate_derivative_triangle(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return (1.0 / gamma) * (1.0 / gamma) * torch.clamp(gamma - torch.abs(x), min=0.0)


class DelRecDelayAwareRegularizer:
    """
    DelRec-specific online HAJR/ASAR with delayed recurrent input sensitivity.

    This extends the standard online beta-sensitivity recursion with an extra
    term coming from the delayed recurrent buffer:

        dI_rec[t]/dbeta = psi_buffer[t]

    where psi_buffer is updated from the current spike sensitivity and routed
    through the learnable delay mask exactly like the recurrent forward buffer.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        surrogate_type: str = "atan",
        surrogate_alpha: float = 2.0,
        detach_state: bool = False,
    ):
        self.threshold = float(threshold)
        surrogate_type = surrogate_type.lower()
        self.surrogate_type = surrogate_type
        self.surrogate_alpha = float(surrogate_alpha)
        self.detach_state = bool(detach_state)

        if surrogate_type == "atan":
            self.surrogate_deriv = lambda x: surrogate_derivative_atan(x, surrogate_alpha)
        elif surrogate_type in {"fast_sigmoid", "sigmoid"}:
            self.surrogate_deriv = lambda x: surrogate_derivative_fast_sigmoid(x, surrogate_alpha)
        elif surrogate_type in {"triangle", "triangular"}:
            self.surrogate_deriv = lambda x: surrogate_derivative_triangle(x, surrogate_alpha)
        else:
            raise ValueError(f"Unsupported surrogate_type={surrogate_type!r}")

        self.states: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.states.clear()

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

    def _ensure_state(
        self,
        layer_id: str,
        batch_size: int,
        feature_dim: int,
        buffer_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        state = self.states.get(layer_id)
        if state is not None:
            psi_buffer = state["psi_buffer"]
            if (
                state["psi"].shape == (batch_size, feature_dim)
                and psi_buffer.shape == (batch_size, feature_dim, buffer_len)
                and state["psi"].device == device
            ):
                return state

        state = {
            "psi": torch.zeros(batch_size, feature_dim, device=device, dtype=dtype),
            "v_prev": torch.zeros(batch_size, feature_dim, device=device, dtype=dtype),
            "spk_prev": torch.zeros(batch_size, feature_dim, device=device, dtype=dtype),
            "psi_buffer": torch.zeros(batch_size, feature_dim, buffer_len, device=device, dtype=dtype),
        }
        self.states[layer_id] = state
        return state

    def step(
        self,
        layer_id: str,
        mem: torch.Tensor,
        spike: torch.Tensor,
        beta: Union[torch.Tensor, float],
        timestep: int,
        reset_mechanism: str = "soft",
        threshold: Optional[Union[torch.Tensor, float]] = None,
        decision_gate: bool = True,
        surrogate_deriv=None,
        recurrent_weight: Optional[torch.Tensor] = None,
        delay_mask: Optional[torch.Tensor] = None,
        buffer_pointer: int = 0,
        rec_input_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mem_flat = mem.reshape(mem.shape[0], -1)
        spk_flat = spike.reshape(spike.shape[0], -1)
        batch_size, feature_dim = mem_flat.shape
        device = mem_flat.device
        dtype = mem_flat.dtype

        beta_val = self._format_param(beta, feature_dim, device, dtype, clamp=(0.01, 0.99))
        thr_val = self._format_param(
            self.threshold if threshold is None else threshold,
            feature_dim,
            device,
            dtype,
        )

        layer_surrogate_deriv = surrogate_deriv or self.surrogate_deriv
        if delay_mask is not None:
            delay_mask = delay_mask.to(device=device, dtype=dtype)
            buffer_len = int(delay_mask.shape[1])
        else:
            buffer_len = 1

        state = self._ensure_state(layer_id, batch_size, feature_dim, buffer_len, device, dtype)
        psi_prev = state["psi"]
        v_prev = state["v_prev"]
        spk_prev = state["spk_prev"]
        psi_buffer = state["psi_buffer"]

        psi_input = psi_buffer[:, :, buffer_pointer]
        if rec_input_mask is not None:
            psi_input = psi_input * rec_input_mask.reshape(batch_size, -1).to(device=device, dtype=dtype)

        if reset_mechanism == "none":
            psi_new = v_prev + beta_val * psi_prev + psi_input
        else:
            surr_prev = layer_surrogate_deriv(v_prev - thr_val)
            if reset_mechanism == "soft":
                alpha = beta_val - thr_val * surr_prev
                psi_new = v_prev + alpha * psi_prev + psi_input
            elif reset_mechanism == "hard":
                gate = 1.0 - spk_prev
                coeff = beta_val * (gate - v_prev * surr_prev)
                psi_new = v_prev * gate + coeff * psi_prev + psi_input
            else:
                raise ValueError(
                    f"Unknown reset_mechanism={reset_mechanism!r}. Expected 'soft', 'hard', or 'none'."
                )

        if reset_mechanism == "none" or not decision_gate:
            step_loss = psi_new.pow(2).mean()
        else:
            surr_curr = layer_surrogate_deriv(mem_flat - thr_val)
            step_loss = (surr_curr * psi_new).pow(2).mean()

        if recurrent_weight is not None and delay_mask is not None:
            spike_sens = layer_surrogate_deriv(mem_flat - thr_val) * psi_new
            delayed_psi = torch.einsum(
                "io,bol->bil",
                recurrent_weight.to(device=device, dtype=dtype),
                spike_sens.unsqueeze(2) * delay_mask.unsqueeze(0),
            )

            psi_buffer = psi_buffer.clone()
            psi_buffer[:, :, buffer_pointer] = 0.0
            next_pointer = (buffer_pointer + 1) % buffer_len

            first_chunk = min(buffer_len - 1, buffer_len - next_pointer)
            if first_chunk > 0:
                psi_buffer[:, :, next_pointer:next_pointer + first_chunk] = (
                    psi_buffer[:, :, next_pointer:next_pointer + first_chunk]
                    + delayed_psi[:, :, 1:1 + first_chunk]
                )

            remaining = buffer_len - 1 - first_chunk
            if remaining > 0:
                psi_buffer[:, :, 0:remaining] = (
                    psi_buffer[:, :, 0:remaining]
                    + delayed_psi[:, :, 1 + first_chunk:1 + first_chunk + remaining]
                )

            state["psi_buffer"] = self._store_state(psi_buffer)

        state["psi"] = self._store_state(psi_new)
        state["v_prev"] = self._store_state(mem_flat)
        state["spk_prev"] = spk_flat.detach()

        return step_loss
