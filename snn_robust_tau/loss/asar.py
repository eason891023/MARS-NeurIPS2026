"""
ASAR Loss Functions for Spiking Neural Networks
=================================================
This module provides ASAR (Autoregressive Spike-driven Activity Regularization) loss functions
for both Hard Reset and Soft Reset LIF neuron mechanisms.

Usage:
    from snn_robust_tau.loss.asar import ASARLoss
"""

import math
import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict, Any

# =============================================================================
# ASARLoss - True Sensitivity Formula (Ignoring Reset)
# =============================================================================
class ASARLoss(nn.Module):
    """
    True ASAR Loss using the original sensitivity derivation.
    
    For LIF neuron dynamics (ignoring reset for simplicity):
        V[t] = β * V[t-1] + I[t]
    
    The sensitivity Ψ[t] = dV[t]/dβ follows:
        Ψ[t] = β * Ψ[t-1] + V[t-1]
    
    ASAR Loss: Σ Ψ[t]²
    
    FRAMEWORK COMPATIBILITY:
    - SNNTorch: stores PRE-reset membrane → set mem_is_post_reset=False
    - SpikingJelly: stores POST-reset membrane → set mem_is_post_reset=True
    
    When mem_is_post_reset=True, reconstructs V_pre = V_post + θ * S
    
    Args:
        threshold: Spike threshold (for reconstruction)
        mem_is_post_reset: True if input mem is post-reset (SpikingJelly), 
                          False if pre-reset (SNNTorch)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 mem_is_post_reset: bool = True,
                 reduction: str = 'mean'):
        super().__init__()
        self.threshold = threshold
        self.mem_is_post_reset = mem_is_post_reset
        self.reduction = reduction
    
    def forward(self, 
                mem_rec_list: List[torch.Tensor], 
                beta_list: List[Union[torch.Tensor, float]],
                spike_rec_list: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute ASAR loss using true sensitivity formula.
        
        Args:
            mem_rec_list: List of membrane potential tensors [T, B, N]
            beta_list: List of beta (decay) values for each layer
            spike_rec_list: List of spike tensors [T, B, N]. 
                           REQUIRED if mem_is_post_reset=True (for reconstruction)
        
        Returns:
            ASAR loss value
        """
        total_loss = 0.0
        count = 0
        
        device = mem_rec_list[0].device if len(mem_rec_list) > 0 else 'cpu'
        
        # Validate spike_rec_list requirement
        if self.mem_is_post_reset and spike_rec_list is None:
            raise ValueError("spike_rec_list is required when mem_is_post_reset=True "
                           "(for SpikingJelly). Use mem_is_post_reset=False for SNNTorch.")
        
        for layer_idx, (mem_rec, beta) in enumerate(zip(mem_rec_list, beta_list)):
            if mem_rec.shape[0] < 2:
                continue
            
            # Flatten spatial dimensions: [T, B, ...] -> [T, B, N]
            T, B = mem_rec.shape[:2]
            mem_flat = mem_rec.view(T, B, -1)
            N = mem_flat.shape[2]
            
            # Reconstruct V_pre if needed (SpikingJelly gives post-reset)
            if self.mem_is_post_reset:
                spike_flat = spike_rec_list[layer_idx].view(T, B, -1)
                v_pre_rec = mem_flat + self.threshold * spike_flat
            else:
                # SNNTorch already gives pre-reset
                v_pre_rec = mem_flat
            
            # Handle beta: can be scalar, 1D tensor [N], or 0-dim tensor
            if isinstance(beta, torch.Tensor):
                beta_val = beta.clamp(0.01, 0.99)
                # Ensure beta broadcasts correctly with [B, N]
                if beta_val.dim() == 0:
                    # Scalar tensor -> broadcasts to all neurons
                    pass
                elif beta_val.dim() == 1:
                    # Per-neuron beta [N] or [hidden_size]
                    if beta_val.shape[0] != N:
                        # If sizes don't match, use mean (e.g., different layer sizes)
                        beta_val = beta_val.mean()
                    else:
                        # Reshape for broadcasting: [N] -> [1, N] for [B, N] operations
                        beta_val = beta_val.unsqueeze(0)  # [1, N]
            else:
                beta_val = max(0.01, min(0.99, beta))
                beta_val = torch.tensor(beta_val, device=device, dtype=mem_flat.dtype)
            
            # Initialize Ψ[-1] = 0, V[-1] = 0
            psi = torch.zeros(B, N, device=device, dtype=mem_flat.dtype)
            v_prev = torch.zeros(B, N, device=device, dtype=mem_flat.dtype)
            
            layer_loss = 0.0
            for t in range(T):
                # Ψ[t] = β * Ψ[t-1] + V_pre[t-1]
                # beta_val broadcasts: scalar or [1, N] * [B, N] -> [B, N]
                psi = beta_val * psi + v_prev
                layer_loss = layer_loss + (psi ** 2).mean()
                v_prev = v_pre_rec[t]
            
            total_loss = total_loss + layer_loss / T
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
# Online ASAR (Memory Efficient) - Original Sensitivity Formula
# =============================================================================
class OnlineASAR:
    """
    Online ASAR computation using the original sensitivity derivation:
        Ψ[t] = β * Ψ[t-1] + V[t-1]
    
    This is the true Active Sensitivity Aware Regularization (ASAR) loss,
    which measures dV/dβ (sensitivity of membrane potential to time constant).
    
    Memory efficient: O(B*N) per layer instead of O(T*B*N).
    Computes ASAR loss incrementally during the forward pass.
    
    Usage:
        asar_state = OnlineASAR()
        asar_state.reset()
        
        for t in range(T):
            # ... forward pass ...
            loss += asar_state.step('layer1', mem, beta, t)
        
        total_loss = asar_state.get_total_loss()
    """
    
    def __init__(self, detach_state: bool = False):
        self.states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.total_psi_squared = 0.0
        self.total_count = 0
        self.detach_state = detach_state
    
    def reset(self):
        """Reset state at the start of each forward pass."""
        self.states.clear()
        self.total_psi_squared = 0.0
        self.total_count = 0
    
    def _store_state(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach() if self.detach_state else tensor

    def step(self,
             layer_id: str, 
             mem: torch.Tensor, 
             beta: Union[torch.Tensor, float], 
             timestep: int) -> torch.Tensor:
        """
        Update ASAR for one layer at one timestep.
        
        Implements: Ψ[t] = β * Ψ[t-1] + V[t-1]
        
        Args:
            layer_id: Unique layer identifier
            mem: Current membrane potential [B, ...] (will be flattened)
            beta: Decay factor (scalar or tensor)
            timestep: Current timestep (0-indexed)
        
        Returns:
            step_loss: Loss contribution for this step (Ψ[t]^2 mean)
        """
        B = mem.shape[0]
        mem_flat = mem.reshape(B, -1)
        N = mem_flat.shape[1]
        device = mem_flat.device
        dtype = mem_flat.dtype
        
        # Ensure beta is tensor
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, device=device, dtype=dtype)
        
        # Initialize state for new layer
        if layer_id not in self.states:
            self.states[layer_id] = {
                'psi': torch.zeros(B, N, device=device, dtype=dtype),
                'v_prev': torch.zeros(B, N, device=device, dtype=dtype),
            }
        
        state = self.states[layer_id]
        psi = state['psi']
        v_prev = state['v_prev']
        
        # Core ASAR formula: Ψ[t] = β * Ψ[t-1] + V[t-1]
        psi_new = beta * psi + v_prev
        
        # Compute step loss (mean squared sensitivity)
        step_loss = (psi_new ** 2).mean()
        
        # Update state for next timestep. By default we keep the recursion
        # differentiable; detach_state=True enables truncated online mode.
        state['psi'] = self._store_state(psi_new)
        state['v_prev'] = self._store_state(mem_flat)
        
        # Track for total loss
        self.total_psi_squared += step_loss.detach().item()
        self.total_count += 1
        
        return step_loss

    def step_from_layer(
        self,
        layer_id: str,
        layer: nn.Module,
        mem: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """
        SNNtorch-compatible wrapper for no-reset ASAR layers.

        This method is intended for output layers configured with
        `reset_mechanism='none'`, where the no-reset ASAR recursion is exact.
        """
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError("step_from_layer requires snntorch to be installed") from exc

        if not isinstance(layer, (snn.Leaky, snn.RLeaky)):
            raise TypeError(f"Expected snntorch Leaky/RLeaky layer, got {type(layer)}")

        reset_mechanism = getattr(layer, 'reset_mechanism', None)
        if reset_mechanism != 'none':
            raise ValueError(
                "OnlineASAR.step_from_layer is only valid for no-reset layers "
                f"(got reset_mechanism={reset_mechanism!r}). Use OnlineHAJR for reset layers."
            )

        beta = getattr(layer, 'beta', None)
        if beta is None:
            raise AttributeError("snntorch layer is missing beta")

        return self.step(layer_id=layer_id, mem=mem, beta=beta, timestep=timestep)
    
    def get_total_loss(self) -> float:
        """Get the average accumulated Psi^2 loss (for logging, not gradient)."""
        if self.total_count == 0:
            return 0.0
        return self.total_psi_squared / self.total_count


# =============================================================================
# Online Prediction Error (Memory Efficient) - Temporal Coherence Loss
# =============================================================================
class OnlinePredictionError:
    """
    Online ASAR computation for Hard Reset mechanism.
    
    Memory efficient: O(B*N) per layer instead of O(T*B*N).
    Computes ASAR loss incrementally during the forward pass.
    
    For Hard Reset: ASAR = Σ (V[t] - β·V[t-1])² / (1 - β²)
    
    Usage:
        asar_state = OnlineASARHardState()
        asar_state.reset()
        
        for t in range(T):
            # ... forward pass ...
            loss += asar_state.step('layer1', mem, beta, t)
        
        total_loss = asar_state.get_total_loss(num_layers, T)
    
    Args:
        eps: Small value for numerical stability
    """
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.accumulated_loss = 0.0
    
    def reset(self):
        """Reset state at the start of each forward pass."""
        self.states.clear()
        self.accumulated_loss = 0.0
    
    def step(self, 
             layer_id: str, 
             mem: torch.Tensor, 
             beta: Union[torch.Tensor, float], 
             timestep: int) -> torch.Tensor:
        """
        Update ASAR Hard for one layer at one timestep.
        
        Args:
            layer_id: Unique layer identifier
            mem: Current membrane potential [B, ...] (will be flattened)
            beta: Decay factor
            timestep: Current timestep (0-indexed)
        
        Returns:
            step_loss: Loss contribution for this step (with gradient)
        """
        B = mem.shape[0]
        mem_flat = mem.reshape(B, -1)
        device = mem_flat.device
        dtype = mem_flat.dtype
        
        # Ensure beta is tensor and clamped
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, device=device, dtype=dtype)
        beta = beta.clamp(0.01, 0.99)
        
        # Initialize state for new layer
        if layer_id not in self.states:
            self.states[layer_id] = {
                'v_prev': torch.zeros(B, mem_flat.shape[1], device=device, dtype=dtype),
            }
        
        state = self.states[layer_id]
        
        if timestep == 0:
            step_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        else:
            v_prev = state['v_prev']
            
            # ASAR Hard: (V[t] - β·V[t-1])² / (1 - β²)
            diff = mem_flat - beta * v_prev
            scaling = 1.0 / (1.0 - beta**2 + self.eps)
            step_loss = (diff**2).mean() * scaling
        
        # Store state (detached for memory efficiency)
        state['v_prev'] = mem_flat.detach()
        
        return step_loss
    
    def get_total_loss(self, num_layers: int, num_timesteps: int) -> float:
        """Get normalized accumulated loss (for logging, not gradient)."""
        return self.accumulated_loss / (num_layers * num_timesteps + 1e-6)


# =============================================================================
# Online ASAR Hard State (Memory Efficient) - Correct Hard Reset Derivation
# =============================================================================
class OnlineASARHardState:
    """
    Online ASAR computation for TRUE Hard Reset mechanism (reset to zero).
    
    This implements the mathematically correct sensitivity derivation for Hard Reset:
        u_t = β * u_{t-1} * (1 - o_{t-1}) + m_t
    
    Where (1 - o_{t-1}) is a gate that resets membrane to 0 when spike occurs.
    
    The sensitivity Ψ_t = ∂u_t/∂β follows the recurrence:
        Ψ_t = u_{t-1}(1 - o_{t-1}) + [β(1 - o_{t-1})I + (W_rec - β·diag(u_{t-1}))D_{t-1}] Ψ_{t-1}
    
    For feedforward networks (no W_rec), this simplifies to:
        Ψ_t = u_{t-1}(1 - o_{t-1}) + β(1 - o_{t-1})Ψ_{t-1} - β·u_{t-1}·D_{t-1}·Ψ_{t-1}
    
    Where D_{t-1} = S'(u_{t-1} - θ) is the surrogate derivative.
    
    Memory efficient: O(B*N) per layer instead of O(T*B*N).
    
    Usage:
        asar_state = OnlineASARTrueHardState(surrogate_alpha=2.0)
        asar_state.reset()
        
        for t in range(T):
            # ... forward pass ...
            loss += asar_state.step('layer1', mem, spike, beta, t, threshold)
    
    Args:
        surrogate_type: 'atan' or 'fast_sigmoid'
        surrogate_alpha: Alpha parameter for surrogate gradient
    """
    
    def __init__(self, 
                 surrogate_type: str = 'atan',
                 surrogate_alpha: float = 2.0):
        self.surrogate_type = surrogate_type
        self.surrogate_alpha = surrogate_alpha
        self.states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.accumulated_loss = 0.0
        
        # Select surrogate derivative function
        if surrogate_type == 'atan':
            self.surrogate_deriv = lambda x: surrogate_derivative_atan(x, surrogate_alpha)
        elif surrogate_type == 'fast_sigmoid':
            self.surrogate_deriv = lambda x: surrogate_derivative_fast_sigmoid(x, surrogate_alpha)
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")
    
    def reset(self):
        """Reset state at the start of each forward pass."""
        self.states.clear()
        self.accumulated_loss = 0.0
    
    def step(self, 
             layer_id: str, 
             mem: torch.Tensor, 
             spike: Optional[torch.Tensor],
             beta: Union[torch.Tensor, float], 
             timestep: int,
             threshold: Union[torch.Tensor, float] = 1.0) -> torch.Tensor:
        """
        Update ASAR True Hard for one layer at one timestep.
        
        Args:
            layer_id: Unique layer identifier
            mem: Current membrane potential [B, ...] (will be flattened)
            spike: Previous timestep's spike output [B, ...] (o_{t-1}), 
                   or None to infer from membrane > threshold
            beta: Decay factor
            timestep: Current timestep (0-indexed)
            threshold: Firing threshold for this layer
        
        Returns:
            step_loss: Loss contribution for this step (with gradient)
        """
        B = mem.shape[0]
        mem_flat = mem.reshape(B, -1)
        N = mem_flat.shape[1]
        device = mem_flat.device
        dtype = mem_flat.dtype
        
        # Ensure beta is tensor and clamped
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, device=device, dtype=dtype)
        beta = beta.clamp(0.01, 0.99)
        
        # Ensure threshold is tensor
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, device=device, dtype=dtype)
        
        # Initialize state for new layer
        if layer_id not in self.states:
            self.states[layer_id] = {
                'psi': torch.zeros(B, N, device=device, dtype=dtype),
                'u_prev': torch.zeros(B, N, device=device, dtype=dtype),
                'o_prev': torch.zeros(B, N, device=device, dtype=dtype),
            }
        
        state = self.states[layer_id]
        
        if timestep == 0:
            psi = torch.zeros_like(mem_flat)
        else:
            u_prev = state['u_prev']
            o_prev = state['o_prev']
            psi_prev = state['psi']
            
            # Gate: (1 - o_{t-1}) - this is the key difference from soft reset
            gate = 1.0 - o_prev
            
            # Surrogate derivative: D_{t-1} = S'(u_{t-1} - θ)
            D_prev = self.surrogate_deriv(u_prev - threshold)
            
            # Feedforward version (no W_rec):
            # Ψ_t = u_{t-1}(1 - o_{t-1}) + β(1 - o_{t-1})Ψ_{t-1} - β·u_{t-1}·D_{t-1}·Ψ_{t-1}
            # Simplify: Ψ_t = u_{t-1}(1 - o_{t-1}) + [β(1 - o_{t-1}) - β·u_{t-1}·D_{t-1}]·Ψ_{t-1}
            
            term1 = u_prev * gate  # u_{t-1}(1 - o_{t-1})
            
            # Coefficient for Ψ_{t-1}
            # α = β(1 - o_{t-1}) - β·u_{t-1}·D_{t-1}
            #   = β[(1 - o_{t-1}) - u_{t-1}·D_{t-1}]
            coeff = beta * (gate - u_prev * D_prev)
            
            psi = term1 + coeff * psi_prev
        
        # Step loss: Ψ²
        step_loss = psi.pow(2).mean()
        
        # Infer spike from current membrane if not provided
        if spike is not None:
            spike_flat = spike.reshape(B, -1).detach()
        else:
            # Infer spike: o = 1 if mem >= threshold
            spike_flat = (mem_flat >= threshold).float().detach()
        
        # Store state (detached for memory efficiency)
        state['psi'] = psi.detach()
        state['u_prev'] = mem_flat.detach()
        state['o_prev'] = spike_flat
        
        return step_loss
    
    def get_total_loss(self, num_layers: int, num_timesteps: int) -> float:
        """Get normalized accumulated loss (for logging, not gradient)."""
        return self.accumulated_loss / (num_layers * num_timesteps + 1e-6)


# =============================================================================
# Online ASAR Soft State (Memory Efficient)
# =============================================================================
class OnlineASARSoftState:
    """
    Online ASAR computation for Soft Reset mechanism.
    
    Memory efficient: O(B*N) per layer instead of O(T*B*N).
    Computes ASAR loss incrementally during the forward pass.
    
    Usage:
        asar_state = OnlineASARSoftState(threshold=1.0, surrogate_alpha=2.0)
        asar_state.reset()
        
        for t in range(T):
            # ... forward pass ...
            loss += asar_state.step('layer1', mem, beta, t)
        
        total_loss = asar_state.get_total_loss(num_layers, T)
    
    Args:
        threshold: Firing threshold
        surrogate_type: 'atan' or 'fast_sigmoid'
        surrogate_alpha: Alpha parameter for surrogate gradient
    """
    
    def __init__(self, 
                 surrogate_type: str = 'atan',
                 surrogate_alpha: float = 2.0):
        self.surrogate_type = surrogate_type
        self.surrogate_alpha = surrogate_alpha
        self.states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.accumulated_loss = 0.0
        
        # Select surrogate derivative function
        if surrogate_type == 'atan':
            self.surrogate_deriv = lambda x: surrogate_derivative_atan(x, surrogate_alpha)
        elif surrogate_type == 'fast_sigmoid':
            self.surrogate_deriv = lambda x: surrogate_derivative_fast_sigmoid(x, surrogate_alpha)
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")
    
    def reset(self):
        """Reset state at the start of each forward pass."""
        self.states.clear()
        self.accumulated_loss = 0.0
    
    def step(self, 
             layer_id: str, 
             mem: torch.Tensor, 
             beta: Union[torch.Tensor, float], 
             timestep: int,
             threshold: Union[torch.Tensor, float] = 1.0) -> torch.Tensor:
        """
        Update ASAR for one layer at one timestep.
        
        Args:
            layer_id: Unique layer identifier
            mem: Current membrane potential [B, ...] (will be flattened)
            beta: Decay factor
            timestep: Current timestep (0-indexed)
            threshold: Current threshold for this layer (supports learnable)
        
        Returns:
            step_loss: Loss contribution for this step (with gradient)
        """
        B = mem.shape[0]
        mem_flat = mem.reshape(B, -1)
        N = mem_flat.shape[1]
        device = mem_flat.device
        dtype = mem_flat.dtype
        
        # Ensure beta is tensor and clamped
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, device=device, dtype=dtype)
        beta = beta.clamp(0.01, 0.99)
        
        # Initialize state for new layer
        if layer_id not in self.states:
            self.states[layer_id] = {
                'psi': torch.zeros(B, N, device=device, dtype=dtype),
                'v_prev': torch.zeros(B, N, device=device, dtype=dtype),
            }
        
        state = self.states[layer_id]
        
        # Ensure threshold is tensor
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, device=device, dtype=dtype)
        
        if timestep == 0:
            psi = torch.zeros_like(mem_flat)
        else:
            v_prev = state['v_prev']
            psi_prev = state['psi']
            
            # α = β - θ * S'(V_prev - θ)
            surr_deriv = self.surrogate_deriv(v_prev - threshold)
            alpha = beta - threshold * surr_deriv
            
            # Ψ = V_prev + α * Ψ_prev
            psi = v_prev + alpha * psi_prev
        
        # Step loss
        step_loss = psi.pow(2).mean()
        
        # Store state (detached for memory efficiency)
        state['psi'] = psi.detach()
        state['v_prev'] = mem_flat.detach()
        
        return step_loss
    
    def get_total_loss(self, num_layers: int, num_timesteps: int) -> float:
        """Get normalized accumulated loss (for logging, not gradient)."""
        return self.accumulated_loss / (num_layers * num_timesteps + 1e-6)


# =============================================================================
# Utility Functions
# =============================================================================
def prepare_mem_for_asar(model, lif_layers: List) -> tuple:
    """
    Prepare membrane history and beta values for ASAR loss computation.
    
    Args:
        model: Model with mem_history attribute (dict: layer -> list of mem tensors)
        lif_layers: List of LIF layer modules
    
    Returns:
        mem_rec_list: List of [T, B, N] membrane tensors
        beta_list: List of beta values
    """
    if not hasattr(model, 'mem_history') or not model.mem_history:
        return [], []
    
    mem_rec_list = []
    beta_list = []
    
    for lif_layer in lif_layers:
        if lif_layer in model.mem_history and len(model.mem_history[lif_layer]) > 0:
            mem_stack = torch.stack(model.mem_history[lif_layer])  # [T, B, ...]
            T, B = mem_stack.shape[:2]
            mem_flat = mem_stack.view(T, B, -1)
            mem_rec_list.append(mem_flat)
            
            beta = lif_layer.beta
            if isinstance(beta, torch.nn.Parameter):
                beta = beta.clamp(0.01, 0.99)
            beta_list.append(beta)
    
    return mem_rec_list, beta_list


# =============================================================================
# Convenience function to get the right ASAR loss
# =============================================================================
def get_asar_loss(loss_type: str = 'hard', **kwargs):
    """
    Factory function to get the appropriate ASAR loss.
    
    Args:
        loss_type: 'hard' or 'soft'
        **kwargs: Additional arguments for the loss function
    
    Returns:
        ASAR loss instance
    """
    if loss_type == 'hard':
        return ASARLossHard(**kwargs)
    elif loss_type == 'soft':
        return ASARLossSoft(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'hard' or 'soft'.")


# =============================================================================
# ASARLossV2 - Unified Multi-Mode ASAR Loss
# =============================================================================
class ASARLossV2(nn.Module):
    """
    Unified ASAR Loss with per-layer sensitivity computation modes.
    
    Supports different neuron types in different layers:
    - Hidden layers: Hard Reset-to-Zero or Subtract Reset
    - Readout layer: No-reset membrane accumulator
    
    Modes:
        'hard_reset_gated': Ψ[t] = (1-S[t-1]) * V[t-1] + β * (1-S[t-1]) * Ψ[t-1]
                           Spike gate stops gradient accumulation on reset.
                           
        'subthreshold':     Ψ[t] = V[t-1] + β * Ψ[t-1]
                           Pure low-pass filter dynamics (no reset term).
                           Use for subtract reset or readout layers.
    
    Loss Definition:
        L = Σ_ℓ w_ℓ · reduce_t( ||Ψ_t^(ℓ)||_p )
    
    Where:
        - w_ℓ = layer_weights[ℓ]
        - ||·||_p = penalty norm (L1, L2, or Huber)
        - reduce_t = time_reduce ('mean' or 'sum')
    
    IMPORTANT:
        - mem_rec_list must contain PRE-RESET membrane potentials!
        - spike_rec_list is used as stop-grad gating constants (no gradient flows through spikes)
    
    Args:
        layer_modes: List of mode strings, one per layer.
                    Options: 'hard_reset_gated', 'subthreshold'
        penalty: 'l1' | 'l2' | 'huber' - how to penalize Ψ
        time_reduce: 'mean' | 'sum' - how to aggregate over time
        layer_weights: Optional list of per-layer weights (default: all 1.0)
        huber_delta: Delta parameter for Huber loss (default: 1.0)
        reduction: 'mean' | 'sum' | 'none' - final aggregation over layers
    
    Example:
        loss_fn = ASARLossV2(
            layer_modes=['hard_reset_gated', 'hard_reset_gated', 'subthreshold'],
            penalty='l2',
            time_reduce='mean',
            layer_weights=[1.0, 1.0, 0.5]
        )
        
        loss = loss_fn(
            mem_rec_list=[mem0, mem1, mem2],  # PRE-RESET membrane [T, B, N]
            beta_list=[beta0, beta1, beta2],
            spike_rec_list=[spk0, spk1, None]  # Only needed for hard_reset_gated
        )
    """
    
    VALID_MODES = {'hard_reset_gated', 'subthreshold', 'subtract_reset'}
    VALID_PENALTIES = {'l1', 'l2', 'huber'}
    VALID_TIME_REDUCE = {'mean', 'sum'}
    
    def __init__(self,
                 layer_modes: List[str],
                 penalty: str = 'l2',
                 time_reduce: str = 'mean',
                 layer_weights: Optional[List[float]] = None,
                 huber_delta: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        
        # Validate modes
        for mode in layer_modes:
            if mode not in self.VALID_MODES:
                raise ValueError(f"Invalid mode '{mode}'. Choose from {self.VALID_MODES}")
        
        # Validate penalty
        if penalty not in self.VALID_PENALTIES:
            raise ValueError(f"Invalid penalty '{penalty}'. Choose from {self.VALID_PENALTIES}")
        
        # Validate time_reduce
        if time_reduce not in self.VALID_TIME_REDUCE:
            raise ValueError(f"Invalid time_reduce '{time_reduce}'. Choose from {self.VALID_TIME_REDUCE}")
        
        self.layer_modes = layer_modes
        self.penalty = penalty
        self.time_reduce = time_reduce
        self.huber_delta = huber_delta
        self.reduction = reduction
        
        # Layer weights (default: all 1.0)
        if layer_weights is None:
            self.layer_weights = [1.0] * len(layer_modes)
        else:
            if len(layer_weights) != len(layer_modes):
                raise ValueError(f"layer_weights length ({len(layer_weights)}) must match "
                               f"layer_modes length ({len(layer_modes)})")
            self.layer_weights = layer_weights
    
    def _compute_penalty(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute penalty on sensitivity tensor Ψ."""
        if self.penalty == 'l1':
            return psi.abs().mean()
        elif self.penalty == 'l2':
            return psi.pow(2).mean()
        elif self.penalty == 'huber':
            # Huber loss: L2 for small values, L1 for large values
            abs_psi = psi.abs()
            quadratic = torch.clamp(abs_psi, max=self.huber_delta)
            linear = abs_psi - quadratic
            return (0.5 * quadratic.pow(2) + self.huber_delta * linear).mean()
        else:
            raise ValueError(f"Unknown penalty: {self.penalty}")
    
    def _compute_layer_loss_hard_reset_gated(self,
                                              mem_rec: torch.Tensor,
                                              beta: torch.Tensor,
                                              spike_rec: torch.Tensor) -> torch.Tensor:
        """
        Compute sensitivity loss for Hard Reset with spike gating.
        
        Formula: Ψ[t] = (1-S[t-1]) * V[t-1] + β * (1-S[t-1]) * Ψ[t-1]
        
        Args:
            mem_rec: [T, B, N] PRE-RESET membrane potentials
            beta: Decay factor (scalar or [N] tensor)
            spike_rec: [T, B, N] Spike indicators (DETACHED - no gradient!)
        
        Returns:
            Layer loss (scalar)
        """
        T, B, N = mem_rec.shape
        device = mem_rec.device
        dtype = mem_rec.dtype
        
        # Ensure spike_rec is detached (stop-grad)
        spike_rec = spike_rec.detach()
        
        # Initialize Ψ[-1] = 0, V[-1] = 0
        psi = torch.zeros(B, N, device=device, dtype=dtype)
        v_prev = torch.zeros(B, N, device=device, dtype=dtype)
        s_prev = torch.zeros(B, N, device=device, dtype=dtype)
        
        time_losses = []
        
        for t in range(T):
            # Gate: (1 - S[t-1]) - zeros out when spike occurred
            gate = 1.0 - s_prev  # [B, N], detached
            
            # Ψ[t] = gate * V[t-1] + β * gate * Ψ[t-1]
            psi = gate * v_prev + beta * gate * psi
            
            # Compute penalty for this timestep
            step_loss = self._compute_penalty(psi)
            time_losses.append(step_loss)
            
            # Update for next timestep
            v_prev = mem_rec[t]
            s_prev = spike_rec[t]
        
        # Aggregate over time
        time_losses = torch.stack(time_losses)
        if self.time_reduce == 'mean':
            return time_losses.mean()
        else:  # sum
            return time_losses.sum()
    
    def _compute_layer_loss_subthreshold(self,
                                          mem_rec: torch.Tensor,
                                          beta: torch.Tensor) -> torch.Tensor:
        """
        Compute sensitivity loss for subthreshold/subtract reset dynamics.
        
        Formula: Ψ[t] = V[t-1] + β * Ψ[t-1]
        
        Args:
            mem_rec: [T, B, N] PRE-RESET membrane potentials
            beta: Decay factor (scalar or [N] tensor)
        
        Returns:
            Layer loss (scalar)
        """
        T, B, N = mem_rec.shape
        device = mem_rec.device
        dtype = mem_rec.dtype
        
        # Initialize Ψ[-1] = 0, V[-1] = 0
        psi = torch.zeros(B, N, device=device, dtype=dtype)
        v_prev = torch.zeros(B, N, device=device, dtype=dtype)
        
        time_losses = []
        
        for t in range(T):
            # Ψ[t] = V[t-1] + β * Ψ[t-1]
            psi = v_prev + beta * psi
            
            # Compute penalty for this timestep
            step_loss = self._compute_penalty(psi)
            time_losses.append(step_loss)
            
            # Update for next timestep
            v_prev = mem_rec[t]
        
        # Aggregate over time
        time_losses = torch.stack(time_losses)
        if self.time_reduce == 'mean':
            return time_losses.mean()
        else:  # sum
            return time_losses.sum()
    
    def forward(self,
                mem_rec_list: List[torch.Tensor],
                beta_list: List[Union[torch.Tensor, float]],
                spike_rec_list: Optional[List[Optional[torch.Tensor]]] = None) -> torch.Tensor:
        """
        Compute unified ASAR loss across all layers.
        
        Args:
            mem_rec_list: List of membrane potential tensors [T, B, N] per layer.
                         MUST be PRE-RESET membrane potentials!
            beta_list: List of beta (decay) values for each layer.
                      Can be float, 0-dim tensor, or [N] tensor.
            spike_rec_list: Optional list of spike tensors [T, B, N].
                           Required for 'hard_reset_gated' mode layers.
                           Set to None for layers that don't need it.
        
        Returns:
            ASAR loss value (scalar tensor with gradient)
        """
        if len(mem_rec_list) != len(self.layer_modes):
            raise ValueError(f"mem_rec_list length ({len(mem_rec_list)}) must match "
                           f"layer_modes length ({len(self.layer_modes)})")
        
        if len(beta_list) != len(self.layer_modes):
            raise ValueError(f"beta_list length ({len(beta_list)}) must match "
                           f"layer_modes length ({len(self.layer_modes)})")
        
        # Handle spike_rec_list default
        if spike_rec_list is None:
            spike_rec_list = [None] * len(self.layer_modes)
        
        device = mem_rec_list[0].device if len(mem_rec_list) > 0 else 'cpu'
        dtype = mem_rec_list[0].dtype if len(mem_rec_list) > 0 else torch.float32
        
        layer_losses = []
        
        for layer_idx, (mem_rec, beta, mode) in enumerate(zip(mem_rec_list, beta_list, self.layer_modes)):
            # Skip if membrane record is too short
            if mem_rec.shape[0] < 2:
                continue
            
            # Flatten spatial dimensions: [T, B, ...] -> [T, B, N]
            T, B = mem_rec.shape[:2]
            mem_flat = mem_rec.view(T, B, -1)
            N = mem_flat.shape[2]
            
            # Handle beta: scalar, 0-dim tensor, or [N] tensor
            # NOTE: Do NOT clamp tensor beta - assume neuron uses sigmoid/softplus parameterization
            # Clamping learnable tensors kills gradient at boundaries!
            if isinstance(beta, torch.Tensor):
                beta_val = beta  # Trust the neuron's parameterization
                if beta_val.dim() == 0:
                    pass  # Scalar broadcasts naturally
                elif beta_val.dim() == 1:
                    if beta_val.shape[0] != N:
                        raise ValueError(f"Layer {layer_idx}: beta shape {beta_val.shape} does not match "
                                       f"flattened neuron count N={N}. Check your beta tensor dimensions.")
                    beta_val = beta_val.unsqueeze(0)  # [1, N] for [B, N] broadcast
                else:
                    raise ValueError(f"Layer {layer_idx}: beta must be scalar or 1D tensor, got dim={beta_val.dim()}")
            else:
                # Only clamp scalar floats (non-learnable)
                beta_val = torch.tensor(max(0.01, min(0.99, beta)), 
                                       device=device, dtype=dtype)
            
            # Compute layer loss based on mode
            if mode == 'hard_reset_gated':
                # Validate spike_rec_list
                if spike_rec_list[layer_idx] is None:
                    raise ValueError(f"spike_rec_list[{layer_idx}] is required for "
                                   f"'hard_reset_gated' mode")
                
                spike_flat = spike_rec_list[layer_idx].view(T, B, -1)
                layer_loss = self._compute_layer_loss_hard_reset_gated(
                    mem_flat, beta_val, spike_flat
                )
            
            elif mode in ('subthreshold', 'subtract_reset'):
                layer_loss = self._compute_layer_loss_subthreshold(mem_flat, beta_val)
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Apply layer weight
            weighted_loss = self.layer_weights[layer_idx] * layer_loss
            layer_losses.append(weighted_loss)
        
        # Handle empty case - keep on compute graph for AMP/optimizer compatibility
        if len(layer_losses) == 0:
            return mem_rec_list[0].sum() * 0.0
        
        # Aggregate over layers
        total_loss = torch.stack(layer_losses)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:  # 'none'
            return total_loss
    
    def __repr__(self):
        return (f"ASARLossV2(layer_modes={self.layer_modes}, penalty='{self.penalty}', "
                f"time_reduce='{self.time_reduce}', layer_weights={self.layer_weights})")
