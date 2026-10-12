import torch

from spikingjelly.activation_based import neuron


class ParametricLIFNodeCompat(neuron.ParametricLIFNode):
    """
    DelRec compatibility wrapper.

    Original DelRec code constructs neurons with `tau=...`, while
    SpikingJelly's learnable-time-constant neuron expects `init_tau=...`.
    This wrapper preserves the original DelRec constructor call pattern.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_reset=None,
        v_threshold: float = 1.0,
        surrogate_function=None,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
        **kwargs,
    ):
        super().__init__(
            init_tau=float(tau),
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
            **kwargs,
        )


def beta_from_plif(node: neuron.ParametricLIFNode):
    return 1.0 - node.w.sigmoid()


def set_beta_on_plif(node: neuron.ParametricLIFNode, beta):
    beta = beta.clamp(0.01, 0.99)
    reciprocal_tau = 1.0 - beta
    node.w.data.copy_(torch.log(reciprocal_tau / beta))
