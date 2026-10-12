# Third-party code: DelRec

Upstream: https://github.com/alexmaxad/DelRec at commit `c9e60a1` ("Remove population_delay init").
License: MIT, Copyright (c) 2025 alexmaxad (see `LICENSE` in this directory).

Only the files needed for the SSC experiments are included. Upstream notebooks, result folders and the
other benchmark scripts (SHD, PSMNIST, spike-penalty, equi-parameter) are omitted.

## Unmodified upstream files

`LICENSE`, `README.md`, `requirements.txt`, `configs/__init__.py`, `src/__init__.py`, `src/utils.py`,
`src/recurrent_neurons.py`, `src/datasets.py`, `src/SHD/SHD_downloader.py`.

## Modified upstream file

- `src/SSC/snn.py`: `SNN_recurrent_delays` optionally replaces the scalar PLIF decay parameter of each
  recurrent-delay layer by a per-neuron vector when `config.vector_beta` is true (7 added lines). The
  attribute defaults to absent/false, in which case the model is identical to upstream; all paper
  experiments use the upstream behaviour (scalar per-layer decay).

## Files added for this paper

- `perf_SSC_p0_spikingjelly.py`: SSC training entry point (Clean / MARS / Noise Aug / surrogate-gated ablation).
- `configs/perf_SSC_p0_spikingjelly.py`: SSC configuration (DelRec protocol, 100 epochs, learnable β via PLIF).
- `src/SSC/trainer_p0_spikingjelly.py`: training loop with the MARS regularizer, β noise augmentation,
  robustness/quantization evaluation and checkpointing.
- `src/SSC/delay_aware_regularizer.py`: delay-aware membrane-sensitivity recursion for the recurrent-delay layers.
- `src/spikingjelly_compat.py`: SpikingJelly ParametricLIFNode compatibility layer (β = 1 − sigmoid(w)).
- `src/SSC/snntorch_p0.py`: SSC HDF5 spike iterator and an snnTorch variant used by the diagnostics.
