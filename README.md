# Membrane Sensitivity and Deployment Fragility of Learnable Time Constants in Spiking Neural Networks

Code for the NeurIPS 2026 paper introducing **MARS** (Membrane-Aware Robustness through Sensitivity Regularization).

Chiu Chang Cheng<sup>1</sup>, Ya Ning Chang<sup>1,*</sup>, Chao Hung Wang<sup>1,2,*</sup>

<sup>1</sup> Miin Wu School of Computing, National Cheng Kung University
<sup>2</sup> Academy of Innovative Semiconductor and Sustainable Manufacturing, National Cheng Kung University
<sup>*</sup> Corresponding authors: Ya Ning Chang (yaningchang@gs.ncku.edu.tw) and Chao Hung Wang (chwang@gs.ncku.edu.tw)

Paper: TBD

```bibtex
@inproceedings{cheng2026mars,
  title     = {Membrane Sensitivity and Deployment Fragility of Learnable Time Constants in Spiking Neural Networks},
  author    = {Cheng, Chiu Chang and Chang, Ya Ning and Wang, Chao Hung},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {TBD}
}
```

## What MARS is

SNNs with a learnable membrane decay factor β become fragile when the deployed β deviates from the trained value
(analog device mismatch, drift). To first order, this fragility is governed by the membrane sensitivity
Ψ_t = ∂V_t/∂β, which obeys a closed-form online recursion carrying one extra state per neuron. MARS adds
λ·(1/T)Σ_t‖Ψ_t‖² to the task loss during training (no inference cost).
The implementation is `snn_robust_tau/loss/hajr.py` (`OnlineHAJR`, called with `decision_gate=False`).

**Naming in the code.** Mode strings used throughout scripts, checkpoints and result files:

| Code mode | Paper name |
|---|---|
| `clean` | Clean (learnable β, no regularizer) |
| `noise_aug` | Noise Aug (Gaussian β perturbation, CV_train = 0.15) |
| `asar` | **MARS** (ungated ‖Ψ_t‖² penalty, λ = 5e-4) |
| `hajr` | surrogate-gated MARS variant (‖σ'(V_t−θ) ⊙ Ψ_t‖²; exploratory, not reported in the paper) |

In the evaluation code (`eval/`) and in `results/partial/*.json` the MARS method key is `mars`; it loads the
`asar` checkpoints.

## Repository layout

| Path | Content |
|---|---|
| `snn_robust_tau/` | MARS / sensitivity regularizers (`loss/hajr.py`, `loss/asar.py`) and repository paths (`paths.py`) |
| `train/` | Training: SHD-FC, SHD-RSNN, DVS-Gesture, CIFAR-10, SHD-FC λ sweep; `baselines/` simple controls; `common/` shared SHD loaders |
| `third_party/DelRec/` | DelRec SSC architecture (MIT) + our SSC training entry point; see `third_party/DelRec/MODIFICATIONS.md` |
| `eval/` | Unified β-mismatch evaluator (per-neuron / layer-wise / global), joint mismatch + PTQ evaluator |
| `analysis/` | Post-hoc analyses (joint/distribution/temporal tests, memory, mechanism, overhead) |
| `diagnostics/` | Worst-case / typical-case λ diagnostics, (B2) coherence checks, and Ψ heat maps (Fig. 3, stability table) |
| `paper_artifacts/` | Scripts that regenerate every table and figure from `results/` |
| `results/` | Path-scrubbed result files used by the paper (JSON / NPZ, all < 5 MB) |
| `MANIFEST.md` | Paper artifact → script → command → input → regenerable from shipped results? |

Generated files go to `outputs/`, logs to `logs/` (both git-ignored). No checkpoints are released.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, torch 2.11.0, torchvision 0.26.0, snntorch 0.9.4, tonic 1.6.0, spikingjelly 0.0.0.0.14,
DCLS 0.1.1, numpy 1.26.4, scipy 1.17.1, matplotlib 3.10.8, h5py 3.16.0 on a single NVIDIA RTX 5090 (32 GB).
The paper checkpoints were trained with an earlier PyTorch version; retrained models and re-run evaluations can
therefore differ slightly from the reported numbers.

## Datasets

Datasets are not redistributed; obtain them from the original sources and follow their terms of use.
Default layout (override the root with `MARS_DATA_ROOT`):

```
data/
  tonic/SHD/shd_{train,test}.h5             SHD: downloaded automatically by tonic (or from https://zenkelab.org/datasets)
  tonic/DVSGesture/ibmGesture{Train,Test}   DVS-Gesture: tonic.datasets.DVSGesture
  cifar10/cifar-10-batches-py               CIFAR-10: torchvision (downloaded by train/cifar10.py)
  SSC/ssc_{train,valid,test}.h5             SSC: download and gunzip from https://zenkelab.org/datasets
```

If the tonic download of the DVS-Gesture archives fails, download `ibmGestureTrain.tar.gz` and
`ibmGestureTest.tar.gz` manually from Zenodo record 8060604 into `data/tonic/DVSGesture/` (tonic verifies the MD5
and extracts them on first use).

## Training

All SHD/DVS/CIFAR scripts train every mode in `CONFIG["modes"]` for the five seeds 42, 52, 72, 92, 102
(edit `CONFIG` to restrict) and write `outputs/train/<run>_<timestamp>/{checkpoints,results.json,plots}`.

```bash
python train/shd_fc.py          # SHD-FC   (clean, hajr, asar, noise_aug; 50 epochs)
python train/shd_rsnn.py        # SHD-RSNN
python train/dvs_gesture.py     # DVS-Gesture
python train/cifar10.py         # CIFAR-10
python third_party/DelRec/perf_SSC_p0_spikingjelly.py --modes clean,asar,noise_aug --seeds 42,52,72,92,102   # SSC (100 epochs)
python train/shd_lambda_sweep.py   # SHD-FC MARS λ sweep (seeds 42/52/62/72/82, no input augmentation; appendix)
```

Simple baselines (Table 3):

```bash
# β-L2, hard β range, fixed homogeneous / heterogeneous β, recurrent DropConnect:
# validation pilots for λ_β and the DropConnect rate, then 5 seeds (selected values: results/controls/selected_hyperparameters.json)
python train/baselines/control_pipeline.py --output-dir outputs/train/controls
# or directly with the selected hyperparameters:
python train/baselines/train_controls.py --architecture ff   --modes beta_l2,beta_clip,fixed_homogeneous,fixed_heterogeneous --lambda-beta-l2 1e-3 --output-dir outputs/train/controls/formal_ff
python train/baselines/train_controls.py --architecture rsnn --modes beta_l2,beta_clip,fixed_homogeneous,fixed_heterogeneous,dropconnect --lambda-beta-l2 1e-1 --dropconnect-p 0.05 --output-dir outputs/train/controls/formal_rsnn
# weight decay and spectral normalization (SHD-FC only)
python train/baselines/shd_extra_baselines.py
```

### Checkpoint layout for evaluation

Copy or symlink the trained `best_s{seed}_{mode}.pth` files into (override the root with `MARS_CKPT_ROOT`):

```
checkpoints/SHD-FC/  checkpoints/SHD-RSNN/  checkpoints/SSC/  checkpoints/DVS-Gesture/  checkpoints/CIFAR-10/
checkpoints/SHD-FC-controls/    beta_l2, beta_clip, fixed_homogeneous, fixed_heterogeneous, weight_decay, spectral_norm
checkpoints/SHD-RSNN-controls/  beta_l2, beta_clip, fixed_homogeneous, fixed_heterogeneous, dropconnect
```

## Evaluation (requires checkpoints and test sets)

β̃ = clip(β(1+CV·z), 0.01, 0.99). Draws depend only on (setting, training seed, mismatch structure, draw index), so all
methods of a seed see identical draws. Results are written to `outputs/eval/partial/` in the same format as
`results/partial/`.

```bash
# per-neuron CV sweep {0.05,...,0.3} + layer-wise/global CV {0.1,0.2}, 10 draws (Fig. 1); one setting:
python eval/run_unified.py --setting SHD-FC            # settings: SHD-FC SHD-RSNN SSC DVS-Gesture CIFAR-10
# per-neuron CV 0.2 with 200 draws (Table 1 SHD per-neuron, Tables 2-3), main and control checkpoints:
python eval/run_unified.py --setting SHD-FC --only-cv 0.2 --draws 200 --tag _mc200
python eval/run_unified.py --setting SHD-FC --controls --only-cv 0.2 --draws 200 --tag _mc200
# layer-wise / global drift at CV 0.2, 100 draws (Table 1):
python eval/run_unified.py --setting SSC --modes layer_shared,global --cvs 0.2 --draws 100 --tag _cv02d100
# β + weight/threshold mismatch and 8/6-bit PTQ, 200 draws (Table 2):
python eval/run_joint_ptq.py --setting SHD-FC --draws 200
# drivers for all of the above
bash eval/run_all.sh; bash eval/run_layerwise_v2.sh; bash eval/run_joint_ptq.sh
```

To regenerate the tables from your own evaluation, point the artifact scripts at it, e.g.
`MARS_RESULTS_ROOT=outputs/eval_results python paper_artifacts/make_main_tables.py` with your files under
`outputs/eval_results/partial/`. The checkpoint-dependent analyses in `analysis/` and `diagnostics/` are listed with
their commands in `MANIFEST.md`.

## Regenerating paper tables and figures from the shipped results

No GPU, dataset or checkpoint is needed:

```bash
bash paper_artifacts/reproduce_all.sh
```

or individually:

```bash
python paper_artifacts/make_main_tables.py         # Tables 1-3 -> outputs/tables/table{1,2,3}_rows.tex
python paper_artifacts/make_appendix_batch1.py     # main-result, per-seed and paired-difference tables
python paper_artifacts/make_appendix_batch2.py     # joint mismatch and PTQ (a) tables
python paper_artifacts/build_tables.py             # distribution, temporal, SSC joint, cost ratios, Ψ fold reductions
python paper_artifacts/recompute_correlation.py    # mechanism correlation table
python paper_artifacts/appendix_cr_extra_numbers.py  # λ sweep, learned β / activity, R², PTQ (b), stability table, cost
python paper_artifacts/make_fig1_variants.py       # Fig. 1  -> outputs/figures/fig1_robustness_perneuron.pdf
python paper_artifacts/make_figures_fixed.py       # Fig. 3  -> outputs/figures/fig3_heatmap_stability.pdf
python paper_artifacts/make_fig4_new.py            # Fig. 4  -> outputs/figures/fig4_sensitivity_drop_perneuron.pdf
```

The regenerated figure PDFs are byte-identical to the paper figures apart from the PDF creation timestamp.
See `MANIFEST.md` for the mapping of every paper number to its script and input file.

## License

MIT (see `LICENSE`). `third_party/DelRec/` is MIT-licensed by its original author (see `third_party/DelRec/LICENSE`).
