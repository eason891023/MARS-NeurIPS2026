# Paper artifact manifest

Every table, figure and quoted number of the paper, the script that produces it, and its input.
"Shipped" = regenerable from the files in `results/` alone (CPU, seconds). "Checkpoints" = the input
result file itself must be recomputed from trained checkpoints (and test sets) with the listed producer.
All commands run from the repository root; outputs go to `outputs/`. `bash paper_artifacts/reproduce_all.sh`
runs every "Shipped" command.

Abbreviations: `PA` = `paper_artifacts/`, `RP` = `results/partial/`, `RA` = `results/analysis/`,
`RD` = `results/diagnostics/`, `RM` = `results/mechanism/`, `extra` = `python PA/appendix_cr_extra_numbers.py`.

## Main text

| Paper artifact | Script | Command | Input result file(s) | Regenerable | Producer of the input |
|---|---|---|---|---|---|
| Fig. 1 (accuracy vs per-neuron CV, 5 settings) | `PA/make_fig1_variants.py` | `python PA/make_fig1_variants.py` → `outputs/figures/fig1_robustness_perneuron.pdf` | `RP/{SHD-FC,SHD-RSNN,SSC,DVS-Gesture,CIFAR-10}.json` | Shipped | `eval/run_unified.py --setting <S>` (checkpoints) |
| Table 1 (tab:structured) | `PA/make_main_tables.py` | `python PA/make_main_tables.py` → `outputs/tables/table1_rows.tex` | per-neuron: `RP/SHD-{FC,RSNN}_mc200.json`, `RP/{SSC,DVS-Gesture,CIFAR-10}.json`; layer-wise/global: `RP/*_cv02d100.json` | Shipped | `eval/run_unified.py` (`--only-cv 0.2 --draws 200 --tag _mc200`; `--modes layer_shared,global --cvs 0.2 --draws 100 --tag _cv02d100`) |
| Fig. 3a (Ψ heat maps) and "51× / 5.8×" | `PA/make_figures_fixed.py`; ratios: `extra` (psi_suppression) | `python PA/make_figures_fixed.py` → `outputs/figures/fig3_heatmap_stability.pdf` | `RD/heatmap_data.npz`, `RD/heatmap_summary.json` | Shipped | `diagnostics/rerun_sensitivity_heatmap.py` (checkpoints) |
| Fig. 3b (λ_r vs λ̃, "0.89/0.94") | `PA/make_figures_fixed.py` | same | `RD/results_fixed_shd_fc.json` | Shipped | `diagnostics/eval_typical_case_stability_fixed.py` (checkpoints) |
| Table 2 (tab:joint) | `PA/make_main_tables.py` | `python PA/make_main_tables.py` → `table2_rows.tex` | `RP/SHD-{FC,RSNN}_jointptq.json`, `RP/SHD-{FC,RSNN}_mc200.json` | Shipped | `eval/run_joint_ptq.py --draws 200` (checkpoints) |
| §4.3 text: weight+threshold alone ≤ 0.34 pp; 6-bit PTQ MARS SHD-RSNN 84.8→83.2 % | `PA/make_appendix_batch2.py` | `python PA/make_appendix_batch2.py` | `RP/SHD-*_jointptq.json` | Shipped | as Table 2 |
| §4.3 text: SSC β+weight+threshold+delay 1.17/0.22/0.23 pp | `PA/build_tables.py` (SSC block of `tab_app_joint.tex`, `digest_auto.md`) | `python PA/build_tables.py` | `RA/ssc_joint.json` | Shipped | `analysis/ssc_joint_delay_eval.py` (checkpoints) |
| §4.3 text: log-normal / t3 / correlated MARS drops 0.24–0.82 pp | `PA/build_tables.py` (`tab_app_distribution.tex`) | `python PA/build_tables.py` | `RA/distribution_shift_{ff,rsnn}.json`, `RA/joint_noise_{ff,rsnn}.json` (nominal) | Shipped | `analysis/distribution_shift_eval.py`, `analysis/joint_noise_eval.py` (checkpoints) |
| Table 3 (tab:baselines); intro / §4.2 "fixed β = 0.5: 75.5 %, 2.3 pp" | `PA/make_main_tables.py` | `python PA/make_main_tables.py` → `table3_rows.tex` | `RP/SHD-{FC,RSNN}_controls_mc200.json`, `RP/SHD-{FC,RSNN}_mc200.json` | Shipped | `eval/run_unified.py --controls --only-cv 0.2 --draws 200 --tag _mc200` (control checkpoints) |
| §4.2 "paired differences ≤ 1.03 pp" | `PA/make_appendix_batch1.py` (tab:paired, SHD-FC per-neuron mean −1.028, printed −1.0) | `python PA/make_appendix_batch1.py` | `RP/SHD-*_mc200.json` | Shipped | as Table 1 |
| §5 "at CV = 0.05 Clean SHD-FC loses 4–5 pp, SHD-RSNN 1–2 pp" | `eval/aggregate.py` (`UNIFIED_RESULTS.md`) / Fig. 1 data | `python eval/aggregate.py` | `RP/<setting>.json`, `results/training/*.json`, `RA/joint_noise_*.json` | Shipped | `eval/run_unified.py` |
| §4.5 ordering MARS < Noise Aug < Clean in 15/15; r = 0.94, ρ = 0.95, r = 0.91 (layer-wise) | `PA/recompute_correlation.py` | `python PA/recompute_correlation.py` | `RM/asar_vs_robustness.json`, `RM/ssc_scatter_points.json`, `RP/*` | Shipped | `analysis/eval_asar_vs_robustness.py`, `analysis/compute_ssc_scatter_points.py` (checkpoints) |
| §4.5 / App. mechanism: Ψ-energy fold 4.67× (NA/Clean), 78.2× (MARS/Clean), 16.7× (MARS/NA) | `PA/build_tables.py` (`digest_auto.md`, "Pooled statistics") or `analysis/mechanism_triangulation.py` | `python PA/build_tables.py` | `RA/mechanism_triangulation.json` (itself regenerated from `RM/*` by `python analysis/mechanism_triangulation.py`) | Shipped | `analysis/mechanism_triangulation.py` |
| §4.5 temporal: horizons; SSC 82.1→9.3/8.0/2.8 %; SHD-RSNN 84.9→27.3/20.1 %; R²₂₀ 0.47→0.08; β ranges | `PA/build_tables.py` (`tab_temporal.tex`) + `extra` (temporal_controls, memory_r2, state_activity) | `python PA/build_tables.py; extra` | `RA/memory_capacity.json`, `RA/temporal_information_{ff,rsnn}.json`, `RA/ssc_temporal.json`, `RA/state_activity.json` | Shipped | `analysis/memory_capacity_eval.py` + `analyze_memory_capacity.py`, `analysis/temporal_information_eval.py`, `analysis/ssc_temporal_eval.py` + `merge_ssc_temporal.py` + `analyze_ssc_temporal.py`, `analysis/state_activity_eval.py` (checkpoints) |
| §5 / Limitations: 1.35–2.06× step time, 1.13–2.30× peak memory | `PA/build_tables.py` (`tab_cost.tex`) | `python PA/build_tables.py` | `RA/overhead.json`, `RA/overhead_real_dvs.json` | Shipped | `analysis/five_setting_overhead.py` (data, no checkpoints) |

## Appendix

| Paper artifact | Script | Command | Input result file(s) | Regenerable | Producer of the input |
|---|---|---|---|---|---|
| App. setup: β-L2 λ_β = 1e-3 / 1e-1, DropConnect p = 0.05 | – | – | `results/controls/selected_hyperparameters.json` | Shipped | `train/baselines/control_pipeline.py` |
| tab:main (nominal ± SD, drop [95 % CI]) | `PA/make_appendix_batch1.py` | `python PA/make_appendix_batch1.py` → `outputs/tables/tab_main_rows.tex` | `RP/SHD-*_mc200.json`, `RP/{SSC,DVS-Gesture,CIFAR-10}.json` | Shipped | `eval/run_unified.py` |
| tab:perseed | `PA/make_appendix_batch1.py` | same → `tab_perseed_rows.tex` | same | Shipped | same |
| tab:paired (3 structures) | `PA/make_appendix_batch1.py` | same → `tab_paired_rows.tex` | same + `RP/*_cv02d100.json` | Shipped | same |
| tab:app_lambda (λ sweep) | `extra` (lambda_sweep) | `python PA/appendix_cr_extra_numbers.py` | `results/lambda_sweep/shd_fc_lambda_sweep.json` (trimmed: per-epoch `history` and per-batch `beta_trace` removed from the 28 MB original; all fields used here kept) | Shipped | `train/shd_lambda_sweep.py` (training) |
| tab:app_distribution | `PA/build_tables.py` (`tab_app_distribution.tex`) | `python PA/build_tables.py` | `RA/distribution_shift_{ff,rsnn}.json`, `RA/joint_noise_{ff,rsnn}.json` | Shipped | `analysis/distribution_shift_eval.py` (checkpoints) |
| tab:app_joint (weight+threshold / β only / joint) | `PA/make_appendix_batch2.py` | `python PA/make_appendix_batch2.py` → `tab_joint_rows.tex` | `RP/SHD-*_jointptq.json`, `RP/SHD-*_mc200.json` | Shipped | `eval/run_joint_ptq.py` |
| App. joint paragraph: SSC 1.17±0.25 / 0.22±0.16 / 0.23±0.16 | `PA/build_tables.py` | `python PA/build_tables.py` (`digest_auto.md`, SSC block) | `RA/ssc_joint.json` | Shipped | `analysis/ssc_joint_delay_eval.py` |
| tab:app_temporal (horizons, original / reversed / shuffled) | `PA/build_tables.py` (`tab_temporal.tex`) | `python PA/build_tables.py` | `RA/memory_capacity.json`, `RA/temporal_information_*.json`, `RA/ssc_temporal.json` | Shipped | see §4.5 temporal |
| tab:app_temporal rate-repeat column | `extra` (temporal_controls) | `extra` | `RA/temporal_information_*.json` | Shipped | `analysis/temporal_information_eval.py` |
| App. temporal: SSC jitter ±1–10 bins / local shuffle, methods differ ≤ 0.8 pp | `analysis/analyze_ssc_temporal.py` | `python analysis/analyze_ssc_temporal.py --results RA/ssc_temporal_raw.json --output-dir outputs/analysis/ssc_temporal` | `RA/ssc_temporal_raw.json` | Shipped | `analysis/ssc_temporal_eval.py` (per mode) + `analysis/merge_ssc_temporal.py` |
| tab:app_memory R²_k | `extra` (memory_r2; 3 decimals, the paper rounds these to 2) | `extra` | `RA/memory_capacity.json` (from `RA/memory_capacity_raw.json` via `analysis/analyze_memory_capacity.py`) | Shipped | `analysis/memory_capacity_eval.py` |
| tab:app_memory Reset@k | `extra` (state_activity) | `extra` | `RA/state_activity.json` | Shipped | `analysis/state_activity_eval.py` |
| tab:app_beta_activity (learned β, spike probability) | `extra` (state_activity) | `extra` | `RA/state_activity.json` | Shipped | `analysis/state_activity_eval.py` |
| tab:app_mechanism (r / ρ, n = 30 / 45) | `PA/recompute_correlation.py` | `python PA/recompute_correlation.py` | `RM/*.json`, `RP/*` | Shipped | `analysis/eval_asar_vs_robustness.py`, `analysis/compute_ssc_scatter_points.py` |
| App. mechanism 4.67× / 78.2× / 16.7× | `PA/build_tables.py` (`digest_auto.md`) | `python PA/build_tables.py` | `RA/mechanism_triangulation.json` | Shipped | `analysis/mechanism_triangulation.py` (from `RM/*`) |
| Fig. 4 (sensitivity energy vs drop; per-setting r) | `PA/make_fig4_new.py` | `python PA/make_fig4_new.py` → `outputs/figures/fig4_sensitivity_drop_perneuron.pdf` | `RM/asar_vs_robustness.json`, `RM/ssc_scatter_points.json`, `RP/SHD-*_mc200.json`, `RP/{SSC,DVS-Gesture}.json` | Shipped | as tab:app_mechanism |
| tab:stability-both (‖W‖, w_eff, κ, η, ᾱ, α̃, λ_r, λ̃); per-element λ_r 249.9 | `extra` (stability_table) | `extra` | `RD/results_fixed_shd_fc.json` | Shipped | `diagnostics/eval_typical_case_stability_fixed.py` (checkpoints) |
| (B1) κ = 26–82 | `extra` (stability_table) | `extra` | `RD/results_fixed_shd_fc.json` | Shipped | same |
| (B2) temporal coherence: pairwise cosine of normalized δV_t 0.50–0.82 (layer 1) / 0.71–0.83 (layer 2); per-neuron autocorrelation < 0.1 only after ≥ 6 steps | `diagnostics/eval_b2_shd_fc.py` | `python diagnostics/eval_b2_shd_fc.py` → `outputs/diagnostics/b2_shd_fc.json` | `RD/b2_shd_fc.json` (shipped; holds the numbers, no table/figure script reads it) | Checkpoints | `diagnostics/eval_b2_shd_fc.py` (SHD-FC checkpoints, 5 seeds × 4 modes, 30 test batches). The recomputed SHD-FC values do **not** reproduce the quoted numbers (same-sample pairwise cosine 0.5–0.9, autocorrelation does not fall below 0.1 within 3 steps); see the JSON (`across_seeds`) |
| (B3) η ∈ [0.008, 0.026]; c_t ∈ [0.03, 0.15]; permuted [0.13, 0.26] | `extra` (stability_table) | `extra` | `RD/typical_case_shd_fc.json` | Shipped | `diagnostics/eval_typical_case_stability_fixed.py` |
| (B4) α̃ ≤ ᾱ < 1 | `extra` (stability_table) | `extra` | `RD/results_fixed_shd_fc.json` | Shipped | same |
| tab:app_ptq (a) PTQ 8/6 bit + β | `PA/make_appendix_batch2.py` | `python PA/make_appendix_batch2.py` → `tab_ptq_rows.tex` | `RP/SHD-*_jointptq.json`, `RP/SHD-*_mc200.json` | Shipped | `eval/run_joint_ptq.py` |
| tab:app_ptq (b) PTQ 4/3 bit, no β mismatch | `extra` (ptq_low_bit) | `extra` | `results/training/SHD-{FC,RSNN}.json` (`quantization`) | Shipped | `train/shd_fc.py`, `train/shd_rsnn.py` (end-of-training evaluation) |
| App. PTQ: ‖W₁‖_op 41.9 (MARS) vs 33.2 (Clean), seed 42 | `extra` (stability_table, weight_norms) | `extra` | `RD/results_fixed_shd_fc.json`, `RD/stability_verification_fixed.json` | Shipped | `diagnostics/eval_stability_verification_fixed.py` |
| tab:app_cost absolute step time / peak memory | `extra` (overhead_abs) | `extra` | `RA/overhead.json`, `RA/overhead_real_dvs.json` | Shipped | `analysis/five_setting_overhead.py` (DVS row: real-batch rerun) |
| tab:app_cost ratios | `PA/build_tables.py` (`tab_cost.tex`) | `python PA/build_tables.py` | same | Shipped | same |
| App. cost: end-to-end SHD-FC epoch 58.3 s → 80.1 s (1.37×) | `extra` (overhead_abs) | `extra` | `RA/training_overhead_v2.json` | Shipped | `analysis/eval_training_overhead_v2.py` (+ `analysis/_overhead_worker.py`) |

## Other files in `results/`

| File | Content |
|---|---|
| `RP/*_lw50.json`, `RP/SSC_lw30.json` | Layer-wise CV sweep (50 / 30 draws) for the layer-wise variant of Fig. 1 (`outputs/figures/fig1_robustness_layerwise.pdf`, not in the paper) |
| `results/training/*.json` | End-of-training summaries of the main runs (training-time evaluator; `SSC-clean.json` is the separate SSC Clean run with the same configuration). Used by `eval/aggregate.py` and the PTQ (b) numbers |
| `RD/assumptions_rsnn_ssc_fixed.json`, `RD/lambda_r_by_dataset_arch*_fixed.json` | Diagnostic checks on SHD-RSNN, SSC, CIFAR-10 (`diagnostics/eval_typical_case_assumptions_fixed.py`, `diagnostics/plot_lambda_r_by_dataset_arch_fixed.py`); not quoted in the paper |

Checkpoint paths inside the result files are repository-relative (`checkpoints/<setting>/best_s{seed}_{mode}.pth`).
