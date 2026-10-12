#!/usr/bin/env bash
# Regenerate every paper table and figure that can be rebuilt from the shipped result files
# (results/). CPU only, no checkpoints or datasets needed. Outputs go to outputs/{tables,figures,analysis,eval}.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python3}"
OUT="${MARS_OUTPUT_ROOT:-outputs}"
mkdir -p "$OUT/tables"

echo "== Main tables 1-3 (outputs/tables/table{1,2,3}_rows.tex)"
$PY paper_artifacts/make_main_tables.py
echo "== Appendix: main / per-seed / paired / joint / PTQ(a) rows"
$PY paper_artifacts/make_appendix_batch1.py > /dev/null
$PY paper_artifacts/make_appendix_batch2.py > /dev/null
echo "== Appendix: distribution, temporal, SSC joint, cost ratios, mechanism digest"
$PY paper_artifacts/build_tables.py
echo "== Appendix: mechanism correlation (r / rho)"
$PY paper_artifacts/recompute_correlation.py
echo "== Appendix: remaining numbers (lambda sweep, beta/activity, R^2, PTQ(b), stability, cost)"
$PY paper_artifacts/appendix_cr_extra_numbers.py > "$OUT/tables/appendix_extra_numbers.txt"
echo "   -> $OUT/tables/appendix_extra_numbers.txt"
echo "== Figures 1, 3, 4 (outputs/figures/)"
$PY paper_artifacts/make_fig1_variants.py
$PY paper_artifacts/make_figures_fixed.py
$PY paper_artifacts/make_fig4_new.py
echo "== Analyses that run from shipped raw results (outputs/analysis/, outputs/eval/)"
$PY analysis/mechanism_triangulation.py > /dev/null
$PY analysis/analyze_memory_capacity.py > /dev/null
$PY analysis/analyze_ssc_temporal.py --results results/analysis/ssc_temporal_raw.json --output-dir "$OUT/analysis/ssc_temporal" > /dev/null
$PY eval/aggregate.py > /dev/null
echo "done"
