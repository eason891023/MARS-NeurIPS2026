#!/usr/bin/env bash
# Main unified beta-mismatch run (per-neuron CV sweep 0.05-0.3, layer-wise/global CV 0.1/0.2;
# 10 draws per checkpoint and CV), all five settings, two GPU queues in parallel.
# Output: outputs/eval/partial/<setting>.json ; logs: logs/eval/<setting>.log
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG=logs/eval; mkdir -p "$LOG"
(
  for s in SHD-FC SHD-RSNN CIFAR-10 DVS-Gesture; do
    python3 eval/run_unified.py --setting "$s" > "$LOG/$s.log" 2>&1
  done
) &
( python3 eval/run_unified.py --setting SSC > "$LOG/SSC.log" 2>&1 ) &
wait
# 200-draw per-neuron estimate at CV 0.2 for SHD (Table 1 / Table 3 / Fig. 4)
for s in SHD-FC SHD-RSNN; do
  python3 eval/run_unified.py --setting "$s" --only-cv 0.2 --draws 200 --tag _mc200 > "$LOG/${s}_mc200.log" 2>&1
  python3 eval/run_unified.py --setting "$s" --controls --only-cv 0.2 --draws 200 --tag _mc200 > "$LOG/${s}_controls_mc200.log" 2>&1
done
echo "done: outputs/eval/partial/"
