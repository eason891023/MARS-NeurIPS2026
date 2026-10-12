#!/usr/bin/env bash
# Layer-wise/global drift at CV 0.2 with 100 draws per checkpoint (Table 1, layer-wise and global
# columns; paired differences), plus the optional layer-wise CV sweep (50 draws; SSC 30 draws).
# Output: outputs/eval/partial/<setting>_cv02d100.json, <setting>_lw50.json / SSC_lw30.json
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG=logs/eval; mkdir -p "$LOG"; SWEEP=0.05,0.1,0.15,0.2,0.3

qA() {  # SHD-FC, SHD-RSNN, DVS-Gesture, CIFAR-10
  for s in SHD-FC SHD-RSNN DVS-Gesture CIFAR-10; do
    python3 eval/run_unified.py --setting "$s" --modes layer_shared,global --cvs 0.2 --draws 100 --tag _cv02d100 > "$LOG/${s}_cv02d100.log" 2>&1
  done
  if [ "${WITH_LW_SWEEP:-0}" = "1" ]; then
    for s in SHD-FC SHD-RSNN DVS-Gesture CIFAR-10; do
      python3 eval/run_unified.py --setting "$s" --modes layer_shared --cvs $SWEEP --draws 50 --tag _lw50 > "$LOG/${s}_lw50.log" 2>&1
    done
  fi
}
qB() {  # SSC
  python3 eval/run_unified.py --setting SSC --modes layer_shared,global --cvs 0.2 --draws 100 --tag _cv02d100 > "$LOG/SSC_cv02d100.log" 2>&1
  if [ "${WITH_LW_SWEEP:-0}" = "1" ]; then
    python3 eval/run_unified.py --setting SSC --modes layer_shared --cvs $SWEEP --draws 30 --tag _lw30 > "$LOG/SSC_lw30.log" 2>&1
  fi
}
qA & qB & wait
echo "done: outputs/eval/partial/"
