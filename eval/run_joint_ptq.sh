#!/usr/bin/env bash
# Table 2: per-neuron beta mismatch combined with weight/threshold mismatch or 8/6-bit PTQ (SHD, 200 draws).
# Output: outputs/eval/partial/SHD-*_jointptq.json ; logs: logs/eval/
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG=logs/eval; mkdir -p "$LOG"
for s in SHD-FC SHD-RSNN; do
  python3 eval/run_joint_ptq.py --setting "$s" --draws 200 > "$LOG/${s}_jointptq.log" 2>&1
done
echo "done: outputs/eval/partial/SHD-*_jointptq.json"
