"""
Training overhead measurement v2: one mode per subprocess to avoid contention.
Runs 5 epochs each (first 2 warmup, last 3 measured) of SHD-FC training (train/shd_fc.py).
Output: outputs/analysis/training_overhead/results_v2.json
        (shipped copy: results/analysis/training_overhead_v2.json; paper: 58.3 s -> 80.1 s per epoch)
"""
import json
import os
import subprocess
import sys


def _find_project_root(start_file: str) -> str:
    cur = os.path.abspath(os.path.dirname(start_file))
    while True:
        if os.path.isdir(os.path.join(cur, "snn_robust_tau")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot locate project root")
        cur = parent


PROJECT_ROOT = _find_project_root(__file__)
sys.path.insert(0, PROJECT_ROOT)
from snn_robust_tau.paths import OUTPUT_ROOT  # noqa: E402

WORKER_SCRIPT = os.path.join(PROJECT_ROOT, "analysis", "_overhead_worker.py")


def main():
    output_dir = str(OUTPUT_ROOT / "analysis" / "training_overhead")
    os.makedirs(output_dir, exist_ok=True)

    venv_python = sys.executable

    results = {}
    for mode in ["clean", "asar", "hajr", "noise_aug"]:
        print(f"\n{'='*40}")
        print(f"Benchmarking: {mode} (isolated process)")
        print(f"{'='*40}")

        result = subprocess.run(
            [venv_python, WORKER_SCRIPT, mode],
            capture_output=True, text=True, timeout=600,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-500:])

        # Parse output
        tmp_path = os.path.join(output_dir, f"_tmp_{mode}.json")
        if os.path.exists(tmp_path):
            with open(tmp_path) as f:
                results[mode] = json.load(f)
            os.remove(tmp_path)
        else:
            print(f"WARNING: no result file for {mode}")

    if not results:
        print("No results collected!")
        return

    clean_t = results.get("clean", {}).get("mean_s", 1.0)
    print(f"\n{'Mode':>12s} {'Time (s)':>10s} {'Overhead':>10s}")
    print("-" * 35)
    for mode in ["clean", "asar", "hajr", "noise_aug"]:
        if mode not in results:
            continue
        t = results[mode]["mean_s"]
        overhead = (t / clean_t - 1) * 100
        results[mode]["overhead_pct"] = float(overhead)
        print(f"{mode:>12s} {t:>10.2f} {overhead:>+9.1f}%")

    with open(os.path.join(output_dir, "results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/results_v2.json")


if __name__ == "__main__":
    main()
