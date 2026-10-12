"""Post-hoc sensitivity energy (MARS loss) and training-time drop at CV 0.2 for the SSC checkpoints.
Output: outputs/analysis/ssc_scatter_points.json (shipped copy: results/mechanism/ssc_scatter_points.json).
Usage : python3 analysis/compute_ssc_scatter_points.py
"""
import json
import sys
from pathlib import Path

import h5py
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS, SSC_ROOT, ckpt_dir  # noqa: E402

DELREC_ROOT = ROOT / "third_party" / "DelRec"
DATASET_PATH = SSC_ROOT / "ssc_test.h5"
OUTPUT_PATH = OUTPUT_ROOT / "analysis" / "ssc_scatter_points.json"
EVAL_CV = "0.2"
MAX_BATCHES = 10
SEEDS = [42, 52, 72, 92, 102]
MODES = ["clean", "hajr", "asar", "noise_aug"]

# Training-run results (drop at CV 0.2 of the training-time evaluator). Clean SSC was trained in a
# separate run (results/training/SSC-clean.json); all SSC checkpoints live in checkpoints/SSC/.
RESULTS_BY_MODE = {
    "clean": RESULTS / "training" / "SSC-clean.json",
    "hajr": RESULTS / "training" / "SSC.json",
    "asar": RESULTS / "training" / "SSC.json",
    "noise_aug": RESULTS / "training" / "SSC.json",
}

CKPT_DIR_BY_MODE = {m: ckpt_dir("SSC") for m in MODES}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing SSC test set at {DATASET_PATH}. "
            "Download `ssc_test.h5.gz` from https://zenkelab.org/datasets first."
        )

    sys.path.insert(0, str(DELREC_ROOT))

    from configs.perf_SSC_p0_spikingjelly import Config
    from src.SSC.snn import SNN_recurrent_delays
    from src.SSC.snntorch_p0 import SSCSpikeIterator
    from src.SSC.trainer_p0_spikingjelly import (
        compute_regularizer_loss,
        get_regularizer,
        load_checkpoint_snapshot,
    )

    config = Config()
    config.datasets_path = str(DATASET_PATH.parent)
    device = torch.device("cpu")
    regularizer = get_regularizer(config, "asar")

    with h5py.File(DATASET_PATH, "r") as handle:
        test_loader = SSCSpikeIterator(
            handle,
            batch_size=config.batch_size,
            nb_steps=config.time_window,
            nb_units=700,
            max_time=1.4,
            n_bins=config.n_bins,
            shuffle=False,
        )

        results_cache = {
            mode: json.loads(RESULTS_BY_MODE[mode].read_text())["per_seed"]
            for mode in MODES
        }

        points = []
        for mode in MODES:
            for seed in SEEDS:
                key = f"s{seed}_{mode}"
                ckpt_path = CKPT_DIR_BY_MODE[mode] / f"best_s{seed}_{mode}.pth"
                model = SNN_recurrent_delays(config).to(device)
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                load_checkpoint_snapshot(model, checkpoint)
                model.eval()

                if hasattr(test_loader, "reset"):
                    test_loader.reset()

                total_loss = 0.0
                n_batches = 0
                with torch.no_grad():
                    for inputs, _targets in test_loader:
                        inputs = inputs.permute(1, 0, 2).float().to(device)
                        _ = model(inputs)
                        loss = compute_regularizer_loss(model, config, "asar", regularizer)
                        total_loss += float(loss.item())
                        n_batches += 1
                        if n_batches >= MAX_BATCHES:
                            break

                robust = results_cache[mode][key]["robustness"]
                clean_acc = float(robust["0.0"]["mean"])
                robust_acc = float(robust[EVAL_CV]["mean"])
                point = {
                    "dataset": "SSC",
                    "mode": mode,
                    "seed": seed,
                    "asar_loss": total_loss / max(n_batches, 1),
                    "clean_acc": clean_acc,
                    "robust_acc": robust_acc,
                    "acc_drop": clean_acc - robust_acc,
                    "max_batches": n_batches,
                }
                print(
                    f"s{seed}_{mode}: loss={point['asar_loss']:.4f}, "
                    f"drop@{EVAL_CV}={point['acc_drop']:.2f}",
                    flush=True,
                )
                points.append(point)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(points, indent=2))
    print(f"Wrote {len(points)} SSC points to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
