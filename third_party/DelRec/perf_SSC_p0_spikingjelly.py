import argparse
import json
import sys

from configs.perf_SSC_p0_spikingjelly import Config
from src.SSC.snn import SNN_recurrent_delays
from src.SSC.trainer_p0_spikingjelly import run_experiment
from src.datasets import load_dataset


def _parse_int_list(raw: str):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_str_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="DelRec SSC p0 benchmark with SpikingJelly ParametricLIFNode")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seed list")
    parser.add_argument("--modes", type=str, default="", help="Comma-separated modes: clean,hajr,asar,noise_aug")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--datasets-path", type=str, default="", help="Path to SSC HDF5 directory")
    parser.add_argument("--results-dir", type=str, default="", help="Override results root")
    args = parser.parse_args()

    config = Config()
    if args.seeds:
        config.seeds = _parse_int_list(args.seeds)
    if args.modes:
        config.modes = _parse_str_list(args.modes)
    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.datasets_path:
        config.datasets_path = args.datasets_path
    if args.results_dir:
        config.results_dir = args.results_dir

    summary = run_experiment(config, load_dataset_fn=load_dataset, model_cls=SNN_recurrent_delays)
    print(json.dumps(summary, indent=2))
