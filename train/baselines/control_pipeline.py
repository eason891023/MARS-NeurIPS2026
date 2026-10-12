#!/usr/bin/env python3
"""Pilot simple-control hyperparameters, select by validation accuracy, then run five seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validation_score(path: Path, mode: str) -> float:
    payload = json.loads((path / "results.json").read_text())
    return float(payload["runs"][f"s42_{mode}"]["best_val"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-epochs", type=int, default=15)
    parser.add_argument("--formal-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train_controls.py")
    base = [
        sys.executable,
        str(script),
        "--seeds",
        "42",
        "--epochs",
        str(args.pilot_epochs),
        "--num-workers",
        str(args.num_workers),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
    ]
    selection: dict[str, dict[str, float]] = {}
    for architecture in ["ff", "rsnn"]:
        candidates = []
        for index, value in enumerate([1e-3, 1e-2, 1e-1]):
            output = args.output_dir / "pilots" / f"{architecture}_beta_l2_{index}"
            run(
                base
                + [
                    "--architecture",
                    architecture,
                    "--modes",
                    "beta_l2",
                    "--lambda-beta-l2",
                    str(value),
                    "--output-dir",
                    str(output),
                ]
            )
            candidates.append((validation_score(output, "beta_l2"), value))
        best_score, best_value = max(candidates)
        selection.setdefault(architecture, {})["lambda_beta_l2"] = best_value
        selection[architecture]["lambda_beta_l2_pilot_val"] = best_score

    drop_candidates = []
    for index, value in enumerate([0.05, 0.10, 0.20]):
        output = args.output_dir / "pilots" / f"rsnn_dropconnect_{index}"
        run(
            base
            + [
                "--architecture",
                "rsnn",
                "--modes",
                "dropconnect",
                "--dropconnect-p",
                str(value),
                "--output-dir",
                str(output),
            ]
        )
        drop_candidates.append((validation_score(output, "dropconnect"), value))
    best_score, best_value = max(drop_candidates)
    selection["rsnn"]["dropconnect_p"] = best_value
    selection["rsnn"]["dropconnect_pilot_val"] = best_score
    (args.output_dir / "selected_hyperparameters.json").write_text(json.dumps(selection, indent=2))

    formal_base = [
        sys.executable,
        str(script),
        "--seeds",
        "42,52,72,92,102",
        "--epochs",
        str(args.formal_epochs),
        "--num-workers",
        str(args.num_workers),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
    ]
    run(
        formal_base
        + [
            "--architecture",
            "ff",
            "--modes",
            "beta_l2,beta_clip,fixed_homogeneous,fixed_heterogeneous",
            "--lambda-beta-l2",
            str(selection["ff"]["lambda_beta_l2"]),
            "--output-dir",
            str(args.output_dir / "formal_ff"),
        ]
    )
    run(
        formal_base
        + [
            "--architecture",
            "rsnn",
            "--modes",
            "beta_l2,beta_clip,fixed_homogeneous,fixed_heterogeneous,dropconnect",
            "--lambda-beta-l2",
            str(selection["rsnn"]["lambda_beta_l2"]),
            "--dropconnect-p",
            str(selection["rsnn"]["dropconnect_p"]),
            "--output-dir",
            str(args.output_dir / "formal_rsnn"),
        ]
    )


if __name__ == "__main__":
    main()
