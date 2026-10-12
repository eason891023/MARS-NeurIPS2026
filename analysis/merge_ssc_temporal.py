#!/usr/bin/env python3
"""Validate and merge separately nohupped SSC method shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--asar", type=Path, required=True)
    parser.add_argument("--noise-aug", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    reference = None
    for expected_mode, path in [
        ("clean", args.clean),
        ("asar", args.asar),
        ("noise_aug", args.noise_aug),
    ]:
        payload = json.loads(path.read_text())
        metadata = payload["metadata"]
        if metadata["modes"] != [expected_mode]:
            raise RuntimeError(f"{path}: expected only {expected_mode}")
        signature = (
            metadata["seeds"],
            metadata["conditions"],
            metadata["dataset"],
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise RuntimeError(f"{path}: metadata differs across shards")
        expected = {
            (expected_mode, int(seed), condition)
            for seed in metadata["seeds"]
            for condition in metadata["conditions"]
        }
        keys = {
            (row["mode"], int(row["seed"]), row["condition"])
            for row in payload["records"]
        }
        if keys != expected or len(payload["records"]) != len(expected):
            raise RuntimeError(
                f"{path}: got {len(keys)}/{len(expected)} unique expected keys"
            )
        records.extend(payload["records"])
    clean_metadata = json.loads(args.clean.read_text())["metadata"]
    metadata = dict(clean_metadata)
    metadata["modes"] = ["clean", "noise_aug", "asar"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, {"metadata": metadata, "records": records})
    print(f"PASS: merged {len(records)} records -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
