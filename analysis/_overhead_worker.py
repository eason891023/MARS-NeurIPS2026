"""
Worker process: benchmark a single training mode.
Called by eval_training_overhead_v2.py as a subprocess.
Usage: python _overhead_worker.py <mode>
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


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
from snn_robust_tau.paths import OUTPUT_ROOT, RESULTS, ckpt_dir  # noqa: E402,F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    mode = sys.argv[1]

    spec = importlib.util.spec_from_file_location(
        "shd", os.path.join(PROJECT_ROOT, "train", "shd_fc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = mod.CONFIG.copy()
    cfg["num_workers"] = 0
    cfg["use_augmentations"] = False

    from snntorch import surrogate
    surrogate_fn = surrogate.atan(alpha=cfg.get("surrogate_alpha", 2.0))

    mod.seed_everything(42)
    train_loader, _, _, input_size = mod.build_loaders(42, cfg)

    model = mod.SHDSNN(
        input_size=input_size,
        beta_init=cfg["beta_init"],
        threshold_init=cfg.get("threshold_init", 1.0),
        surrogate_fn=surrogate_fn,
        learn_beta=cfg.get("learn_beta", True),
        learn_threshold=cfg.get("learn_threshold", False),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    if mode == "asar":
        lam = cfg["lambda_asar"]
    elif mode == "hajr":
        lam = cfg["lambda_hajr"]
    else:
        lam = 0.0

    n_warmup = 2
    n_measure = 3

    print(f"Device: {DEVICE}, mode: {mode}, batches/epoch: {len(train_loader)}")

    all_times = []
    for epoch in range(n_warmup + n_measure):
        model.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for data, targets in train_loader:
            data, targets = data.to(DEVICE).float(), targets.to(DEVICE)

            beta_originals = None
            if mode == "noise_aug":
                beta_originals = {}
                with torch.no_grad():
                    for idx, (_, layer) in enumerate(model._lif_layers):
                        beta_originals[idx] = layer.beta.data.clone()
                        noise = torch.randn_like(layer.beta) * cfg["noise_aug_cv"]
                        layer.beta.data.copy_((layer.beta.data * (1.0 + noise)).clamp(0.01, 0.99))

            reg_mode = mode if mode in {"hajr", "asar"} else None
            logits_t, loss_reg = model(data, reg_mode=reg_mode)
            loss = criterion(logits_t.sum(0), targets) + lam * loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if beta_originals is not None:
                with torch.no_grad():
                    for idx, (_, layer) in enumerate(model._lif_layers):
                        layer.beta.data.copy_(beta_originals[idx])

            optimizer.step()
            with torch.no_grad():
                for _, layer in model._lif_layers:
                    layer.beta.clamp_(0.01, 0.99)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        label = "WARMUP" if epoch < n_warmup else "MEASURE"
        print(f"  [{label}] epoch {epoch+1}: {elapsed:.2f}s")

        if epoch >= n_warmup:
            all_times.append(elapsed)

    mean_t = float(np.mean(all_times))
    std_t = float(np.std(all_times))
    print(f"  Result: {mean_t:.2f} ± {std_t:.2f} s/epoch")

    output_dir = str(OUTPUT_ROOT / "analysis" / "training_overhead")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"_tmp_{mode}.json"), "w") as f:
        json.dump({
            "mode": mode,
            "epoch_times": all_times,
            "mean_s": mean_t,
            "std_s": std_t,
        }, f, indent=2)


if __name__ == "__main__":
    main()
