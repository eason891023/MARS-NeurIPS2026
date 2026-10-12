import copy
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    class _DummyWandb:
        def init(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

        def login(self, *args, **kwargs):
            return None

    wandb = _DummyWandb()


def _find_project_root(start_file: str) -> str:
    cur = os.path.abspath(os.path.dirname(start_file))
    while True:
        if os.path.isdir(os.path.join(cur, "snn_robust_tau")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Cannot locate project root containing 'snn_robust_tau'.")
        cur = parent


PROJECT_ROOT = _find_project_root(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from snn_robust_tau.loss.hajr import HAJRLoss
from src.recurrent_neurons import axonal_recdel
from src.SSC.snn import spike_registrator
from src.spikingjelly_compat import beta_from_plif, set_beta_on_plif
from src.utils import calc_loss_SSC, calc_metric_SSC, progress_bar, reset_states, seed_everything

PRINT_EVERY = 30


def _stdout_is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def report_progress(current: int, total: int, msg: str) -> None:
    if _stdout_is_tty():
        progress_bar(current, total, msg, every=PRINT_EVERY)
        return
    if current == 0 or (current + 1) % PRINT_EVERY == 0 or (current + 1) == total:
        print(f"[{current + 1}/{total}] {msg}", flush=True)


def _config_to_dict(config) -> Dict[str, Any]:
    def _jsonable(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        return getattr(value, "__name__", repr(value))

    if is_dataclass(config):
        return {k: _jsonable(v) for k, v in asdict(config).items()}
    out = {}
    for key in dir(config):
        if key.startswith("__"):
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        out[key] = _jsonable(value)
    return out


def collect_regularizer_tensors(model) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[Any], List[str]]:
    mem_rec_list: List[torch.Tensor] = []
    spike_rec_list: List[torch.Tensor] = []
    beta_list: List[torch.Tensor] = []
    threshold_list: List[Any] = []
    reset_mechanism_list: List[str] = []

    for layer in model.layers:
        if isinstance(layer, axonal_recdel):
            if not hasattr(layer, "v_seq"):
                raise RuntimeError("axonal_recdel layer is missing v_seq. Ensure store_v_seq=True.")
            mem_rec_list.append(layer.v_seq)
            beta_list.append(beta_from_plif(layer.neuron_module))
            threshold_list.append(layer.neuron_module.v_threshold)
            reset_mechanism_list.append("soft" if layer.neuron_module.v_reset is None else "hard")
        elif isinstance(layer, spike_registrator):
            if layer.spikes is None:
                raise RuntimeError("spike_registrator has no recorded spikes. Forward pass must run before regularization.")
            spike_rec_list.append(layer.spikes)

    if len(mem_rec_list) != len(spike_rec_list):
        raise RuntimeError(
            f"Mismatched membrane/spike records: {len(mem_rec_list)} vs {len(spike_rec_list)}."
        )

    return mem_rec_list, spike_rec_list, beta_list, threshold_list, reset_mechanism_list


def get_regularizer(config, mode: str):
    if mode not in {"hajr", "asar"}:
        return None
    return HAJRLoss(
        threshold=float(config.v_threshold),
        mem_is_post_reset=True,
        surrogate_type=str(config.reg_surrogate_type),
        surrogate_alpha=float(config.reg_surrogate_alpha),
        reduction="mean",
    )


def compute_regularizer_loss(model, config, mode: str, regularizer) -> torch.Tensor:
    if regularizer is None:
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

    mem_rec_list, spike_rec_list, beta_list, threshold_list, reset_mechanism_list = collect_regularizer_tensors(model)
    decision_gate_list = [mode == "hajr"] * len(mem_rec_list)
    return regularizer(
        mem_rec_list,
        beta_list,
        threshold_list=threshold_list,
        spike_rec_list=spike_rec_list,
        reset_mechanism_list=reset_mechanism_list,
        decision_gate_list=decision_gate_list,
    )


def get_plif_layers(model) -> List[Tuple[str, Any]]:
    layers = []
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, axonal_recdel):
            layers.append((f"recdel{idx}", layer.neuron_module))
    return layers


def perturb_betas_for_training(model, cv: float) -> Dict[str, torch.Tensor]:
    originals: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, node in get_plif_layers(model):
            beta = beta_from_plif(node)
            originals[name] = beta.detach().clone()
            noise = torch.randn_like(beta) * cv
            perturbed = (beta * (1.0 + noise)).clamp(0.01, 0.99)
            set_beta_on_plif(node, perturbed)
    return originals


def restore_betas(model, originals: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, node in get_plif_layers(model):
            set_beta_on_plif(node, originals[name])


def clamp_plif_betas(model) -> None:
    with torch.no_grad():
        for _, node in get_plif_layers(model):
            set_beta_on_plif(node, beta_from_plif(node).clamp(0.01, 0.99))


def collect_lif_snapshot(model) -> List[Dict[str, Any]]:
    snapshot = []
    for name, node in get_plif_layers(model):
        snapshot.append(
            {
                "name": name,
                "beta": float(beta_from_plif(node).detach().mean().item()),
                "threshold": float(node.v_threshold),
                "reset_mechanism": "soft" if node.v_reset is None else "hard",
            }
        )
    return snapshot


def collect_delay_snapshot(model) -> List[Dict[str, Any]]:
    snapshot = []
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, axonal_recdel):
            snapshot.append(
                {
                    "name": f"recdel{idx}",
                    "sigma": float(layer.sigma),
                }
            )
    return snapshot


def restore_delay_snapshot(model, snapshot: List[Dict[str, Any]]) -> None:
    sigma_by_name = {item["name"]: float(item["sigma"]) for item in snapshot}
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, axonal_recdel):
            name = f"recdel{idx}"
            if name in sigma_by_name:
                layer.sigma = sigma_by_name[name]


def build_checkpoint_snapshot(model, optimizer=None, scheduler=None, epoch: int | None = None) -> Dict[str, Any]:
    snapshot = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        # DelRec forward depends on sigma but sigma is not part of state_dict.
        "delay_state": collect_delay_snapshot(model),
    }
    if optimizer is not None:
        snapshot["optimizer_state_dicts"] = [copy.deepcopy(opt.state_dict()) for opt in optimizer]
    if scheduler is not None:
        snapshot["scheduler_state_dicts"] = [copy.deepcopy(sch.state_dict()) for sch in scheduler]
    if epoch is not None:
        snapshot["epoch"] = int(epoch)
    return snapshot


def load_checkpoint_snapshot(model, snapshot: Dict[str, Any]) -> None:
    model.load_state_dict(snapshot["model_state_dict"])
    restore_delay_snapshot(model, snapshot.get("delay_state", []))


def _quantize_tensor_inplace(param: torch.Tensor, bits: int) -> None:
    max_abs = param.abs().max()
    if max_abs.item() == 0.0:
        return
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    param.copy_(torch.round(param / scale).clamp(-qmax, qmax) * scale)


def init_optim_sche(model, config):
    weights_norm = []
    weights = []
    positions = []
    seen = set()

    def add_param(bucket, param):
        if param is None or not param.requires_grad:
            return
        key = id(param)
        if key in seen:
            return
        seen.add(key)
        bucket.append(param)

    from src.SSC.snn import dcls_module, modified_batchnorm

    for m in model.layers:
        if isinstance(m, torch.nn.Linear):
            add_param(weights, m.weight)
            if config.bias:
                add_param(weights, m.bias)

        elif isinstance(m, axonal_recdel):
            add_param(weights, m.recurrent_weights)
            add_param(weights, getattr(m.neuron_module, "w", None))
            add_param(positions, m.recurrent_delays)
            if hasattr(m, "p_spread"):
                add_param(positions, m.p_spread)

        elif isinstance(m, dcls_module):
            add_param(weights, m.weight)
            if config.bias:
                add_param(weights, m.bias)
            add_param(positions, m.P)

        elif isinstance(m, modified_batchnorm):
            add_param(weights_norm, m.weight)
            if config.bias:
                add_param(weights_norm, m.bias)

    optimizer = [
        torch.optim.Adam(
            [
                {"params": weights, "lr": config.lr_w, "weight_decay": config.weight_decay},
                {"params": weights_norm, "lr": config.lr_w, "weight_decay": 0.0},
            ]
        )
    ]
    if positions:
        optimizer.append(torch.optim.Adam([{"params": positions, "lr": config.lr_positions, "weight_decay": 0.0}]))

    scheduler = []
    if config.scheduler_weights == "cos":
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=config.epochs))
    elif config.scheduler_weights == "onecycle":
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[0], max_lr=config.lr_w, total_steps=config.epochs))
    else:
        raise NotImplementedError

    if len(optimizer) > 1:
        if config.scheduler_pos == "cos":
            scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=config.epochs))
        elif config.scheduler_pos == "onecycle":
            scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[1], max_lr=config.lr_positions, total_steps=config.epochs))
        else:
            raise NotImplementedError

    return optimizer, scheduler


@torch.no_grad()
def evaluate(loader, model, device, config, desc: str = "Eval", show_progress: bool = False) -> float:
    model.eval()
    if hasattr(loader, "reset"):
        loader.reset()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.permute(1, 0, 2).float().to(device)
        targets = targets.to(device)
        reset_states(model=model)
        outputs = model(inputs)
        loss = calc_loss_SSC(outputs, targets)
        correct += calc_metric_SSC(outputs, targets)
        total += targets.size(0)
        loss_sum += float(loss.item())
        if show_progress:
            report_progress(
                batch_idx,
                len(loader),
                f"{desc} Loss: {loss_sum / (batch_idx + 1):.3f} | Acc: {100.0 * correct / max(total, 1):.3f}%",
            )
    return 100.0 * correct / max(total, 1)


def apply_beta_noise(model, cv: float) -> Dict[str, torch.Tensor]:
    originals: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, node in get_plif_layers(model):
            beta = beta_from_plif(node)
            originals[name] = beta.detach().clone()
            noise = torch.randn_like(beta) * cv
            perturbed = (beta * (1.0 + noise)).clamp(0.01, 0.99)
            set_beta_on_plif(node, perturbed)
    return originals


def evaluate_robustness(model, loader, device, config) -> Dict[str, Dict[str, Any]]:
    results = {}
    for cv in config.eval_cvs:
        cv = float(cv)
        print(f"    robustness iid cv={cv:.2f}", flush=True)
        if cv == 0.0:
            acc = evaluate(loader, model, device, config)
            results[str(cv)] = {"mean": acc, "std": 0.0, "runs": [acc]}
            continue
        runs = []
        for _ in range(int(config.eval_mc_runs)):
            original = apply_beta_noise(model, cv)
            runs.append(evaluate(loader, model, device, config))
            restore_betas(model, original)
        results[str(cv)] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0,
            "runs": runs,
        }
    return results


def apply_beta_noise_layerwise(model, cv: float) -> Dict[str, torch.Tensor]:
    originals: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, node in get_plif_layers(model):
            beta = beta_from_plif(node)
            originals[name] = beta.detach().clone()
            delta = float(torch.randn(1).item()) * cv
            perturbed = (beta * (1.0 + delta)).clamp(0.01, 0.99)
            set_beta_on_plif(node, perturbed)
    return originals


def apply_beta_noise_global(model, cv: float) -> Dict[str, torch.Tensor]:
    originals: Dict[str, torch.Tensor] = {}
    delta = float(torch.randn(1).item()) * cv
    with torch.no_grad():
        for name, node in get_plif_layers(model):
            beta = beta_from_plif(node)
            originals[name] = beta.detach().clone()
            perturbed = (beta * (1.0 + delta)).clamp(0.01, 0.99)
            set_beta_on_plif(node, perturbed)
    return originals


def evaluate_robustness_with_noise(model, loader, device, config, noise_fn) -> Dict[str, Dict[str, Any]]:
    results = {}
    for cv in config.eval_cvs:
        cv = float(cv)
        print(f"    robustness structured cv={cv:.2f}", flush=True)
        if cv == 0.0:
            acc = evaluate(loader, model, device, config)
            results[str(cv)] = {"mean": acc, "std": 0.0, "runs": [acc]}
            continue
        runs = []
        for _ in range(int(config.eval_mc_runs)):
            original = noise_fn(model, cv)
            runs.append(evaluate(loader, model, device, config))
            restore_betas(model, original)
        results[str(cv)] = {
            "mean": float(np.mean(runs)),
            "std": float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0,
            "runs": runs,
        }
    return results


def evaluate_quantization(model, loader, device, config) -> Dict[str, float]:
    base_state = copy.deepcopy(model.state_dict())
    results = {}
    with torch.no_grad():
        for bits in config.quant_bits:
            print(f"    quantization bits={bits}", flush=True)
            for layer in model.layers:
                if isinstance(layer, torch.nn.Linear):
                    _quantize_tensor_inplace(layer.weight, int(bits))
                    if layer.bias is not None:
                        _quantize_tensor_inplace(layer.bias, int(bits))
                elif isinstance(layer, axonal_recdel):
                    _quantize_tensor_inplace(layer.recurrent_weights, int(bits))
            results[str(bits)] = evaluate(loader, model, device, config)
            model.load_state_dict(base_state)
    return results


def train_one_mode(train_loader, valid_loader, test_loader, model, mode, device, config, seed):
    optimizer, scheduler = init_optim_sche(model, config)
    regularizer = get_regularizer(config, mode)

    best_val = -1.0
    best_epoch = -1
    best_snapshot = None
    history = []
    beta_trace = {"epoch": []}
    for idx, _ in enumerate(get_plif_layers(model), start=1):
        beta_trace[f"layer{idx}"] = []

    for epoch in range(config.epochs):
        model.train()
        if hasattr(train_loader, "reset"):
            train_loader.reset()
        if hasattr(valid_loader, "reset"):
            valid_loader.reset()

        for layer in model.layers:
            if isinstance(layer, axonal_recdel):
                layer.update_sigma(epoch)

        samples = 0
        correct = 0
        task_sum = 0.0
        reg_sum = 0.0

        ramp = min(1.0, float(epoch + 1) / max(int(config.rampup_epochs), 1))
        if mode == "hajr":
            lam = float(config.lambda_hajr) * ramp
        elif mode == "asar":
            lam = float(config.lambda_asar) * ramp
        else:
            lam = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.permute(1, 0, 2).float().to(device)
            targets = targets.to(device)

            beta_originals = None
            if mode == "noise_aug":
                beta_originals = perturb_betas_for_training(model, float(config.noise_aug_cv))

            reset_states(model=model)
            outputs = model(inputs)
            loss_task = calc_loss_SSC(outputs, targets)
            loss_reg = compute_regularizer_loss(model, config, mode, regularizer)
            loss = loss_task + lam * loss_reg

            for opt in optimizer:
                opt.zero_grad()
            loss.backward()

            if beta_originals is not None:
                restore_betas(model, beta_originals)

            for opt in optimizer:
                opt.step()

            clamp_plif_betas(model)
            if hasattr(model, "clamp_delays"):
                model.clamp_delays()

            batch_size = targets.size(0)
            samples += batch_size
            correct += calc_metric_SSC(outputs, targets)
            task_sum += loss_task.item() * batch_size
            reg_sum += float(loss_reg.detach().item()) * batch_size

        for sch in scheduler:
            sch.step()

        train_acc = 100.0 * correct / max(samples, 1)
        val_acc = evaluate(valid_loader, model, device, config, desc="Val", show_progress=False)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_snapshot = build_checkpoint_snapshot(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
            )

        beta_values = [float(beta_from_plif(node).detach().mean().item()) for _, node in get_plif_layers(model)]
        history.append(
            {
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "task_loss": task_sum / max(samples, 1),
                "reg_loss": reg_sum / max(samples, 1),
                "beta_mean": float(np.mean(beta_values)) if beta_values else 0.0,
                "lambda": lam,
            }
        )
        beta_trace["epoch"].append(epoch + 1)
        for idx, beta_val in enumerate(beta_values, start=1):
            beta_trace[f"layer{idx}"].append(beta_val)
        print(
            f"  [s{seed}][{mode}] e{epoch + 1:03d} "
            f"train={train_acc:.2f} val={val_acc:.2f} beta_mean={np.mean(beta_values):.3f} lambda={lam:.5f}",
            flush=True,
        )

    if best_snapshot is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    load_checkpoint_snapshot(model, best_snapshot)
    test_acc = evaluate(test_loader, model, device, config, desc="Test", show_progress=False)
    print(f"  [s{seed}][{mode}] BEST val={best_val:.2f}@e{best_epoch} -> test={test_acc:.2f}", flush=True)
    return model, test_acc, best_val, best_epoch, history, beta_trace, best_snapshot


def run_experiment(config, load_dataset_fn, model_cls):
    os.environ["WANDB_MODE"] = "disabled"
    wandb.login(key=None)

    if not os.path.isabs(config.datasets_path):
        config.datasets_path = os.path.join(PROJECT_ROOT, config.datasets_path)

    seed_list = list(config.seeds)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_root = Path(config.results_dir or os.path.join(PROJECT_ROOT, "outputs", "train", f"ssc_{timestamp}"))
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (results_root / "traces").mkdir(parents=True, exist_ok=True)

    config_dict = _config_to_dict(config)
    with open(results_root / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    all_results: Dict[str, Any] = {}
    all_histories: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    summary = {"device": str(device), "results_dir": str(results_root), "modes": {}}

    for seed in seed_list:
        config.seed = int(seed)
        seed_everything(seed=int(seed), is_cuda=torch.cuda.is_available())
        for mode in config.modes:
            train_loader, valid_loader, test_loader = load_dataset_fn(config)
            model = model_cls(config).to(device)

            run_name = f"{model.__class__.__name__}/{config.dataset}_{mode}_s{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="SSC", config=config_dict, name=run_name, reinit=True)

            print(f"\n=== Seed {seed} | Mode: {mode} ===", flush=True)
            model, test_acc, best_val, best_epoch, history, beta_trace, best_snapshot = train_one_mode(
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                model=model,
                mode=mode,
                device=device,
                config=config,
                seed=int(seed),
            )
            print(f"  [s{seed}][{mode}] running robustness + quantization...", flush=True)
            all_histories[(int(seed), mode)] = history
            robustness = evaluate_robustness(model, test_loader, device, config)
            robustness_layerwise = evaluate_robustness_with_noise(model, test_loader, device, config, apply_beta_noise_layerwise)
            robustness_global = evaluate_robustness_with_noise(model, test_loader, device, config, apply_beta_noise_global)
            quantization = evaluate_quantization(model, test_loader, device, config)

            trace_path = results_root / "traces" / f"beta_trace_s{seed}_{mode}.json"
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump(beta_trace, handle, indent=2)

            ckpt_path = results_root / "checkpoints" / f"best_s{seed}_{mode}.pth"
            checkpoint_payload = {
                **best_snapshot,
                "config": config_dict,
                "seed": int(seed),
                "mode": mode,
                "history": history,
                "test_acc": float(test_acc),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
                "beta_trace": beta_trace,
                "lif_parameters": collect_lif_snapshot(model),
            }
            torch.save(checkpoint_payload, ckpt_path)

            all_results[f"s{seed}_{mode}"] = {
                "seed": int(seed),
                "mode": mode,
                "test_acc": float(test_acc),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch),
                "checkpoint": str(ckpt_path),
                "beta_trace": str(trace_path),
                "lif_parameters": collect_lif_snapshot(model),
                "history": history,
                "robustness": robustness,
                "robustness_layerwise": robustness_layerwise,
                "robustness_global": robustness_global,
                "quantization": quantization,
            }
            with open(results_root / "results.json", "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, indent=2)

            wandb.finish()

    for mode in config.modes:
        test_accs = [all_results[f"s{seed}_{mode}"]["test_acc"] for seed in seed_list]
        robustness = {}
        for cv in config.eval_cvs:
            vals = [all_results[f"s{seed}_{mode}"]["robustness"][str(cv)]["mean"] for seed in seed_list]
            robustness[str(cv)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        quantization = {}
        for bits in config.quant_bits:
            vals = [all_results[f"s{seed}_{mode}"]["quantization"][str(bits)] for seed in seed_list]
            quantization[str(bits)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        robustness_layerwise = {}
        robustness_global = {}
        for cv in config.eval_cvs:
            vals_lw = [all_results[f"s{seed}_{mode}"]["robustness_layerwise"][str(cv)]["mean"] for seed in seed_list]
            vals_gl = [all_results[f"s{seed}_{mode}"]["robustness_global"][str(cv)]["mean"] for seed in seed_list]
            robustness_layerwise[str(cv)] = {
                "mean": float(np.mean(vals_lw)),
                "std": float(np.std(vals_lw, ddof=1)) if len(vals_lw) > 1 else 0.0,
            }
            robustness_global[str(cv)] = {
                "mean": float(np.mean(vals_gl)),
                "std": float(np.std(vals_gl, ddof=1)) if len(vals_gl) > 1 else 0.0,
            }
        summary["modes"][mode] = {
            "test_acc": {
                "mean": float(np.mean(test_accs)),
                "std": float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
            },
            "robustness": robustness,
            "robustness_layerwise": robustness_layerwise,
            "robustness_global": robustness_global,
            "quantization": quantization,
        }

    with open(results_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    final = {"config": config_dict, "per_seed": all_results, "summary": summary["modes"]}
    with open(results_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)

    return final
