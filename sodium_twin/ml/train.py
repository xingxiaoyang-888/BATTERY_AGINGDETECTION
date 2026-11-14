"""Reproducible training recipe. Dry-run is the default; execution is explicit."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import save_file

from ..data import FEATURES, parse_csv
from .dataset import TrainScaler, make_windows, split_cells
from .networks import BatteryForecaster, ModelConfig, discrete_survival_nll, pinball_loss
from .uncertainty import apply_calibration, calibrate, metrics


def predict_batches(model, windows, device):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(windows[0]), 64):
            x = torch.from_numpy(windows[0][offset : offset + 64]).to(device)
            anchor = torch.from_numpy(windows[1][offset : offset + 64]).to(device)
            outputs.append(model(x, anchor).cpu().numpy())
    return np.concatenate(outputs)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    content = Path(args.csv).read_text(encoding="utf-8-sig")
    frame = parse_csv(content, web_limits=False)
    splits = split_cells(frame, args.seed)
    frames = {name: frame[frame.cell_id.isin(ids)] for name, ids in splits.items()}
    scaler = TrainScaler.fit(frames["train"])
    cfg = ModelConfig(architecture=args.architecture, lookback=args.lookback, horizon=args.horizon)
    windows = {
        name: make_windows(data, scaler, cfg.lookback, cfg.horizon) for name, data in frames.items()
    }
    model = BatteryForecaster(cfg)
    report = {
        "mode": "train" if args.execute else "dry-run",
        "seed": args.seed,
        "model": model.metadata(),
        "split_cells": splits,
        "windows": {name: len(values[0]) for name, values in windows.items()},
        "optimizer_steps": 0,
    }
    if not args.execute:
        print(json.dumps(report, indent=2))
        return report
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epoch count and batch size must be positive.")
    destination = Path(args.output) / args.architecture
    if destination.exists():
        raise FileExistsError(
            "Output directory already exists. Choose a new output root to preserve artifacts."
        )
    device = torch.device(args.device)
    model.to(device)
    train = TensorDataset(*(torch.from_numpy(array) for array in windows["train"][:3]))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best, best_state, patience = float("inf"), None, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, anchor, target in loader:
            x, anchor, target = x.to(device), anchor.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, logits = model.forward_with_hazard(x, anchor)
            loss = pinball_loss(prediction, target)
            at_risk = anchor > args.eol_pct
            if at_risk.any():
                crossing = target[at_risk] <= args.eol_pct
                observed = crossing.any(dim=1)
                first = crossing.int().argmax(dim=1)
                bins = torch.where(observed, first, torch.full_like(first, cfg.horizon - 1))
                loss = loss + 0.01 * discrete_survival_nll(logits[at_risk], bins, observed)
            if args.architecture == "patchtst" and args.masked_weight > 0:
                loss = loss + args.masked_weight * model.encoder.masked_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            report["optimizer_steps"] += 1
            total += float(loss.detach()) * len(x)
        validation = predict_batches(model, windows["validation"], device)
        score = metrics(validation, windows["validation"][2])["mae_pp"]
        history.append({"epoch": epoch + 1, "loss": total / len(train), "validation_mae_pp": score})
        print(json.dumps(history[-1]))
        if score < best:
            best, patience = score, 0
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            patience += 1
        scheduler.step()
        if patience >= 8:
            break
    if best_state is None:
        raise RuntimeError("No finite validation score; no model artifact was saved.")
    model.load_state_dict(best_state)
    calibration_prediction = predict_batches(model, windows["calibration"], device)
    calibration = calibrate(
        calibration_prediction, windows["calibration"][2], windows["calibration"][3]
    )
    test_prediction = apply_calibration(
        predict_batches(model, windows["test"], device), calibration
    )
    evaluation = metrics(test_prediction, windows["test"][2])
    persistence = np.repeat(windows["test"][1][:, None], cfg.horizon, axis=1)
    evaluation["persistence_mae_pp"] = float(np.mean(abs(persistence - windows["test"][2])))
    passed = evaluation["mae_pp"] < evaluation["persistence_mae_pp"]
    # Artifacts remain unavailable to the API unless they beat the held-out baseline.
    manifest = {
        "schema_version": 2,
        "weights_format": "safetensors",
        "status": "validated" if passed else "rejected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "features": FEATURES,
        "chemistry": args.chemistry,
        "provenance": args.provenance,
        "data_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "scaler": scaler.to_dict(),
        "calibration": calibration,
        "metrics": evaluation,
        "split_cells": splits,
        "seed": args.seed,
        "history": history,
        "torch_version": torch.__version__,
        "optimizer_steps": report["optimizer_steps"],
        "eol_pct": args.eol_pct,
    }
    destination.mkdir(parents=True)
    weights = destination / "weights.safetensors"
    save_file({name: value.contiguous() for name, value in best_state.items()}, str(weights))
    manifest["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"artifact": str(destination), "status": manifest["status"], "metrics": evaluation},
            indent=2,
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--architecture",
        choices=["patchtst", "itransformer", "selective-ssm", "tcn"],
        default="patchtst",
    )
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--chemistry", choices=["sodium-ion", "lithium-ion"], default="sodium-ion")
    parser.add_argument("--provenance", choices=["measured", "synthetic"], default="measured")
    parser.add_argument("--eol-pct", type=float, default=80)
    parser.add_argument("--masked-weight", type=float, default=0.02)
    parser.add_argument("--execute", action="store_true", help="Explicitly enable optimizer steps.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
