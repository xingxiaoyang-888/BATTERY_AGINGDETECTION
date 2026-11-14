"""Guarded inference from local, validated tensor-only checkpoints."""

import numpy as np
import torch
import hashlib
from safetensors.torch import load as load_tensors
from ..data import FEATURES
from ..registry import inspect_artifact
from .dataset import TrainScaler
from .networks import BatteryForecaster, ModelConfig
from .uncertainty import apply_calibration


def learned_forecast(frame, request, dataset, model_dir):
    directory = model_dir / request.model
    manifest = inspect_artifact(directory)
    if manifest is None:
        raise ValueError(
            "This architecture has no validated checkpoint. Use the baseline until training and evaluation are complete."
        )
    if manifest["chemistry"] != dataset["chemistry"]:
        raise ValueError("Dataset chemistry differs from the model's training chemistry.")
    if manifest["features"] != FEATURES:
        raise ValueError("Feature order does not match the serving contract.")
    cfg = ModelConfig(**manifest["config"])
    history = frame[frame.cell_id == request.cell_id].sort_values("cycle_index")
    if len(history) < cfg.lookback or request.horizon > cfg.horizon:
        raise ValueError(
            f"Model requires {cfg.lookback} observations and a horizon of at most {cfg.horizon} cycles."
        )
    recent = history.tail(cfg.lookback)
    if not (np.diff(recent.cycle_index) == 1).all():
        raise ValueError("Learned inference requires consecutive observed cycles.")
    last = recent.iloc[-1]
    if (
        abs(request.temperature_c - last.temperature_c) > 0.1
        or abs(request.c_rate - last.c_rate) > 0.01
    ):
        raise ValueError(
            "Learned models currently extrapolate observed operating conditions. Use the baseline for changed scenarios."
        )
    scaler = TrainScaler.from_dict(manifest["scaler"])
    values = recent[FEATURES].to_numpy()
    outside = (
        (values < scaler.minimum - 0.1 * scaler.scale)
        | (values > scaler.maximum + 0.1 * scaler.scale)
    ).any(axis=0)
    if outside.any():
        raise ValueError(
            "Outside training support: "
            + ", ".join(np.asarray(FEATURES)[outside])
            + ". Use the explicit baseline or collect calibration data."
        )
    model = BatteryForecaster(cfg)
    payload = (directory / "weights.safetensors").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Model weight checksum changed before inference.")
    state = load_tensors(payload)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Model weights must be finite.")
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(scaler.transform(values)[None]), torch.tensor([float(last.soh_pct)])
        )
    result = apply_calibration(prediction.numpy(), manifest["calibration"])[0, : request.horizon]
    crossing = np.flatnonzero(result[:, 1] <= request.eol_pct)
    rul = (
        0 if last.soh_pct <= request.eol_pct else (int(crossing[0]) + 1 if len(crossing) else None)
    )
    points = [
        {
            "cycle_index": int(last.cycle_index),
            "lower": float(last.soh_pct),
            "median": float(last.soh_pct),
            "upper": float(last.soh_pct),
        }
    ]
    points.extend(
        {
            "cycle_index": int(last.cycle_index) + i + 1,
            "lower": float(v[0]),
            "median": float(v[1]),
            "upper": float(v[2]),
        }
        for i, v in enumerate(result)
    )
    return {
        "model": request.model,
        "model_label": request.model,
        "cell_id": request.cell_id,
        "provenance": dataset["provenance"],
        "current_soh_pct": float(last.soh_pct),
        "current_cycle": int(last.cycle_index),
        "rul_cycles": rul,
        "rul_censored": rul is None,
        "horizon": request.horizon,
        "eol_pct": request.eol_pct,
        "rate_pct_per_100_cycles": float((last.soh_pct - result[-1, 1]) / request.horizon * 100),
        "uncertainty": {
            "kind": "cell-conformal",
            "calibrated": True,
            "coverage": 1 - manifest["calibration"]["alpha"],
        },
        "history": history[["cycle_index", "soh_pct"]].to_dict("records"),
        "trajectory": points,
        "scenario": {"temperature_c": request.temperature_c, "c_rate": request.c_rate},
        "warnings": [manifest["calibration"]["assumption"]],
    }
