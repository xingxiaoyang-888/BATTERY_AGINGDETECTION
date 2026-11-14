"""Cell-grouped split conformal calibration with an explicit finite-sample rank."""

import math
import numpy as np


def calibrate(predictions, targets, cell_ids, alpha=0.2):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != (*targets.shape, 3) or targets.ndim != 2:
        raise ValueError("Expected predictions [N,H,3] and targets [N,H].")
    if len(cell_ids) != len(targets) or not 0 < alpha < 1:
        raise ValueError("Invalid calibration groups or alpha.")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Calibration values must be finite.")
    if (np.diff(predictions, axis=-1) < 0).any():
        raise ValueError("Prediction quantiles must be ordered.")
    scores = np.maximum(predictions[..., 0] - targets, targets - predictions[..., 2])
    cell_ids = np.asarray(cell_ids)
    grouped = [float(np.max(scores[cell_ids == cell])) for cell in sorted(set(cell_ids))]
    rank = math.ceil((len(grouped) + 1) * (1 - alpha))
    if rank > len(grouped):
        raise ValueError("Too few independent calibration cells for finite nominal coverage.")
    return {
        "radius": max(0.0, float(np.sort(grouped)[rank - 1])),
        "alpha": alpha,
        "cells": len(grouped),
        "unit": "cell",
        "scope": "simultaneous trajectory",
        "assumption": "Exchangeable independent cells and the same window sampling protocol; no guarantee under domain shift.",
    }


def apply_calibration(predictions, calibration):
    out = np.asarray(predictions).copy()
    out[..., 0] = np.maximum(0, out[..., 0] - calibration["radius"])
    out[..., 2] = np.minimum(105, out[..., 2] + calibration["radius"])
    return out


def metrics(predictions, target):
    return {
        "mae_pp": float(np.mean(abs(predictions[..., 1] - target))),
        "rmse_pp": float(np.sqrt(np.mean((predictions[..., 1] - target) ** 2))),
        "point_coverage": float(
            np.mean((target >= predictions[..., 0]) & (target <= predictions[..., 2]))
        ),
        "mean_width_pp": float(np.mean(predictions[..., 2] - predictions[..., 0])),
    }
