"""Transparent history-only baseline, separate from learned model inference."""

import numpy as np
import pandas as pd
from .schemas import ForecastRequest


def robust_forecast(frame: pd.DataFrame, request: ForecastRequest, provenance: str) -> dict:
    history = frame[frame.cell_id == request.cell_id].sort_values("cycle_index")
    if len(history) < 8:
        raise ValueError("At least eight observed points are required for the selected cell.")
    recent = history.tail(64)
    x = recent.cycle_index.to_numpy(dtype=float)
    y = recent.soh_pct.to_numpy(dtype=float)
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    slopes = dy[dx > 0] / dx[dx > 0]
    observed_rate = max(0.0, -float(np.median(slopes)))
    historical_temp = float(recent.temperature_c.median())
    historical_rate = float(recent.c_rate.median())
    thermal = np.exp(np.clip((request.temperature_c - historical_temp) / 35, -1, 1))
    electrical = (request.c_rate / historical_rate) ** 0.45
    stress = float(np.clip(thermal * electrical, 0.3, 4))
    rate = observed_rate * stress
    last_x, last_y = float(x[-1]), float(y[-1])
    residual = y - (last_y + observed_rate * (last_x - x))
    noise = max(0.05, float(np.median(abs(residual - np.median(residual)))) * 1.4826)
    slope_spread = max(0.0001, float(np.quantile(slopes, 0.8) - np.quantile(slopes, 0.2)))
    offsets = np.unique(np.r_[0, np.linspace(1, request.horizon, 100).astype(int)])
    median = np.clip(last_y - rate * offsets, 0, 105)
    # This envelope is a sensitivity estimate, not calibrated probabilistic coverage.
    fast = np.clip(
        last_y - (rate + slope_spread * stress) * offsets - noise * np.sqrt(offsets), 0, 105
    )
    slow = np.clip(last_y - max(0, rate - slope_spread * stress) * offsets, 0, 105)
    fast[0] = slow[0] = last_y
    crossings = offsets[median <= request.eol_pct]
    already_eol = last_y <= request.eol_pct
    rul = 0.0 if already_eol else ((last_y - request.eol_pct) / rate if rate > 1e-9 else None)
    within = rul is not None and rul <= request.horizon
    warnings = ["Uncalibrated sensitivity envelope; nominal coverage is not available."]
    if provenance == "synthetic":
        warnings.append("Synthetic demonstration data; results are not experimental evidence.")
    if (
        abs(request.temperature_c - historical_temp) > 10
        or abs(request.c_rate - historical_rate) > 1
    ):
        warnings.append(
            "Scenario differs substantially from observed conditions; extrapolation risk is high."
        )
    if not within:
        warnings.append(
            "EOL is not reached within the requested horizon; no RUL point estimate is reported."
        )
    return {
        "model": "robust-trend",
        "model_label": "Robust history baseline",
        "provenance": provenance,
        "cell_id": request.cell_id,
        "current_soh_pct": last_y,
        "current_cycle": int(last_x),
        "rul_cycles": round(rul, 1) if within else None,
        "rul_censored": not within,
        "horizon": request.horizon,
        "eol_pct": request.eol_pct,
        "rate_pct_per_100_cycles": round(rate * 100, 4),
        "uncertainty": {"kind": "sensitivity", "calibrated": False, "coverage": None},
        "scenario": {
            "temperature_c": request.temperature_c,
            "c_rate": request.c_rate,
            "relative_stress": round(stress, 3),
        },
        "history": history[["cycle_index", "soh_pct"]].to_dict("records"),
        "trajectory": [
            {
                "cycle_index": int(last_x + h),
                "median": round(float(m), 4),
                "lower": round(float(l), 4),
                "upper": round(float(u), 4),
            }
            for h, m, l, u in zip(offsets, median, fast, slow)
        ],
        "warnings": warnings,
    }
