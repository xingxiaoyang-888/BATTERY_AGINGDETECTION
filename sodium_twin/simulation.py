"""Inspectable lumped electrical/thermal model for portable scenario exploration."""

import numpy as np
from .schemas import SimulationRequest


def simulate(request: SimulationRequest) -> dict:
    cell_current = request.current_a / request.parallel
    effective_capacity = request.capacity_ah * request.initial_soh_pct / 100
    soc, temperature = request.initial_soc_pct / 100, request.ambient_c
    polarization = 0.0
    rows = []
    stopped = False
    cutoff_reason = None
    resistance = 0.004 * (50 / request.capacity_ah) * (100 / request.initial_soh_pct)
    heat_capacity = 900 * request.capacity_ah / 50
    # Integrate at one-second resolution and retain an economical output grid.
    interval = max(1, request.duration_s // 180)
    for second in range(request.duration_s + 1):
        current = 0.0 if stopped else cell_current
        r0 = resistance * np.exp((25 - temperature) / 80)
        ocv = 2.0 + 1.35 * soc + 0.1 * np.tanh((soc - 0.5) * 8)
        cell_voltage = float(ocv - current * r0 - polarization)
        if not stopped and (cell_voltage < 1.5 or cell_voltage > 4.2 or temperature >= 70):
            stopped = True
            cutoff_reason = "voltage" if temperature < 70 else "temperature"
            current = 0.0
            cell_voltage = float(ocv - polarization)
        if second % interval == 0 or second == request.duration_s:
            rows.append(
                {
                    "time_s": second,
                    "voltage_v": round(cell_voltage * request.series, 3),
                    "current_a": 0.0 if stopped else request.current_a,
                    "soc_pct": round(soc * 100, 3),
                    "temperature_c": round(temperature, 3),
                    "power_kw": round(
                        cell_voltage * request.series * current * request.parallel / 1000, 3
                    ),
                }
            )
        if second == request.duration_s:
            break
        next_soc = soc - current / (effective_capacity * 3600)
        if not 0.02 < next_soc < 0.98:
            stopped = True
            cutoff_reason = "soc"
            current = 0
            next_soc = soc
        soc = next_soc
        polarization += (current * r0 * 0.5 - polarization) * (1 - np.exp(-1 / 45))
        heat_w = current**2 * r0 + polarization**2 / max(r0 * 0.5, 1e-8)
        cooling_decay = np.exp(-request.cooling_w_k / heat_capacity)
        temperature = (
            request.ambient_c
            + (temperature - request.ambient_c) * cooling_decay
            + heat_w / request.cooling_w_k * (1 - cooling_decay)
        )
    return {
        "engine": "lumped-rc",
        "label": "Equivalent-circuit scenario model",
        "provenance": "simulation",
        "calibrated": False,
        "series": request.series,
        "parallel": request.parallel,
        "trajectory": rows,
        "summary": {
            "peak_temperature_c": max(row["temperature_c"] for row in rows),
            "final_soc_pct": rows[-1]["soc_pct"],
            "final_voltage_v": rows[-1]["voltage_v"],
            "cutoff_reached": stopped,
            "cutoff_reason": cutoff_reason,
        },
        "cell_temperatures": [
            [round(temperature, 2)] * request.parallel for _ in range(request.series)
        ],
        "warnings": [
            "Uncalibrated lumped-parameter simulation. Identical cells are assumed; no spatial gradients are resolved.",
            "Use the Modelica/FMU solver for heterogeneous cell and fault studies.",
        ]
        + (
            [f"Protective {cutoff_reason} cutoff reached; current was set to zero."]
            if stopped
            else []
        ),
    }
