"""Strict cycle CSV ingestion and explicitly synthetic demonstration data."""

from io import StringIO
import re
import numpy as np
import pandas as pd

FEATURES = ["soh_pct", "temperature_c", "c_rate", "capacity_ah"]
REQUIRED = ["cell_id", "cycle_index", *FEATURES]


def parse_csv(content: str, *, web_limits: bool = True) -> pd.DataFrame:
    if web_limits and len(content.encode("utf-8")) > 2_000_000:
        raise ValueError("CSV exceeds the 2 MB limit.")
    try:
        frame = pd.read_csv(StringIO(content), dtype={"cell_id": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("CSV could not be parsed.") from exc
    return validate_frame(frame, web_limits=web_limits)


def validate_frame(frame: pd.DataFrame, *, web_limits: bool = True) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
    if len(frame) < 8 or (web_limits and len(frame) > 20000):
        raise ValueError("Provide at least eight rows; web imports support at most 20,000.")
    frame = frame[REQUIRED].copy()
    if frame.isna().any().any():
        raise ValueError("Missing values are not allowed; repair the source data explicitly.")
    frame["cell_id"] = frame.cell_id.astype(str).str.strip()
    if not frame.cell_id.map(lambda x: bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", x))).all():
        raise ValueError(
            "Cell IDs must use 1–80 letters, digits, periods, underscores, or hyphens."
        )
    for col in REQUIRED[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[REQUIRED[1:]].to_numpy()).all():
        raise ValueError("All measurements must be finite.")
    bounds = {
        "cycle_index": (0, 1000000),
        "soh_pct": (1, 105),
        "temperature_c": (-40, 100),
        "c_rate": (0.01, 15),
        "capacity_ah": (0.01, 2000),
    }
    for col, (lower, upper) in bounds.items():
        if not frame[col].between(lower, upper).all():
            raise ValueError(f"{col} must be between {lower} and {upper}.")
    if (frame.cycle_index % 1 != 0).any():
        raise ValueError("Cycle indices must be integers.")
    if frame.duplicated(["cell_id", "cycle_index"]).any():
        raise ValueError("Duplicate cell/cycle pairs are not allowed.")
    if frame.groupby("cell_id").size().min() < 8:
        raise ValueError("Each cell needs at least eight observations.")
    if web_limits and frame.cell_id.nunique() > 200:
        raise ValueError("A workspace import supports at most 200 cells.")
    frame["cycle_index"] = frame.cycle_index.astype(int)
    return frame.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)


def demo_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for index in range(12):
        cycles = np.arange(0, 641, 5)
        rate = 0.0085 + index * 0.00065
        soh = 100 - rate * cycles - 0.000003 * cycles**2
        soh += rng.normal(0, 0.10, len(cycles))
        for cycle, health in zip(cycles, soh):
            rows.append(
                {
                    "cell_id": f"NA-{index + 1:03d}",
                    "cycle_index": int(cycle),
                    "soh_pct": round(float(health), 3),
                    "temperature_c": 24 + index * 0.45,
                    "c_rate": 1.0,
                    "capacity_ah": 50.0,
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    cells = []
    for cell, group in frame.groupby("cell_id", sort=True):
        last = group.iloc[-1]
        cells.append(
            {
                "id": cell,
                "cycles": int(last.cycle_index),
                "soh_pct": round(float(last.soh_pct), 2),
                "temperature_c": round(float(last.temperature_c), 1),
                "capacity_ah": float(last.capacity_ah),
                "observations": len(group),
                "c_rate": float(last.c_rate),
                "status": "review" if last.soh_pct < 90 else "stable",
            }
        )
    return {"rows": len(frame), "cell_count": len(cells), "cells": cells}
