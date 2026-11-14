"""Cell-disjoint splits, train-only scaling, and strictly causal observed windows."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..data import FEATURES


def split_cells(frame: pd.DataFrame, seed=42):
    cells = np.array(sorted(frame.cell_id.unique()))
    if len(cells) < 20:
        raise ValueError(
            "At least 20 independent cells are required for four-way splitting and calibration."
        )
    np.random.default_rng(seed).shuffle(cells)
    n = len(cells)
    test, validation, calibration = max(1, n // 7), max(1, n // 7), max(4, n // 5)
    indices = [0, n - test - validation - calibration, n - test - calibration, n - test, n]
    names = ["train", "validation", "calibration", "test"]
    return {name: cells[indices[i] : indices[i + 1]].tolist() for i, name in enumerate(names)}


@dataclass
class TrainScaler:
    center: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, frame):
        values = frame[FEATURES].to_numpy(dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Scaler requires finite training observations.")
        center = np.median(values, axis=0)
        scale = np.maximum(
            np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0), 1e-3
        )
        return cls(center, scale, values.min(axis=0), values.max(axis=0))

    def transform(self, values):
        return ((np.asarray(values) - self.center) / self.scale).astype(np.float32)

    def to_dict(self):
        return {
            name: getattr(self, name).tolist() for name in ("center", "scale", "minimum", "maximum")
        }

    @classmethod
    def from_dict(cls, data):
        arrays = {
            name: np.asarray(data[name], dtype=float)
            for name in ("center", "scale", "minimum", "maximum")
        }
        if any(
            array.shape != (len(FEATURES),) or not np.isfinite(array).all()
            for array in arrays.values()
        ):
            raise ValueError("Invalid scaler metadata.")
        if (arrays["scale"] <= 0).any():
            raise ValueError("Scaler scale must be positive.")
        return cls(**arrays)


def make_windows(frame, scaler, lookback, horizon, stride=8):
    xs, anchors, ys, groups = [], [], [], []
    if stride < 1:
        raise ValueError("Window stride must be positive.")
    for cell, group in frame.groupby("cell_id"):
        group = group.sort_values("cycle_index")
        # One model step means one physical cycle; never silently compress gaps.
        if not (np.diff(group.cycle_index) == 1).all():
            raise ValueError(f"Cell {cell}: deep models require consecutive cycle indices.")
        raw = group[FEATURES].to_numpy(dtype=float)
        normalized = scaler.transform(raw)
        for end in range(lookback, len(group) - horizon + 1, stride):
            xs.append(normalized[end - lookback : end])
            anchors.append(raw[end - 1, 0])
            ys.append(raw[end : end + horizon, 0])
            groups.append(cell)
    if not xs:
        raise ValueError(
            "Split has no complete windows; add longer cell histories or shorten the model horizon."
        )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(anchors, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(groups),
    )
