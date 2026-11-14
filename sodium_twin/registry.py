"""Artifact-backed model availability. Architecture support never implies training."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from .data import FEATURES

MODELS = [
    {
        "id": "robust-trend",
        "name": "Robust trend",
        "family": "Statistical baseline",
        "description": "Outlier-resistant historical slope with an explicit scenario sensitivity envelope.",
        "tags": ["No training", "Interpretable", "Scenario exploration"],
    },
    {
        "id": "patchtst",
        "name": "Physics-guided Patch Transformer",
        "family": "PatchTST-inspired",
        "description": "Channel-independent temporal patches, masked reconstruction, and ordered degradation quantiles.",
        "tags": ["Self-supervised auxiliary", "Attention", "Monotonic SOH"],
    },
    {
        "id": "itransformer",
        "name": "Inverted Transformer",
        "family": "iTransformer-inspired",
        "description": "Sensor histories become tokens, allowing attention to learn cross-variable dependencies.",
        "tags": ["Multivariate attention", "Quantiles", "Censored survival"],
    },
    {
        "id": "selective-ssm",
        "name": "Selective State Space",
        "family": "Mamba-inspired reference",
        "description": "Input-selective diagonal recurrence for linear-time sequence modeling without custom CUDA kernels.",
        "tags": ["Linear-time scan", "Selective memory", "Portable PyTorch"],
    },
    {
        "id": "tcn",
        "name": "Temporal ConvNet",
        "family": "Causal convolution baseline",
        "description": "Dilated causal convolutions provide an efficient deep-learning comparison.",
        "tags": ["Causal", "Dilated convolution", "Quantiles"],
    },
]


def inspect_artifact(directory: Path):
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    if path.stat().st_size > 5_000_000:
        raise ValueError("Manifest exceeds the metadata size limit.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 2
        or data.get("status") != "validated"
    ):
        raise ValueError("Artifact has not passed validation.")
    if data.get("weights_format") != "safetensors":
        raise ValueError("Only safetensors artifacts can be activated.")
    for key in ("config", "scaler", "metrics", "calibration", "split_cells"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"Invalid {key} metadata.")
    if data.get("provenance") != "measured":
        raise ValueError("Synthetic artifacts cannot be activated for operational inference.")
    weights = directory / "weights.safetensors"
    if not weights.is_file() or weights.stat().st_size > 200_000_000:
        raise ValueError("Missing or oversized weights.")
    with weights.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != data.get("weights_sha256"):
        raise ValueError("Model weight checksum mismatch.")
    groups = data.get("split_cells", {})
    if set(groups) != {"train", "validation", "calibration", "test"}:
        raise ValueError("Four independent data partitions are required.")
    if any(
        not isinstance(group, list) or any(not isinstance(cell, str) for cell in group)
        for group in groups.values()
    ):
        raise ValueError("Invalid cell partition metadata.")
    flattened = [cell for group in groups.values() for cell in group]
    if len(set(flattened)) != len(flattened) or any(not group for group in groups.values()):
        raise ValueError("Model metadata contains overlapping or empty cell partitions.")
    if data.get("config", {}).get("architecture") != directory.name:
        raise ValueError("Architecture does not match the registry directory.")
    config = data["config"]
    limits = {
        "features": (4, 4),
        "lookback": (8, 1024),
        "horizon": (1, 3000),
        "width": (8, 256),
        "heads": (1, 16),
        "layers": (1, 8),
        "patch_length": (1, 1024),
        "stride": (1, 1024),
    }
    if set(config) != {*limits, "architecture", "dropout"}:
        raise ValueError("Model configuration fields do not match the serving contract.")
    for key, (low, high) in limits.items():
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("Model configuration exceeds serving limits.")
    if (
        config["width"] % config["heads"]
        or config["patch_length"] > config["lookback"]
        or math.ceil((config["lookback"] - config["patch_length"]) / config["stride"]) + 1 > 256
        or not isinstance(config["dropout"], (int, float))
        or not 0 <= config["dropout"] < 1
    ):
        raise ValueError("Invalid attention, patch, or dropout configuration.")
    scaler = data["scaler"]
    for key in ("center", "scale", "minimum", "maximum"):
        values = scaler.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise ValueError("Invalid scaler metadata.")
    if any(v <= 0 for v in scaler["scale"]) or any(
        lo > hi for lo, hi in zip(scaler["minimum"], scaler["maximum"])
    ):
        raise ValueError("Invalid scaler domain or scale.")
    if data.get("features") != FEATURES or data.get("chemistry") not in {
        "sodium-ion",
        "lithium-ion",
    }:
        raise ValueError("Missing or incompatible feature/chemistry metadata.")
    calibration = data.get("calibration", {})
    alpha = calibration.get("alpha", 0)
    radius = calibration.get("radius", -1)
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("Invalid calibration coverage metadata.")
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
        raise ValueError("Invalid calibration radius.")
    count = len(groups["calibration"])
    if calibration.get("cells") != count or math.ceil((count + 1) * (1 - alpha)) > count:
        raise ValueError("Insufficient or inconsistent independent calibration cells.")
    evaluation = data.get("metrics", {})
    for field in ("mae_pp", "rmse_pp", "point_coverage", "mean_width_pp", "persistence_mae_pp"):
        value = evaluation.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing or invalid held-out evaluation metrics.")
    if evaluation["mae_pp"] >= evaluation["persistence_mae_pp"] or evaluation["point_coverage"] > 1:
        raise ValueError("Artifact does not satisfy the held-out evaluation gate.")
    return data


def model_registry(root):
    result = []
    for item in MODELS:
        entry = {
            **item,
            "status": "available" if item["id"] == "robust-trend" else "untrained",
            "metrics": None,
            "parameters": None,
        }
        if item["id"] != "robust-trend":
            try:
                manifest = inspect_artifact(root / item["id"])
                if manifest:
                    if not importlib.util.find_spec("torch") or not importlib.util.find_spec(
                        "safetensors"
                    ):
                        raise ValueError(
                            "Install the optional PyTorch runtime to activate this artifact."
                        )
                    entry.update(status="validated", metrics=manifest["metrics"])
            except (ValueError, OSError, KeyError, TypeError) as exc:
                entry.update(status="unavailable", reason=str(exc))
        result.append(entry)
    return result
