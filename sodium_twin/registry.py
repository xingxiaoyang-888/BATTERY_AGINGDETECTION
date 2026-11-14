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
    if data.get("schema_version") != 1 or data.get("status") != "validated":
        raise ValueError("Artifact has not passed validation.")
    if data.get("provenance") != "measured":
        raise ValueError("Synthetic artifacts cannot be activated for operational inference.")
    weights = directory / "weights.pt"
    if not weights.is_file() or weights.stat().st_size > 200_000_000:
        raise ValueError("Missing or oversized weights.")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != data.get("weights_sha256"):
        raise ValueError("Model weight checksum mismatch.")
    groups = data.get("split_cells", {})
    if set(groups) != {"train", "validation", "calibration", "test"}:
        raise ValueError("Four independent data partitions are required.")
    flattened = [cell for group in groups.values() for cell in group]
    if len(set(flattened)) != len(flattened) or any(not group for group in groups.values()):
        raise ValueError("Model metadata contains overlapping or empty cell partitions.")
    if data.get("config", {}).get("architecture") != directory.name:
        raise ValueError("Architecture does not match the registry directory.")
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
                    if not importlib.util.find_spec("torch"):
                        raise ValueError(
                            "Install the optional PyTorch runtime to activate this artifact."
                        )
                    entry.update(status="validated", metrics=manifest["metrics"])
            except (ValueError, OSError, KeyError, TypeError) as exc:
                entry.update(status="unavailable", reason=str(exc))
        result.append(entry)
    return result
