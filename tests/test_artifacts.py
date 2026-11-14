"""Model activation must fail closed for missing, corrupted, or leaked artifacts."""

import hashlib
import json
import pytest
from sodium_twin.registry import inspect_artifact, model_registry


def manifest_at(tmp_path):
    directory = tmp_path / "patchtst"
    directory.mkdir()
    weights = b"test fixture; not a trained checkpoint"
    (directory / "weights.safetensors").write_bytes(weights)
    manifest = {
        "schema_version": 2,
        "weights_format": "safetensors",
        "status": "validated",
        "provenance": "measured",
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "split_cells": {
            "train": ["a"],
            "validation": ["b"],
            "calibration": ["c1", "c2", "c3", "c4"],
            "test": ["d"],
        },
        "config": {
            "architecture": "patchtst",
            "features": 4,
            "lookback": 64,
            "horizon": 128,
            "patch_length": 8,
            "stride": 4,
            "width": 64,
            "heads": 4,
            "layers": 3,
            "dropout": 0.1,
        },
        "scaler": {
            "center": [90, 25, 1, 50],
            "scale": [1, 1, 1, 1],
            "minimum": [50, 0, 0, 1],
            "maximum": [100, 60, 5, 100],
        },
        "features": ["soh_pct", "temperature_c", "c_rate", "capacity_ah"],
        "chemistry": "sodium-ion",
        "calibration": {"alpha": 0.2, "radius": 1, "cells": 4},
        "metrics": {
            "mae_pp": 1,
            "rmse_pp": 1,
            "point_coverage": 0.8,
            "mean_width_pp": 2,
            "persistence_mae_pp": 2,
        },
    }
    return directory, manifest


@pytest.mark.parametrize(
    "issue",
    [
        "checksum",
        "overlap",
        "synthetic",
        "rejected",
        "version",
        "architecture",
        "calibration",
        "metrics",
    ],
)
def test_bad_artifact_is_never_available(tmp_path, issue):
    directory, manifest = manifest_at(tmp_path)
    if issue == "checksum":
        manifest["weights_sha256"] = "incorrect"
    if issue == "overlap":
        manifest["split_cells"]["test"] = ["a"]
    if issue == "synthetic":
        manifest["provenance"] = "synthetic"
    if issue == "rejected":
        manifest["status"] = "rejected"
    if issue == "version":
        manifest["schema_version"] = 1
    if issue == "architecture":
        manifest["config"]["architecture"] = "tcn"
    if issue == "calibration":
        manifest["calibration"]["cells"] = 1
    if issue == "metrics":
        manifest["metrics"]["mae_pp"] = 3
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_artifact(directory)
    assert model_registry(tmp_path)[1]["status"] == "unavailable"


def test_missing_artifact_does_not_create_random_inference_weights(tmp_path):
    assert inspect_artifact(tmp_path / "patchtst") is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("issue", ["root", "scaler", "shape", "format", "dimensions"])
def test_malformed_or_unsafe_metadata_cannot_activate(tmp_path, issue):
    directory, manifest = manifest_at(tmp_path)
    if issue == "root":
        manifest = []
    if issue == "scaler":
        manifest["scaler"]["scale"] = [0, 1, 1, 1]
    if issue == "shape":
        manifest["config"] = []
    if issue == "format":
        manifest["weights_format"] = "pickle"
    if issue == "dimensions":
        manifest["config"]["width"] = 1000000
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert model_registry(tmp_path)[1]["status"] == "unavailable"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_payload_returns_json_validation_error(client, value):
    response = client.post(
        "/api/v2/simulations",
        content='{"current_a":' + value + "}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "finite" in response.text
