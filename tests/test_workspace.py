"""Exercise complete upload, prediction, persistence, and export flows."""

import numpy as np
import pytest
from sodium_twin.data import parse_csv
from sodium_twin.schemas import ForecastRequest, SimulationRequest
from sodium_twin.forecast import robust_forecast
from sodium_twin.simulation import simulate


def test_upload_forecast_export_and_restart(client, cycle_csv):
    uploaded = client.post("/api/v2/datasets", json={"name": "Measured cohort", "csv": cycle_csv})
    assert uploaded.status_code == 201
    dataset = uploaded.json()
    assert dataset["provenance"] == "user-upload"
    response = client.post(
        "/api/v2/forecasts",
        json={"dataset_id": dataset["id"], "cell_id": "CELL-01", "horizon": 1200},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["uncertainty"]["calibrated"] is False
    assert result["rul_cycles"] == pytest.approx(901, abs=0.1)
    assert result["provenance"] == "user-upload"
    saved = client.get(f"/api/v2/runs/{result['id']}").json()
    assert saved["request"]["dataset_id"] == dataset["id"]
    assert saved["result"]["trajectory"] == result["trajectory"]
    exported = client.get(f"/api/v2/runs/{result['id']}/export.csv")
    assert exported.status_code == 200
    assert exported.text.startswith("cycle_index,median,lower,upper")
    assert client.get("/api/v2/runs").json()[0]["id"] == result["id"]
    from sodium_twin.store import Store

    reopened = Store(client.app.state.store.path.parent)
    assert reopened.frame(dataset["id"])[1]["rows"] == 100


def test_demo_and_untrained_models_are_explicit(client):
    assert client.get("/api/v2/health").json()["training_enabled"] is False
    assert client.get("/api/v2/datasets").json()[0]["provenance"] == "synthetic"
    models = client.get("/api/v2/models").json()
    assert len(models) == 5
    assert all(m["status"] == "untrained" and m["metrics"] is None for m in models[1:])
    response = client.post(
        "/api/v2/forecasts", json={"dataset_id": "demo", "cell_id": "NA-001", "model": "patchtst"}
    )
    assert response.status_code in {422, 503}
    assert client.get("/api/v2/runs").json() == []


@pytest.mark.parametrize(
    "change",
    [
        {"horizon": 999999},
        {"temperature_c": 80},
        {"eol_pct": 100},
        {"c_rate": 0},
        {"model": "imaginary-model"},
        {"typo_field": 5},
    ],
)
def test_bounded_request_contracts(client, change):
    result = client.post(
        "/api/v2/forecasts", json={"dataset_id": "demo", "cell_id": "NA-001", **change}
    )
    assert result.status_code == 422


def test_cross_origin_writes_and_missing_resources(client, cycle_csv):
    assert (
        client.post(
            "/api/v2/datasets",
            json={"name": "Test", "csv": cycle_csv},
            headers={"origin": "https://unrelated.example"},
        ).status_code
        == 403
    )
    assert client.get("/api/v2/runs/absent").status_code == 404
    assert client.get("/api/v2/datasets/demo/cells/absent").status_code == 404
    assert client.get("/api/v2/health").headers["X-Request-ID"]


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "nan", "infinite", "missing", "fraction", "out-of-range", "formula-id"],
)
def test_bad_data_is_rejected(cycle_csv, mutation):
    frame = parse_csv(cycle_csv)
    if mutation == "duplicate":
        frame.loc[1, "cycle_index"] = 0
    if mutation == "nan":
        frame.loc[1, "soh_pct"] = np.nan
    if mutation == "infinite":
        frame.loc[1, "soh_pct"] = np.inf
    if mutation == "missing":
        frame = frame.drop(columns="capacity_ah")
    if mutation == "fraction":
        frame["cycle_index"] = frame.cycle_index.astype(float)
        frame.loc[1, "cycle_index"] = 1.5
    if mutation == "out-of-range":
        frame.loc[1, "soh_pct"] = 120
    if mutation == "formula-id":
        frame["cell_id"] = "=1+2"
    with pytest.raises(ValueError):
        parse_csv(frame.to_csv(index=False))


def test_forecast_respects_horizon_monotonicity_and_stress(cycle_csv):
    frame = parse_csv(cycle_csv)
    request = ForecastRequest(dataset_id="x", cell_id="CELL-01", horizon=600)
    result = robust_forecast(frame, request, "user-upload")
    assert result["rul_cycles"] is None and result["rul_censored"]
    prediction = np.array(
        [[p[k] for k in ["lower", "median", "upper"]] for p in result["trajectory"]]
    )
    assert (np.diff(prediction, axis=0) <= 0).all()
    assert (np.diff(prediction, axis=1) >= 0).all()
    stressed = robust_forecast(
        frame, request.model_copy(update={"temperature_c": 45, "c_rate": 3}), "user-upload"
    )
    assert stressed["rate_pct_per_100_cycles"] > result["rate_pct_per_100_cycles"]


def test_flat_or_improving_history_does_not_invent_finite_life(cycle_csv):
    frame = parse_csv(cycle_csv)
    frame["soh_pct"] = 98.0
    result = robust_forecast(frame, ForecastRequest(dataset_id="x", cell_id="CELL-01"), "synthetic")
    assert result["rul_cycles"] is None
    assert result["trajectory"][-1]["median"] == 98
    assert any("Synthetic" in warning for warning in result["warnings"])


def test_already_below_eol_is_zero_rul(cycle_csv):
    frame = parse_csv(cycle_csv)
    frame["soh_pct"] -= 25
    result = robust_forecast(
        frame, ForecastRequest(dataset_id="x", cell_id="CELL-01"), "user-upload"
    )
    assert result["rul_cycles"] == 0 and not result["rul_censored"]


@pytest.mark.parametrize("current,comparison", [(40, "discharge"), (-40, "charge"), (0, "rest")])
def test_simulation_has_correct_current_and_energy_direction(current, comparison):
    req = SimulationRequest(current_a=current)
    result = simulate(req)
    final = result["summary"]
    if comparison == "discharge":
        assert final["final_soc_pct"] < req.initial_soc_pct
    if comparison == "charge":
        assert final["final_soc_pct"] > req.initial_soc_pct
    if comparison == "rest":
        assert final["final_soc_pct"] == req.initial_soc_pct
    assert final["peak_temperature_c"] >= req.ambient_c
    assert result["calibrated"] is False
    assert len(result["trajectory"]) <= 201


def test_soc_cutoff_removes_current_and_initial_conditions_are_preserved():
    req = SimulationRequest(
        series=8, parallel=2, capacity_ah=1, current_a=100, duration_s=200, initial_soc_pct=10
    )
    result = simulate(req)
    assert result["summary"]["cutoff_reached"]
    assert result["trajectory"][-1]["current_a"] == 0
    assert result["trajectory"][0]["temperature_c"] == req.ambient_c
    assert result["summary"]["final_soc_pct"] >= 2


def test_simulation_api_is_persisted(client):
    response = client.post("/api/v2/simulations", json={"duration_s": 10})
    assert response.status_code == 200
    result = response.json()
    assert client.get(f"/api/v2/runs/{result['id']}").json()["kind"] == "simulation"


def test_extreme_small_cell_parameters_remain_finite_and_trigger_protection():
    result = simulate(SimulationRequest(capacity_ah=0.1, current_a=500, cooling_w_k=20))
    assert result["summary"]["cutoff_reason"] == "voltage"
    assert all(np.isfinite(row["temperature_c"]) for row in result["trajectory"])
    assert all(row["current_a"] == 0 for row in result["trajectory"])


def test_offline_training_is_not_restricted_by_web_import_size(cycle_csv):
    frame = parse_csv(cycle_csv)
    import pandas as pd

    large = pd.concat([frame.assign(cell_id=f"C{i}") for i in range(201)], ignore_index=True)
    text = large.to_csv(index=False)
    with pytest.raises(ValueError):
        parse_csv(text)
    assert len(parse_csv(text, web_limits=False)) == 20100
