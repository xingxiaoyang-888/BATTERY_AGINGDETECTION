"""Structural, gradient, leakage, and calibration checks without optimizer updates."""

from argparse import Namespace
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from sodium_twin.ml.dataset import TrainScaler, make_windows, split_cells
from sodium_twin.ml.networks import (
    BatteryForecaster,
    ModelConfig,
    discrete_survival_nll,
    pinball_loss,
)
from sodium_twin.ml.uncertainty import apply_calibration, calibrate

torch.set_num_threads(2)


@pytest.mark.parametrize("architecture", ["patchtst", "itransformer", "selective-ssm", "tcn"])
def test_models_preserve_order_physics_batch_independence_and_gradients(architecture):
    torch.manual_seed(42)
    model = BatteryForecaster(
        ModelConfig(
            architecture=architecture, lookback=17, horizon=20, width=16, layers=2, dropout=0
        )
    )
    x, anchor = torch.randn(3, 17, 4), torch.tensor([90.0, 93.0, 96.0])
    model.eval()
    prediction = model(x, anchor)
    assert prediction.shape == (3, 20, 3)
    assert torch.all(torch.diff(prediction, dim=1) <= 0)
    assert torch.all(torch.diff(prediction, dim=2) >= 0)
    assert torch.all(prediction <= anchor[:, None, None])
    assert torch.all(prediction >= 0)
    assert torch.allclose(model(x[[2, 0, 1]], anchor[[2, 0, 1]]), prediction[[2, 0, 1]], atol=1e-5)
    target = anchor[:, None] - torch.arange(1, 21) * 0.03
    loss = pinball_loss(prediction, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_masked_reconstruction_and_independent_attention_initialization():
    model = BatteryForecaster(ModelConfig(lookback=16, horizon=20, width=16, layers=2))
    encoder = model.encoder
    assert not torch.equal(encoder.blocks[0].linear1.weight, encoder.blocks[1].linear1.weight)
    loss = encoder.masked_loss(torch.randn(2, 16, 4))
    assert torch.isfinite(loss)
    loss.backward()
    assert encoder.reconstruct.weight.grad.abs().sum() > 0


def test_survival_likelihood_accounts_for_censoring_without_fake_events():
    logits = torch.zeros(2, 3, requires_grad=True)
    result = discrete_survival_nll(logits, torch.tensor([1, 2]), torch.tensor([True, False]))
    assert float(result) == pytest.approx(2.5 * np.log(2))
    result.backward()
    assert logits.grad[1].gt(0).all()
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 2] == 0


def fixture_frame():
    return pd.DataFrame(
        [
            {
                "cell_id": f"C{i:02d}",
                "cycle_index": t,
                "soh_pct": 100 - 0.02 * t,
                "temperature_c": 20 + i,
                "c_rate": 1.0,
                "capacity_ah": 50.0,
            }
            for i in range(20)
            for t in range(48)
        ]
    )


def test_splits_and_scaling_never_observe_heldout_cells():
    frame = fixture_frame()
    groups = split_cells(frame)
    assert groups == split_cells(frame)
    all_ids = sum(groups.values(), [])
    assert len(all_ids) == len(set(all_ids)) == 20
    training = frame[frame.cell_id.isin(groups["train"])]
    scaler = TrainScaler.fit(training)
    before = scaler.to_dict()
    heldout = frame[~frame.cell_id.isin(groups["train"])].copy()
    heldout["temperature_c"] = 9999
    scaler.transform(heldout[["soh_pct", "temperature_c", "c_rate", "capacity_ah"]])
    assert scaler.to_dict() == before
    x, anchors, y, ids = make_windows(training, scaler, 16, 20)
    assert x.shape[1:] == (16, 4) and y.shape[1] == 20
    assert set(ids) <= set(groups["train"])
    assert np.all(y[:, 0] < anchors)


def test_gapped_cycles_are_rejected_instead_of_compressed():
    frame = fixture_frame()
    scaler = TrainScaler.fit(frame)
    frame = frame[frame.cycle_index != 17]
    with pytest.raises(ValueError, match="consecutive"):
        make_windows(frame, scaler, 16, 20)


def test_group_calibration_uses_cells_and_finite_sample_rank():
    target = np.full((8, 3), 95.0)
    prediction = np.repeat(np.array([94.0, 95.0, 96.0])[None, None, :], 8, axis=0)
    prediction = np.repeat(prediction, 3, axis=1)
    target[-1, -1] = 99
    calibration = calibrate(prediction, target, np.repeat(["A", "B", "C", "D"], 2))
    assert calibration["cells"] == 4
    assert calibration["radius"] == 3
    corrected = apply_calibration(prediction, calibration)
    assert ((target >= corrected[..., 0]) & (target <= corrected[..., 2])).all()
    with pytest.raises(ValueError, match="Too few"):
        calibrate(prediction, target, ["same-cell"] * 8)


def test_dry_run_never_steps_optimizer_or_creates_weights(tmp_path, monkeypatch):
    from sodium_twin.ml.train import run

    source = tmp_path / "cycles.csv"
    fixture_frame().to_csv(source, index=False)
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("Training is forbidden in dry-run")
    )
    args = Namespace(
        csv=source, seed=42, architecture="patchtst", lookback=16, horizon=20, execute=False
    )
    result = run(args)
    assert result["optimizer_steps"] == 0
    assert result["mode"] == "dry-run"
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.safetensors"))
