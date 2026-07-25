"""SOH 残差学习与持久性锚点测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.soh_ai.models import BiLSTMAttention, TemporalTransformer, XGBoostWrapper
from models.soh_ai.trainer import EnsembleTrainer


class _ZeroResidualModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=np.float32)


class TestResidualSOHModels(unittest.TestCase):
    def test_xgboost_zero_residual_equals_persistence(self):
        model = XGBoostWrapper(residual_learning=True, soh_feature_index=1)
        model._model = _ZeroResidualModel()
        model._is_fitted = True
        X = np.array([[3.0, 0.98], [4.0, 0.91]], dtype=np.float32)

        prediction = model.predict(X).ravel()

        np.testing.assert_allclose(prediction, X[:, 1])

    def test_lstm_initial_prediction_equals_current_soh(self):
        model = BiLSTMAttention(input_dim=3, soh_feature_index=0).eval()
        X = torch.randn(4, 8, 3)
        X[:, -1, 0] = torch.tensor([0.99, 0.95, 0.90, 0.85])

        with torch.no_grad():
            prediction = model(X).squeeze(-1)

        torch.testing.assert_close(prediction, X[:, -1, 0])

    def test_transformer_initial_prediction_equals_current_soh(self):
        model = TemporalTransformer(input_dim=3, soh_feature_index=0).eval()
        X = torch.randn(2, 8, 3)
        X[:, -1, 0] = torch.tensor([0.97, 0.88])

        with torch.no_grad():
            prediction = model(X).squeeze(-1)

        torch.testing.assert_close(prediction, X[:, -1, 0])

    def test_test_metrics_always_include_persistence(self):
        trainer = EnsembleTrainer()
        trainer.soh_feature_index = 0
        X = np.zeros((3, 2, 2), dtype=np.float32)
        X[:, -1, 0] = np.array([0.99, 0.95, 0.90])
        y = np.array([[0.98], [0.94], [0.89]], dtype=np.float32)

        metrics = trainer._evaluate_test(X, y)

        self.assertIn("persistence", metrics)
        self.assertAlmostEqual(metrics["persistence"]["RMSE"], 0.01, places=6)

    def test_ensemble_weights_are_persisted(self):
        trainer = EnsembleTrainer()
        trainer.ensemble.set_weights(xgb=0.2, lstm=0.8, transformer=0.0)
        trainer.results = {"history": {}, "test_results": {}}

        with tempfile.TemporaryDirectory() as tmp:
            trainer.save_all(tmp)
            weights = json.loads(
                (Path(tmp) / "ensemble_weights.json").read_text(encoding="utf-8")
            )

        self.assertAlmostEqual(weights["xgb"], 0.2)
        self.assertAlmostEqual(weights["lstm"], 0.8)


if __name__ == "__main__":
    unittest.main()
