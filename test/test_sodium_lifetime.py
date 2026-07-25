"""钠电删失标签与寿命推理契约测试。"""

import unittest

import numpy as np
import pandas as pd

from models.soh_ai.config import ACTUAL_FEATURE_COLUMNS
from models.soh_ai.evaluate import CovariateRolloutConfig, ModelEvaluator
from models.soh_ai.lifetime import (
    SodiumLifetimePredictor,
    build_eol_labels,
    build_landmark_labels,
)


class _DecayModel:
    def predict(self, batch):
        return np.asarray([[batch[0, -1, 1] - 0.01]], dtype=np.float32)


def _reference_frame():
    rows = []
    conditions = [
        ('a', 25.0, 1.0, 1.0, 0.008),
        ('b', 25.0, 2.0, 2.0, 0.015),
        ('c', 40.0, 1.0, 1.0, 0.020),
        ('d', -10.0, 1.0, 1.0, 0.030),
    ]
    for cell_id, temp, charge, discharge, loss in conditions:
        for cycle in range(1, 21):
            rows.append({
                'cell_id': cell_id,
                'condition': cell_id,
                'dataset_id': 'fixture',
                'cycle_index': cycle,
                'soh': 1.0 - loss * cycle,
                'temperature_c': temp,
                'c_rate_charge': charge,
                'c_rate_discharge': discharge,
                'soc_min': 0.0,
                'soc_max': 1.0,
                'nominal_capacity_ah': 1.2,
            })
    return pd.DataFrame(rows)


class TestSodiumLifetime(unittest.TestCase):
    def test_eol_labels_preserve_right_censoring(self):
        labels = build_eol_labels(_reference_frame(), thresholds=(0.8,))

        self.assertEqual(len(labels), 4)
        self.assertEqual(int(labels['event_observed'].sum()), 3)
        censored = labels[labels['cell_id'] == 'a'].iloc[0]
        self.assertFalse(bool(censored['event_observed']))
        self.assertEqual(censored['censoring'], 'right_censored')
        self.assertTrue(pd.isna(censored['event_cycle']))

    def test_landmarks_do_not_invent_censored_rul(self):
        landmarks = build_landmark_labels(
            _reference_frame(),
            thresholds=(0.8,),
            observation_fractions=(0.5,),
        )
        censored = landmarks[landmarks['cell_id'] == 'a'].iloc[0]

        self.assertFalse(bool(censored['rul_event_observed']))
        self.assertTrue(pd.isna(censored['actual_rul_cycles']))
        self.assertGreater(censored['rul_lower_bound_cycles'], 0)

    def test_rollout_returns_soh_column_and_is_monotonic(self):
        window = np.zeros((4, len(ACTUAL_FEATURE_COLUMNS)), dtype=np.float32)
        window[:, ACTUAL_FEATURE_COLUMNS.index('nominal_capacity_ah')] = 1.2
        window[:, ACTUAL_FEATURE_COLUMNS.index('soh')] = 0.95
        window[:, ACTUAL_FEATURE_COLUMNS.index('soc_max')] = 1.0
        window[:, ACTUAL_FEATURE_COLUMNS.index('soc_mean')] = 0.5
        window[:, ACTUAL_FEATURE_COLUMNS.index('coulombic_efficiency')] = 0.99

        trajectory = ModelEvaluator().rollout_sequence(
            _DecayModel(),
            window,
            steps=3,
            rollout_cfg=CovariateRolloutConfig(enforce_monotonic=True),
        )

        np.testing.assert_allclose(trajectory, [0.94, 0.93, 0.92], atol=1e-6)
        self.assertTrue(np.all(np.diff(trajectory) <= 0))

    def test_predictor_returns_monotonic_trajectory_and_ood_warning(self):
        predictor = SodiumLifetimePredictor(_reference_frame(), use_ai=False)
        result = predictor.predict(
            current_soh=0.95,
            current_cycle=100,
            eol_threshold=0.80,
            scenario={
                'temperature_c': 25.0,
                'c_rate_charge': 1.0,
                'c_rate_discharge': 1.0,
                'soc_min': 0.1,
                'soc_max': 0.9,
                'nominal_capacity_ah': 50.0,
            },
        )

        soh = [point['soh'] for point in result['trajectory']]
        self.assertTrue(np.all(np.diff(soh) <= 0))
        self.assertTrue(result['domain_check']['is_ood'])
        self.assertLess(result['uncertainty']['confidence_score'], 0.5)
        self.assertGreater(result['prediction']['rul_cycles'], 0)

    def test_predictor_rejects_invalid_physical_domain(self):
        predictor = SodiumLifetimePredictor(_reference_frame(), use_ai=False)

        with self.assertRaises(ValueError):
            predictor.predict(0.95, 10, eol_threshold=1.2)
        with self.assertRaises(ValueError):
            predictor.predict(0.95, 10, scenario={'c_rate_discharge': -1})

    def test_higher_thermal_and_electrical_stress_cannot_extend_rul(self):
        predictor = SodiumLifetimePredictor(_reference_frame(), use_ai=False)
        history = [
            {'cycle_index': cycle, 'soh': 1.0 - 0.001 * cycle}
            for cycle in range(1, 33)
        ]
        nominal = predictor.predict(
            0.968,
            32,
            eol_threshold=0.80,
            history=history,
            scenario={'temperature_c': 25, 'c_rate_charge': 1, 'c_rate_discharge': 1},
        )
        stressed = predictor.predict(
            0.968,
            32,
            eol_threshold=0.80,
            history=history,
            scenario={
                'temperature_c': 55,
                'temperature_spread_c': 10,
                'c_rate_charge': 5,
                'c_rate_discharge': 8,
            },
        )

        self.assertLessEqual(
            stressed['prediction']['rul_cycles'],
            nominal['prediction']['rul_cycles'],
        )
        self.assertGreater(stressed['stress']['history_rate_multiplier'], 1.0)
        self.assertLess(stressed['uncertainty']['confidence_score'], 0.8)

    def test_rul_ai_is_gated_above_98_percent_soh(self):
        predictor = SodiumLifetimePredictor(_reference_frame(), use_ai=False)
        predictor._ai_rate = lambda history, scenario: 0.001
        history = [
            {'cycle_index': cycle, 'soh': 1.0 - 0.0002 * cycle}
            for cycle in range(1, 33)
        ]

        early = predictor.predict(0.99, 32, 0.80, history=history)
        later = predictor.predict(0.97, 32, 0.80, history=history)

        self.assertFalse(any(
            source['source'] == 'xgboost_128_cycle'
            for source in early['rate_sources']
        ))
        self.assertTrue(any(
            source['source'] == 'xgboost_128_cycle'
            for source in later['rate_sources']
        ))


if __name__ == '__main__':
    unittest.main()
