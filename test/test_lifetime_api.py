"""钠电寿命API的参数与单位转换测试。"""

import asyncio
import unittest

from pydantic import ValidationError

import server
from server import (
    LifetimeHistoryPoint,
    LifetimePredictionRequest,
    LifetimeScenarioRequest,
)


class _FakeLifetimeService:
    def __init__(self):
        self.call = None

    def predict(self, **kwargs):
        self.call = kwargs
        return {'status': 'ok', 'prediction': {'rul_cycles': 123}}

    def evidence_summary(self):
        return {'cells': 47, 'thresholds': []}


class TestLifetimeAPI(unittest.TestCase):
    def test_request_rejects_invalid_soc_and_history(self):
        with self.assertRaises(ValidationError):
            LifetimeScenarioRequest(soc_min_pct=90, soc_max_pct=10)
        with self.assertRaises(ValidationError):
            LifetimePredictionRequest(
                current_soh_pct=95,
                current_cycle=20,
                history=[
                    LifetimeHistoryPoint(cycle_index=10, soh_pct=98),
                    LifetimeHistoryPoint(cycle_index=9, soh_pct=97),
                ],
            )

    def test_predict_endpoint_converts_percent_units(self):
        fake = _FakeLifetimeService()
        previous = server._lifetime_service
        server._lifetime_service = fake
        try:
            request = LifetimePredictionRequest(
                current_soh_pct=95,
                current_cycle=100,
                eol_soh_pct=80,
                scenario=LifetimeScenarioRequest(soc_min_pct=10, soc_max_pct=90),
                history=[LifetimeHistoryPoint(cycle_index=90, soh_pct=96)],
            )
            response = asyncio.run(server.predict_sodium_lifetime(request))
        finally:
            server._lifetime_service = previous

        self.assertEqual(response['code'], 200)
        self.assertAlmostEqual(fake.call['current_soh'], 0.95)
        self.assertAlmostEqual(fake.call['eol_threshold'], 0.80)
        self.assertAlmostEqual(fake.call['scenario']['soc_min'], 0.10)
        self.assertAlmostEqual(fake.call['scenario']['soc_max'], 0.90)
        self.assertEqual(fake.call['history'][-1], {'cycle_index': 100, 'soh': 0.95})

    def test_evidence_endpoint_uses_cached_service(self):
        fake = _FakeLifetimeService()
        previous = server._lifetime_service
        server._lifetime_service = fake
        try:
            response = asyncio.run(server.get_lifetime_evidence())
        finally:
            server._lifetime_service = previous

        self.assertEqual(response['payload']['cells'], 47)


if __name__ == '__main__':
    unittest.main()
