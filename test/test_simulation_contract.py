"""仿真请求、CSV工况与电热老化数据闭环测试。"""

import io
import unittest

import numpy as np
import pandas as pd
from pydantic import ValidationError

from models.data_structures import SimulationConfig
from models.simulation_engine import BatteryDigitalTwin
from server import CurrentProfilePoint, SimulationRequest
from utils.data_parser import parse_profile_csv
from utils.fmu_interface import FMUClient


class _FakeFMU:
    Ns = 2
    Np = 1

    def __init__(self):
        self.call = None

    def run_simulation(self, stop_time, inputs, parameters, current_profile):
        self.call = {
            'stop_time': stop_time,
            'inputs': inputs,
            'parameters': parameters,
            'current_profile': current_profile,
        }
        frame = pd.DataFrame({
            'time': [0.0, 3600.0],
            'pack.V_pack': [6.2, 6.0],
            'pack.I_pack': [10.0, 10.0],
            'pack.T_max': [300.15, 310.15],
            'pack.T_min': [299.15, 305.15],
            'pack.SOC_min': [0.8, 0.6],
            'pack.SOH_min': [1.0, 0.99],
        })
        temp = [[[27.0], [26.0]], [[37.0], [32.0]]]
        soc = [[[0.8], [0.8]], [[0.6], [0.6]]]
        soh = [[[1.0], [1.0]], [[0.99], [0.99]]]
        return frame, temp, soc, soh


class TestSimulationContract(unittest.TestCase):
    def test_csv_profile_is_normalized_and_serialized(self):
        source = io.BytesIO(b"Time,Current\n10,0\n15,80\n20,-30\n")
        profile, message = parse_profile_csv(source)

        self.assertIsNotNone(profile, message)
        self.assertEqual(profile['duration'], 10.0)
        self.assertEqual(profile['points'][0], {'time_s': 0.0, 'current_a': 0.0})
        self.assertEqual(profile['points'][-1], {'time_s': 10.0, 'current_a': -30.0})

    def test_request_rejects_invalid_profile_and_fault_index(self):
        with self.assertRaises(ValidationError):
            SimulationRequest(
                duration_s=10,
                current_profile=[
                    CurrentProfilePoint(time_s=5, current_a=1),
                    CurrentProfilePoint(time_s=4, current_a=2),
                ],
            )
        with self.assertRaises(ValidationError):
            SimulationRequest(series_num=8, parallel_num=2, fault_s_index=9)

    def test_fmpy_profile_table_contract(self):
        table = FMUClient._build_current_profile([(2, 10), (5, -5)], stop_time=8)

        self.assertEqual(table['time'].tolist(), [0.0, 2.0, 5.0, 8.0])
        self.assertEqual(table['I_load_external'].tolist(), [10.0, 10.0, -5.0, -5.0])

    def test_engine_uses_fmu_soh_and_true_temperature_spread(self):
        engine = BatteryDigitalTwin.__new__(BatteryDigitalTwin)
        engine.fmu_engine = _FakeFMU()
        engine.ai_compensator = None
        config = SimulationConfig(
            sim_duration_s=3600,
            pack_current_a=10,
            current_profile=[(0, 10), (3600, 10)],
            env_temp_c=25,
            init_soc=80,
            init_soh=95,
            cell_capacity_ah=50,
            series_num=2,
            parallel_num=1,
            fault_mode=3,
            fault_s_index=2,
            fault_p_index=1,
            fault_severity=0.7,
        )

        _, kpis, _ = engine.run_profile(config)

        self.assertAlmostEqual(kpis.final_soh, 94.0)
        self.assertAlmostEqual(kpis.avg_delta_t, 3.0)
        self.assertAlmostEqual(kpis.max_delta_t, 5.0)
        self.assertAlmostEqual(kpis.ah_throughput, 10.0)
        self.assertAlmostEqual(kpis.equivalent_full_cycles, 0.1)
        params = engine.fmu_engine.call['parameters']
        self.assertEqual(params['pack.faultMode'], 3)
        self.assertAlmostEqual(params['pack.cellData.Q_nominal'], 180000.0)


if __name__ == '__main__':
    unittest.main()
