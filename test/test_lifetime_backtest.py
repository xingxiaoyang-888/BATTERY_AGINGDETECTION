"""寿命按电芯留一回测协议测试。"""

import unittest

import pandas as pd

from utils.evaluate_lifetime_backtest import evaluate_leave_one_cell_out


class TestLifetimeBacktest(unittest.TestCase):
    def test_held_out_cell_is_excluded_and_censoring_is_preserved(self):
        rows = []
        for cell_id, loss in [('a', 0.010), ('b', 0.015), ('c', 0.020), ('d', 0.008)]:
            for cycle in range(1, 21):
                rows.append({
                    'cell_id': cell_id,
                    'condition': cell_id,
                    'dataset_id': 'fixture',
                    'cycle_index': cycle,
                    'soh': 1.0 - loss * cycle,
                    'temperature_c': 25.0,
                    'c_rate_charge': 1.0,
                    'c_rate_discharge': 1.0,
                    'rest_time_h': 0.0,
                    'soc_min': 0.0,
                    'soc_max': 1.0,
                    'nominal_capacity_ah': 1.2,
                })
        result = evaluate_leave_one_cell_out(
            pd.DataFrame(rows),
            thresholds=(0.80,),
            observation_fractions=(0.25,),
            use_ai=False,
            progress_every=0,
        )

        self.assertEqual(result['protocol']['cells'], 4)
        self.assertEqual(len(result['predictions']), 4)
        censored = next(row for row in result['predictions'] if row['cell_id'] == 'd')
        self.assertFalse(censored['event_observed'])
        self.assertIsNone(censored['actual_rul'])
        self.assertGreater(censored['rul_lower_bound'], 0)
        summary = result['summary']['per_threshold']['0.8']
        self.assertEqual(summary['right_censored_samples'], 1)


if __name__ == '__main__':
    unittest.main()
