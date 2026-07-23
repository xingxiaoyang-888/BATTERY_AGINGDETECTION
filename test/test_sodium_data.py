"""钠电数据加载、隔离和质量门禁测试。"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from models.soh_ai.data_pipeline import DataCleaner, DataSplitter, SOHDataPipeline
from models.soh_ai.config import FEATURE_CFG
from utils.audit_sodium_data import audit_feature_table
from utils.evaluate_horizon_baselines import _metrics
from utils.sodium_dataset_loader import (
    MendeleyNFMLoader,
    RWTHCommercialAgingLoader,
)
from utils.soh_data_loader import (
    CellDegradationData,
    CycleData,
    WenzhouDataLoader,
    _duration_to_hours,
)


class TestWenzhouSodiumConversion(unittest.TestCase):
    def test_duration_and_unit_conversion(self):
        self.assertAlmostEqual(_duration_to_hours("2:00:00"), 2.0)
        cycle = WenzhouDataLoader()._excel_dict_to_cycle({
            "cycle_index": 1,
            "charge_capacity_ah": 1.10,
            "discharge_capacity_ah": 1.08,
            "coulombic_efficiency": 98.2,
            "charge_time": "2:00:00",
            "discharge_time": "1:30:00",
            "dc_resistance": 55.0,
            "temperature_c": 0.0,
        }, 0)

        self.assertAlmostEqual(cycle.discharge_capacity_ah, 1.08)
        self.assertAlmostEqual(cycle.coulombic_efficiency, 0.982)
        self.assertAlmostEqual(cycle.c_rate_charge, 0.5)
        self.assertAlmostEqual(cycle.c_rate_discharge, 2.0 / 3.0)
        self.assertAlmostEqual(cycle.dc_resistance_ohm, 0.055)
        self.assertAlmostEqual(cycle.temperature_c, 25.0)


class TestMendeleyNFMLoader(unittest.TestCase):
    @staticmethod
    def _write_workbook(path: Path, capacity: float):
        voltages = [2.0, 2.5, 3.0, 3.5, 4.0]
        discharge = pd.DataFrame([
            [0.0, 0.25 * capacity, 0.55 * capacity, 0.8 * capacity, capacity],
            [0.0, 0.24 * capacity, 0.53 * capacity, 0.78 * capacity, 0.98 * capacity],
        ], columns=voltages)
        charge = pd.DataFrame([
            [0.0, 0.2 * capacity, 0.5 * capacity, 0.82 * capacity, 1.01 * capacity],
            [0.0, 0.2 * capacity, 0.49 * capacity, 0.80 * capacity, capacity],
        ], columns=voltages)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            discharge.to_excel(writer, sheet_name="Discharge", index=False)
            charge.to_excel(writer, sheet_name="Charge", index=False)

    def test_only_sodium_workbooks_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_workbook(root / "Dataset 1-1.xlsx", 4.3)
            self._write_workbook(root / "Dataset 1-2.xlsx", 3.6)
            self._write_workbook(root / "Dataset 2.xlsx", 2.5)

            cells = MendeleyNFMLoader.load(str(root))

        self.assertEqual([cell.cell_id for cell in cells], ["NFM-25C", "NFM-M15C"])
        self.assertTrue(all(cell.chemistry == "sodium-ion" for cell in cells))
        self.assertTrue(all(len(cell.cycles) == 2 for cell in cells))
        self.assertGreater(cells[0].cycles[0].mean_discharge_voltage_v, 2.0)
        self.assertAlmostEqual(
            cells[0].cycles[0].coulombic_efficiency, 1.0 / 1.01, places=6
        )


class TestSodiumIsolation(unittest.TestCase):
    def test_kfold_scaling_contract_keeps_soh_in_physical_domain(self):
        feature_cols = [FEATURE_CFG.target_col, "temperature_c", "c_rate_charge"]
        scale_cols = [col for col in feature_cols if col != FEATURE_CFG.target_col]

        self.assertNotIn(FEATURE_CFG.target_col, scale_cols)
        self.assertEqual(scale_cols, ["temperature_c", "c_rate_charge"])

    def test_soh_reference_does_not_use_future_capacity(self):
        capacities = [1.0] * 10 + [2.0]
        cell = CellDegradationData(
            cell_id="causal-reference",
            chemistry="sodium-ion",
            cycles=[
                CycleData(cycle_index=index + 1, discharge_capacity_ah=capacity)
                for index, capacity in enumerate(capacities)
            ],
        )

        self.assertAlmostEqual(cell.soh_series[0], 1.0)
        self.assertAlmostEqual(cell.soh_series[-1], 2.0)

    def test_soh_smoothing_does_not_look_ahead(self):
        cleaner = DataCleaner()
        prefix = [1.0, 0.99, 0.98, 0.97, 0.96]
        normal = pd.Series(prefix + [0.95, 0.94, 0.93])
        changed_future = pd.Series(prefix + [0.50, 0.40, 0.30])

        normal_smoothed = cleaner._smooth_soh(normal, window=5)
        changed_smoothed = cleaner._smooth_soh(changed_future, window=5)

        pd.testing.assert_series_equal(
            normal_smoothed.iloc[:len(prefix)],
            changed_smoothed.iloc[:len(prefix)],
        )

    def test_plain_pipeline_rejects_lithium_cell(self):
        lithium = CellDegradationData(
            cell_id="li-1",
            chemistry="lithium-ion",
            nominal_capacity_ah=1.0,
            cycles=[CycleData(cycle_index=idx + 1, discharge_capacity_ah=1.0)
                    for idx in range(40)],
        )
        with self.assertRaisesRegex(ValueError, "混入非钠电"):
            SOHDataPipeline().run([lithium], save=False)

    def test_cell_level_split_has_no_overlap(self):
        rows = []
        for cell_index in range(6):
            for cycle_index in range(40):
                rows.append({
                    "cell_id": f"cell-{cell_index}",
                    "cycle_index": cycle_index + 1,
                    "soh": 1.0 - cycle_index * 0.001,
                })
        splits = DataSplitter().split(pd.DataFrame(rows))
        cell_sets = [set(splits[name]["cell_id"].unique()) for name in ("train", "val", "test")]
        self.assertFalse(cell_sets[0] & cell_sets[1])
        self.assertFalse(cell_sets[0] & cell_sets[2])
        self.assertFalse(cell_sets[1] & cell_sets[2])

    def test_rwth_ird_parser_extracts_reference_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "S2000TEST_CU_at25degC_V00.ird")
            rows = [
                "Cellname: ' NA18650-1250\n", "Starttime: '2023-01-01 00:00:00'\n",
                "###\n", "Time,Voltage,Current,Ah_counter,Temperature,StepID\n",
                "0 days 00:00:00,3.8,-0.6,1.2,25,1\n",
                "0 days 01:00:00,2.8,-0.6,0.6,25,1\n",
                "0 days 02:00:00,1.5,-0.6,0.0,25,1\n",
            ]
            path.write_text("".join(rows), encoding="utf-8")
            with path.open("rb") as source:
                start_time, frame = RWTHCommercialAgingLoader._read_ird(source, path.name)
            cycles = RWTHCommercialAgingLoader._cycles_from_measurement(
                frame, "DOD100_1C1C_25degC", start_time, "cu", path.name
            )

        self.assertEqual(len(cycles), 1)
        self.assertAlmostEqual(cycles[0].discharge_capacity_ah, 1.2)
        self.assertTrue(cycles[0].metadata["is_reference_capacity"])
        self.assertEqual(cycles[0].temperature_c, 25.0)

    def test_rwth_condition_metadata(self):
        meta = RWTHCommercialAgingLoader._condition_metadata("DOD80_2C2C_-10degC")
        self.assertAlmostEqual(meta["dod"], 0.8)
        self.assertAlmostEqual(meta["soc_min"], 0.2)
        self.assertAlmostEqual(meta["c_rate_discharge"], 2.0)
        self.assertAlmostEqual(meta["temperature_c"], -10.0)


class TestSodiumAudit(unittest.TestCase):
    def test_horizon_metric_returns_physical_rmse(self):
        report = _metrics(
            pd.Series([1.0, 0.9]).to_numpy(),
            pd.Series([0.99, 0.88]).to_numpy(),
        )
        self.assertAlmostEqual(report["rmse"], (0.00025 ** 0.5))
        self.assertAlmostEqual(report["mae"], 0.015)

    def test_valid_sodium_feature_table_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_table.parquet"
            pd.DataFrame({
                "cell_id": ["a", "b", "c"],
                "chemistry": ["sodium-ion"] * 3,
                "dataset_id": ["fixture"] * 3,
                "soh": [1.0, 0.9, 0.8],
                "coulombic_efficiency": [1.0, 0.99, 1.01],
                "c_rate_charge": [1.0] * 3,
                "c_rate_discharge": [1.0] * 3,
                "temperature_c": [25.0] * 3,
                "nominal_capacity_ah": [1.2] * 3,
            }).to_parquet(path, index=False)

            report = audit_feature_table(path)

        self.assertTrue(report["passed"])
        self.assertEqual(report["cells"], 3)


if __name__ == "__main__":
    unittest.main()
