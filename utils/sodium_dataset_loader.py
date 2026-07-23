"""公开钠离子电池数据集适配器。

不同来源的数据统一转换为 ``CellDegradationData``，供现有 SOH 数据管线使用。
默认入口只会返回明确标记为钠离子电池的数据。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from utils.soh_data_loader import CellDegradationData, CycleData, WenzhouDataLoader


logger = logging.getLogger(__name__)


def _valid_curve_columns(columns: Iterable[object]) -> List[float]:
    """返回工作簿中用作固定电压采样点的数值列。"""
    values = []
    for value in columns:
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            values.append(float(value))
    return sorted(values)


def _curve_capacity(row: pd.Series, voltage_columns: List[float]) -> float:
    values = pd.to_numeric(row[voltage_columns], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    return float(values.max()) if values.size else 0.0


def _mean_voltage(row: pd.Series, voltage_columns: List[float]) -> float:
    """按容量增量积分计算循环平均电压。"""
    capacities = pd.to_numeric(row[voltage_columns], errors="coerce").to_numpy(dtype=float)
    voltages = np.asarray(voltage_columns, dtype=float)
    valid = np.isfinite(capacities) & (capacities >= 0)
    if valid.sum() < 2:
        return 0.0

    capacities = capacities[valid]
    voltages = voltages[valid]
    order = np.argsort(capacities)
    capacities = capacities[order]
    voltages = voltages[order]
    capacities, unique_idx = np.unique(capacities, return_index=True)
    voltages = voltages[unique_idx]
    if capacities.size < 2 or capacities[-1] <= capacities[0]:
        return 0.0
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(voltages, capacities) / (capacities[-1] - capacities[0]))


class MendeleyNFMLoader:
    """读取 Mendeley NFM 4.5 Ah 钠电循环数据。

    ``Dataset 2.xlsx`` 是锂电对照，故意不在文件清单中，纯钠电流程不会加载它。
    """

    FILES = {
        "Dataset 1-1.xlsx": {"cell_id": "NFM-25C", "temperature_c": 25.0},
        "Dataset 1-2.xlsx": {"cell_id": "NFM-M15C", "temperature_c": -15.0},
    }
    DATASET_ID = "mendeley-nfm-2026"

    @classmethod
    def load(cls, base_dir: str) -> List[CellDegradationData]:
        base = Path(base_dir)
        cells = []
        for filename, metadata in cls.FILES.items():
            path = base / filename
            if not path.exists():
                logger.warning("NFM 数据缺失: %s", path)
                continue
            cells.append(cls._load_workbook(path, **metadata))
        return cells

    @classmethod
    def _load_workbook(
        cls, path: Path, cell_id: str, temperature_c: float
    ) -> CellDegradationData:
        with pd.ExcelFile(path) as workbook:
            sheet_names = list(workbook.sheet_names)
        discharge_sheet = next(
            (name for name in sheet_names if "discharge" in name.lower()), None
        )
        charge_sheet = next(
            (
                name for name in sheet_names
                if "charge" in name.lower() and "discharge" not in name.lower()
            ),
            None,
        )
        if discharge_sheet is None:
            raise ValueError(f"NFM 工作簿缺少放电表: {path}")

        discharge_df = pd.read_excel(path, sheet_name=discharge_sheet)
        charge_df = pd.read_excel(path, sheet_name=charge_sheet) if charge_sheet else None
        discharge_voltage = _valid_curve_columns(discharge_df.columns)
        charge_voltage = _valid_curve_columns(charge_df.columns) if charge_df is not None else []
        if len(discharge_voltage) < 5:
            raise ValueError(f"NFM 工作簿没有足够的固定电压采样列: {path}")

        cycles: List[CycleData] = []
        for idx, discharge_row in discharge_df.iterrows():
            charge_row = charge_df.iloc[idx] if charge_df is not None and idx < len(charge_df) else None
            discharge_capacity = _curve_capacity(discharge_row, discharge_voltage)
            charge_capacity = _curve_capacity(charge_row, charge_voltage) if charge_row is not None else 0.0
            if discharge_capacity <= 0:
                continue

            provided_soh = discharge_row.get("SOH", np.nan)
            if not np.isfinite(provided_soh) and charge_row is not None:
                provided_soh = charge_row.get("SOH", np.nan)

            discharge_curve = pd.DataFrame({
                "V": discharge_voltage,
                "Q": pd.to_numeric(
                    discharge_row[discharge_voltage], errors="coerce"
                ).to_numpy(dtype=float),
            })
            charge_curve = None
            if charge_row is not None:
                charge_curve = pd.DataFrame({
                    "V": charge_voltage,
                    "Q": pd.to_numeric(
                        charge_row[charge_voltage], errors="coerce"
                    ).to_numpy(dtype=float),
                })

            cycles.append(CycleData(
                cycle_index=idx + 1,
                charge_capacity_ah=charge_capacity,
                discharge_capacity_ah=discharge_capacity,
                coulombic_efficiency=(
                    discharge_capacity / charge_capacity if charge_capacity > 0 else 0.0
                ),
                mean_charge_voltage_v=(
                    _mean_voltage(charge_row, charge_voltage) if charge_row is not None else 0.0
                ),
                mean_discharge_voltage_v=_mean_voltage(discharge_row, discharge_voltage),
                max_charge_voltage_v=max(charge_voltage) if charge_voltage else 0.0,
                min_discharge_voltage_v=min(discharge_voltage),
                temperature_c=temperature_c,
                temp_max_c=temperature_c,
                temp_min_c=temperature_c,
                c_rate_charge=1.0,
                c_rate_discharge=1.0,
                charge_curve=charge_curve,
                discharge_curve=discharge_curve,
                metadata={
                    "dataset_id": cls.DATASET_ID,
                    "source_file": str(path),
                    "provided_soh": float(provided_soh) if np.isfinite(provided_soh) else None,
                    "soc_min": 0.0,
                    "soc_max": 1.0,
                    "cathode_type": "NFM",
                    "voltage_min": 2.0,
                    "voltage_max": 4.0,
                },
            ))

        return CellDegradationData(
            cell_id=cell_id,
            chemistry="sodium-ion",
            nominal_capacity_ah=4.5,
            cycles=cycles,
            metadata={
                "dataset_id": cls.DATASET_ID,
                "source_file": str(path),
                "temperature_c": temperature_c,
                "nominal_capacity_ah": 4.5,
                "voltage_window_v": [2.0, 4.0],
                "test_rate_c": 1.0,
            },
        )


class RWTHCommercialAgingLoader:
    """RWTH 67 只商业钠电数据的受控入口。

    官方整包尚未落地，当前无法对内部通道字段和循环聚合规则做实测验证。
    因而这里采用失败关闭策略：目录为空时跳过；发现疑似整包数据时明确报错，
    不用猜测字段生成可能错误的 SOH 标签。
    """

    DATASET_ID = "rwth-commercial-sib-aging-67"
    DATA_SUFFIXES = {".csv", ".txt", ".parquet", ".pkl", ".pickle", ".mat", ".h5"}

    @classmethod
    def load(cls, base_dir: str) -> List[CellDegradationData]:
        base = Path(base_dir)
        if not base.exists():
            return []
        candidates = [
            path for path in base.rglob("*")
            if path.is_file()
            and path.suffix.lower() in cls.DATA_SUFFIXES
            and not re.search(r"read|helper|example", path.name, re.IGNORECASE)
        ]
        if not candidates:
            logger.info("RWTH 整包尚未解压到 %s，本次跳过", base)
            return []
        raise RuntimeError(
            "检测到 RWTH 原始数据，但该数据源的字段映射尚未通过真实整包验证。"
            "请先按 docs/sodium_server_training.md 完成结构核验，再启用 RWTH 训练。"
        )


class SodiumDatasetLoader:
    """统一加载当前可用的纯钠电数据源。"""

    SUPPORTED_DATASETS = {"wenzhou", "mendeley-nfm", "rwth"}

    def __init__(self, wenzhou_dir: Optional[str] = None, public_dir: Optional[str] = None):
        if wenzhou_dir is None:
            from models.soh_ai.config import RAW_DATA_DIR
            wenzhou_dir = RAW_DATA_DIR
        if public_dir is None:
            from models.soh_ai.config import SODIUM_RAW_DATA_DIR
            public_dir = SODIUM_RAW_DATA_DIR
        self.wenzhou_dir = Path(wenzhou_dir)
        self.public_dir = Path(public_dir)

    def load_all(self, include: Optional[Iterable[str]] = None) -> List[CellDegradationData]:
        selected = set(include or self.SUPPORTED_DATASETS)
        unknown = selected - self.SUPPORTED_DATASETS
        if unknown:
            raise ValueError(f"未知钠电数据源: {sorted(unknown)}")

        cells: List[CellDegradationData] = []
        if "wenzhou" in selected:
            cells.extend(WenzhouDataLoader(str(self.wenzhou_dir)).load_dataset("sodium-ion"))
        if "mendeley-nfm" in selected:
            cells.extend(MendeleyNFMLoader.load(str(self.public_dir / "mendeley_nfm")))
        if "rwth" in selected:
            cells.extend(RWTHCommercialAgingLoader.load(str(self.public_dir / "rwth_67_cells")))

        foreign = [cell.cell_id for cell in cells if cell.chemistry != "sodium-ion"]
        if foreign:
            raise ValueError(f"钠电管线混入了非钠电电芯: {foreign}")
        logger.info("钠电数据加载完成: %d 个数据实体", len(cells))
        return cells
