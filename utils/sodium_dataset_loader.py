"""公开钠离子电池数据集适配器。

不同来源的数据统一转换为 ``CellDegradationData``，供现有 SOH 数据管线使用。
默认入口只会返回明确标记为钠离子电池的数据。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple

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
    """流式读取 RWTH 67 只商业钠电循环老化数据。"""

    DATASET_ID = "rwth-commercial-sib-aging-67"
    NOMINAL_CAPACITY_AH = 1.2
    CONDITION_PATTERN = re.compile(
        r"^DOD(?P<dod>\d+)_(?P<charge>[\d.]+)C(?P<discharge>[\d.]+)C_"
        r"(?P<temperature>-?\d+|RT)deg?C?$|"
        r"^DOD(?P<dod_rt>\d+)_(?P<charge_rt>[\d.]+)C(?P<discharge_rt>[\d.]+)C_RT$"
    )

    @classmethod
    def _condition_metadata(cls, condition: str) -> Dict[str, float]:
        match = cls.CONDITION_PATTERN.match(condition)
        if not match:
            raise ValueError(f"无法识别 RWTH 循环工况: {condition}")
        groups = match.groupdict()
        dod = float(groups.get("dod") or groups.get("dod_rt")) / 100.0
        charge = float(groups.get("charge") or groups.get("charge_rt"))
        discharge = float(groups.get("discharge") or groups.get("discharge_rt"))
        temperature_text = groups.get("temperature")
        temperature = 25.0 if temperature_text in {None, "RT"} else float(temperature_text)
        return {
            "dod": dod,
            "soc_min": 1.0 - dod,
            "soc_max": 1.0,
            "temperature_c": temperature,
            "c_rate_charge": charge,
            "c_rate_discharge": discharge,
        }

    @staticmethod
    def _open_text(source: BinaryIO) -> TextIOWrapper:
        return TextIOWrapper(source, encoding="utf-8", errors="replace", newline="")

    @classmethod
    def _read_ird(cls, source: BinaryIO, source_name: str) -> Tuple[pd.Timestamp, pd.DataFrame]:
        text = cls._open_text(source)
        start_time = None
        for line in text:
            if line.startswith("Starttime:"):
                start_time = pd.to_datetime(line.split("'", 2)[1])
            if line.startswith("###"):
                break
        if start_time is None:
            raise ValueError(f"RWTH 文件缺少 Starttime: {source_name}")
        frame = pd.read_csv(text)
        expected = {"Time", "Voltage", "Current", "Ah_counter", "Temperature", "StepID"}
        if frame.empty:
            return start_time, frame
        if set(frame.columns) != expected:
            raise ValueError(f"RWTH 通道不匹配 {source_name}: {list(frame.columns)}")
        for column in expected - {"Time"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["Time"] = pd.to_timedelta(frame["Time"], errors="coerce")
        frame = frame.dropna(subset=["Time", "Voltage", "Current", "Ah_counter"])
        return start_time, frame

    @classmethod
    def _capacity_segments(cls, frame: pd.DataFrame) -> List[Dict[str, float]]:
        """提取连续恒流段；电流负值为放电，正值为充电。"""
        if frame.empty:
            return []
        segments = []
        for step_id, step in frame.groupby("StepID", sort=False):
            current = step["Current"].median()
            if not np.isfinite(current) or abs(current) < 0.02:
                continue
            duration_h = (step["Time"].iloc[-1] - step["Time"].iloc[0]).total_seconds() / 3600.0
            capacity = abs(float(step["Ah_counter"].iloc[-1] - step["Ah_counter"].iloc[0]))
            if duration_h <= 0 or capacity < 0.05:
                continue
            voltage = step["Voltage"].to_numpy(dtype=float)
            amp_hours = step["Ah_counter"].to_numpy(dtype=float)
            delta_ah = np.abs(np.diff(amp_hours))
            mean_voltage = float(
                np.sum((voltage[:-1] + voltage[1:]) * 0.5 * delta_ah) / delta_ah.sum()
            ) if delta_ah.sum() > 0 else float(np.nanmean(voltage))
            segments.append({
                "step_id": int(step_id),
                "direction": "discharge" if current < 0 else "charge",
                "capacity_ah": capacity,
                "mean_voltage_v": mean_voltage,
                "mean_current_a": abs(float(current)),
                "temperature_c": float(step["Temperature"].median()),
                "duration_h": duration_h,
            })
        return segments

    @classmethod
    def _cycles_from_measurement(
        cls, frame: pd.DataFrame, condition: str, start_time: pd.Timestamp,
        file_kind: str, source_name: str,
    ) -> List[CycleData]:
        meta = cls._condition_metadata(condition)
        segments = cls._capacity_segments(frame)
        discharges = [item for item in segments if item["direction"] == "discharge"]
        charges = [item for item in segments if item["direction"] == "charge"]
        cycles = []
        for discharge in discharges:
            c_rate = discharge["mean_current_a"] / cls.NOMINAL_CAPACITY_AH
            is_reference = 0.35 <= c_rate <= 0.65
            is_aging_rate = abs(c_rate - meta["c_rate_discharge"]) <= max(0.15, 0.15 * meta["c_rate_discharge"])
            # DOD20 的单次容量约为 0.24 Ah，不能使用固定 0.3 Ah 门槛。
            expected_capacity = cls.NOMINAL_CAPACITY_AH * meta["dod"]
            if discharge["capacity_ah"] < max(0.05, 0.5 * expected_capacity):
                continue
            if file_kind == "cu" and not is_reference:
                continue
            if file_kind != "cu" and not (is_reference or is_aging_rate):
                continue

            nearest_charge = min(
                charges,
                key=lambda item: abs(item["step_id"] - discharge["step_id"]),
                default=None,
            )
            charge_capacity = nearest_charge["capacity_ah"] if nearest_charge else 0.0
            cycle_metadata = {
                "dataset_id": cls.DATASET_ID,
                "condition": condition,
                "source_file": source_name,
                "measurement_start": start_time.isoformat(),
                "is_reference_capacity": is_reference,
                "file_kind": file_kind,
                "soc_min": meta["soc_min"],
                "soc_max": meta["soc_max"],
                "dod": meta["dod"],
            }
            cycles.append(CycleData(
                cycle_index=0,
                timestamp=start_time.timestamp(),
                charge_capacity_ah=charge_capacity,
                discharge_capacity_ah=discharge["capacity_ah"],
                coulombic_efficiency=(
                    discharge["capacity_ah"] / charge_capacity if charge_capacity > 0 else 0.0
                ),
                mean_charge_voltage_v=(nearest_charge["mean_voltage_v"] if nearest_charge else 0.0),
                mean_discharge_voltage_v=discharge["mean_voltage_v"],
                temperature_c=discharge["temperature_c"],
                temp_max_c=discharge["temperature_c"],
                temp_min_c=discharge["temperature_c"],
                c_rate_charge=meta["c_rate_charge"],
                c_rate_discharge=meta["c_rate_discharge"],
                metadata=cycle_metadata,
            ))
        return cycles

    @classmethod
    def _load_zip(cls, archive_path: Path) -> List[CellDegradationData]:
        import zipfile

        by_cell: Dict[str, List[CycleData]] = defaultdict(list)
        condition_by_cell: Dict[str, str] = {}
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".ird"):
                    continue
                parts = info.filename.split("/")
                condition = parts[0]
                if not condition.startswith("DOD"):
                    continue
                filename = parts[-1]
                cell_id = filename.split("_", 1)[0].split(".", 1)[0]
                file_kind = "cu" if "/CU_at_25degC/" in info.filename else "cycling"
                try:
                    with archive.open(info) as source:
                        start_time, frame = cls._read_ird(source, info.filename)
                    cycles = cls._cycles_from_measurement(
                        frame, condition, start_time, file_kind, info.filename
                    )
                except (ValueError, pd.errors.ParserError) as exc:
                    logger.warning("跳过 RWTH 文件 %s: %s", info.filename, exc)
                    continue
                by_cell[cell_id].extend(cycles)
                condition_by_cell[cell_id] = condition

        cells = []
        for cell_id, cycles in sorted(by_cell.items()):
            cycles.sort(key=lambda item: (item.timestamp or 0, item.metadata.get("source_file", "")))
            for index, cycle in enumerate(cycles, start=1):
                cycle.cycle_index = index
            condition = condition_by_cell[cell_id]
            cells.append(CellDegradationData(
                cell_id=f"RWTH-{cell_id}",
                chemistry="sodium-ion",
                nominal_capacity_ah=cls.NOMINAL_CAPACITY_AH,
                cycles=cycles,
                metadata={
                    "dataset_id": cls.DATASET_ID,
                    "condition": condition,
                    **cls._condition_metadata(condition),
                },
            ))
        return cells

    @classmethod
    def load(cls, base_dir: str) -> List[CellDegradationData]:
        base = Path(base_dir)
        if not base.exists():
            return []
        archives = sorted(base.glob("*.zip"))
        if not archives:
            logger.info("RWTH 整包尚未解压到 %s，本次跳过", base)
            return []
        cells = cls._load_zip(archives[0])
        if len(cells) != 67:
            raise ValueError(f"RWTH 循环老化电芯应为 67 只，实际解析到 {len(cells)} 只")
        return cells


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
