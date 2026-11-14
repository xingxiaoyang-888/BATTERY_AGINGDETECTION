# utils/soh_data_loader.py
"""Multi-format Wenzhou battery aging data adapters."""

import os
import re
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _duration_to_hours(value: Any) -> float:
    """Normalize Excel durations, timedeltas, or h:mm:ss text to hours."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 3600.0
    if hasattr(value, 'total_seconds'):
        return float(value.total_seconds()) / 3600.0
    if isinstance(value, (int, float, np.number)):

        numeric = float(value)
        return numeric * 24.0 if 0 <= numeric <= 2 else numeric / 3600.0
    try:
        return pd.to_timedelta(str(value)).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return 0.0

# ============================================================

# ============================================================

@dataclass
class CycleData:
    """Measurements and metadata for one charge/discharge cycle."""
    cycle_index: int
    timestamp: Optional[float] = None

    charge_capacity_ah: float = 0.0
    discharge_capacity_ah: float = 0.0

    coulombic_efficiency: float = 0.0

    charge_energy_wh: float = 0.0
    discharge_energy_wh: float = 0.0

    mean_charge_voltage_v: float = 0.0
    mean_discharge_voltage_v: float = 0.0
    max_charge_voltage_v: float = 0.0
    min_discharge_voltage_v: float = 0.0

    temperature_c: float = 25.0
    temp_max_c: float = 25.0
    temp_min_c: float = 25.0

    c_rate_charge: float = 0.0
    c_rate_discharge: float = 0.0
    rest_time_h: float = 0.0

    dc_resistance_ohm: Optional[float] = None

    charge_curve: Optional[pd.DataFrame] = None   # columns: [V, Q, Time]
    discharge_curve: Optional[pd.DataFrame] = None

    metadata: Dict[str, Any] = None


@dataclass
class CellDegradationData:
    """All available aging observations for one cell."""
    cell_id: str
    chemistry: str = "unknown"
    nominal_capacity_ah: float = None
    cycles: List[CycleData] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.cycles is None:
            self.cycles = []
        if self.metadata is None:
            self.metadata = {}

    def __len__(self):
        return len(self.cycles)

    @property
    def soh_series(self) -> np.ndarray:
        """Compute SOH relative to initial observed capacity without future-capacity normalization."""
        if not self.cycles:
            return np.array([])
        valid_caps = [c.discharge_capacity_ah for c in self.cycles
                      if c.discharge_capacity_ah > 0]
        initial_caps = valid_caps[:10]
        ref_cap = max(initial_caps) if initial_caps else 1.0
        return np.array([c.discharge_capacity_ah / ref_cap for c in self.cycles])


# ============================================================

# ============================================================

class MATFileLoader:
    """Load MATLAB v5 and HDF5-based v7.3 data files."""

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """Load."""
        import scipy.io as sio


        try:
            data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)

            clean_data = {k: v for k, v in data.items() if not k.startswith('__')}
            logger.info(f"  [MAT v5] 加载成功: {filepath} → {list(clean_data.keys())}")
            return clean_data
        except NotImplementedError:

            logger.info(f"  检测到 MATLAB v7.3 格式，切换到 h5py 读取...")
            return MATFileLoader._load_v73(filepath)
        except Exception as e:
            logger.error(f"  MAT 文件加载失败 [{filepath}]: {e}")
            raise

    @staticmethod
    def _load_v73(filepath: str) -> Dict[str, Any]:
        """Load v73."""
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = MATFileLoader._h5_to_dict(f[key])
        logger.info(f"  [MAT v7.3] 加载成功: {filepath} → {list(data.keys())}")
        return data

    @staticmethod
    def _h5_to_dict(h5obj) -> Any:
        """Recursively convert HDF5 values to native Python containers."""
        import h5py
        if isinstance(h5obj, h5py.Dataset):
            val = h5obj[()]

            if isinstance(val, np.ndarray) and val.dtype.kind == 'U':
                return ''.join(val.flatten())
            return val
        elif isinstance(h5obj, h5py.Group):
            result = {}
            for k, v in h5obj.items():
                result[k] = MATFileLoader._h5_to_dict(v)
            return result
        return h5obj

    @staticmethod
    def extract_cycle_info(mat_data: Dict[str, Any], key: str = None) -> List[Dict]:
        """Extract cycle info."""

        if key is None:
            candidates = ['Cycle_information', 'cycle_information',
                          'CycleInformation', 'cycleInfo', 'cycle_data']
            for c in candidates:
                if c in mat_data:
                    key = c
                    break

        if key is None or key not in mat_data:
            logger.warning(f"  未找到循环信息键，可用键: {list(mat_data.keys())}")
            return []

        raw = mat_data[key]


        if hasattr(raw, 'dtype') and hasattr(raw, 'flat'):
            # MATLAB struct array → Python list of dicts
            cycles = []
            field_names = raw.dtype.names
            for i in range(len(raw)):
                cycle_dict = {}
                for fn in field_names:
                    val = raw[i][fn]

                    if isinstance(val, np.ndarray):
                        val = val.item() if val.size == 1 else val.flatten()

                    if isinstance(val, np.ndarray) and val.dtype.kind == 'U':
                        val = str(val)
                    cycle_dict[fn] = val
                cycles.append(cycle_dict)
            logger.info(f"  提取到 {len(cycles)} 个循环记录")
            return cycles


        if isinstance(raw, dict) and not isinstance(raw, np.ndarray):

            return MATFileLoader._extract_h5_struct_cycles(raw)

        logger.warning(f"  无法识别的 cycle_info 数据格式: {type(raw)}")
        return []

    @staticmethod
    def _extract_h5_struct_cycles(h5_struct: Dict) -> List[Dict]:
        """Extract h5 struct cycles."""

        fields = list(h5_struct.keys())
        n_cycles = 0
        for f in fields:
            arr = h5_struct[f]
            if isinstance(arr, np.ndarray):
                n_cycles = max(n_cycles, len(arr))
        if n_cycles == 0:
            return []

        cycles = []
        for i in range(n_cycles):
            cycle_dict = {}
            for f in fields:
                arr = h5_struct[f]
                if isinstance(arr, np.ndarray) and i < len(arr):
                    val = arr[i]
                    if isinstance(val, np.ndarray):
                        val = val.item() if val.size == 1 else val.flatten()
                    cycle_dict[f] = val
            cycles.append(cycle_dict)
        return cycles


# ============================================================

# ============================================================

class XLSFileLoader:
    """Load cycle summaries from sodium-ion Excel workbooks."""

    @staticmethod
    def load_sheets(filepath: str) -> Dict[str, pd.DataFrame]:
        """Load sheets."""
        xls = pd.ExcelFile(filepath)
        sheets = {}
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if not df.empty:
                    sheets[sheet_name] = df
            except Exception as e:
                logger.warning(f"  跳过损坏的 sheet '{sheet_name}': {e}")
        logger.info(f"  [XLS] 加载成功: {filepath} → {len(sheets)} 个 sheet")
        return sheets

    @staticmethod
    def load_single(filepath: str, sheet_name: str = 0) -> pd.DataFrame:
        """Load single."""
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        logger.info(f"  [XLS] 加载: {filepath} [{sheet_name}] → {len(df)} 行 × {len(df.columns)} 列")
        return df

    @staticmethod
    def detect_cycle_structure(df: pd.DataFrame) -> Dict[str, str]:
        """Detect cycle structure."""

        col_map = {c: c for c in df.columns}
        clean_cols = {c: c.lower().replace(' ', '_').replace('-', '_') for c in df.columns}


        match_rules = {
            'cycle_index':        [r'cycle', r'循环', r'index', r'序号'],
            'charge_capacity_ah': [r'charge.*cap', r'充电.*容', r'cha.*cap', r'cap.*cha'],
            'discharge_capacity_ah': [r'discharge.*cap', r'放电.*容', r'dis.*cap', r'cap.*dis'],
            'coulombic_efficiency': [r'coulomb', r'库仑', r'efficiency', r'效率', r'ce'],
            'charge_energy_wh':     [r'charge.*energy', r'充电能量'],
            'discharge_energy_wh':  [r'discharge.*energy', r'放电能量'],
            'temperature_c':      [r'temp', r'温度', r'temperature'],
            'voltage_mean':       [r'voltage.*mean', r'平均.*压', r'v_mean'],
            'current_a':          [r'current', r'电流', r'i_'],
            'charge_time':        [r'charge.*time', r'充电时间'],
            'discharge_time':     [r'discharge.*time', r'放电时间'],
            'dc_resistance':      [r'discharge.*dc.*resistance', r'放电直流内阻',
                                   r'dc.*resistance', r'内阻', r'dcr'],
            'mean_voltage_v':     [r'median.*voltage', r'中值电压', r'voltage.*mean'],
        }

        mapping = {}
        for std_name, patterns in match_rules.items():
            for orig_col, clean_col in clean_cols.items():
                if any(re.search(p, clean_col) for p in patterns):
                    if std_name not in mapping:
                        mapping[std_name] = orig_col
                    break


        for orig_col, clean_col in clean_cols.items():
            if re.search(r'discharge.*dc.*resistance|放电直流内阻', clean_col):
                mapping['dc_resistance'] = orig_col
                break

        logger.info(f"  列结构检测: {len(mapping)}/{len(match_rules)} 个字段已识别")
        return mapping

    @staticmethod
    def parse_cycle_summary(df: pd.DataFrame,
                            col_mapping: Dict[str, str] = None) -> List[dict]:
        """Parse cycle summary."""
        if col_mapping is None:
            col_mapping = XLSFileLoader.detect_cycle_structure(df)

        cycles = []
        for i, row in df.iterrows():
            cycle = {}
            for std_name, orig_col in col_mapping.items():
                if orig_col in df.columns:
                    val = row[orig_col]
                    if isinstance(val, (np.floating, float)) and np.isnan(val):
                        continue
                    cycle[std_name] = val
            if cycle:
                cycles.append(cycle)

        return cycles


# ============================================================

# ============================================================

class NDAXFileLoader:
    """Legacy adapter for raw NDAX electrochemical files."""

    @staticmethod
    def parse(filepath: str) -> Optional[Dict[str, Any]]:
        """Parse."""

        try:
            with open(filepath, 'rb') as f:
                header = f.read(256)
        except Exception as e:
            logger.error(f"  无法读取 .ndax 文件 [{filepath}]: {e}")
            return None


        if header[:4] != b'G4\2\0' and header[:4] != b'G4\3\0':
            logger.warning(f"  文件可能不是标准 Gamry .ndax 格式: {filepath}")


        logger.info(f"  [NDAX] 解析: {filepath} → 暂用二进制扫描模式")


        return {
            'filepath': filepath,
            'file_size': os.path.getsize(filepath),
            'header_version': header[:4].decode('latin-1', errors='ignore'),
            'raw_data_available': True,
        }


# ============================================================

# ============================================================

class WenzhouDataLoader:
    """Unified loading interface for the Wenzhou dataset families."""

    EXTENSION_HANDLERS = {
        '.mat':  MATFileLoader.load,
        '.xls':  XLSFileLoader.load_sheets,
        '.xlsx': XLSFileLoader.load_sheets,
        '.nda':  NDAXFileLoader.parse,
        '.ndax': NDAXFileLoader.parse,
    }

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            from models.soh_ai.config import RAW_DATA_DIR
            base_dir = RAW_DATA_DIR
        self.base_dir = base_dir

    def detect_format(self, filepath: str) -> str:
        """Detect format."""
        ext = Path(filepath).suffix.lower()
        return ext

    def load_cell(self, filepath: str,
                  cell_id: str = None,
                  chemistry: str = "unknown",
                  nominal_capacity_ah: float = None) -> CellDegradationData:
        """Load cell."""
        ext = self.detect_format(filepath)
        if ext not in self.EXTENSION_HANDLERS:
            raise ValueError(f"不支持的文件格式: {ext}")

        if cell_id is None:
            cell_id = Path(filepath).stem

        logger.info(f"  加载电芯 [{cell_id}] → {filepath}")


        raw_data = self.EXTENSION_HANDLERS[ext](filepath)


        cycles = self._convert_to_cycles(raw_data, ext, filepath)


        if nominal_capacity_ah is None and cycles:
            valid_caps = [c.discharge_capacity_ah for c in cycles
                          if c.discharge_capacity_ah > 0]
            if valid_caps:
                nominal_capacity_ah = max(valid_caps)

        return CellDegradationData(
            cell_id=cell_id,
            chemistry=chemistry,
            nominal_capacity_ah=nominal_capacity_ah,
            cycles=cycles,
            metadata={'source_file': filepath, 'format': ext}
        )

    def _convert_to_cycles(self, raw_data: Any, ext: str,
                           filepath: str) -> List[CycleData]:
        """Normalize parsed records into CycleData instances."""
        cycles = []

        if ext == '.mat':

            if 'Discharge_capacity' in raw_data:

                cycles = self._final_mat_to_cycles(raw_data, filepath)
            else:

                mat_cycles = MATFileLoader.extract_cycle_info(raw_data)
                for i, mc in enumerate(mat_cycles):
                    cd = self._mat_dict_to_cycle(mc, i)
                    if cd:
                        cycles.append(cd)

        elif ext in ('.xls', '.xlsx'):

            for sheet_name, df in raw_data.items():
                col_map = XLSFileLoader.detect_cycle_structure(df)
                summary = XLSFileLoader.parse_cycle_summary(df, col_map)
                for i, s in enumerate(summary):
                    cd = self._excel_dict_to_cycle(s, i + len(cycles))
                    if cd:
                        cycles.append(cd)

        elif ext in ('.nda', '.ndax'):

            logger.info(f"  NDAX 文件已索引，详细解析需 Gamry SDK: {filepath}")

        return cycles

    def _mat_dict_to_cycle(self, d: dict, idx: int) -> Optional[CycleData]:
        """Mat dict to cycle."""
        try:
            return CycleData(
                cycle_index=d.get('cycle_index', d.get('cycle', idx + 1)),
                charge_capacity_ah=float(d.get('capacity_charge',
                                        d.get('charge_capacity',
                                        d.get('Q_charge', 0)))),
                discharge_capacity_ah=float(d.get('capacity_discharge',
                                           d.get('discharge_capacity',
                                           d.get('Q_discharge', 0)))),
                coulombic_efficiency=float(d.get('coulombic_efficiency',
                                          d.get('CE',
                                          d.get('efficiency', 0)))),
                temperature_c=float(d.get('temperature',
                                   d.get('temp',
                                   d.get('T', 25.0)))),
                c_rate_charge=float(d.get('c_rate_charge',
                                   d.get('C_rate_charge', 0))),
                c_rate_discharge=float(d.get('c_rate_discharge',
                                      d.get('C_rate_discharge', 0))),
                dc_resistance_ohm=float(d.get('dc_resistance',
                                       d.get('DCR',
                                       d.get('resistance', np.nan)))),
                metadata={k: v for k, v in d.items()
                          if k not in ['cycle_index', 'cycle', 'capacity_charge',
                                       'charge_capacity', 'Q_charge',
                                       'capacity_discharge', 'discharge_capacity',
                                       'Q_discharge', 'coulombic_efficiency',
                                       'CE', 'efficiency', 'temperature', 'temp',
                                       'T', 'c_rate_charge', 'C_rate_charge',
                                       'c_rate_discharge', 'C_rate_discharge',
                                       'dc_resistance', 'DCR', 'resistance']}
            )
        except Exception as e:
            logger.warning(f"  转换 MAT 循环 {idx} 失败: {e}")
            return None

    def _load_paired_info(self, final_path: str) -> Optional[Dict[str, Any]]:
        """Load paired info."""
        final_file = Path(final_path)
        info_file = final_file.parent / final_file.name.replace(
            'Detail_Final.mat', 'Detail_Cycle_information.mat')
        if info_file.exists():
            try:
                return MATFileLoader.load(str(info_file))
            except Exception:
                logger.debug(f"  配对文件加载失败: {info_file.name}")
        return None

    def _infer_temperature_from_mode_current(self, info_data: Optional[Dict[str, Any]], fallback: float = 25.0) -> float:
        """Infer temperature from mode current."""
        if not info_data:
            return fallback
        for key in ('temperature', 'Temperature', 'temp', 'TEMP', 'ambient_temperature', 'ambient_temp'):
            val = info_data.get(key)
            if val is not None:
                try:
                    return float(np.asarray(val).squeeze())
                except Exception:
                    pass
        mode_current = info_data.get('MODE_current')
        if mode_current is None:
            return fallback
        try:
            code = int(np.asarray(mode_current).squeeze())
        except Exception:
            return fallback
        mapping = {
            -33311000: -20.0,
            -22211000: -10.0,
            -11111000: 0.0,
            -5561000: 25.0,
        }
        return mapping.get(code, fallback)

    def _final_mat_to_cycles(self, mat_data: Dict[str, Any],
                             filepath: str) -> List[CycleData]:
        """Parse the flat-array layout used by Wenzhou Final.mat files."""

        def _get_1d(arr, n=None):
            """Safely extract a one-dimensional array across squeeze settings."""
            a = np.atleast_1d(np.asarray(arr).squeeze())
            return a

        dis_cap_arr = _get_1d(mat_data['Discharge_capacity'])
        n_cycles = len(dis_cap_arr)


        info_data = self._load_paired_info(filepath)
        ambient_temp = self._infer_temperature_from_mode_current(info_data, fallback=25.0)


        chg_cap_arr = _get_1d(mat_data['Charge_capacity'])
        dis_volt_arr = _get_1d(mat_data.get('Discharge_voltage_mean', np.zeros(n_cycles)))
        chg_volt_arr = _get_1d(mat_data.get('Charge_voltage_mean', np.zeros(n_cycles)))
        dis_curr_arr = _get_1d(mat_data.get('Discharge_current_mean', np.zeros(n_cycles)))
        chg_curr_arr = _get_1d(mat_data.get('Charge_current_mean', np.zeros(n_cycles)))


        dis_res_arr = None
        chg_res_arr = None
        if info_data is not None:
            if 'Discharge_resistance' in info_data:
                dis_res_arr = _get_1d(info_data['Discharge_resistance'])
            if 'Charge_resistance' in info_data:
                chg_res_arr = _get_1d(info_data['Charge_resistance'])

        cycles = []
        for i in range(n_cycles):
            try:
                dis_cap = abs(float(dis_cap_arr[i]))
                chg_cap = float(chg_cap_arr[i])


                ce = dis_cap / chg_cap if chg_cap > 0 else 0.0
                ce = min(ce, 1.0)


                dcr = np.nan
                if dis_res_arr is not None:
                    dcr = float(dis_res_arr[i])
                if (np.isnan(dcr) or dcr == 0) and chg_res_arr is not None:
                    dcr = float(chg_res_arr[i])



                nominal_q = dis_cap if dis_cap > 0 else 1.0
                c_rate_dis = (abs(float(dis_curr_arr[i])) * 3600) / nominal_q
                c_rate_chg = (float(chg_curr_arr[i]) * 3600) / nominal_q if chg_cap > 0 else 0.0

                cd = CycleData(
                    cycle_index=i + 1,
                    discharge_capacity_ah=dis_cap,
                    charge_capacity_ah=chg_cap,
                    coulombic_efficiency=ce,
                    temperature_c=25.0,
                    c_rate_charge=c_rate_chg,
                    c_rate_discharge=c_rate_dis,
                    mean_discharge_voltage_v=float(dis_volt_arr[i]),
                    mean_charge_voltage_v=float(chg_volt_arr[i]),
                    dc_resistance_ohm=dcr if not np.isnan(dcr) else None,
                    metadata={'source_file': filepath}
                )
                cycles.append(cd)
            except Exception as e:
                logger.warning(f"  转换 Final.mat 循环 {i} 失败 [{filepath}]: {e}")

        logger.info(f"  [Final.mat] 解析完成: {len(cycles)} 个循环 (含电阻={dis_res_arr is not None})")
        return cycles

    def _excel_dict_to_cycle(self, d: dict, idx: int) -> Optional[CycleData]:
        """Excel dict to cycle."""
        try:
            charge_capacity = float(d.get('charge_capacity_ah', 0) or 0)
            discharge_capacity = float(d.get('discharge_capacity_ah', 0) or 0)




            ce = float(d.get('coulombic_efficiency', 0) or 0)
            if ce > 1.5:
                ce /= 100.0

            charge_hours = _duration_to_hours(d.get('charge_time'))
            discharge_hours = _duration_to_hours(d.get('discharge_time'))
            c_rate_charge = (1.0 / charge_hours) if charge_hours > 0 else 0.0
            c_rate_discharge = (1.0 / discharge_hours) if discharge_hours > 0 else 0.0

            resistance = d.get('dc_resistance')
            if resistance is not None and not pd.isna(resistance):
                resistance = float(resistance)

                resistance = resistance / 1000.0 if resistance > 1.0 else resistance
            else:
                resistance = None

            temperature = float(d.get('temperature_c', 25.0) or 0)
            if temperature == 0:
                temperature = 25.0

            mean_voltage = float(d.get('mean_voltage_v', 0) or 0)
            return CycleData(
                cycle_index=d.get('cycle_index', idx + 1),
                charge_capacity_ah=charge_capacity,
                discharge_capacity_ah=discharge_capacity,
                coulombic_efficiency=min(max(ce, 0.0), 1.2),
                charge_energy_wh=float(d.get('charge_energy_wh', 0) or 0),
                discharge_energy_wh=float(d.get('discharge_energy_wh', 0) or 0),
                mean_charge_voltage_v=mean_voltage,
                mean_discharge_voltage_v=mean_voltage,
                temperature_c=temperature,
                temp_max_c=temperature,
                temp_min_c=temperature,
                c_rate_charge=c_rate_charge,
                c_rate_discharge=c_rate_discharge,
                dc_resistance_ohm=resistance,
                metadata={k: v for k, v in d.items()
                          if k not in ['cycle_index', 'charge_capacity_ah',
                                       'discharge_capacity_ah', 'coulombic_efficiency',
                                       'temperature_c', 'c_rate_charge',
                                       'c_rate_discharge', 'dc_resistance']}
            )
        except Exception as e:
            logger.warning(f"  转换 Excel 循环 {idx} 失败: {e}")
            return None

    def load_dataset(self, dataset_name: str) -> List[CellDegradationData]:
        """Load dataset."""
        dataset_map = {
            "sodium-ion": self._find_sodium_ion_files,
            "cold-curse": self._find_cold_curse_files,
            "randomized": self._find_randomized_files,
            "pack": self._find_pack_files,
        }

        if dataset_name not in dataset_map:
            raise ValueError(f"未知数据集: {dataset_name}，可选: {list(dataset_map.keys())}")

        files = dataset_map[dataset_name]()
        results = []
        for filepath, cell_id, chemistry in files:
            try:
                cell_data = self.load_cell(
                    filepath=filepath,
                    cell_id=cell_id,
                    chemistry=chemistry
                )
                results.append(cell_data)
            except Exception as e:
                logger.error(f"  加载失败 [{cell_id}]: {e}")

        logger.info(f" 数据集 [{dataset_name}] 加载完成: {len(results)}/{len(files)} 个电芯")
        return results

    def _find_sodium_ion_files(self) -> List[Tuple[str, str, str]]:
        """Find sodium ion files."""
        files = []

        for ext in ['*.xls', '*.xlsx']:
            for f in Path(self.base_dir).glob(f'**/{ext}'):
                fname = f.name

                if re.match(r'[A-Z]-\d+-\d+', fname.upper().replace(' ', '')):
                    cell_id = f.stem
                    files.append((str(f), cell_id, "sodium-ion"))
        logger.info(f"  发现 {len(files)} 个钠离子电芯数据文件")
        return files

    def _find_cold_curse_files(self) -> List[Tuple[str, str, str]]:
        """Find cold curse files."""
        files = []
        for f in Path(self.base_dir).glob('**/*Detail_Final.mat'):
            fname = f.name

            m = re.search(r'Dongzhen-(\d+)', fname)
            if m:
                cell_id = f"Dongzhen-{m.group(1)}"
                files.append((str(f), cell_id, "lithium-ion"))
        logger.info(f"  发现 {len(files)} 个冷诅咒数据文件 (Final.mat)")
        return files

    def _find_randomized_files(self) -> List[Tuple[str, str, str]]:
        """Find randomized files."""
        files = []
        for f in Path(self.base_dir).glob('**/*.mat'):
            fname = f.name.lower()
            if 'random' in fname or 'batch' in fname:
                cell_id = f.stem
                files.append((str(f), cell_id, "lithium-ion"))
        logger.info(f"  发现 {len(files)} 个随机工况数据文件")
        return files

    def _find_pack_files(self) -> List[Tuple[str, str, str]]:
        """Find pack files."""
        files = []
        for ext in ['*.mat', '*.xlsx', '*.csv']:
            for f in Path(self.base_dir).glob(f'**/{ext}'):
                fname = f.name.lower()
                if 'pack' in fname or 'module' in fname:
                    cell_id = f.stem
                    files.append((str(f), cell_id, "lithium-ion-pack"))
        logger.info(f"  发现 {len(files)} 个 Pack 级数据文件")
        return files


# ============================================================

# ============================================================

def load_wenzhou_sodium_ion(base_dir: str = None) -> List[CellDegradationData]:
    """Load wenzhou sodium ion."""
    loader = WenzhouDataLoader(base_dir)
    return loader.load_dataset("sodium-ion")


def load_wenzhou_cold_curse(base_dir: str = None) -> List[CellDegradationData]:
    """Load wenzhou cold curse."""
    loader = WenzhouDataLoader(base_dir)
    return loader.load_dataset("cold-curse")


def load_all_wenzhou(base_dir: str = None) -> Dict[str, List[CellDegradationData]]:
    """Load all wenzhou."""
    loader = WenzhouDataLoader(base_dir)
    results = {}
    for ds_name in ["sodium-ion", "cold-curse", "randomized", "pack"]:
        try:
            data = loader.load_dataset(ds_name)
            if data:
                results[ds_name] = data
        except Exception as e:
            logger.error(f"  加载 [{ds_name}] 失败: {e}")
    return results
