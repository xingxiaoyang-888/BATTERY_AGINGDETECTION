# utils/soh_data_loader.py
"""
Wenzhou 系列电池老化数据集 — 多格式数据加载器
====================================================
支持格式:
  - .mat  (MATLAB v7.3 / v5 格式) → 冷诅咒 & 随机工况数据
  - .xls / .xlsx (Excel)         → 钠离子电池循环数据
  - .ndax (Gamry 电化学工作站)     → 原始电化学测量数据

设计原则:
  1. 每种格式独立解析器，统一输出为 pandas DataFrame
  2. 自动检测文件格式并路由到对应解析器
  3. 批量加载 + 元数据提取
"""

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

# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class CycleData:
    """单个充放电循环的全部数据"""
    cycle_index: int
    timestamp: Optional[float] = None          # 循环开始时间戳
    # 充/放电容量
    charge_capacity_ah: float = 0.0
    discharge_capacity_ah: float = 0.0
    # 库仑效率
    coulombic_efficiency: float = 0.0
    # 能量
    charge_energy_wh: float = 0.0
    discharge_energy_wh: float = 0.0
    # 电压
    mean_charge_voltage_v: float = 0.0
    mean_discharge_voltage_v: float = 0.0
    max_charge_voltage_v: float = 0.0
    min_discharge_voltage_v: float = 0.0
    # 温度
    temperature_c: float = 25.0
    temp_max_c: float = 25.0
    temp_min_c: float = 25.0
    # 工况
    c_rate_charge: float = 0.0
    c_rate_discharge: float = 0.0
    rest_time_h: float = 0.0
    # 内阻 (可选)
    dc_resistance_ohm: Optional[float] = None
    # 原始充放电曲线数据 (可选，用于电化学特征提取)
    charge_curve: Optional[pd.DataFrame] = None   # columns: [V, Q, Time]
    discharge_curve: Optional[pd.DataFrame] = None
    # 元数据
    metadata: Dict[str, Any] = None


@dataclass
class CellDegradationData:
    """单个电芯的完整老化数据"""
    cell_id: str
    chemistry: str = "unknown"          # "sodium-ion" | "lithium-ion" | "LFP" 等
    nominal_capacity_ah: float = None   # 标称容量
    cycles: List[CycleData] = None
    metadata: Dict[str, Any] = None     # 实验条件、制造商等

    def __post_init__(self):
        if self.cycles is None:
            self.cycles = []
        if self.metadata is None:
            self.metadata = {}

    def __len__(self):
        return len(self.cycles)

    @property
    def soh_series(self) -> np.ndarray:
        """计算 SOH 序列（基于全部循环中的最大放电容量作为 BOL 参考）"""
        if not self.cycles:
            return np.array([])
        # 取全生命周期最大放电容量作为标称容量 (BOL)
        valid_caps = [c.discharge_capacity_ah for c in self.cycles
                      if c.discharge_capacity_ah > 0]
        ref_cap = max(valid_caps) if valid_caps else 1.0
        return np.array([c.discharge_capacity_ah / ref_cap for c in self.cycles])


# ============================================================
# MAT 文件解析器（冷诅咒 & 随机工况数据）
# ============================================================

class MATFileLoader:
    """
    MATLAB .mat 文件加载器

    支持格式:
      - v5  (标准 HDF5 容器, scipy.io.loadmat 可读)
      - v7.3 (HDF5 格式, 需要 h5py)
      - v4  (旧格式, 兼容处理)

    Wenzhou 数据集典型结构:
      Dongzhen-XXX-Detail_Cycle_information.mat
        ├── cycle_info: struct array
        │   ├── capacity_discharge
        │   ├── capacity_charge
        │   ├── temperature
        │   ├── current
        │   ├── ...
        └── ...
    """

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        加载 .mat 文件，自动处理 v5 和 v7.3 格式

        Returns:
            dict: MATLAB 工作区变量字典
        """
        import scipy.io as sio

        # 尝试标准 scipy 加载（v5/v4 格式）
        try:
            data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
            # 过滤掉 MATLAB 内部变量（__xxx__）
            clean_data = {k: v for k, v in data.items() if not k.startswith('__')}
            logger.info(f"  [MAT v5] 加载成功: {filepath} → {list(clean_data.keys())}")
            return clean_data
        except NotImplementedError:
            # v7.3 格式需要 h5py
            logger.info(f"  检测到 MATLAB v7.3 格式，切换到 h5py 读取...")
            return MATFileLoader._load_v73(filepath)
        except Exception as e:
            logger.error(f"  MAT 文件加载失败 [{filepath}]: {e}")
            raise

    @staticmethod
    def _load_v73(filepath: str) -> Dict[str, Any]:
        """使用 h5py 加载 MATLAB v7.3 格式"""
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = MATFileLoader._h5_to_dict(f[key])
        logger.info(f"  [MAT v7.3] 加载成功: {filepath} → {list(data.keys())}")
        return data

    @staticmethod
    def _h5_to_dict(h5obj) -> Any:
        """递归转换 h5py 对象为 Python 原生类型"""
        import h5py
        if isinstance(h5obj, h5py.Dataset):
            val = h5obj[()]
            # 处理 MATLAB 字符串
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
        """
        从 Cycle_information.mat 中提取循环数据

        Wenzhou 冷诅咒数据典型结构:
          mat_data['Cycle_information'] → struct array (每个元素 = 一个循环)

        Returns:
            List[Dict]: 每个循环的字典列表
        """
        # 自动寻找 cycle 相关键
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

        # 处理 scipy.io.loadmat 的 struct 数组
        if hasattr(raw, 'dtype') and hasattr(raw, 'flat'):
            # MATLAB struct array → Python list of dicts
            cycles = []
            field_names = raw.dtype.names
            for i in range(len(raw)):
                cycle_dict = {}
                for fn in field_names:
                    val = raw[i][fn]
                    # 处理标量数组
                    if isinstance(val, np.ndarray):
                        val = val.item() if val.size == 1 else val.flatten()
                    # 处理 MATLAB string
                    if isinstance(val, np.ndarray) and val.dtype.kind == 'U':
                        val = str(val)
                    cycle_dict[fn] = val
                cycles.append(cycle_dict)
            logger.info(f"  提取到 {len(cycles)} 个循环记录")
            return cycles

        # 处理 h5py Group
        if isinstance(raw, dict) and not isinstance(raw, np.ndarray):
            # h5py 加载的 struct
            return MATFileLoader._extract_h5_struct_cycles(raw)

        logger.warning(f"  无法识别的 cycle_info 数据格式: {type(raw)}")
        return []

    @staticmethod
    def _extract_h5_struct_cycles(h5_struct: Dict) -> List[Dict]:
        """从 h5py 加载的 struct 中提取循环数组"""
        # 获取所有字段名和长度
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
# XLS/XLSX 文件解析器（钠离子电池循环数据）
# ============================================================

class XLSFileLoader:
    """
    钠离子电池 Excel 循环数据加载器

    Wenzhou 钠电数据典型结构:
      每个 .xls 文件包含多个 sheet（可能按循环组织）
      或单个 sheet 包含所有循环的时序数据

    列名可能包括:
      - Cycle Index / 循环序号
      - Charge Capacity (Ah) / 充电容量
      - Discharge Capacity (Ah) / 放电容量
      - Coulombic Efficiency / 库仑效率
      - Temperature / 温度
      - ...
    """

    @staticmethod
    def load_sheets(filepath: str) -> Dict[str, pd.DataFrame]:
        """加载 Excel 文件的所有 sheet"""
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
        """加载单个 sheet 为 DataFrame"""
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        logger.info(f"  [XLS] 加载: {filepath} [{sheet_name}] → {len(df)} 行 × {len(df.columns)} 列")
        return df

    @staticmethod
    def detect_cycle_structure(df: pd.DataFrame) -> Dict[str, str]:
        """
        自动检测 DataFrame 中的列结构

        通过列名模糊匹配识别关键字段。

        Returns:
            Dict: {标准字段名: 实际列名} 映射
        """
        # 标准化列名（去除空格、转换为小写）
        col_map = {c: c for c in df.columns}
        clean_cols = {c: c.lower().replace(' ', '_').replace('-', '_') for c in df.columns}

        # 关键字段匹配规则
        match_rules = {
            'cycle_index':        [r'cycle', r'循环', r'index', r'序号'],
            'charge_capacity_ah': [r'charge.*cap', r'充电.*容', r'cha.*cap', r'cap.*cha'],
            'discharge_capacity_ah': [r'discharge.*cap', r'放电.*容', r'dis.*cap', r'cap.*dis'],
            'coulombic_efficiency': [r'coulomb', r'库仑', r'efficiency', r'效率', r'ce'],
            'temperature_c':      [r'temp', r'温度', r'temperature'],
            'voltage_mean':       [r'voltage.*mean', r'平均.*压', r'v_mean'],
            'current_a':          [r'current', r'电流', r'i_'],
            'time_s':             [r'time', r'时间', r'duration'],
            'dc_resistance':      [r'resistance', r'内阻', r'dcr', r'ir'],
        }

        mapping = {}
        for std_name, patterns in match_rules.items():
            for orig_col, clean_col in clean_cols.items():
                if any(re.search(p, clean_col) for p in patterns):
                    if std_name not in mapping:  # 优先第一个匹配
                        mapping[std_name] = orig_col
                    break

        logger.info(f"  列结构检测: {len(mapping)}/{len(match_rules)} 个字段已识别")
        return mapping

    @staticmethod
    def parse_cycle_summary(df: pd.DataFrame,
                            col_mapping: Dict[str, str] = None) -> List[dict]:
        """
        将 DataFrame 解析为循环级汇总数据

        如果 df 本身就是每行一个循环的汇总表，直接按行解析；
        如果是原始时序数据，需要先按 cycle 聚合。
        """
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
# NDAX 文件解析器（Gamry 电化学原始数据）
# ============================================================

class NDAXFileLoader:
    """
    Gamry .ndax 原始电化学数据解析器

    .ndax 是 Gamry Instruments 的二进制数据格式，包含:
      - 充放电曲线的原始 V/I/Q/t 时序
      - 实验元数据（温度、日期、设置等）

    由于 .ndax 是专有二进制格式，我们提供两种解析路径:
      1. 直接二进制解析（已知结构）
      2. 通过 Gamry 提供的 COM/DLL 接口（Windows only）
    """

    @staticmethod
    def parse(filepath: str) -> Optional[Dict[str, Any]]:
        """
        尝试解析 .ndax 文件

        Returns:
            Dict: {'curves': [...], 'metadata': {...}} 或 None
        """
        # .ndax 格式为二进制，尝试读取文件头识别版本
        try:
            with open(filepath, 'rb') as f:
                header = f.read(256)
        except Exception as e:
            logger.error(f"  无法读取 .ndax 文件 [{filepath}]: {e}")
            return None

        # 检查 Gamry 文件签名
        if header[:4] != b'G4\2\0' and header[:4] != b'G4\3\0':
            logger.warning(f"  文件可能不是标准 Gamry .ndax 格式: {filepath}")
            # 不直接放弃，继续尝试

        logger.info(f"  [NDAX] 解析: {filepath} → 暂用二进制扫描模式")

        # 返回基本结构供下游使用
        return {
            'filepath': filepath,
            'file_size': os.path.getsize(filepath),
            'header_version': header[:4].decode('latin-1', errors='ignore'),
            'raw_data_available': True,
        }


# ============================================================
# 统一数据加载接口
# ============================================================

class WenzhouDataLoader:
    """
    Wenzhou 系列数据集 — 统一加载门面

    自动根据文件扩展名路由到对应解析器，
    输出标准化的 CellDegradationData 对象。

    用法:
        loader = WenzhouDataLoader()
        cell_data = loader.load_cell("F:/path/to/H-1-2.xls")
        # 或
        dataset = loader.load_dataset("sodium-ion")
    """

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
        """检测文件格式"""
        ext = Path(filepath).suffix.lower()
        return ext

    def load_cell(self, filepath: str,
                  cell_id: str = None,
                  chemistry: str = "unknown",
                  nominal_capacity_ah: float = None) -> CellDegradationData:
        """
        加载单个电芯的数据文件

        Args:
            filepath: 数据文件路径 (.xls / .mat / .ndax)
            cell_id: 电芯标识
            chemistry: 化学体系
            nominal_capacity_ah: 标称容量

        Returns:
            CellDegradationData 对象
        """
        ext = self.detect_format(filepath)
        if ext not in self.EXTENSION_HANDLERS:
            raise ValueError(f"不支持的文件格式: {ext}")

        if cell_id is None:
            cell_id = Path(filepath).stem

        logger.info(f"  加载电芯 [{cell_id}] → {filepath}")

        # 路由到对应解析器
        raw_data = self.EXTENSION_HANDLERS[ext](filepath)

        # 转换为标准化的 CycleData 列表
        cycles = self._convert_to_cycles(raw_data, ext, filepath)

        # 推断标称容量 (全生命周期最大放电容量)
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
        """
        将原始解析结果转换为标准化的 CycleData 列表。

        这是整个数据管线的关键转换点——所有格式统一收敛到这里。
        """
        cycles = []

        if ext == '.mat':
            # 检测 MAT 文件格式类型
            if 'Discharge_capacity' in raw_data:
                # Final.mat 格式: flat arrays，每列一个循环
                cycles = self._final_mat_to_cycles(raw_data, filepath)
            else:
                # 旧格式: Cycle_information struct array
                mat_cycles = MATFileLoader.extract_cycle_info(raw_data)
                for i, mc in enumerate(mat_cycles):
                    cd = self._mat_dict_to_cycle(mc, i)
                    if cd:
                        cycles.append(cd)

        elif ext in ('.xls', '.xlsx'):
            # Excel 数据 → 多个 sheet
            for sheet_name, df in raw_data.items():
                col_map = XLSFileLoader.detect_cycle_structure(df)
                summary = XLSFileLoader.parse_cycle_summary(df, col_map)
                for i, s in enumerate(summary):
                    cd = self._excel_dict_to_cycle(s, i + len(cycles))
                    if cd:
                        cycles.append(cd)

        elif ext in ('.nda', '.ndax'):
            # NDAX 二进制 → 暂存元数据
            logger.info(f"  NDAX 文件已索引，详细解析需 Gamry SDK: {filepath}")

        return cycles

    def _mat_dict_to_cycle(self, d: dict, idx: int) -> Optional[CycleData]:
        """将 MATLAB struct 字段转为 CycleData"""
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
        """
        加载与 Final.mat 配对的 Cycle_information.mat

        从 Cycle_information.mat 中提取:
          - Discharge_resistance / Charge_resistance → 直流内阻
          - MODE_current → 电流模式
          - 各类分类标签

        Returns:
            Dict 或 None (配对文件不存在时)
        """
        final_file = Path(final_path)
        info_file = final_file.parent / final_file.name.replace(
            'Detail_Final.mat', 'Detail_Cycle_information.mat')
        if info_file.exists():
            try:
                return MATFileLoader.load(str(info_file))
            except Exception:
                logger.debug(f"  配对文件加载失败: {info_file.name}")
        return None

    def _final_mat_to_cycles(self, mat_data: Dict[str, Any],
                             filepath: str) -> List[CycleData]:
        """
        解析 Wenzhou Final.mat 的 flat-array 结构

        Final.mat 结构 (每个变量为 shape=(1, N_cycles) 的数组):
          - Discharge_capacity / Charge_capacity → 放/充电容量 (单位: mAs)
          - Discharge_voltage_mean / Charge_voltage_mean → 平均电压 (V)
          - Discharge_current_mean / Charge_current_mean → 平均电流 (mA)
          - Discharge_datetime_* / Charge_datetime_* → 时间戳 (MATLAB datenum)
          - Charging_power / Discharging_power → 原始功率时序 (cell array)

        同时尝试加载配对的 Cycle_information.mat 获取电阻数据。
        """
        # 处理 squeeze 后的数组 (可能是 1D 或 2D)
        def _get_1d(arr, n=None):
            """安全获取 1D 数组 (兼容 squeeze_me=True/False)"""
            a = np.atleast_1d(np.asarray(arr).squeeze())
            return a

        dis_cap_arr = _get_1d(mat_data['Discharge_capacity'])
        n_cycles = len(dis_cap_arr)

        # 尝试加载配对的 Cycle_information.mat
        info_data = self._load_paired_info(filepath)

        # 预提取数组 (避免循环中重复索引)
        chg_cap_arr = _get_1d(mat_data['Charge_capacity'])
        dis_volt_arr = _get_1d(mat_data.get('Discharge_voltage_mean', np.zeros(n_cycles)))
        chg_volt_arr = _get_1d(mat_data.get('Charge_voltage_mean', np.zeros(n_cycles)))
        dis_curr_arr = _get_1d(mat_data.get('Discharge_current_mean', np.zeros(n_cycles)))
        chg_curr_arr = _get_1d(mat_data.get('Charge_current_mean', np.zeros(n_cycles)))

        # 从 Cycle_information.mat 提取电阻 (如有)
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
                dis_cap = abs(float(dis_cap_arr[i]))   # 取绝对值
                chg_cap = float(chg_cap_arr[i])

                # 库仑效率 (考虑部分循环的情况)
                ce = dis_cap / chg_cap if chg_cap > 0 else 0.0
                ce = min(ce, 1.0)  # 钳制到 [0, 1]

                # 直流内阻 (优先放电内阻)
                dcr = np.nan
                if dis_res_arr is not None:
                    dcr = float(dis_res_arr[i])
                if (np.isnan(dcr) or dcr == 0) and chg_res_arr is not None:
                    dcr = float(chg_res_arr[i])

                # C-rate 估算: C_rate ≈ |I_mean| / (|Q|/3600)
                # Q 单位 mAs, I 单位 mA → C_rate = I / (Q/3600) = I*3600/Q
                nominal_q = dis_cap if dis_cap > 0 else 1.0
                c_rate_dis = (abs(float(dis_curr_arr[i])) * 3600) / nominal_q
                c_rate_chg = (float(chg_curr_arr[i]) * 3600) / nominal_q if chg_cap > 0 else 0.0

                cd = CycleData(
                    cycle_index=i + 1,
                    discharge_capacity_ah=dis_cap,
                    charge_capacity_ah=chg_cap,
                    coulombic_efficiency=ce,
                    temperature_c=25.0,  # 冷诅咒数据: 温度从实验元数据获取
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
        """将 Excel 行字典转为 CycleData"""
        try:
            return CycleData(
                cycle_index=d.get('cycle_index', idx + 1),
                charge_capacity_ah=float(d.get('charge_capacity_ah', 0)),
                discharge_capacity_ah=float(d.get('discharge_capacity_ah', 0)),
                coulombic_efficiency=float(d.get('coulombic_efficiency', 0)),
                temperature_c=float(d.get('temperature_c', 25.0)),
                c_rate_charge=float(d.get('c_rate_charge', 0)),
                c_rate_discharge=float(d.get('c_rate_discharge', 0)),
                dc_resistance_ohm=float(d.get('dc_resistance', np.nan))
                    if d.get('dc_resistance') is not None else None,
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
        """
        按数据集名称批量加载

        Args:
            dataset_name: "sodium-ion" | "cold-curse" | "randomized" | "pack"

        Returns:
            List[CellDegradationData]: 该数据集下所有电芯的数据
        """
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
        """扫描钠离子电池数据文件"""
        files = []
        # 扫描 xls/xlsx 文件（H-1-2, H-2-1, H-5-1 等）
        for ext in ['*.xls', '*.xlsx']:
            for f in Path(self.base_dir).glob(f'**/{ext}'):
                fname = f.name
                # 匹配钠离子电芯命名模式 H-X-Y
                if re.match(r'[A-Z]-\d+-\d+', fname.upper().replace(' ', '')):
                    cell_id = f.stem
                    files.append((str(f), cell_id, "sodium-ion"))
        logger.info(f"  发现 {len(files)} 个钠离子电芯数据文件")
        return files

    def _find_cold_curse_files(self) -> List[Tuple[str, str, str]]:
        """扫描冷诅咒数据集文件 — 只匹配 Detail_Final.mat (主数据源)"""
        files = []
        for f in Path(self.base_dir).glob('**/*Detail_Final.mat'):
            fname = f.name
            # 匹配 Dongzhen-XXX 命名模式
            m = re.search(r'Dongzhen-(\d+)', fname)
            if m:
                cell_id = f"Dongzhen-{m.group(1)}"
                files.append((str(f), cell_id, "lithium-ion"))
        logger.info(f"  发现 {len(files)} 个冷诅咒数据文件 (Final.mat)")
        return files

    def _find_randomized_files(self) -> List[Tuple[str, str, str]]:
        """扫描随机工况数据集文件"""
        files = []
        for f in Path(self.base_dir).glob('**/*.mat'):
            fname = f.name.lower()
            if 'random' in fname or 'batch' in fname:
                cell_id = f.stem
                files.append((str(f), cell_id, "lithium-ion"))
        logger.info(f"  发现 {len(files)} 个随机工况数据文件")
        return files

    def _find_pack_files(self) -> List[Tuple[str, str, str]]:
        """扫描 Pack 级数据文件"""
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
# 便捷函数
# ============================================================

def load_wenzhou_sodium_ion(base_dir: str = None) -> List[CellDegradationData]:
    """快捷函数：加载 Wenzhou 钠离子电池老化数据"""
    loader = WenzhouDataLoader(base_dir)
    return loader.load_dataset("sodium-ion")


def load_wenzhou_cold_curse(base_dir: str = None) -> List[CellDegradationData]:
    """快捷函数：加载 Wenzhou 冷诅咒电池老化数据"""
    loader = WenzhouDataLoader(base_dir)
    return loader.load_dataset("cold-curse")


def load_all_wenzhou(base_dir: str = None) -> Dict[str, List[CellDegradationData]]:
    """快捷函数：加载所有可用的 Wenzhou 数据集"""
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
