# models/soh_ai/data_pipeline.py
"""
SOH AI 数据管线 — 从原始循环数据到模型就绪的训练样本
============================================================
处理流程:
  Stage 1: 循环级特征提取 ──→ 从 CycleData → 标准化特征向量
  Stage 2: 数据清洗 ────────→ 缺失值 / 异常值 / 平滑
  Stage 3: 序列窗口构建 ────→ (N, lookback, features) 张量
  Stage 4: 训练集划分 ──────→ train / val / test split
  Stage 5: 持久化 ──────────→ Parquet 存储

设计原则:
  - 所有处理步骤可追溯、可复现
  - 支持增量处理（新增电芯数据无需重跑全部）
  - 自动生成数据质量报告
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

from .config import (
    FEATURE_CFG, ACTUAL_FEATURE_COLUMNS, DQ_CFG, TRAIN_CFG,
    PROCESSED_DATA_DIR, WEIGHTS_DIR,
)
from utils.soh_data_loader import CellDegradationData, CycleData

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _make_json_serializable(obj):
    """递归转换 numpy 类型为 JSON 可序列化的 Python 原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ============================================================
# Stage 1: 循环级特征提取
# ============================================================

class CycleFeatureExtractor:
    """
    从原始 CycleData 提取标准化特征向量

    输入: CellDegradationData (一个电芯的完整循环列表)
    输出: pd.DataFrame (每行 = 一个循环，每列 = 一个特征)
    """

    def __init__(self, cfg = None):
        self.cfg = cfg or FEATURE_CFG

    def extract(self, cell: CellDegradationData) -> pd.DataFrame:
        """
        从单个电芯的全部循环中提取特征 DataFrame
        """
        if not cell.cycles:
            raise ValueError(f"电芯 {cell.cell_id} 没有循环数据")

        n_cycles = len(cell.cycles)
        features = {}
        cumulative_ah = np.cumsum([max(0.0, float(c.discharge_capacity_ah or 0.0)) for c in cell.cycles])

        # --- A. 当前状态特征 ---
        soh_series = cell.soh_series  # shape: (n_cycles,)
        for i, cycle in enumerate(cell.cycles):
            features.setdefault('soh', [None] * n_cycles)[i] = soh_series[i]
            features.setdefault('cycle_index', [None] * n_cycles)[i] = cycle.cycle_index
            features.setdefault('cumulative_ah_throughput', [None] * n_cycles)[i] = cumulative_ah[i]
            features.setdefault('internal_resistance', [None] * n_cycles)[i] = cycle.dc_resistance_ohm
            features.setdefault('coulombic_efficiency', [None] * n_cycles)[i] = cycle.coulombic_efficiency

        # --- B. 工况特征 ---
        for i, cycle in enumerate(cell.cycles):
            features.setdefault('temperature_c', [None] * n_cycles)[i] = cycle.temperature_c
            features.setdefault('c_rate_charge', [None] * n_cycles)[i] = cycle.c_rate_charge
            features.setdefault('c_rate_discharge', [None] * n_cycles)[i] = cycle.c_rate_discharge
            features.setdefault('rest_time_h', [None] * n_cycles)[i] = cycle.rest_time_h

        # SOC 窗口特征（从充放电曲线或元数据推断）
        for i, cycle in enumerate(cell.cycles):
            soc_info = self._extract_soc_window(cycle)
            features.setdefault('soc_min', [None] * n_cycles)[i] = soc_info.get('soc_min', 0)
            features.setdefault('soc_max', [None] * n_cycles)[i] = soc_info.get('soc_max', 0)
            features.setdefault('soc_mean', [None] * n_cycles)[i] = soc_info.get('soc_mean', 0)

        # --- C. 趋势特征（从历史序列计算） ---
        df_temp = pd.DataFrame(features)
        # 确保 soh 列存在且无全空
        if 'soh' in df_temp.columns and df_temp['soh'].notna().any():
            soh_vals = df_temp['soh'].values

            df_temp['soh_diff_1'] = self._diff(soh_vals, lag=1)
            df_temp['soh_diff_3'] = self._diff(soh_vals, lag=3)
            df_temp['soh_diff_5'] = self._diff(soh_vals, lag=5)
            df_temp['soh_decay_rate'] = self._rolling_decay_rate(soh_vals, window=10)

            # 容量衰减加速度（SOH 二阶差分）
            soh_diff1 = self._diff(soh_vals, lag=1)
            df_temp['capacity_fade_acceleration'] = self._diff(soh_diff1, lag=3)

        if 'internal_resistance' in df_temp.columns:
            r_vals = df_temp['internal_resistance'].values
            df_temp['r_diff_1'] = self._diff(r_vals, lag=1)

        if 'coulombic_efficiency' in df_temp.columns:
            ce_vals = df_temp['coulombic_efficiency'].values
            df_temp['ce_trend'] = self._rolling_mean(ce_vals, window=5)

        # --- D. 电化学特征（从充放电曲线提取） ---
        for i, cycle in enumerate(cell.cycles):
            echem = self._extract_echem_features(cycle)
            for echem_key in self.cfg.electrochemical_features:
                features.setdefault(echem_key, [None] * n_cycles)[i] = echem.get(echem_key, None)

        # 构建最终 DataFrame
        result_df = pd.DataFrame(features)
        for col in df_temp.columns:
            if col not in result_df.columns:
                result_df[col] = df_temp[col].values
        result_df.insert(0, 'cell_id', cell.cell_id)
        result_df.insert(1, 'chemistry', cell.chemistry)
        result_df.insert(2, 'dataset_id', cell.metadata.get('dataset_id', 'wenzhou-sodium-ion'))
        result_df.insert(3, 'condition', cell.metadata.get('condition', 'unknown'))
        result_df.insert(4, 'nominal_capacity_ah', cell.nominal_capacity_ah)
        for col in result_df.columns:
            if col not in {'cell_id', 'chemistry', 'dataset_id', 'condition'}:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        return result_df

    def _diff(self, arr: np.ndarray, lag: int = 1) -> np.ndarray:
        """计算滞后差分（前面填充 NaN）"""
        numeric = pd.to_numeric(pd.Series(arr), errors='coerce').to_numpy(dtype=float)
        result = np.full(numeric.shape, np.nan, dtype=float)
        if len(numeric) > lag:
            result[lag:] = numeric[lag:] - numeric[:-lag]
        return result

    def _rolling_mean(self, arr: np.ndarray, window: int = 5) -> np.ndarray:
        """滚动均值（前面用较小的窗口）"""
        numeric = pd.to_numeric(pd.Series(arr), errors='coerce').to_numpy(dtype=float)
        result = np.full(numeric.shape, np.nan, dtype=float)
        for i in range(len(numeric)):
            start = max(0, i - window + 1)
            window_values = numeric[start:i+1]
            if np.isfinite(window_values).any():
                result[i] = np.nanmean(window_values)
        return result

    def _rolling_decay_rate(self, arr: np.ndarray, window: int = 10) -> np.ndarray:
        """
        计算滚动指数衰减率
        对 log(SOH) vs cycle 做线性拟合，返回斜率
        """
        numeric = pd.to_numeric(pd.Series(arr), errors='coerce').to_numpy(dtype=float)
        result = np.full(numeric.shape, np.nan, dtype=float)
        for i in range(window, len(numeric)):
            y = numeric[max(0, i-window):i+1]
            x = np.arange(len(y))
            # 只对有效数据拟合
            valid = ~np.isnan(y) & (y > 0)
            if valid.sum() < 3:
                continue
            log_y = np.log(y[valid])
            x_valid = x[valid]
            if len(x_valid) >= 3:
                slope = np.polyfit(x_valid, log_y, 1)[0]
                result[i] = -slope  # 正值 = 衰减
        return result

    def _extract_soc_window(self, cycle: CycleData) -> Dict[str, float]:
        """从循环数据中提取/推断 SOC 窗口"""
        soc_info = {'soc_min': 0.0, 'soc_max': 1.0, 'soc_mean': 0.5}

        # 方法1: 从 metadata 中获取
        if cycle.metadata:
            for key in ['soc_min', 'soc_max', 'soc_window']:
                if key in cycle.metadata:
                    val = cycle.metadata[key]
                    if isinstance(val, (int, float)):
                        soc_info['soc_min'] = min(val, soc_info.get('soc_min', val))
                        soc_info['soc_max'] = max(val, soc_info.get('soc_max', val))

        # 方法2: 从充放电曲线推断
        if cycle.charge_curve is not None:
            q_max = cycle.charge_curve['Q'].max() if 'Q' in cycle.charge_curve.columns else 0
            if q_max > 0 and cycle.discharge_capacity_ah > 0:
                soc_info['soc_max'] = min(1.0, q_max / cycle.discharge_capacity_ah)

        soc_info['soc_mean'] = (soc_info['soc_min'] + soc_info['soc_max']) / 2
        return soc_info

    def _extract_echem_features(self, cycle: CycleData) -> Dict[str, float]:
        """
        从充放电曲线中提取电化学特征

        需要原始 V-Q 数据，没有则返回空字典（后续用 NaN 填充）
        """
        features = {
            'mean_charge_voltage': cycle.mean_charge_voltage_v or np.nan,
            'mean_discharge_voltage': cycle.mean_discharge_voltage_v or np.nan,
        }
        if cycle.mean_charge_voltage_v and cycle.mean_discharge_voltage_v:
            features['voltage_hysteresis'] = (
                cycle.mean_charge_voltage_v - cycle.mean_discharge_voltage_v
            )

        curve = cycle.discharge_curve
        if curve is None or not {'V', 'Q'}.issubset(curve.columns):
            return features

        clean = curve[['V', 'Q']].replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean.sort_values('V').drop_duplicates('V')
        if len(clean) < 5:
            return features
        dq_dv = np.gradient(clean['Q'].to_numpy(dtype=float), clean['V'].to_numpy(dtype=float))
        if np.isfinite(dq_dv).any():
            peak_idx = int(np.nanargmax(np.abs(dq_dv)))
            features['dq_dv_peak_shift'] = float(clean['V'].iloc[peak_idx])
            features['dq_dv_peak_height_ratio'] = float(abs(dq_dv[peak_idx]))
        return features


# ============================================================
# Stage 2: 数据清洗与质量检查
# ============================================================

class DataCleaner:
    """
    数据清洗与质量保证

    处理:
      - 缺失值填充/标记
      - 异常值检测（基于统计阈值）
      - SOH 平滑（去除测量噪声）
      - 质量报告生成
    """

    def __init__(self, cfg = None):
        self.cfg = cfg or DQ_CFG
        self.quality_report = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整的数据清洗流程"""
        df = df.copy()
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'cells': df['cell_id'].nunique() if 'cell_id' in df.columns else 0,
        }

        # Step 1: 丢弃全空的辅助列
        df = df.dropna(axis=1, how='all')

        # Step 2: 检查每列缺失率
        missing_rates = df.isnull().mean()
        cols_to_drop = missing_rates[missing_rates > self.cfg.max_missing_rate].index.tolist()
        if cols_to_drop:
            logger.info(f"  丢弃高缺失率列 ({len(cols_to_drop)}): {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        self.quality_report['dropped_columns'] = cols_to_drop

        # Step 3: 丢弃循环数过少的电芯
        if 'cell_id' in df.columns:
            cycle_counts = df.groupby('cell_id').size()
            bad_cells = cycle_counts[cycle_counts < self.cfg.min_cycles_per_cell].index.tolist()
            if bad_cells:
                logger.info(f"  丢弃循环数不足的电芯 ({len(bad_cells)}): {bad_cells}")
                df = df[~df['cell_id'].isin(bad_cells)]
            self.quality_report['rejected_cells'] = bad_cells

        # Step 4: 异常值检测与处理
        df = self._handle_outliers(df)

        # Step 5: SOH 平滑
        if 'soh' in df.columns and 'cell_id' in df.columns:
            df['soh_raw'] = df['soh'].copy()
            df['soh'] = df.groupby('cell_id')['soh'].transform(
                lambda x: self._smooth_soh(x, self.cfg.soh_smooth_window)
            )

        # Step 6: 容量跳变检测
        if 'soh' in df.columns:
            df = self._detect_capacity_jumps(df)

        # Step 7: 仅用历史值前向填充，避免未来循环信息泄漏到当前循环。
        if 'cell_id' in df.columns:
            fill_cols = [c for c in df.columns
                         if c not in ['cell_id', 'chemistry', 'dataset_id', 'condition', 'soh_raw']]
            for col in fill_cols:
                if df[col].isnull().any():
                    df[col] = df.groupby('cell_id')[col].transform(
                        lambda x: x.ffill()
                    )
        else:
            df = df.ffill()

        # 最终检查：仍有 NaN 的列用 0 填充
        df = df.fillna(0)

        self.quality_report['final_shape'] = df.shape
        self.quality_report['final_nan_count'] = df.isnull().sum().sum()
        self.quality_report['chemistry_counts'] = df.groupby('chemistry')['cell_id'].nunique().to_dict()
        if 'dataset_id' in df.columns:
            self.quality_report['dataset_counts'] = df.groupby('dataset_id')['cell_id'].nunique().to_dict()
        self.quality_report['physical_ranges'] = self._physical_ranges(df)

        return df

    @staticmethod
    def _physical_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        tracked = [
            'soh', 'nominal_capacity_ah', 'temperature_c',
            'c_rate_charge', 'c_rate_discharge', 'internal_resistance',
            'coulombic_efficiency',
        ]
        ranges = {}
        for col in tracked:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            ranges[col] = {
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
            }
        return ranges

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        异常值处理:
          1. 检测：偏离滚动中位数 > N * rolling_std 的点
          2. 处理：用滚动中位数替换（保留趋势，去除尖峰）
        """
        outlier_count = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        bounded_cols = {
            'soh', 'coulombic_efficiency', 'soc_min', 'soc_max', 'soc_mean',
            'c_rate_charge', 'c_rate_discharge', 'temperature_c',
        }

        for col in numeric_cols:
            if col in ['cell_id', 'chemistry', 'cycle_index']:
                continue
            if df[col].isnull().all():
                continue
            if col in bounded_cols:
                continue

            # 按电芯分组处理
            clean_col = df[col].copy()
            if 'cell_id' in df.columns:
                for cell_id in df['cell_id'].unique():
                    mask = df['cell_id'] == cell_id
                    vals = df.loc[mask, col]
                    if len(vals) < 10:
                        continue
                    rolling_med = vals.rolling(window=10, min_periods=3).median()
                    rolling_std = vals.rolling(window=10, min_periods=3).std()
                    deviation = np.abs(vals - rolling_med)
                    is_outlier = deviation > self.cfg.max_outlier_std * rolling_std
                    n_outliers = is_outlier.sum()
                    if n_outliers > 0:
                        outlier_count += n_outliers
                        clean_col.loc[mask & is_outlier] = rolling_med[is_outlier]
            df[col] = clean_col

        self.quality_report['outliers_replaced'] = int(outlier_count)
        if outlier_count > 0:
            logger.info(f"  替换 {outlier_count} 个异常值")
        return df

    def _smooth_soh(self, series: pd.Series, window: int = 5) -> pd.Series:
        """因果中值滤波与 EMA 平滑，不使用未来循环。"""
        if len(series) < window:
            return series
        # 使用中值滤波 + EMA 组合平滑
        median_smoothed = series.rolling(window=window, min_periods=1).median()
        # EMA: S_t = α*x_t + (1-α)*S_{t-1}
        alpha = 2.0 / (window + 1)
        ema = median_smoothed.copy()
        for i in range(1, len(ema)):
            if pd.notna(ema.iloc[i]):
                ema.iloc[i] = alpha * ema.iloc[i] + (1 - alpha) * ema.iloc[i-1]
        return ema

    def _detect_capacity_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测并标记容量跳变（可能是数据记录错误）"""
        df['soh_jump_flag'] = False
        if 'cell_id' in df.columns:
            for cell_id in df['cell_id'].unique():
                mask = df['cell_id'] == cell_id
                soh = df.loc[mask, 'soh'].values
                if len(soh) < 3:
                    continue
                soh_diff = np.diff(soh)
                jump_mask = np.abs(soh_diff) > self.cfg.max_capacity_jump
                # 标记跳变的起点
                jump_indices = np.where(jump_mask)[0]
                if len(jump_indices) > 0:
                    df_indices = df.loc[mask].index[jump_indices + 1]  # +1: 差分偏移
                    df.loc[df_indices, 'soh_jump_flag'] = True
        self.quality_report['capacity_jumps'] = int(df['soh_jump_flag'].sum())
        return df

    def generate_report(self) -> Dict:
        """生成数据质量报告"""
        return self.quality_report


# ============================================================
# Stage 3: 序列窗口构建 (用于 LSTM / Transformer)
# ============================================================

class SequenceBuilder:
    """
    从特征 DataFrame 构建序列样本

    滑动窗口:
      输入窗口: [t-lookback, ..., t-1, t]  →  X (lookback × n_features)
      预测目标: [t+1, ..., t+horizon]     →  y (horizon × n_targets)
    """

    def __init__(self, cfg = None):
        self.cfg = cfg or FEATURE_CFG

    def build_sequences(self, df: pd.DataFrame,
                        lookback: int = None,
                        horizon: int = None,
                        target_col: str = None,
                        feature_cols: List[str] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建 (X, y) 序列样本

        Args:
            df: 特征 DataFrame
            lookback: 输入序列长度
            horizon: 预测步长
            target_col: 目标列名
            feature_cols: 使用的特征列（None = 自动选择）

        Returns:
            X: (n_samples, lookback, n_features)
            y: (n_samples, horizon)
            cell_ids: (n_samples,) — 每个样本所属电芯
            cycle_indices: (n_samples,) — 每个样本的起始循环序号
        """
        lookback = lookback or self.cfg.lookback_window
        horizon = horizon or self.cfg.prediction_horizon
        target_col = target_col or self.cfg.target_col

        if feature_cols is None:
            # 优先使用配置里定义的真实输入列，避免把趋势/标签/派生列误喂给模型
            feature_cols = [c for c in FEATURE_CFG.all_features if c in df.columns]
            if not feature_cols:
                exclude = ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                           'soh_raw', 'soh_jump_flag', target_col]
                feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
        # 确保目标列可用
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在 DataFrame 中。列: {df.columns.tolist()}")

        X_list, y_list, cell_list, idx_list = [], [], [], []

        # 按电芯独立构建（不跨电芯截窗口）
        cells = df['cell_id'].unique() if 'cell_id' in df.columns else ['_single_']

        for cell_id in cells:
            if 'cell_id' in df.columns:
                cell_df = df[df['cell_id'] == cell_id].reset_index(drop=True)
            else:
                cell_df = df.reset_index(drop=True)

            n = len(cell_df)
            total_window = lookback + horizon
            if n < total_window:
                continue

            for i in range(n - total_window + 1):
                # 输入窗口
                X_window = cell_df.loc[i:i+lookback-1, feature_cols].values.astype(np.float32)
                # 目标窗口
                y_window = cell_df.loc[i+lookback:i+lookback+horizon-1, target_col].values.astype(np.float32)

                # NaN 检查
                if np.isnan(X_window).any() or np.isnan(y_window).any():
                    continue

                X_list.append(X_window)
                y_list.append(y_window)
                cell_list.append(cell_id)
                idx_list.append(cell_df.loc[i, 'cycle_index']
                                if 'cycle_index' in cell_df.columns else i)

        X = np.array(X_list) if X_list else np.empty((0, lookback, len(feature_cols)))
        y = np.array(y_list) if y_list else np.empty((0, horizon))
        cell_arr = np.array(cell_list)
        idx_arr = np.array(idx_list)

        logger.info(f"  序列构建: {len(X)} 个样本, X.shape={X.shape}, y.shape={y.shape}")
        return X, y, cell_arr, idx_arr


# ============================================================
# Stage 4: 训练/验证/测试集划分
# ============================================================

class DataSplitter:
    """
    电池老化数据的科学划分

    关键原则: 同一电芯的循环数据不能跨集合（防止数据泄漏）

    两种模式:
      1. leave_one_cell_out: 针对钠电数据量少（3 个电芯）
      2. stratified_split: 针对冷诅咒数据量大（141 个电芯）
    """

    def __init__(self, cfg = None):
        self.cfg = cfg or TRAIN_CFG

    def split(self, df: pd.DataFrame,
              method: str = "auto",
              random_state: int = None) -> Dict[str, pd.DataFrame]:
        """
        划分数据集

        Args:
            df: 特征 DataFrame（必须包含 cell_id 列）
            method: "leave_one_cell_out" | "random_cell" | "auto"

        Returns:
            {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
        """
        random_state = random_state or self.cfg.seed
        cells = df['cell_id'].unique()
        n_cells = len(cells)

        if method == "auto":
            method = (
                "leave_one_cell_out"
                if self.cfg.leave_one_cell_out and n_cells < 10
                else "random_cell"
            )

        logger.info(f"  数据划分方法: {method}, 共 {n_cells} 个电芯")

        if method == "leave_one_cell_out":
            return self._leave_one_cell_out(df, cells, random_state)
        else:
            return self._random_cell_split(df, cells, random_state)

    def _leave_one_cell_out(self, df: pd.DataFrame,
                            cells: np.ndarray,
                            random_state: int) -> Dict[str, pd.DataFrame]:
        """
        留一电芯交叉验证划分

        对于只有 3 个电芯的钠电数据:
          - 如果 n_cells == 3: train=2个cell, val=1个cell, test=该cell的30%
          - 如果 n_cells >= 4: train=n-2, val=1, test=1
        """
        np.random.seed(random_state)
        shuffled = np.random.permutation(cells)

        if len(cells) == 3:
            train_cells = shuffled[:2]
            val_cell = shuffled[2]
            test_cell = shuffled[2]  # 与 val 相同，但用不同循环范围
            logger.info(f"  LOCO 划分 (n=3): train={train_cells.tolist()}, val/test={val_cell}")
        elif len(cells) >= 4:
            train_cells = shuffled[:-2]
            val_cell = shuffled[-2]
            test_cell = shuffled[-1]
            logger.info(f"  LOCO 划分: train={train_cells.tolist()}, "
                       f"val={val_cell}, test={test_cell}")
        else:
            # n_cells <= 2: 按循环 70/15/15 划分（不够理想但可用）
            return self._cycle_level_split(df, cells)

        train_df = df[df['cell_id'].isin(train_cells)]
        val_df = df[df['cell_id'] == val_cell]
        test_df = df[df['cell_id'] == test_cell]
        # 留一电芯策略下，val/test 必须来自不同电芯，避免 early stopping 泄露
        if val_cell == test_cell:
            remaining = [c for c in cells if c not in {val_cell, test_cell}]
            if remaining:
                test_cell = remaining[0]
            else:
                cell_data = df[df['cell_id'] == val_cell].sort_values('cycle_index')
                n = len(cell_data)
                val_df = cell_data.iloc[:int(n * 0.5)]
                test_df = cell_data.iloc[int(n * 0.5):]

        return {'train': train_df, 'val': val_df, 'test': test_df}

    def _random_cell_split(self, df: pd.DataFrame,
                           cells: np.ndarray,
                           random_state: int) -> Dict[str, pd.DataFrame]:
        """按电芯随机划分"""
        np.random.seed(random_state)
        shuffled = np.random.permutation(cells)

        n_train = int(len(cells) * self.cfg.train_ratio)
        n_val = int(len(cells) * self.cfg.val_ratio)

        train_cells = shuffled[:n_train]
        val_cells = shuffled[n_train:n_train + n_val]
        test_cells = shuffled[n_train + n_val:]

        logger.info(f"  按电芯划分: train={len(train_cells)}, "
                   f"val={len(val_cells)}, test={len(test_cells)} 个电芯")

        return {
            'train': df[df['cell_id'].isin(train_cells)],
            'val':   df[df['cell_id'].isin(val_cells)],
            'test':  df[df['cell_id'].isin(test_cells)],
        }

    def _cycle_level_split(self, df: pd.DataFrame,
                           cells: np.ndarray) -> Dict[str, pd.DataFrame]:
        """按循环比例划分（最后的选择）"""
        logger.warning("  ⚠️ 电芯数量不足，使用循环级划分（可能泄漏）")
        df = df.sort_values(['cell_id', 'cycle_index'])
        n = len(df)
        n_train = int(n * self.cfg.train_ratio)
        n_val = int(n * self.cfg.val_ratio)
        return {
            'train': df.iloc[:n_train],
            'val':   df.iloc[n_train:n_train + n_val],
            'test':  df.iloc[n_train + n_val:],
        }

    # ── 化学体系感知划分 ─────────────────────────────────
    def split_by_chemistry(self, df: pd.DataFrame,
                           target_chemistry: str = "sodium-ion",
                           random_state: int = None) -> Dict[str, any]:
        """
        跨化学体系的迁移学习数据划分。

        核心原则:
          1. 目标域 (target, 如 sodium-ion) 的全部电芯用于 train/val/test —
             绝不让目标域数据混入源域预训练集
          2. 源域 (source, 如 lithium-ion) 的全部电芯用于 pretrain —
             作为迁移学习的预训练语料
          3. 源域数据量 >> 目标域时，源域内部做 train/val 划分用于
             预训练阶段的 early stopping

        Args:
            df: 特征 DataFrame，必须包含 'cell_id' 和 'chemistry' 列
            target_chemistry: 目标化学体系标识
            random_state: 随机种子

        Returns:
            {
                'pretrain': DataFrame | None,   # 源域预训练数据 (源域全部电芯)
                'train': DataFrame,             # 目标域训练集
                'val': DataFrame,               # 目标域验证集
                'test': DataFrame,              # 目标域测试集
                'source_chemistry': str,        # 源域化学体系名 (如 "lithium-ion")
                'target_chemistry': str,        # 目标域化学体系名
                'source_cells': List[str],      # 源域电芯 ID 列表
                'target_cells': List[str],      # 目标域电芯 ID 列表
                'is_cross_chemistry': bool,     # 是否为跨化学体系迁移学习
            }
        """
        random_state = random_state or self.cfg.seed

        if 'chemistry' not in df.columns:
            logger.warning("  DataFrame 缺少 'chemistry' 列，回退到普通划分")
            splits = self.split(df, random_state=random_state)
            splits['pretrain'] = None
            splits['source_chemistry'] = None
            splits['target_chemistry'] = target_chemistry
            splits['source_cells'] = []
            splits['target_cells'] = list(df['cell_id'].unique())
            splits['is_cross_chemistry'] = False
            return splits

        # 1. 按化学体系分拆电芯
        chemistries = df['chemistry'].unique()
        logger.info(f"  检测到化学体系: {chemistries.tolist()}")

        target_mask = df['chemistry'] == target_chemistry
        target_df = df[target_mask].copy()
        source_df = df[~target_mask].copy()

        target_cells = target_df['cell_id'].unique().tolist() if len(target_df) > 0 else []
        source_cells = source_df['cell_id'].unique().tolist() if len(source_df) > 0 else []

        logger.info(f"  目标域 ({target_chemistry}): {len(target_cells)} 个电芯 — {target_cells}")
        logger.info(f"  源域: {len(source_cells)} 个电芯")

        is_cross = len(source_cells) > 0 and len(target_cells) > 0

        if not is_cross:
            # 只有一种化学体系 → 普通划分
            logger.info("  单一化学体系，使用普通划分")
            splits = self.split(df, random_state=random_state)
            splits['pretrain'] = None
            splits['source_chemistry'] = None
            splits['target_chemistry'] = target_chemistry
            splits['source_cells'] = []
            splits['target_cells'] = target_cells if target_cells else source_cells
            splits['is_cross_chemistry'] = False
            return splits

        # 2. 源域: 全部用于预训练（内部做 train/val 分以支持 early stopping）
        source_chemistry_name = source_df['chemistry'].iloc[0]
        if len(source_cells) >= 4:
            np.random.seed(random_state)
            s_shuffled = np.random.permutation(source_cells)
            n_s_train = int(len(source_cells) * 0.85)
            s_train_cells = s_shuffled[:n_s_train]
            s_val_cells = s_shuffled[n_s_train:]
            pretrain_df = source_df[source_df['cell_id'].isin(s_train_cells)]
            pretrain_val_df = source_df[source_df['cell_id'].isin(s_val_cells)]
            logger.info(f"  源域预训练: train={len(s_train_cells)} 电芯, val={len(s_val_cells)} 电芯")
        else:
            pretrain_df = source_df
            pretrain_val_df = None
            logger.info(f"  源域预训练: {len(source_cells)} 电芯 (无独立验证集)")

        # 3. 目标域: LOCO 划分 (钠电只有 3 个电芯，必须谨慎)
        target_splits = self._leave_one_cell_out(target_df, np.array(target_cells), random_state)
        # 如果 LOCO 返回的 train 里有目标域之外的 cell（不可能发生，但安全起见）
        for key in ['train', 'val', 'test']:
            if key in target_splits:
                actual_cells = target_splits[key]['cell_id'].unique().tolist()
                foreign = [c for c in actual_cells if c not in target_cells]
                if foreign:
                    logger.warning(f"  ⚠️ 目标域 {key} 集混入了非目标电芯: {foreign}，已清除")
                    target_splits[key] = target_splits[key][
                        target_splits[key]['cell_id'].isin(target_cells)
                    ]

        logger.info(f"  目标域划分: train={len(target_splits['train']['cell_id'].unique())} 电芯, "
                   f"val={len(target_splits['val']['cell_id'].unique())} 电芯, "
                   f"test={len(target_splits['test']['cell_id'].unique())} 电芯")

        return {
            'pretrain': pretrain_df,
            'pretrain_val': pretrain_val_df,
            'train': target_splits['train'],
            'val': target_splits['val'],
            'test': target_splits['test'],
            'source_chemistry': source_chemistry_name,
            'target_chemistry': target_chemistry,
            'source_cells': source_cells,
            'target_cells': target_cells,
            'is_cross_chemistry': True,
        }


# ============================================================
# Stage 5: 数据管线编排器
# ============================================================

class SOHDataPipeline:
    """
    SOH 数据管线 — 统一编排所有处理阶段

    用法:
        pipeline = SOHDataPipeline()
        processed = pipeline.run(cells_data)
        # processed 包含: train/val/test DataFrame + scaler + 质量报告
    """

    def __init__(self):
        self.extractor = CycleFeatureExtractor()
        self.cleaner = DataCleaner()
        self.seq_builder = SequenceBuilder()
        self.splitter = DataSplitter()
        self.scalers: Dict[str, StandardScaler] = {}
        self.quality_report: Dict = {}

    def run(self, cells: List[CellDegradationData],
            save: bool = True,
            chemistry_aware: bool = False,
            target_chemistry: str = "sodium-ion") -> Dict[str, any]:
        """
        执行完整数据管线

        Args:
            cells: CellDegradationData 列表
            save: 是否保存中间产物到磁盘
            chemistry_aware: 是否启用化学体系感知模式（迁移学习场景）
            target_chemistry: 目标化学体系标识（仅在 chemistry_aware=True 时生效）

        Returns:
            普通模式:
            {
                'feature_df': pd.DataFrame,
                'splits': {'train': df, ...},
                'sequences': {'train': (X,y), ...},
                'scalers': {'X': scaler, 'y': scaler},
                'quality_report': dict,
                'feature_cols': list,
                'is_cross_chemistry': bool,
            }
            迁移学习模式 (chemistry_aware=True 且检测到多化学体系):
            {
                ... 以上所有字段 ...
                'pretrain_sequences': {'train': (X,y), 'val': (X,y)|None},
                'source_chemistry': str,
                'target_chemistry': str,
                'source_cells': List[str],
                'target_cells': List[str],
                'is_cross_chemistry': True,
            }
        """
        logger.info(f"━━━ SOH 数据管线启动 ─ {len(cells)} 个电芯 ━━━")
        if chemistry_aware:
            logger.info(f"  化学体系感知模式: target={target_chemistry}")

        # Stage 1: 特征提取
        logger.info("[Stage 1/5] 循环级特征提取...")
        all_features = []
        chemistry_counts = {}
        for cell in cells:
            try:
                cell_df = self.extractor.extract(cell)
                all_features.append(cell_df)
                chem = cell.chemistry
                chemistry_counts[chem] = chemistry_counts.get(chem, 0) + 1
                logger.info(f"  ✓ {cell.cell_id} [{chem}]: {len(cell_df)} 个循环")
            except Exception as e:
                logger.error(f"  ✗ {cell.cell_id}: {e}")

        if not all_features:
            raise RuntimeError("没有成功提取任何电芯的特征！")

        feature_df = pd.concat(all_features, ignore_index=True)
        logger.info(f"  合并特征表: {feature_df.shape}")
        logger.info(f"  化学体系分布: {chemistry_counts}")
        foreign = feature_df.loc[feature_df['chemistry'] != target_chemistry, 'cell_id'].unique().tolist()
        if not chemistry_aware and foreign:
            raise ValueError(f"纯钠电管线混入非钠电电芯: {foreign}")

        # Stage 2: 数据清洗
        logger.info("[Stage 2/5] 数据清洗与质量检查...")
        clean_df = self.cleaner.clean(feature_df)
        self.quality_report = self.cleaner.generate_report()
        logger.info(f"  清洗后: {clean_df.shape}, "
                   f"NaN 数: {clean_df.isnull().sum().sum()}")

        # 判断是否触发跨化学体系模式
        has_multiple_chem = (
            chemistry_aware
            and 'chemistry' in clean_df.columns
            and clean_df['chemistry'].nunique() > 1
        )

        # Stage 3: 数据划分
        logger.info("[Stage 3/5] 训练/验证/测试集划分...")
        if has_multiple_chem:
            logger.info("  检测到多化学体系 → 启用跨域划分")
            splits = self.splitter.split_by_chemistry(
                clean_df, target_chemistry=target_chemistry
            )
        else:
            if chemistry_aware:
                logger.info("  单一化学体系 → 使用普通划分")
            splits = self.splitter.split(clean_df)
            splits['pretrain'] = None
            splits['pretrain_val'] = None
            splits['source_chemistry'] = None
            splits['target_chemistry'] = target_chemistry
            splits['source_cells'] = []
            splits['target_cells'] = []
            splits['is_cross_chemistry'] = False

        is_cross = splits.get('is_cross_chemistry', False)

        # Stage 4: 标准化
        logger.info("[Stage 4/5] 特征标准化...")
        if is_cross and splits.get('pretrain') is not None:
            # 迁移学习模式: 用源域 (数据量大) 拟合 scaler，保证特征空间一致性
            scaled_splits = self._fit_scale_cross_domain(splits, clean_df)
        else:
            scaled_splits = self._fit_scale(splits, clean_df)

        # Stage 5: 序列构建
        logger.info("[Stage 5/5] 序列窗口构建...")
        feature_cols = [c for c in clean_df.columns
                       if c not in ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                                    'soh_raw', 'soh_jump_flag']
                       and clean_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        sequences = {}
        for split_name, sdf in scaled_splits.items():
            # 跳过 pretrain DataFrame（它们在 scaled_splits 中以 'pretrain'/'pretrain_val' 键存在）
            if split_name in ('pretrain', 'pretrain_val') or sdf is None:
                continue
            X, y, _, _ = self.seq_builder.build_sequences(
                sdf, feature_cols=feature_cols
            )
            sequences[split_name] = (X, y)
            logger.info(f"  {split_name}: X{X.shape}, y{y.shape}")

        # 迁移学习模式额外构建源域序列
        pretrain_sequences = None
        if is_cross and splits.get('pretrain') is not None:
            logger.info("  构建源域预训练序列...")
            pretrain_sequences = {}
            for key in ['pretrain', 'pretrain_val']:
                sdf = scaled_splits.get(key)
                if sdf is not None and len(sdf) > 0:
                    X, y, _, _ = self.seq_builder.build_sequences(
                        sdf, feature_cols=feature_cols
                    )
                    # 统一键名：pretrain → train, pretrain_val → val
                    out_key = 'train' if key == 'pretrain' else 'val'
                    pretrain_sequences[out_key] = (X, y)
                    logger.info(f"  source_{out_key}: X{X.shape}, y{y.shape}")

        # 持久化
        if save:
            self._save(clean_df,
                      {k: v for k, v in splits.items()
                       if k in ('train', 'val', 'test') and v is not None},
                      feature_cols)

        result = {
            'feature_df': clean_df,
            'splits': splits,
            'sequences': sequences,
            'scalers': self.scalers,
            'quality_report': self.quality_report,
            'feature_cols': feature_cols,
            'is_cross_chemistry': is_cross,
            'source_chemistry': splits.get('source_chemistry'),
            'target_chemistry': splits.get('target_chemistry'),
            'source_cells': splits.get('source_cells', []),
            'target_cells': splits.get('target_cells', []),
        }

        if is_cross and pretrain_sequences:
            result['pretrain_sequences'] = pretrain_sequences

        logger.info("━━━ SOH 数据管线完成 ━━━")
        if is_cross:
            logger.info(f"  跨域迁移学习: 源域={splits.get('source_chemistry')} → 目标域={target_chemistry}")
        return result

    def _fit_scale(self, splits: Dict[str, pd.DataFrame],
                   full_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        拟合并应用特征标准化。

        ⚠️ 设计原则：
          - X (特征) → RobustScaler 标准化（去量纲，抑制异常值）
          - y (SOH)  → 不做任何缩放！SOH 物理定义域 [0, 1]，天然归一化
                       对 y 做 StandardScaler 会导致模型预测缩放域的值，
                       评估时 RMSE 失去物理意义，且推理部署必须耦合 scaler。
        """
        target_col = FEATURE_CFG.target_col

        # 确定需要标准化的特征列（排除目标列和元数据列）
        exclude = ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                   'soh_raw', 'soh_jump_flag', target_col]
        feature_cols = [c for c in full_df.columns
                       if c not in exclude and full_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        # X scaler: 对特征列做鲁棒标准化
        x_scaler = RobustScaler(quantile_range=(5, 95))
        x_scaler.fit(splits['train'][feature_cols].values)
        self.scalers['X'] = x_scaler
        self.scalers['y'] = None  # 不再缩放 y

        # 变换所有划分：X 标准化，y 保持原值
        scaled = {}
        for split_name, sdf in splits.items():
            if not isinstance(sdf, pd.DataFrame):
                continue
            sdf_scaled = sdf.copy()
            sdf_scaled[feature_cols] = x_scaler.transform(sdf[feature_cols].values)
            # y 不缩放：SOH 保留真实的 [0, 1] 物理值
            scaled[split_name] = sdf_scaled

        logger.info(f"  特征标准化完成: {len(feature_cols)} 个特征列已缩放, "
                    f"SOH 保持原始物理域 [0, 1]")
        return scaled

    def _fit_scale_cross_domain(self, splits: Dict[str, any],
                                full_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        跨化学体系标准化 — 用源域数据拟合 scaler，统一变换源域和目标域。

        设计原则:
          - 源域 (lithium-ion, 数据量大) 拟合 RobustScaler → 提供鲁棒的
            特征分布估计
          - 同一 scaler 变换目标域 (sodium-ion) 数据 → 保证特征空间一致
          - SOH 不缩放，保持物理域 [0, 1]
          - 如果源域不可用，回退到目标域 train 集拟合
        """
        target_col = FEATURE_CFG.target_col
        exclude = ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                   'soh_raw', 'soh_jump_flag', target_col]
        feature_cols = [c for c in full_df.columns
                       if c not in exclude and full_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        # 确定 scaler 拟合数据: 优先源域 (数据量大，分布更广)
        fit_df = None
        if splits.get('pretrain') is not None and len(splits['pretrain']) > 0:
            fit_df = splits['pretrain']
            logger.info(f"  使用源域数据拟合 scaler (n={len(fit_df)})")
        elif splits.get('train') is not None and len(splits['train']) > 0:
            fit_df = splits['train']
            logger.info(f"  源域不可用，回退到目标域 train 拟合 scaler (n={len(fit_df)})")
        else:
            # 极端情况: 用全部数据
            fit_df = full_df
            logger.warning(f"  ⚠️ 无标准拟合数据，使用全量数据 (n={len(fit_df)})")

        x_scaler = RobustScaler(quantile_range=(5, 95))
        x_scaler.fit(fit_df[feature_cols].values)
        self.scalers['X'] = x_scaler
        self.scalers['y'] = None

        # 变换所有 split（源域 + 目标域）
        scaled = {}
        all_split_keys = ['pretrain', 'pretrain_val', 'train', 'val', 'test']
        for split_name in all_split_keys:
            sdf = splits.get(split_name)
            if sdf is None or len(sdf) == 0:
                scaled[split_name] = None
                continue
            sdf_scaled = sdf.copy()
            sdf_scaled[feature_cols] = x_scaler.transform(sdf[feature_cols].values)
            scaled[split_name] = sdf_scaled

        logger.info(f"  跨域标准化完成: scaler 基于源域拟合 → 统一变换源域+目标域 "
                    f"({len(feature_cols)} 个特征列)")
        return scaled

    def _save(self, feature_df: pd.DataFrame,
              splits: Dict[str, pd.DataFrame],
              feature_cols: List[str]):
        """持久化管线产物"""
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(WEIGHTS_DIR), exist_ok=True)

        # 保存特征表
        feature_df.to_parquet(
            os.path.join(PROCESSED_DATA_DIR, 'feature_table.parquet'),
            index=False
        )

        # 保存划分
        for split_name, sdf in splits.items():
            sdf.to_parquet(
                os.path.join(PROCESSED_DATA_DIR, f'{split_name}.parquet'),
                index=False
            )

        # 保存标准化器
        joblib.dump(self.scalers, os.path.join(WEIGHTS_DIR, 'soh_scalers.pkl'))

        # 保存质量报告（处理 numpy 类型）
        report_serializable = _make_json_serializable(self.quality_report)
        with open(os.path.join(PROCESSED_DATA_DIR, 'quality_report.json'), 'w') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)

        # 保存特征列清单
        with open(os.path.join(PROCESSED_DATA_DIR, 'feature_columns.json'), 'w') as f:
            json.dump(list(feature_cols), f, indent=2, ensure_ascii=False)

        logger.info(f"  数据已持久化至: {PROCESSED_DATA_DIR}")
        logger.info(f"  标准化器已保存至: {os.path.join(WEIGHTS_DIR, 'soh_scalers.pkl')}")


# ============================================================
# 便捷函数
# ============================================================

def create_synthetic_test_data(n_cells: int = 5,
                               n_cycles: int = 500,
                               seed: int = 42) -> List[CellDegradationData]:
    """
    生成合成测试数据 — 用于开发阶段无真实数据时的管线调试

    合成数据模拟钠离子电池老化曲线:
      - 前 200 循环: 缓慢线性衰减 (~0.01% SOH / 循环)
      - 200~400 循环: 加速衰减 (~0.03% SOH / 循环, 拐点)
      - 400+ 循环: 快速衰减 (~0.08% SOH / 循环)

    每个电芯的衰减速度略有差异（模拟 cell-to-cell variability）
    """
    np.random.seed(seed)
    cells = []

    for cell_idx in range(n_cells):
        cell_id = f"SYN-Na-{cell_idx+1:03d}"
        cycles = []

        # 电芯特异性参数（引入差异）
        base_decay = 0.0001 + np.random.uniform(-0.00003, 0.00003)   # 基础衰减率
        knee_point = 200 + int(np.random.uniform(-50, 50))            # 拐点位置
        post_knee_factor = 3.0 + np.random.uniform(-0.5, 0.5)        # 拐点后加速倍数
        temp_variation = 25.0 + np.random.uniform(-5, 5)              # 温度差异
        c_rate = 0.5 + np.random.choice([0, 0.5, 1.0])                # 倍率差异

        current_soh = 1.0  # 起始 100%
        initial_cap = 50.0 + np.random.uniform(-2, 2)  # 标称容量 ~50Ah
        current_cap = initial_cap

        for cyc in range(n_cycles):
            # 模拟衰减速率变化
            if cyc < knee_point:
                decay_rate = base_decay
            elif cyc < knee_point + 150:
                # 拐点过渡区
                progress = (cyc - knee_point) / 150
                decay_rate = base_decay * (1 + (post_knee_factor - 1) * progress)
            else:
                decay_rate = base_decay * post_knee_factor

            # 添加随机噪声
            noise = np.random.normal(0, base_decay * 0.3)
            current_soh -= (decay_rate + noise)
            current_soh = max(0.5, current_soh)

            current_cap = initial_cap * current_soh * (1 + np.random.normal(0, 0.001))

            # 库仑效率（随老化逐渐降低）
            ce = 0.998 - (1 - current_soh) * 0.02 + np.random.normal(0, 0.0005)

            # 内阻（随老化逐渐增长）
            r_base = 0.003  # 初始内阻 3mΩ
            r_current = r_base * (1 + (1 - current_soh) * 20) + np.random.normal(0, 0.0001)

            cycle = CycleData(
                cycle_index=cyc + 1,
                charge_capacity_ah=current_cap * ce,
                discharge_capacity_ah=current_cap,
                coulombic_efficiency=ce,
                temperature_c=temp_variation + np.random.normal(0, 1),
                c_rate_charge=c_rate,
                c_rate_discharge=c_rate,
                rest_time_h=max(0.1, 1.0 + np.random.normal(0, 0.2)),
                dc_resistance_ohm=r_current if cyc % 50 == 0 else None,
                metadata={
                    'soc_min': 0.1,
                    'soc_max': 0.9,
                    'soc_mean': 0.5,
                    'test_cell': True,
                }
            )
            cycles.append(cycle)

        cell_data = CellDegradationData(
            cell_id=cell_id,
            chemistry="sodium-ion",
            nominal_capacity_ah=initial_cap,
            cycles=cycles,
            metadata={'synthetic': True, 'seed': seed}
        )
        cells.append(cell_data)

    logger.info(f"  生成 {len(cells)} 个合成电芯，每个 {n_cycles} 个循环")
    return cells
