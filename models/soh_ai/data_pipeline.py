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
    FEATURE_CFG, DQ_CFG, TRAIN_CFG,
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

        # --- A. 状态特征（来自当前循环） ---
        soh_series = cell.soh_series  # shape: (n_cycles,)
        for i, cycle in enumerate(cell.cycles):
            features.setdefault('soh', [None] * n_cycles)[i] = soh_series[i]
            features.setdefault('cycle_index', [None] * n_cycles)[i] = cycle.cycle_index
            features.setdefault('cumulative_ah_throughput', [None] * n_cycles)[i] = (
                sum(c.discharge_capacity_ah for c in cell.cycles[:i+1])
            )
            features.setdefault('internal_resistance', [None] * n_cycles)[i] = (
                cycle.dc_resistance_ohm
            )
            features.setdefault('coulombic_efficiency', [None] * n_cycles)[i] = (
                cycle.coulombic_efficiency
            )

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
        result_df.insert(0, 'cell_id', cell.cell_id)
        result_df.insert(1, 'chemistry', cell.chemistry)

        return result_df

    def _diff(self, arr: np.ndarray, lag: int = 1) -> np.ndarray:
        """计算滞后差分（前面填充 NaN）"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) > lag:
            result[lag:] = arr[lag:] - arr[:-lag]
        return result

    def _rolling_mean(self, arr: np.ndarray, window: int = 5) -> np.ndarray:
        """滚动均值（前面用较小的窗口）"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = np.nanmean(arr[start:i+1])
        return result

    def _rolling_decay_rate(self, arr: np.ndarray, window: int = 10) -> np.ndarray:
        """
        计算滚动指数衰减率
        对 log(SOH) vs cycle 做线性拟合，返回斜率
        """
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window, len(arr)):
            y = arr[max(0, i-window):i+1]
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
        features = {}
        # 这些特征需要原始充放电曲线的 V-Q 数据
        # 如果没有原始曲线数据，返回空（后续由 data_quality 模块处理 NaN）
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

        # Step 7: 前向/后向填充剩余缺失值
        # 按电芯分组填充（保持电芯独立性）
        if 'cell_id' in df.columns:
            fill_cols = [c for c in df.columns if c not in ['cell_id', 'chemistry', 'soh_raw']]
            for col in fill_cols:
                if df[col].isnull().any():
                    df[col] = df.groupby('cell_id')[col].transform(
                        lambda x: x.ffill().bfill()
                    )
        else:
            df = df.ffill().bfill()

        # 最终检查：仍有 NaN 的列用 0 填充
        df = df.fillna(0)

        self.quality_report['final_shape'] = df.shape
        self.quality_report['final_nan_count'] = df.isnull().sum().sum()

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        异常值处理:
          1. 检测：偏离滚动中位数 > N * rolling_std 的点
          2. 处理：用滚动中位数替换（保留趋势，去除尖峰）
        """
        outlier_count = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['cell_id', 'chemistry', 'cycle_index']:
                continue
            if df[col].isnull().all():
                continue

            # 按电芯分组处理
            clean_col = df[col].copy()
            if 'cell_id' in df.columns:
                for cell_id in df['cell_id'].unique():
                    mask = df['cell_id'] == cell_id
                    vals = df.loc[mask, col]
                    if len(vals) < 10:
                        continue
                    rolling_med = vals.rolling(window=10, center=True, min_periods=3).median()
                    rolling_std = vals.rolling(window=10, center=True, min_periods=3).std()
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
        """Savitzky-Golay 风格平滑（保留趋势，去噪）"""
        if len(series) < window:
            return series
        # 使用中值滤波 + EMA 组合平滑
        median_smoothed = series.rolling(window=window, center=True, min_periods=1).median()
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
            # 自动选择数值列（排除非特征列）
            exclude = ['cell_id', 'chemistry', 'cycle_index', 'soh_raw', 'soh_jump_flag',
                       target_col]
            feature_cols = [c for c in df.columns
                           if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

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
            method = "leave_one_cell_out" if n_cells <= 5 else "random_cell"

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

        # val/test 可能相同，进一步按循环划分
        if val_cell == test_cell:
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
            save: bool = True) -> Dict[str, any]:
        """
        执行完整数据管线

        Args:
            cells: CellDegradationData 列表
            save: 是否保存中间产物到磁盘

        Returns:
            {
                'feature_df': pd.DataFrame,       # 全量特征表
                'splits': {'train': df, ...},     # 划分后的数据
                'sequences': {'train': (X,y), ...},  # 序列样本
                'scalers': {'X': scaler, 'y': scaler},  # 标准化器
                'quality_report': dict,            # 数据质量报告
                'feature_cols': list,              # 使用的特征列
            }
        """
        logger.info(f"━━━ SOH 数据管线启动 ─ {len(cells)} 个电芯 ━━━")

        # Stage 1: 特征提取
        logger.info("[Stage 1/5] 循环级特征提取...")
        all_features = []
        for cell in cells:
            try:
                cell_df = self.extractor.extract(cell)
                all_features.append(cell_df)
                logger.info(f"  ✓ {cell.cell_id}: {len(cell_df)} 个循环")
            except Exception as e:
                logger.error(f"  ✗ {cell.cell_id}: {e}")

        if not all_features:
            raise RuntimeError("没有成功提取任何电芯的特征！")

        feature_df = pd.concat(all_features, ignore_index=True)
        logger.info(f"  合并特征表: {feature_df.shape}")

        # Stage 2: 数据清洗
        logger.info("[Stage 2/5] 数据清洗与质量检查...")
        clean_df = self.cleaner.clean(feature_df)
        self.quality_report = self.cleaner.generate_report()
        logger.info(f"  清洗后: {clean_df.shape}, "
                   f"NaN 数: {clean_df.isnull().sum().sum()}")

        # Stage 3: 数据划分
        logger.info("[Stage 3/5] 训练/验证/测试集划分...")
        splits = self.splitter.split(clean_df)

        # Stage 4: 标准化
        logger.info("[Stage 4/5] 特征标准化...")
        scaled_splits = self._fit_scale(splits, clean_df)

        # Stage 5: 序列构建
        logger.info("[Stage 5/5] 序列窗口构建...")
        feature_cols = [c for c in clean_df.columns
                       if c not in ['cell_id', 'chemistry', 'cycle_index',
                                    'soh_raw', 'soh_jump_flag']
                       and clean_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        sequences = {}
        for split_name, sdf in scaled_splits.items():
            X, y, _, _ = self.seq_builder.build_sequences(
                sdf, feature_cols=feature_cols
            )
            sequences[split_name] = (X, y)
            logger.info(f"  {split_name}: X{X.shape}, y{y.shape}")

        # 持久化
        if save:
            self._save(clean_df, splits, feature_cols)

        result = {
            'feature_df': clean_df,
            'splits': splits,
            'sequences': sequences,
            'scalers': self.scalers,
            'quality_report': self.quality_report,
            'feature_cols': feature_cols,
        }

        logger.info("━━━ SOH 数据管线完成 ━━━")
        return result

    def _fit_scale(self, splits: Dict[str, pd.DataFrame],
                   full_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """拟合并应用特征标准化"""
        # 确定需要标准化的列
        exclude = ['cell_id', 'chemistry', 'cycle_index', 'soh_raw', 'soh_jump_flag']
        scale_cols = [c for c in full_df.columns
                     if c not in exclude and full_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        # 目标列单独 scalar（用于逆变换）
        target_col = FEATURE_CFG.target_col
        feature_cols = [c for c in scale_cols if c != target_col]

        # X scaler: 对所有特征列拟合
        x_scaler = RobustScaler(quantile_range=(5, 95))  # 对异常值鲁棒
        x_scaler.fit(splits['train'][feature_cols].values)

        # y scaler: 仅对目标列拟合
        y_scaler = StandardScaler()
        if target_col in splits['train'].columns:
            y_scaler.fit(splits['train'][[target_col]].values)

        self.scalers['X'] = x_scaler
        self.scalers['y'] = y_scaler

        # 变换所有划分
        scaled = {}
        for split_name, sdf in splits.items():
            sdf_scaled = sdf.copy()
            sdf_scaled[feature_cols] = x_scaler.transform(sdf[feature_cols].values)
            if target_col in sdf.columns:
                sdf_scaled[target_col] = y_scaler.transform(sdf[[target_col]].values)
            scaled[split_name] = sdf_scaled

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
