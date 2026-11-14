# models/soh_ai/data_pipeline.py
"""Causal feature extraction and cell-separated data preparation."""

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

# ============================================================

def _make_json_serializable(obj):
    """Recursively convert NumPy values to JSON-compatible Python values."""
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

# ============================================================

class CycleFeatureExtractor:
    """Extract cycle-level features from observed cell histories."""

    def __init__(self, cfg = None):
        self.cfg = cfg or FEATURE_CFG

    def extract(self, cell: CellDegradationData) -> pd.DataFrame:
        """Extract."""
        if not cell.cycles:
            raise ValueError(f"电芯 {cell.cell_id} 没有循环数据")

        n_cycles = len(cell.cycles)
        features = {}
        cumulative_ah = np.cumsum([max(0.0, float(c.discharge_capacity_ah or 0.0)) for c in cell.cycles])


        soh_series = cell.soh_series  # shape: (n_cycles,)
        for i, cycle in enumerate(cell.cycles):
            features.setdefault('soh', [None] * n_cycles)[i] = soh_series[i]
            features.setdefault('cycle_index', [None] * n_cycles)[i] = cycle.cycle_index
            features.setdefault('cumulative_ah_throughput', [None] * n_cycles)[i] = cumulative_ah[i]
            features.setdefault('internal_resistance', [None] * n_cycles)[i] = cycle.dc_resistance_ohm
            features.setdefault('coulombic_efficiency', [None] * n_cycles)[i] = cycle.coulombic_efficiency


        for i, cycle in enumerate(cell.cycles):
            features.setdefault('temperature_c', [None] * n_cycles)[i] = cycle.temperature_c
            features.setdefault('c_rate_charge', [None] * n_cycles)[i] = cycle.c_rate_charge
            features.setdefault('c_rate_discharge', [None] * n_cycles)[i] = cycle.c_rate_discharge
            features.setdefault('rest_time_h', [None] * n_cycles)[i] = cycle.rest_time_h


        for i, cycle in enumerate(cell.cycles):
            soc_info = self._extract_soc_window(cycle)
            features.setdefault('soc_min', [None] * n_cycles)[i] = soc_info.get('soc_min', 0)
            features.setdefault('soc_max', [None] * n_cycles)[i] = soc_info.get('soc_max', 0)
            features.setdefault('soc_mean', [None] * n_cycles)[i] = soc_info.get('soc_mean', 0)


        df_temp = pd.DataFrame(features)

        if 'soh' in df_temp.columns and df_temp['soh'].notna().any():
            soh_vals = df_temp['soh'].values

            df_temp['soh_diff_1'] = self._diff(soh_vals, lag=1)
            df_temp['soh_diff_3'] = self._diff(soh_vals, lag=3)
            df_temp['soh_diff_5'] = self._diff(soh_vals, lag=5)
            df_temp['soh_decay_rate'] = self._rolling_decay_rate(soh_vals, window=10)


            soh_diff1 = self._diff(soh_vals, lag=1)
            df_temp['capacity_fade_acceleration'] = self._diff(soh_diff1, lag=3)

        if 'internal_resistance' in df_temp.columns:
            r_vals = df_temp['internal_resistance'].values
            df_temp['r_diff_1'] = self._diff(r_vals, lag=1)

        if 'coulombic_efficiency' in df_temp.columns:
            ce_vals = df_temp['coulombic_efficiency'].values
            df_temp['ce_trend'] = self._rolling_mean(ce_vals, window=5)


        for i, cycle in enumerate(cell.cycles):
            echem = self._extract_echem_features(cycle)
            for echem_key in self.cfg.electrochemical_features:
                features.setdefault(echem_key, [None] * n_cycles)[i] = echem.get(echem_key, None)


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
        """Calculate lagged differences with NaN values before the first valid lag."""
        numeric = pd.to_numeric(pd.Series(arr), errors='coerce').to_numpy(dtype=float)
        result = np.full(numeric.shape, np.nan, dtype=float)
        if len(numeric) > lag:
            result[lag:] = numeric[lag:] - numeric[:-lag]
        return result

    def _rolling_mean(self, arr: np.ndarray, window: int = 5) -> np.ndarray:
        """Compute trailing means using smaller windows at the start of history."""
        numeric = pd.to_numeric(pd.Series(arr), errors='coerce').to_numpy(dtype=float)
        result = np.full(numeric.shape, np.nan, dtype=float)
        for i in range(len(numeric)):
            start = max(0, i - window + 1)
            window_values = numeric[start:i+1]
            if np.isfinite(window_values).any():
                result[i] = np.nanmean(window_values)
        return result

    def _rolling_decay_rate(self, arr: np.ndarray, window: int = 10) -> np.ndarray:
        """Estimate exponential decay from trailing observations."""
        numeric = pd.to_numeric(pd.Series(arr), errors='coerce').to_numpy(dtype=float)
        result = np.full(numeric.shape, np.nan, dtype=float)
        for i in range(window, len(numeric)):
            y = numeric[max(0, i-window):i+1]
            x = np.arange(len(y))

            valid = ~np.isnan(y) & (y > 0)
            if valid.sum() < 3:
                continue
            log_y = np.log(y[valid])
            x_valid = x[valid]
            if len(x_valid) >= 3:
                slope = np.polyfit(x_valid, log_y, 1)[0]
                result[i] = -slope
        return result

    def _extract_soc_window(self, cycle: CycleData) -> Dict[str, float]:
        """Extract soc window."""
        soc_info = {'soc_min': 0.0, 'soc_max': 1.0, 'soc_mean': 0.5}


        if cycle.metadata:
            for key in ['soc_min', 'soc_max', 'soc_window']:
                if key in cycle.metadata:
                    val = cycle.metadata[key]
                    if isinstance(val, (int, float)):
                        soc_info['soc_min'] = min(val, soc_info.get('soc_min', val))
                        soc_info['soc_max'] = max(val, soc_info.get('soc_max', val))


        if cycle.charge_curve is not None:
            q_max = cycle.charge_curve['Q'].max() if 'Q' in cycle.charge_curve.columns else 0
            if q_max > 0 and cycle.discharge_capacity_ah > 0:
                soc_info['soc_max'] = min(1.0, q_max / cycle.discharge_capacity_ah)

        soc_info['soc_mean'] = (soc_info['soc_min'] + soc_info['soc_max']) / 2
        return soc_info

    def _extract_echem_features(self, cycle: CycleData) -> Dict[str, float]:
        """Extract echem features."""
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

# ============================================================

class DataCleaner:
    """Validate features and apply causal signal cleaning."""

    def __init__(self, cfg = None):
        self.cfg = cfg or DQ_CFG
        self.quality_report = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean."""
        df = df.copy()
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'cells': df['cell_id'].nunique() if 'cell_id' in df.columns else 0,
        }


        df = df.dropna(axis=1, how='all')


        missing_rates = df.isnull().mean()
        cols_to_drop = missing_rates[missing_rates > self.cfg.max_missing_rate].index.tolist()
        if cols_to_drop:
            logger.info(f"  丢弃高缺失率列 ({len(cols_to_drop)}): {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        self.quality_report['dropped_columns'] = cols_to_drop


        if 'cell_id' in df.columns:
            cycle_counts = df.groupby('cell_id').size()
            bad_cells = cycle_counts[cycle_counts < self.cfg.min_cycles_per_cell].index.tolist()
            if bad_cells:
                logger.info(f"  丢弃循环数不足的电芯 ({len(bad_cells)}): {bad_cells}")
                df = df[~df['cell_id'].isin(bad_cells)]
            self.quality_report['rejected_cells'] = bad_cells


        df = self._handle_outliers(df)


        if 'soh' in df.columns and 'cell_id' in df.columns:
            df['soh_raw'] = df['soh'].copy()
            df['soh'] = df.groupby('cell_id')['soh'].transform(
                lambda x: self._smooth_soh(x, self.cfg.soh_smooth_window)
            )


        if 'soh' in df.columns:
            df = self._detect_capacity_jumps(df)


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
        """Handle outliers."""
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
        """Apply a causal median filter and EMA without inspecting future cycles."""
        if len(series) < window:
            return series

        median_smoothed = series.rolling(window=window, min_periods=1).median()
        # EMA: S_t = α*x_t + (1-α)*S_{t-1}
        alpha = 2.0 / (window + 1)
        ema = median_smoothed.copy()
        for i in range(1, len(ema)):
            if pd.notna(ema.iloc[i]):
                ema.iloc[i] = alpha * ema.iloc[i] + (1 - alpha) * ema.iloc[i-1]
        return ema

    def _detect_capacity_jumps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag abrupt capacity changes for data quality review."""
        df['soh_jump_flag'] = False
        if 'cell_id' in df.columns:
            for cell_id in df['cell_id'].unique():
                mask = df['cell_id'] == cell_id
                soh = df.loc[mask, 'soh'].values
                if len(soh) < 3:
                    continue
                soh_diff = np.diff(soh)
                jump_mask = np.abs(soh_diff) > self.cfg.max_capacity_jump

                jump_indices = np.where(jump_mask)[0]
                if len(jump_indices) > 0:
                    df_indices = df.loc[mask].index[jump_indices + 1]
                    df.loc[df_indices, 'soh_jump_flag'] = True
        self.quality_report['capacity_jumps'] = int(df['soh_jump_flag'].sum())
        return df

    def generate_report(self) -> Dict:
        """Generate report."""
        return self.quality_report


# ============================================================

# ============================================================

class SequenceBuilder:
    """Build supervised history/target windows from cycle feature tables."""

    def __init__(self, cfg = None):
        self.cfg = cfg or FEATURE_CFG

    def build_sequences(self, df: pd.DataFrame,
                        lookback: int = None,
                        horizon: int = None,
                        target_col: str = None,
                        feature_cols: List[str] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build sequences."""
        lookback = lookback or self.cfg.lookback_window
        horizon = horizon or self.cfg.prediction_horizon
        target_col = target_col or self.cfg.target_col

        if feature_cols is None:

            feature_cols = [c for c in FEATURE_CFG.all_features if c in df.columns]
            if not feature_cols:
                exclude = ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                           'soh_raw', 'soh_jump_flag', target_col]
                feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在 DataFrame 中。列: {df.columns.tolist()}")

        X_list, y_list, cell_list, idx_list = [], [], [], []


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

                X_window = cell_df.loc[i:i+lookback-1, feature_cols].values.astype(np.float32)

                y_window = cell_df.loc[i+lookback:i+lookback+horizon-1, target_col].values.astype(np.float32)


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

# ============================================================

class DataSplitter:
    """Partition cells for training and held-out evaluation."""

    def __init__(self, cfg = None):
        self.cfg = cfg or TRAIN_CFG

    def split(self, df: pd.DataFrame,
              method: str = "auto",
              random_state: int = None) -> Dict[str, pd.DataFrame]:
        """Split."""
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
        """Leave one cell out."""
        np.random.seed(random_state)
        shuffled = np.random.permutation(cells)

        if len(cells) == 3:
            train_cells = shuffled[:2]
            val_cell = shuffled[2]
            test_cell = shuffled[2]
            logger.info(f"  LOCO 划分 (n=3): train={train_cells.tolist()}, val/test={val_cell}")
        elif len(cells) >= 4:
            train_cells = shuffled[:-2]
            val_cell = shuffled[-2]
            test_cell = shuffled[-1]
            logger.info(f"  LOCO 划分: train={train_cells.tolist()}, "
                       f"val={val_cell}, test={test_cell}")
        else:

            return self._cycle_level_split(df, cells)

        train_df = df[df['cell_id'].isin(train_cells)]
        val_df = df[df['cell_id'] == val_cell]
        test_df = df[df['cell_id'] == test_cell]

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
        """Random cell split."""
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
        """Fallback chronological cycle split; cell separation is preferred."""
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


    def split_by_chemistry(self, df: pd.DataFrame,
                           target_chemistry: str = "sodium-ion",
                           random_state: int = None) -> Dict[str, any]:
        """Keep source and target chemistries separate for transfer experiments."""
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

            logger.info("  单一化学体系，使用普通划分")
            splits = self.split(df, random_state=random_state)
            splits['pretrain'] = None
            splits['source_chemistry'] = None
            splits['target_chemistry'] = target_chemistry
            splits['source_cells'] = []
            splits['target_cells'] = target_cells if target_cells else source_cells
            splits['is_cross_chemistry'] = False
            return splits


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


        target_splits = self._leave_one_cell_out(target_df, np.array(target_cells), random_state)

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

# ============================================================

class SOHDataPipeline:
    """Orchestrate feature extraction, validation, splitting, scaling, and serialization."""

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
        """Run."""
        logger.info(f"━━━ SOH 数据管线启动 ─ {len(cells)} 个电芯 ━━━")
        if chemistry_aware:
            logger.info(f"  化学体系感知模式: target={target_chemistry}")


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


        logger.info("[Stage 2/5] 数据清洗与质量检查...")
        clean_df = self.cleaner.clean(feature_df)
        self.quality_report = self.cleaner.generate_report()
        logger.info(f"  清洗后: {clean_df.shape}, "
                   f"NaN 数: {clean_df.isnull().sum().sum()}")


        has_multiple_chem = (
            chemistry_aware
            and 'chemistry' in clean_df.columns
            and clean_df['chemistry'].nunique() > 1
        )


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


        logger.info("[Stage 4/5] 特征标准化...")
        if is_cross and splits.get('pretrain') is not None:

            scaled_splits = self._fit_scale_cross_domain(splits, clean_df)
        else:
            scaled_splits = self._fit_scale(splits, clean_df)


        logger.info("[Stage 5/5] 序列窗口构建...")
        feature_cols = [c for c in clean_df.columns
                       if c not in ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                                    'soh_raw', 'soh_jump_flag']
                       and clean_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

        sequences = {}
        for split_name, sdf in scaled_splits.items():

            if split_name in ('pretrain', 'pretrain_val') or sdf is None:
                continue
            X, y, _, _ = self.seq_builder.build_sequences(
                sdf, feature_cols=feature_cols
            )
            sequences[split_name] = (X, y)
            logger.info(f"  {split_name}: X{X.shape}, y{y.shape}")


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

                    out_key = 'train' if key == 'pretrain' else 'val'
                    pretrain_sequences[out_key] = (X, y)
                    logger.info(f"  source_{out_key}: X{X.shape}, y{y.shape}")


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

    def load_processed(self,
                       processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
                       scaler_path: Optional[Union[str, Path]] = None,
                       lookback: Optional[int] = None,
                       horizon: Optional[int] = None) -> Dict[str, any]:
        """Restore audited Parquet features and their persisted scaling contract."""
        processed_dir = Path(processed_dir)
        scaler_path = Path(scaler_path or (Path(WEIGHTS_DIR) / 'soh_scalers.pkl'))

        required = [
            processed_dir / 'feature_table.parquet',
            processed_dir / 'feature_columns.json',
            *(processed_dir / f'{name}.parquet' for name in ('train', 'val', 'test')),
            scaler_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"已处理训练输入不完整，缺少: {missing}")

        feature_df = pd.read_parquet(processed_dir / 'feature_table.parquet')
        splits = {
            name: pd.read_parquet(processed_dir / f'{name}.parquet')
            for name in ('train', 'val', 'test')
        }
        with (processed_dir / 'feature_columns.json').open(encoding='utf-8') as source:
            feature_cols = json.load(source)

        if not isinstance(feature_cols, list) or not feature_cols:
            raise ValueError("feature_columns.json 必须包含非空特征列列表")
        missing_columns = sorted({
            column for column in feature_cols
            if any(column not in frame.columns for frame in splits.values())
        })
        if missing_columns:
            raise ValueError(f"已处理划分缺少特征列: {missing_columns}")
        if FEATURE_CFG.target_col not in feature_cols:
            raise ValueError(f"特征列必须包含当前 SOH 输入列 '{FEATURE_CFG.target_col}'")

        cell_sets = {
            name: set(frame['cell_id'].astype(str).unique())
            for name, frame in splits.items()
        }
        overlaps = (
            (cell_sets['train'] & cell_sets['val'])
            | (cell_sets['train'] & cell_sets['test'])
            | (cell_sets['val'] & cell_sets['test'])
        )
        if overlaps:
            raise ValueError(f"训练/验证/测试划分存在电芯重叠: {sorted(overlaps)}")

        for name, frame in splits.items():
            values = frame[feature_cols].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} 划分包含 NaN 或无穷值")

        scalers = joblib.load(scaler_path)
        if not isinstance(scalers, dict) or scalers.get('X') is None:
            raise ValueError(f"标准化器格式无效: {scaler_path}")
        scale_cols = [column for column in feature_cols if column != FEATURE_CFG.target_col]
        x_scaler = scalers['X']
        expected_features = getattr(x_scaler, 'n_features_in_', len(scale_cols))
        if expected_features != len(scale_cols):
            raise ValueError(
                f"标准化器需要 {expected_features} 个特征，当前处理数据为 {len(scale_cols)} 个"
            )

        scaled_splits = {}
        sequences = {}
        for name, frame in splits.items():
            scaled = frame.copy()
            scaled.loc[:, scale_cols] = x_scaler.transform(frame[scale_cols].values)
            scaled_splits[name] = scaled
            X, y, _, _ = self.seq_builder.build_sequences(
                scaled,
                lookback=lookback,
                horizon=horizon,
                feature_cols=feature_cols,
            )
            if len(X) == 0:
                raise ValueError(f"{name} 划分无法构建序列，请检查 lookback/horizon")
            sequences[name] = (X, y)

        quality_report = {}
        quality_path = processed_dir / 'quality_report.json'
        if quality_path.is_file():
            with quality_path.open(encoding='utf-8') as source:
                quality_report = json.load(source)

        self.scalers = scalers
        self.quality_report = quality_report
        logger.info(
            "已恢复处理数据: %d 行, train/val/test 电芯=%d/%d/%d",
            len(feature_df),
            len(cell_sets['train']), len(cell_sets['val']), len(cell_sets['test']),
        )
        return {
            'feature_df': feature_df,
            'splits': splits,
            'scaled_splits': scaled_splits,
            'sequences': sequences,
            'scalers': scalers,
            'quality_report': quality_report,
            'feature_cols': feature_cols,
            'is_cross_chemistry': False,
            'source_chemistry': None,
            'target_chemistry': 'sodium-ion',
            'source_cells': [],
            'target_cells': sorted(set().union(*cell_sets.values())),
        }

    def _fit_scale(self, splits: Dict[str, pd.DataFrame],
                   full_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Fit feature normalization on the training partition; preserve physical SOH units."""
        target_col = FEATURE_CFG.target_col


        exclude = ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                   'soh_raw', 'soh_jump_flag', target_col]
        feature_cols = [c for c in full_df.columns
                       if c not in exclude and full_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


        x_scaler = RobustScaler(quantile_range=(5, 95))
        x_scaler.fit(splits['train'][feature_cols].values)
        self.scalers['X'] = x_scaler
        self.scalers['y'] = None


        scaled = {}
        for split_name, sdf in splits.items():
            if not isinstance(sdf, pd.DataFrame):
                continue
            sdf_scaled = sdf.copy()
            sdf_scaled[feature_cols] = x_scaler.transform(sdf[feature_cols].values)

            scaled[split_name] = sdf_scaled

        logger.info(f"  特征标准化完成: {len(feature_cols)} 个特征列已缩放, "
                    f"SOH 保持原始物理域 [0, 1]")
        return scaled

    def _fit_scale_cross_domain(self, splits: Dict[str, any],
                                full_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Fit normalization on source-domain training data and reuse it in both domains."""
        target_col = FEATURE_CFG.target_col
        exclude = ['cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index',
                   'soh_raw', 'soh_jump_flag', target_col]
        feature_cols = [c for c in full_df.columns
                       if c not in exclude and full_df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


        fit_df = None
        if splits.get('pretrain') is not None and len(splits['pretrain']) > 0:
            fit_df = splits['pretrain']
            logger.info(f"  使用源域数据拟合 scaler (n={len(fit_df)})")
        elif splits.get('train') is not None and len(splits['train']) > 0:
            fit_df = splits['train']
            logger.info(f"  源域不可用，回退到目标域 train 拟合 scaler (n={len(fit_df)})")
        else:

            fit_df = full_df
            logger.warning(f"  ⚠️ 无标准拟合数据，使用全量数据 (n={len(fit_df)})")

        x_scaler = RobustScaler(quantile_range=(5, 95))
        x_scaler.fit(fit_df[feature_cols].values)
        self.scalers['X'] = x_scaler
        self.scalers['y'] = None


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
        """Save."""
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(WEIGHTS_DIR), exist_ok=True)


        feature_df.to_parquet(
            os.path.join(PROCESSED_DATA_DIR, 'feature_table.parquet'),
            index=False
        )


        for split_name, sdf in splits.items():
            sdf.to_parquet(
                os.path.join(PROCESSED_DATA_DIR, f'{split_name}.parquet'),
                index=False
            )


        joblib.dump(self.scalers, os.path.join(WEIGHTS_DIR, 'soh_scalers.pkl'))


        report_serializable = _make_json_serializable(self.quality_report)
        with open(os.path.join(PROCESSED_DATA_DIR, 'quality_report.json'), 'w') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)


        with open(os.path.join(PROCESSED_DATA_DIR, 'feature_columns.json'), 'w') as f:
            json.dump(list(feature_cols), f, indent=2, ensure_ascii=False)

        logger.info(f"  数据已持久化至: {PROCESSED_DATA_DIR}")
        logger.info(f"  标准化器已保存至: {os.path.join(WEIGHTS_DIR, 'soh_scalers.pkl')}")


# ============================================================

# ============================================================

def create_synthetic_test_data(n_cells: int = 5,
                               n_cycles: int = 500,
                               seed: int = 42) -> List[CellDegradationData]:
    """Generate explicitly synthetic fixtures for pipeline development."""
    np.random.seed(seed)
    cells = []

    for cell_idx in range(n_cells):
        cell_id = f"SYN-Na-{cell_idx+1:03d}"
        cycles = []


        base_decay = 0.0001 + np.random.uniform(-0.00003, 0.00003)
        knee_point = 200 + int(np.random.uniform(-50, 50))
        post_knee_factor = 3.0 + np.random.uniform(-0.5, 0.5)
        temp_variation = 25.0 + np.random.uniform(-5, 5)
        c_rate = 0.5 + np.random.choice([0, 0.5, 1.0])

        current_soh = 1.0
        initial_cap = 50.0 + np.random.uniform(-2, 2)
        current_cap = initial_cap

        for cyc in range(n_cycles):

            if cyc < knee_point:
                decay_rate = base_decay
            elif cyc < knee_point + 150:

                progress = (cyc - knee_point) / 150
                decay_rate = base_decay * (1 + (post_knee_factor - 1) * progress)
            else:
                decay_rate = base_decay * post_knee_factor


            noise = np.random.normal(0, base_decay * 0.3)
            current_soh -= (decay_rate + noise)
            current_soh = max(0.5, current_soh)

            current_cap = initial_cap * current_soh * (1 + np.random.normal(0, 0.001))


            ce = 0.998 - (1 - current_soh) * 0.02 + np.random.normal(0, 0.0005)


            r_base = 0.003
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
