# models/soh_ai/evaluate.py
"""
SOH AI 模型评估与可视化
==========================

功能:
  - 回归指标计算 (RMSE, MAE, R², MAPE, Max Error)
  - 预测 vs 实际对比图
  - 残差分析与正态性检验
  - SOH 轨迹推演可视化
  - 多模型横向对比

用法:
  from models.soh_ai.evaluate import ModelEvaluator

  evaluator = ModelEvaluator()
  metrics = evaluator.compute_metrics(y_true, y_pred)          # 数值指标
  evaluator.plot_predictions(y_true, y_pred, save_path=...)    # 预测对比图
  evaluator.plot_trajectory(history, future_pred, ...)         # 轨迹推演图

或命令行:
  python -m models.soh_ai.evaluate --model_dir models/weights/
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Matplotlib 后端兼容（无 GUI 服务器）
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合服务器
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

# 全局样式配置
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})
# 尝试使用中文字体（无则回退）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════


@dataclass
class RolloutScenario:
    """????????"""
    temperature_c: Optional[Union[float, List[float], np.ndarray]] = None
    c_rate_charge: Optional[Union[float, List[float], np.ndarray]] = None
    c_rate_discharge: Optional[Union[float, List[float], np.ndarray]] = None
    rest_time_h: Optional[Union[float, List[float], np.ndarray]] = None
    soc_min: Optional[Union[float, List[float], np.ndarray]] = None
    soc_max: Optional[Union[float, List[float], np.ndarray]] = None
    soc_mean: Optional[Union[float, List[float], np.ndarray]] = None
    cumulative_ah_throughput: Optional[Union[float, List[float], np.ndarray]] = None
    internal_resistance: Optional[Union[float, List[float], np.ndarray]] = None
    coulombic_efficiency: Optional[Union[float, List[float], np.ndarray]] = None

    @classmethod
    def from_any(cls, scenario_data=None):
        if scenario_data is None:
            return cls()
        if isinstance(scenario_data, cls):
            return scenario_data
        if isinstance(scenario_data, dict):
            return cls(**scenario_data)
        if isinstance(scenario_data, pd.DataFrame):
            return cls(**{col: scenario_data[col].to_numpy(dtype=np.float32) for col in scenario_data.columns})

        path_obj = Path(scenario_data)
        suffix = path_obj.suffix.lower()
        if suffix == '.json':
            with open(path_obj, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
            return cls(**payload)
        if suffix == '.parquet':
            df = pd.read_parquet(path_obj)
            return cls(**{col: df[col].to_numpy(dtype=np.float32) for col in df.columns})
        if suffix == '.npz':
            archive = np.load(path_obj, allow_pickle=True)
            payload = {key: archive[key] for key in archive.files}
            return cls(**payload)
        if suffix == '.npy':
            loaded = np.load(path_obj, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and loaded.dtype == object:
                return cls(**loaded.item())
            raise ValueError('npy ????????? dict ??')
        raise ValueError(f'??????????: {path_obj}')

    def as_step_arrays(self, future_steps: int) -> Dict[str, np.ndarray]:
        result = {}
        for key, value in self.__dict__.items():
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            result[key] = np.repeat(arr[0], future_steps) if arr.size == 1 else np.resize(arr, future_steps).astype(np.float32)
        return result


@dataclass
class CovariateRolloutConfig:
    """??????????"""
    strategy: str = "hybrid_default"
    scenario: Optional[RolloutScenario] = None
    nominal_capacity_ah: Optional[float] = None
    step_duration_h: float = 1.0
    delta_ah_per_step: Optional[float] = None
    resistance_growth_rate: float = 0.002
    ce_decay_rate: float = 0.00005
    temperature_drift: float = 0.0
    min_coulombic_efficiency: float = 0.85
    min_soh: float = 0.0
    max_internal_resistance: Optional[float] = None
    enforce_monotonic: bool = True
    max_soh_uplift_ratio: float = 1.01


class RegressionMetrics:
    """回归指标计算器"""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方根误差"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """平均绝对误差"""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray,
             eps: float = 1e-8) -> float:
        """平均绝对百分比误差 (%)"""
        mask = np.abs(y_true) > eps
        if mask.sum() == 0:
            return float('nan')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """决定系数 R²"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return float('nan')
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """最大绝对误差"""
        return float(np.max(np.abs(y_true - y_pred)))

    @staticmethod
    def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """解释方差得分"""
        var_resid = np.var(y_true - y_pred)
        var_total = np.var(y_true)
        if var_total == 0:
            return float('nan')
        return float(1 - var_resid / var_total)

    @classmethod
    def compute_all(cls, y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = "") -> Dict[str, float]:
        """
        计算全部回归指标。

        Args:
            y_true: 真实值, shape (n_samples,) 或 (n_samples, horizon)
            y_pred: 预测值, shape 同 y_true
            model_name: 模型名称标签

        Returns:
            dict: {指标名: 值}
        """
        # 确保扁平化
        yt = y_true.ravel()
        yp = y_pred.ravel()

        metrics = {
            'model': model_name or 'model',
            'RMSE': cls.rmse(yt, yp),
            'MAE': cls.mae(yt, yp),
            'R²': cls.r2_score(yt, yp),
            'MAPE_%': cls.mape(yt, yp),
            'Max_Error': cls.max_error(yt, yp),
            'Explained_Var': cls.explained_variance(yt, yp),
            'n_samples': len(yt),
        }
        return metrics


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

class Visualizer:
    """评估可视化 — 每个方法返回 Figure 对象（可保存、可嵌入）"""

    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "plots"

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or self.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_predictions(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         model_name: str = "Model",
                         title: str = None,
                         save: bool = False,
                         ) -> Figure:
        """
        预测值 vs 真实值散点图（含理想对角线）

        越接近对角线 = 模型越准。
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        yt = y_true.ravel()
        yp = y_pred.ravel()

        # 散点
        ax.scatter(yt, yp, alpha=0.4, s=8, c='#2b6cb0', edgecolors='none',
                  label=f'n={len(yt)}')

        # 理想对角线
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims[0] -= margin
        lims[1] += margin
        ax.plot(lims, lims, '--', color='#e53e3e', linewidth=1.5, label='Ideal')

        # 标注指标
        rmse = RegressionMetrics.rmse(yt, yp)
        r2 = RegressionMetrics.r2_score(yt, yp)
        ax.text(0.05, 0.93,
                f'RMSE = {rmse:.4f}\nR² = {r2:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('True SOH')
        ax.set_ylabel('Predicted SOH')
        ax.set_title(title or f'{model_name} — Predictions vs True')
        ax.legend(loc='lower right')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save:
            path = self.output_dir / f'{model_name}_predictions.png'
            fig.savefig(path)
            logger.info(f"  图表已保存: {path}")
        return fig

    def plot_residuals(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       model_name: str = "Model",
                       save: bool = False,
                       ) -> Figure:
        """
        残差分析图（残差分布 + Q-Q 检验）

        用于判断模型是否存在系统性偏差。
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        yt = y_true.ravel()
        yp = y_pred.ravel()
        residuals = yt - yp

        # 左: 残差散点图
        ax = axes[0]
        ax.scatter(yp, residuals, alpha=0.4, s=8, c='#2b6cb0', edgecolors='none')
        ax.axhline(y=0, color='#e53e3e', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Predicted SOH')
        ax.set_ylabel('Residual (True - Pred)')
        ax.set_title(f'{model_name} — Residuals')
        ax.grid(True, alpha=0.3)

        # 标注残差统计
        mean_r = np.mean(residuals)
        std_r = np.std(residuals)
        ax.text(0.05, 0.93,
                f'Mean = {mean_r:.4f}\nStd = {std_r:.4f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 右: 残差直方图
        ax = axes[1]
        ax.hist(residuals, bins=40, density=True, alpha=0.7, color='#2b6cb0',
               edgecolor='white', linewidth=0.5)
        # 叠加正态分布
        from scipy import stats as sp_stats
        x_range = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(x_range, sp_stats.norm.pdf(x_range, mean_r, std_r),
                '--', color='#e53e3e', linewidth=2, label=f'N({mean_r:.3f}, {std_r:.3f})')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name} — Residual Distribution')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if save:
            path = self.output_dir / f'{model_name}_residuals.png'
            fig.savefig(path)
            logger.info(f"  图表已保存: {path}")
        return fig

    def plot_trajectory(self,
                        cycle_indices: np.ndarray,
                        true_soh: np.ndarray,
                        pred_soh: np.ndarray = None,
                        future_pred: np.ndarray = None,
                        knee_point: int = None,
                        eol_threshold: float = 0.80,
                        cell_id: str = "",
                        model_name: str = "Model",
                        save: bool = False,
                        ) -> Figure:
        """
        SOH 老化轨迹图（历史 + 未来预测 + 拐点标记）

        Args:
            cycle_indices: 循环序号
            true_soh: 真实 SOH 序列
            pred_soh: 模型预测 SOH（与 true_soh 同长度）
            future_pred: 未来推演的 SOH 序列
            knee_point: 拐点位置（循环序号）
            eol_threshold: EOL SOH 阈值线
            cell_id: 电芯标识
            model_name: 模型名称
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        # 历史真实值
        ax.plot(cycle_indices, true_soh, '-', color='#2b6cb0', linewidth=1.5,
               label='True SOH', zorder=3)

        # 模型预测（拟合值）
        if pred_soh is not None and len(pred_soh) == len(cycle_indices):
            ax.plot(cycle_indices, pred_soh, '--', color='#3182ce', linewidth=1.2,
                   alpha=0.8, label=f'{model_name} (fit)', zorder=2)

        # 未来推演
        if future_pred is not None and len(future_pred) > 0:
            future_start = cycle_indices[-1]
            future_cycles = np.arange(future_start + 1,
                                     future_start + len(future_pred) + 1)
            ax.plot(future_cycles, future_pred, '--', color='#e53e3e',
                   linewidth=1.5, label=f'Predicted ({len(future_pred)} cycles)', zorder=4)

        # EOL 阈值线
        ax.axhline(y=eol_threshold, color='#d69e2e', linestyle=':',
                  linewidth=1.5, label=f'EOL ({eol_threshold:.0%})')

        # 拐点标记
        if knee_point is not None and 0 <= knee_point < len(cycle_indices):
            ax.axvline(x=cycle_indices[knee_point], color='#e53e3e',
                      linestyle='--', linewidth=1, alpha=0.6, label=f'Knee @ cycle {knee_point}')
            ax.scatter([cycle_indices[knee_point]], [true_soh[knee_point]],
                      c='#e53e3e', s=60, zorder=5)

        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('SOH')
        title = f'SOH Degradation Trajectory'
        if cell_id:
            title += f' — {cell_id}'
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        y_min = float(np.nanmin([np.nanmin(true_soh), eol_threshold]))
        y_max_candidates = [np.nanmax(true_soh)]
        if pred_soh is not None and len(pred_soh) > 0:
            y_max_candidates.append(np.nanmax(pred_soh))
        if future_pred is not None and len(future_pred) > 0:
            y_max_candidates.append(np.nanmax(future_pred))
        y_max = float(np.nanmax(y_max_candidates)) if y_max_candidates else 1.0
        ax.set_ylim(bottom=max(0.0, y_min - 0.05), top=min(1.05, max(0.9, y_max + 0.05)))

        fig.tight_layout()
        if save:
            fname = f'trajectory_{cell_id or model_name}.png'
            fig.savefig(self.output_dir / fname)
            logger.info(f"  图表已保存: {self.output_dir / fname}")
        return fig

    def plot_model_comparison(self,
                              metrics_list: List[Dict[str, float]],
                              metric_keys: List[str] = None,
                              save: bool = False,
                              ) -> Figure:
        """
        多模型指标柱状图对比。

        Args:
            metrics_list: [{'model': 'xgb', 'RMSE': 0.01, ...}, ...]
            metric_keys: 要展示的指标 ['RMSE', 'MAE', 'R²']
        """
        if metric_keys is None:
            metric_keys = ['RMSE', 'MAE', 'R²']

        df = pd.DataFrame(metrics_list)
        models = df['model'].tolist()
        n_models = len(models)
        n_metrics = len(metric_keys)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5))
        if n_metrics == 1:
            axes = [axes]

        colors = ['#2b6cb0', '#3182ce', '#63b3ed', '#e53e3e', '#d69e2e']

        for i, metric in enumerate(metric_keys):
            ax = axes[i]
            if metric not in df.columns:
                ax.text(0.5, 0.5, f'No data for\n{metric}', ha='center', va='center')
                continue

            values = df[metric].values
            bars = ax.bar(range(n_models), values, color=colors[:n_models],
                         edgecolor='white', linewidth=0.5)

            # 数值标签
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           f'{val:.4f}', ha='center', va='bottom', fontsize=8)

            ax.set_xticks(range(n_models))
            ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
            ax.set_title(metric, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Model Comparison', fontsize=14, y=1.02)
        fig.tight_layout()

        if save:
            path = self.output_dir / 'model_comparison.png'
            fig.savefig(path)
            logger.info(f"  图表已保存: {path}")
        return fig


# ═══════════════════════════════════════════════════════════════
# 评估编排器
# ═══════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    SOH 模型评估编排器

    整合指标计算 + 可视化，提供一站式评估接口。

    用法:
        evaluator = ModelEvaluator(output_dir='models/data/plots')
        results = evaluator.evaluate(ensemble, X_test, y_test)
        evaluator.print_summary(results)
    """

    def __init__(self, output_dir: str = None):
        self.vis = Visualizer(output_dir)
        self.metrics = RegressionMetrics()

    def evaluate(self,
                 ensemble_or_model,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 model_name: str = "Model",
                 save_plots: bool = True,
                 ) -> Dict[str, any]:
        """
        对单个模型或集成模型进行完整评估。

        Returns:
            dict: {
                'metrics': {...},
                'figures': [Figure, ...],
                'predictions': np.ndarray,
            }
        """
        # 预测
        if hasattr(ensemble_or_model, 'predict'):
            preds = ensemble_or_model.predict(X_test)
        else:
            # 裸 PyTorch 模型
            import torch
            device = next(ensemble_or_model.parameters()).device
            ensemble_or_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_test).to(device)
                preds = ensemble_or_model(X_t).cpu().numpy()

        # 如果是集成模型返回字典，取 ensemble 预测
        if isinstance(preds, dict):
            y_pred = preds.get('ensemble', list(preds.values())[0])
        else:
            y_pred = preds

        # 指标
        metrics = self.metrics.compute_all(y_test, y_pred, model_name)
        band_metrics = {}
        y_flat = y_test.ravel()
        for band_name, band_mask in {
            'high_soh': y_flat >= 0.9,
            'mid_soh': (y_flat < 0.9) & (y_flat >= 0.7),
            'low_soh': y_flat < 0.7,
        }.items():
            if band_mask.any():
                band_metrics[band_name] = self.metrics.compute_all(y_test[band_mask], y_pred[band_mask], f'{model_name}_{band_name}')

        # 图表
        figures = []
        if save_plots:
            figures.append(self.vis.plot_predictions(y_test, y_pred, model_name, save=True))
            figures.append(self.vis.plot_residuals(y_test, y_pred, model_name, save=True))

        return {
            'metrics': metrics,
            'band_metrics': band_metrics,
            'figures': figures,
            'predictions': y_pred,
        }

    def evaluate_all(self,
                     ensemble,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     save_plots: bool = True,
                     ) -> Dict[str, any]:
        """
        对集成模型的所有子模型分别评估 + 横向对比。
        """
        all_metrics = []

        # 分别评估
        preds = ensemble.predict(X_test)
        for name in ensemble.available_models:
            if name in preds:
                m = self.metrics.compute_all(y_test, preds[name], name)
                all_metrics.append(m)

        # 集成模型
        if 'ensemble' in preds:
            m = self.metrics.compute_all(y_test, preds['ensemble'], 'Ensemble')
            all_metrics.append(m)

        # 横向对比图
        comp_fig = None
        if save_plots and len(all_metrics) > 1:
            comp_fig = self.vis.plot_model_comparison(all_metrics, save=True)

        return {
            'per_model': all_metrics,
            'comparison_figure': comp_fig,
        }

    def _build_xgb_features(self, seq: np.ndarray) -> np.ndarray:
        """?? XGBoost ??????????"""
        last_step = seq[:, -1, :]
        mean_vals = seq.mean(axis=1)
        std_vals = seq.std(axis=1)
        min_vals = seq.min(axis=1)
        max_vals = seq.max(axis=1)
        trend = seq[:, -1, :] - seq[:, 0, :]
        return np.concatenate([last_step, mean_vals, std_vals, min_vals, max_vals, trend], axis=1)

    def _predict_next_soh(self, model, seq: np.ndarray, mask: np.ndarray = None) -> float:
        """????? SOH??? ensemble / PyTorch / XGBoost?"""
        if hasattr(model, 'available_models') and hasattr(model, 'predict'):
            pred = model.predict(seq[None, ...])
            if isinstance(pred, dict):
                pred = pred.get('ensemble', list(pred.values())[0])
            return float(np.asarray(pred).reshape(-1)[-1])

        if model.__class__.__name__ == 'XGBoostWrapper':
            pred = model.predict(self._build_xgb_features(seq))
            return float(np.asarray(pred).reshape(-1)[-1])

        if isinstance(model, torch.nn.Module):
            model.eval()
            device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
            X_t = torch.as_tensor(seq[None, ...], dtype=torch.float32, device=device)
            mask_t = None if mask is None else torch.as_tensor(mask[None, ...], dtype=torch.bool, device=device)
            with torch.no_grad():
                pred = model(X_t, mask_t) if mask_t is not None else model(X_t)
            return float(pred.detach().cpu().numpy().reshape(-1)[-1])

        if hasattr(model, 'predict'):
            pred = model.predict(seq[None, ...])
            if isinstance(pred, dict):
                pred = pred.get('ensemble', list(pred.values())[0])
            return float(np.asarray(pred).reshape(-1)[-1])

        raise TypeError('???????????????? SOH')

    @staticmethod
    def _select_rollout_value(value, step_idx: int, default_value: float):
        """???/?????? step_idx ????"""
        if value is None:
            return default_value
        if np.isscalar(value):
            return float(value)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(array) == 0:
            return default_value
        return float(array[min(step_idx, len(array) - 1)])

    @staticmethod
    def _load_rollout_scenario(scenario_data, future_steps: int) -> RolloutScenario:
        """??????????????????"""
        return RolloutScenario.from_any(scenario_data)

    @staticmethod
    def _apply_covariate_strategy(seq: np.ndarray,
                                  step_idx: int,
                                  pred_soh: float,
                                  rollout_cfg: CovariateRolloutConfig,
                                  scenario: Dict[str, np.ndarray]) -> np.ndarray:
        """????????????????"""
        from models.soh_ai.config import ACTUAL_FEATURE_COLUMNS
        feature_index = {name: idx for idx, name in enumerate(ACTUAL_FEATURE_COLUMNS)}

        next_step = seq[-1:, :].copy()
        current = seq[-1, :]
        previous = seq[-2, :] if seq.shape[0] > 1 else current
        strategy = (rollout_cfg.strategy or 'hybrid_default').lower()

        previous_soh = float(current[feature_index['soh']])
        capped_soh = float(np.clip(pred_soh, rollout_cfg.min_soh, 1.0))
        if rollout_cfg.enforce_monotonic:
            capped_soh = min(capped_soh, previous_soh * rollout_cfg.max_soh_uplift_ratio)
            capped_soh = min(capped_soh, previous_soh)
        soh_drop = max(previous_soh - capped_soh, 0.0)
        soh_rate = soh_drop / max(previous_soh, 1e-6)

        def choose(name: str, default_value: float) -> float:
            if name in scenario:
                return float(scenario[name][step_idx])
            if strategy == 'scenario_driven':
                return default_value
            if name == 'temperature_c' and strategy in {'linear_trend', 'hybrid_default'}:
                return float(current[feature_index[name]] + rollout_cfg.temperature_drift)
            if name == 'cumulative_ah_throughput':
                if rollout_cfg.delta_ah_per_step is not None:
                    delta = rollout_cfg.delta_ah_per_step
                else:
                    c_rate = max(float(current[feature_index['c_rate_discharge']]), 0.0)
                    nominal_capacity = rollout_cfg.nominal_capacity_ah or 1.0
                    delta = nominal_capacity * c_rate * rollout_cfg.step_duration_h
                return float(current[feature_index[name]] + delta * (1.0 + soh_rate))
            if name == 'internal_resistance':
                base = float(current[feature_index[name]])
                prev_base = float(previous[feature_index[name]])
                trend = max(base - prev_base, 0.0)
                growth = rollout_cfg.resistance_growth_rate * soh_rate * max(base, 1e-6)
                return float(base + trend + growth)
            if name == 'coulombic_efficiency':
                decay = rollout_cfg.ce_decay_rate * soh_rate
                return float(max(rollout_cfg.min_coulombic_efficiency, current[feature_index[name]] - decay))
            return default_value

        next_step[0, feature_index['soh']] = capped_soh

        for name in ['temperature_c', 'c_rate_charge', 'c_rate_discharge', 'rest_time_h',
                     'soc_min', 'soc_max', 'soc_mean', 'cumulative_ah_throughput',
                     'internal_resistance', 'coulombic_efficiency']:
            if name in feature_index:
                next_step[0, feature_index[name]] = choose(name, current[feature_index[name]])

        if 'soc_min' in feature_index and 'soc_max' in feature_index:
            next_step[0, feature_index['soc_min']] = float(np.clip(next_step[0, feature_index['soc_min']], 0.0, 1.0))
            next_step[0, feature_index['soc_max']] = float(np.clip(next_step[0, feature_index['soc_max']], 0.0, 1.0))
            if next_step[0, feature_index['soc_min']] > next_step[0, feature_index['soc_max']]:
                next_step[0, feature_index['soc_min']], next_step[0, feature_index['soc_max']] = (
                    next_step[0, feature_index['soc_max']], next_step[0, feature_index['soc_min']]
                )
        if 'soc_mean' in feature_index:
            soc_min = next_step[0, feature_index['soc_min']] if 'soc_min' in feature_index else 0.0
            soc_max = next_step[0, feature_index['soc_max']] if 'soc_max' in feature_index else 1.0
            next_step[0, feature_index['soc_mean']] = float(np.clip(next_step[0, feature_index['soc_mean']], min(soc_min, soc_max), max(soc_min, soc_max)))
        if 'internal_resistance' in feature_index and rollout_cfg.max_internal_resistance is not None:
            next_step[0, feature_index['internal_resistance']] = float(min(next_step[0, feature_index['internal_resistance']], rollout_cfg.max_internal_resistance))
        if 'coulombic_efficiency' in feature_index:
            next_step[0, feature_index['coulombic_efficiency']] = float(np.clip(next_step[0, feature_index['coulombic_efficiency']], rollout_cfg.min_coulombic_efficiency, 1.0))

        return np.concatenate([seq[1:], next_step], axis=0)

    def rollout_sequence(self,
                         model,
                         X_window: np.ndarray,
                         steps: int,
                         mask: np.ndarray = None,
                         rollout_cfg: CovariateRolloutConfig = None,
                         scenario_data=None,
                         x_scaler=None) -> np.ndarray:
        """?????????? SOH????????????"""
        if steps <= 0:
            raise ValueError('steps ?? > 0')

        current = np.asarray(X_window, dtype=np.float32).copy()
        preds = []
        cfg = rollout_cfg or CovariateRolloutConfig()
        scenario = self._load_rollout_scenario(scenario_data, steps).as_step_arrays(steps)

        for step_idx in range(steps):
            model_input = current.copy()
            if x_scaler is not None:
                model_input[:, 1:] = x_scaler.transform(current[:, 1:])
            pred_value = self._predict_next_soh(model, model_input, mask=mask)
            current = self._apply_covariate_strategy(current, step_idx, pred_value, cfg, scenario)
            preds.append(float(current[-1, 0]))
            if mask is not None:
                mask = np.concatenate([mask[1:], np.ones_like(mask[:1])], axis=0)

        return np.asarray(preds, dtype=np.float32)

    @staticmethod
    def _infer_input_dim(state_dict: dict, fallback: int) -> int:
        """从权重字典推断输入维度。"""
        candidate_keys = [
            'lstm.weight_ih_l0',
            'input_proj.weight',
            'input_layer.weight',
        ]
        for key in candidate_keys:
            if key in state_dict:
                return int(state_dict[key].shape[-1])
        return int(fallback)

    @classmethod
    def _load_window(cls, window_data: Union[str, Path, np.ndarray, pd.DataFrame],
                     history_cycles: int = 200,
                     cell_id: str = "") -> np.ndarray:
        """??????????????? 32 ??"""
        if isinstance(window_data, pd.DataFrame):
            window = window_data.to_numpy(dtype=np.float32)
        elif isinstance(window_data, np.ndarray):
            window = np.asarray(window_data, dtype=np.float32)
        else:
            path_obj = Path(window_data)
            suffix = path_obj.suffix.lower()
            if suffix == '.npy':
                window = np.load(path_obj).astype(np.float32)
            elif suffix == '.npz':
                archive = np.load(path_obj)
                if 'window' in archive:
                    window = archive['window'].astype(np.float32)
                elif 'x' in archive:
                    window = archive['x'].astype(np.float32)
                else:
                    window = archive[archive.files[0]].astype(np.float32)
            elif suffix == '.parquet':
                fname = path_obj.name.lower()
                # test.parquet 等已缩放 → 回退到 feature_table.parquet (原始物理值)
                if fname in {'train.parquet', 'val.parquet', 'test.parquet'}:
                    raw_path = path_obj.parent / 'feature_table.parquet'
                    if raw_path.exists():
                        path_obj = raw_path  # 切换到未缩放数据源
                df = pd.read_parquet(path_obj)
                if 'cell_id' in df.columns:
                    target_cell = cell_id if cell_id else str(df['cell_id'].iloc[-1])
                    df = df[df['cell_id'] == target_cell]
                df = df.sort_values('cycle_index') if 'cycle_index' in df.columns else df
                feature_cols_path = path_obj.parent.parent / 'processed' / 'feature_columns.json'
                if not feature_cols_path.exists():
                    feature_cols_path = path_obj.parent / 'feature_columns.json'
                if feature_cols_path.exists():
                    with open(feature_cols_path, 'r', encoding='utf-8') as fh:
                        feature_cols = json.load(fh)
                    feature_cols = [c for c in feature_cols if c in df.columns]
                else:
                    from models.soh_ai.config import ACTUAL_FEATURE_COLUMNS
                    feature_cols = [c for c in ACTUAL_FEATURE_COLUMNS if c in df.columns]
                    if not feature_cols:
                        feature_cols = df.select_dtypes(include=['number']).columns.tolist()
                window = df[feature_cols].tail(32).to_numpy(dtype=np.float32)
            else:
                raise ValueError(f'??????????: {path_obj}')

        if window.ndim != 2:
            raise ValueError(f'window_data ????: {window.shape}')
        return window[-32:] if window.shape[0] > 32 else window

    @classmethod
    def _load_history_context(cls, window_data: Union[str, Path, np.ndarray, pd.DataFrame],
                              history_cycles: int = 200,
                              cell_id: str = "") -> Dict[str, any]:
        """??????????????"""
        context = {
            'history_cycle_indices': None,
            'history_soh': None,
            'resolved_cell_id': cell_id or '',
            'input_is_scaled': False,
        }

        if isinstance(window_data, pd.DataFrame):
            df = window_data.copy()
            if 'cell_id' in df.columns and cell_id:
                df = df[df['cell_id'] == cell_id]
                context['resolved_cell_id'] = cell_id
            if 'cycle_index' in df.columns:
                df = df.sort_values('cycle_index')
            df = df.tail(history_cycles)
            context['history_cycle_indices'] = df['cycle_index'].to_numpy() if 'cycle_index' in df.columns else np.arange(1, len(df) + 1)
            context['history_soh'] = df['soh'].to_numpy(dtype=np.float32) if 'soh' in df.columns else None
            return context

        if isinstance(window_data, np.ndarray):
            arr = np.asarray(window_data, dtype=np.float32)
            context['history_cycle_indices'] = np.arange(1, arr.shape[0] + 1)
            context['history_soh'] = arr[:, 0] if arr.shape[1] > 0 else None
            return context

        path_obj = Path(window_data)
        suffix = path_obj.suffix.lower()
        if suffix == '.parquet':
            df = pd.read_parquet(path_obj)
            split_name = path_obj.name.lower()
            context['input_is_scaled'] = split_name in {'train.parquet', 'val.parquet', 'test.parquet'}

            if 'cell_id' in df.columns:
                if cell_id:
                    df = df[df['cell_id'] == cell_id]
                    context['resolved_cell_id'] = cell_id
                else:
                    context['resolved_cell_id'] = str(df['cell_id'].iloc[-1])
                    df = df[df['cell_id'] == context['resolved_cell_id']]

            if 'cycle_index' in df.columns:
                df = df.sort_values('cycle_index')

            history_df = df.tail(history_cycles)
            if split_name in {'train.parquet', 'val.parquet', 'test.parquet'}:
                feature_table = path_obj.parent / 'feature_table.parquet'
                if feature_table.exists():
                    full_df = pd.read_parquet(feature_table)
                    if 'cell_id' in full_df.columns and context['resolved_cell_id']:
                        full_df = full_df[full_df['cell_id'] == context['resolved_cell_id']]
                    if 'cycle_index' in full_df.columns:
                        full_df = full_df.sort_values('cycle_index')
                    history_df = full_df.tail(history_cycles)
                    context['input_is_scaled'] = True

            context['history_cycle_indices'] = history_df['cycle_index'].to_numpy() if 'cycle_index' in history_df.columns else np.arange(1, len(history_df) + 1)
            context['history_soh'] = history_df['soh'].to_numpy(dtype=np.float32) if 'soh' in history_df.columns else None
            return context

        # npy / npz 的 fallback
        window = cls._load_window(window_data, history_cycles=history_cycles)
        context['history_cycle_indices'] = np.arange(1, window.shape[0] + 1)
        context['history_soh'] = window[:, 0] if window.shape[1] > 0 else None
        return context

    @staticmethod
    def _load_scaler() -> Optional[object]:
        """?? X ???????????"""
        from models.soh_ai.config import WEIGHTS_DIR
        scaler_path = Path(WEIGHTS_DIR) / 'soh_scalers.pkl'
        if scaler_path.exists():
            import joblib
            scalers = joblib.load(scaler_path)
            return scalers.get('X', None)
        return None

    def predict_future_soh(self,
                           ensemble_or_model,
                           window_data: Union[str, Path, np.ndarray, pd.DataFrame],
                           future_steps: int = 100,
                           mask: np.ndarray = None,
                           rollout_cfg: CovariateRolloutConfig = None,
                           scenario_data=None,
                           history_cycles: int = 200,
                           cell_id: str = "") -> Dict[str, any]:
        """SOH ???????"""
        context = self._load_history_context(window_data, history_cycles=history_cycles, cell_id=cell_id)
        X_window = self._load_window(window_data, history_cycles=history_cycles, cell_id=cell_id)
        x_scaler = self._load_scaler()  # 始终用 scaler，数据已保证是原始物理值
        scenario_payload = rollout_cfg.scenario if rollout_cfg and rollout_cfg.scenario is not None else scenario_data
        future_pred = self.rollout_sequence(
            ensemble_or_model,
            X_window,
            future_steps,
            mask=mask,
            rollout_cfg=rollout_cfg,
            scenario_data=scenario_payload,
            x_scaler=x_scaler,
        )
        future_pred = np.asarray(future_pred, dtype=np.float32).reshape(-1)

        start_soh = float(X_window[-1, 0]) if X_window.shape[1] > 0 else float('nan')
        result = {
            'window_shape': tuple(X_window.shape),
            'future_steps': int(future_steps),
            'start_soh': start_soh,
            'future_soh': future_pred,
            'end_soh': float(future_pred[-1]) if len(future_pred) else start_soh,
            'history_cycle_indices': context['history_cycle_indices'],
            'history_soh': context['history_soh'],
            'resolved_cell_id': context['resolved_cell_id'],
        }

        if context['history_cycle_indices'] is not None and context['history_soh'] is not None:
            fig = self.vis.plot_trajectory(
                cycle_indices=np.asarray(context['history_cycle_indices']),
                true_soh=np.asarray(context['history_soh']),
                future_pred=future_pred,
                cell_id=context['resolved_cell_id'],
                model_name='FutureRollout',
                save=True,
            )
            result['figure'] = fig

        return result

    @staticmethod
    def print_summary(results: Dict[str, any]):
        """格式化打印评估摘要"""
        print("\n" + "=" * 60)
        print("  SOH 模型评估报告")
        print("=" * 60)

        if 'per_model' in results:
            # 多模型对比
            print(f"\n{'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'MAPE%':>10}")
            print("-" * 60)
            for m in results['per_model']:
                print(f"{m['model']:<15} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} "
                     f"{m['R²']:>10.4f} {m['MAPE_%']:>10.2f}")
        elif 'metrics' in results:
            m = results['metrics']
            print(f"\n  Model:    {m['model']}")
            print(f"  RMSE:     {m['RMSE']:.6f}")
            print(f"  MAE:      {m['MAE']:.6f}")
            print(f"  R²:       {m['R²']:.6f}")
            print(f"  MAPE:     {m['MAPE_%']:.2f}%")
            print(f"  Max Err:  {m['Max_Error']:.6f}")
            print(f"  N:        {m['n_samples']}")

        print("=" * 60)

    @classmethod
    def from_trained(cls,
                     weights_dir: str = None,
                     test_data: tuple = None,
                     window_data: Union[str, Path, np.ndarray, pd.DataFrame] = None,
                     future_steps: int = 100,
                     output_dir: str = None,
                     rollout_cfg: CovariateRolloutConfig = None,
                     scenario_data=None,
                     history_cycles: int = 200,
                     cell_id: str = '',
                     ) -> Dict[str, any]:
        """
        ?????????????????

        ?????????????

        Args:
            weights_dir: ??????
            test_data: (X_test, y_test) ????
            window_data: ?????????????
            future_steps: ????
            output_dir: ??????
            rollout_cfg: ???????
            scenario_data: ???????

        Returns:
            dict: ????
        """
        from models.soh_ai.config import WEIGHTS_DIR, ACTUAL_FEATURE_COLUMNS
        from models.soh_ai.models import XGBoostWrapper, BiLSTMAttention, TemporalTransformer, EnsembleModel

        weights_dir = Path(weights_dir or WEIGHTS_DIR)
        evaluator = cls(output_dir=output_dir)

        ensemble = EnsembleModel()

        # ?????
        xgb_path = weights_dir / 'xgb_model.pkl'
        lstm_path = weights_dir / 'lstm_attention.pt'
        tf_path = weights_dir / 'transformer.pt'

        if xgb_path.exists():
            ensemble.register('xgb', XGBoostWrapper.load(str(xgb_path)))
        if lstm_path.exists():
            # 优先加载 best checkpoint
            best_path = lstm_path.parent / 'checkpoints' / 'lstm_attention_best.pt'
            load_path = best_path if best_path.exists() else lstm_path
            lstm_state = torch.load(str(load_path), map_location='cpu')
            # 兼容 checkpoint dict vs 纯 state_dict
            if 'model_state_dict' in lstm_state:
                lstm_state = lstm_state['model_state_dict']
            lstm_input_dim = evaluator._infer_input_dim(lstm_state, len(ACTUAL_FEATURE_COLUMNS))
            m = BiLSTMAttention(input_dim=lstm_input_dim)
            m.load_state_dict(lstm_state)
            m.eval()
            ensemble.register('lstm', m)
        if tf_path.exists():
            tf_state = torch.load(str(tf_path), map_location='cpu')
            tf_input_dim = evaluator._infer_input_dim(tf_state, len(ACTUAL_FEATURE_COLUMNS))
            m = TemporalTransformer(input_dim=tf_input_dim)
            m.load_state_dict(tf_state)
            m.eval()
            ensemble.register('transformer', m)

        ensemble_weights_path = weights_dir / 'ensemble_weights.json'
        if ensemble_weights_path.exists():
            with open(ensemble_weights_path, 'r', encoding='utf-8') as fh:
                weights = json.load(fh)
            ensemble.set_weights(
                xgb=weights.get('xgb', 0.0),
                lstm=weights.get('lstm', 0.0),
                transformer=weights.get('transformer', 0.0),
            )

        if test_data is not None:
            X_test, y_test = test_data
            return evaluator.evaluate_all(ensemble, X_test, y_test)
        if window_data is not None:
            scenario_payload = rollout_cfg.scenario if rollout_cfg and rollout_cfg.scenario is not None else scenario_data
            return evaluator.predict_future_soh(
                ensemble,
                window_data=window_data,
                future_steps=future_steps,
                rollout_cfg=rollout_cfg,
                scenario_data=scenario_payload,
                history_cycles=history_cycles,
                cell_id=cell_id,
            )

        logger.warning("  ??????????????")
        return {'ensemble': ensemble}
def main():
    import argparse
    parser = argparse.ArgumentParser(description='SOH AI ????')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='?????? (??: models/weights/)')
    parser.add_argument('--test_data', type=str, default=None,
                       help='?????? (.npy ? .parquet)')
    parser.add_argument('--window_data', type=str, default=None,
                       help='????????? (.npy/.npz/.parquet)')
    parser.add_argument('--future_steps', type=int, default=100,
                       help='?????????')
    parser.add_argument('--rollout_strategy', type=str, default='hybrid_default',
                       choices=['hold_last', 'linear_trend', 'scenario_driven', 'hybrid_default'],
                       help='???????')
    parser.add_argument('--scenario_data', type=str, default=None,
                       help='?????? (.json/.npy/.npz/.parquet)')
    parser.add_argument('--nominal_capacity_ah', type=float, default=None,
                       help='?? Ah ????????? (Ah)')
    parser.add_argument('--step_duration_h', type=float, default=1.0,
                       help='???????? (h)')
    parser.add_argument('--delta_ah_per_step', type=float, default=None,
                       help='?? Ah ??????????')
    parser.add_argument('--cell_id', type=str, default='',
                       help='???????? ID')
    parser.add_argument('--history_cycles', type=int, default=200,
                       help='????????')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='??????')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(levelname)s] %(message)s')

    test_data = None
    if args.test_data:
        test_path = Path(args.test_data)
        if test_path.suffix.lower() == '.npy':
            data = np.load(test_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and len(data) >= 2:
                test_data = (data[0], data[1])
            else:
                raise ValueError(f'????????: {test_path}')
        elif test_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(test_path)
            if 'y_true' not in df.columns:
                raise ValueError('parquet ???????? y_true ?')
            y_test = df['y_true'].to_numpy(dtype=np.float32)
            X_test = df.drop(columns=['y_true']).to_numpy(dtype=np.float32)
            test_data = (X_test, y_test)
        else:
            raise ValueError(f'??????????: {test_path}')

    rollout_cfg = CovariateRolloutConfig(
        strategy=args.rollout_strategy,
        scenario=RolloutScenario.from_any(args.scenario_data) if args.scenario_data else None,
        nominal_capacity_ah=args.nominal_capacity_ah,
        step_duration_h=args.step_duration_h,
        delta_ah_per_step=args.delta_ah_per_step,
    )

    result = ModelEvaluator.from_trained(
        weights_dir=args.model_dir,
        test_data=test_data,
        window_data=args.window_data,
        future_steps=args.future_steps,
        output_dir=args.output_dir,
        rollout_cfg=rollout_cfg,
        scenario_data=args.scenario_data,
        history_cycles=args.history_cycles,
        cell_id=args.cell_id,
    )
    ModelEvaluator.print_summary(result)

    if 'future_soh' in result:
        future_soh = result['future_soh']
        print(f"\n  ?? SOH: {result['start_soh']:.6f}")
        print(f"  ?? SOH: {result['end_soh']:.6f}")
        print(f"  ????: {result['future_steps']}")
        print(f"  ? 5 ?: {np.array2string(future_soh[:5], precision=6, separator=', ')}")
        print(f"  ? 5 ?: {np.array2string(future_soh[-5:], precision=6, separator=', ')}")


if __name__ == '__main__':
    main()
