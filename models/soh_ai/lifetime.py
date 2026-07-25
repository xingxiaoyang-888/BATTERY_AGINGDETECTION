"""钠离子电池寿命标签、删失统计与工程化 RUL 推理。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import ACTUAL_FEATURE_COLUMNS


DEFAULT_EOL_THRESHOLDS = (0.90, 0.85, 0.80)
MODEL_VALIDATED_HORIZON = 128
RUL_AI_MAX_SOH = 0.98


def _as_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    total = float(sorted_weights.sum())
    if total <= 0:
        return float(np.quantile(sorted_values, quantile))
    cumulative = np.cumsum(sorted_weights) / total
    return float(np.interp(quantile, cumulative, sorted_values))


def build_eol_labels(
    frame: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_EOL_THRESHOLDS,
) -> pd.DataFrame:
    """按电芯生成多阈值 EOL 标签，并显式保留右删失样本。"""
    required = {'cell_id', 'cycle_index', 'soh'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'寿命标签数据缺少字段: {missing}')

    rows: List[Dict[str, Any]] = []
    for cell_id, group in frame.groupby('cell_id', sort=True):
        ordered = group.sort_values('cycle_index')
        cycle = pd.to_numeric(ordered['cycle_index'], errors='coerce').to_numpy(dtype=float)
        soh = pd.to_numeric(ordered['soh'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(cycle) & np.isfinite(soh)
        cycle, soh = cycle[valid], soh[valid]
        if len(cycle) == 0:
            continue
        monotonic_soh = np.minimum.accumulate(np.clip(soh, 0.0, 1.2))
        first_cycle, last_cycle = int(cycle[0]), int(cycle[-1])
        metadata = ordered.iloc[-1]
        for threshold in thresholds:
            threshold = float(threshold)
            if not 0 < threshold < 1.2:
                raise ValueError(f'EOL 阈值必须位于 (0, 1.2): {threshold}')
            crossed = np.flatnonzero(monotonic_soh <= threshold)
            event_observed = bool(len(crossed))
            event_cycle = int(cycle[crossed[0]]) if event_observed else None
            duration = (event_cycle if event_observed else last_cycle) - first_cycle
            rows.append({
                'cell_id': str(cell_id),
                'dataset_id': str(metadata.get('dataset_id', 'unknown')),
                'condition': str(metadata.get('condition', 'unknown')),
                'eol_soh_threshold': threshold,
                'first_cycle': first_cycle,
                'last_observed_cycle': last_cycle,
                'event_observed': event_observed,
                'event_cycle': event_cycle,
                'duration_cycles': int(max(duration, 0)),
                'start_soh': float(monotonic_soh[0]),
                'last_observed_soh': float(monotonic_soh[-1]),
                'censoring': 'observed' if event_observed else 'right_censored',
            })
    return pd.DataFrame(rows)


def build_landmark_labels(
    frame: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_EOL_THRESHOLDS,
    observation_fractions: Sequence[float] = (0.25, 0.50, 0.75),
) -> pd.DataFrame:
    """生成早期寿命回测标签；删失样本只提供 RUL 下界。"""
    outcomes = build_eol_labels(frame, thresholds)
    outcome_index = outcomes.set_index(['cell_id', 'eol_soh_threshold'])
    rows: List[Dict[str, Any]] = []
    for cell_id, group in frame.groupby('cell_id', sort=True):
        ordered = group.sort_values('cycle_index')
        cycle = ordered['cycle_index'].to_numpy(dtype=int)
        soh = np.minimum.accumulate(ordered['soh'].to_numpy(dtype=float))
        for fraction in observation_fractions:
            if not 0 < fraction < 1:
                raise ValueError(f'观测比例必须位于 (0, 1): {fraction}')
            position = min(max(int(math.ceil(len(cycle) * fraction)) - 1, 0), len(cycle) - 1)
            observation_cycle = int(cycle[position])
            for threshold in thresholds:
                outcome = outcome_index.loc[(str(cell_id), float(threshold))]
                event_cycle = outcome['event_cycle']
                if bool(outcome['event_observed']) and observation_cycle >= int(event_cycle):
                    continue
                observed = bool(outcome['event_observed'])
                rows.append({
                    'cell_id': str(cell_id),
                    'eol_soh_threshold': float(threshold),
                    'observation_fraction': float(fraction),
                    'observation_cycle': observation_cycle,
                    'observation_soh': float(soh[position]),
                    'rul_event_observed': observed,
                    'actual_rul_cycles': int(event_cycle - observation_cycle) if observed else None,
                    'rul_lower_bound_cycles': int(outcome['last_observed_cycle'] - observation_cycle),
                })
    return pd.DataFrame(rows)


def summarize_eol_labels(labels: pd.DataFrame) -> Dict[str, Any]:
    """生成可写入审计报告的删失统计。"""
    thresholds = []
    for threshold, group in labels.groupby('eol_soh_threshold', sort=False):
        events = int(group['event_observed'].sum())
        cells = int(len(group))
        thresholds.append({
            'eol_soh_threshold': float(threshold),
            'cells': cells,
            'events': events,
            'right_censored': cells - events,
            'event_fraction': round(events / cells, 6) if cells else 0.0,
            'median_observed_event_cycle': (
                float(group.loc[group['event_observed'], 'event_cycle'].median()) if events else None
            ),
        })
    return {'cells': int(labels['cell_id'].nunique()), 'thresholds': thresholds}


class SodiumLifetimePredictor:
    """RWTH 队列、运行历史和短期 AI 预测融合的钠电 RUL 引擎。"""

    scenario_defaults = {
        'temperature_c': 25.0,
        'temperature_spread_c': 3.0,
        'c_rate_charge': 1.0,
        'c_rate_discharge': 1.0,
        'rest_time_h': 0.0,
        'soc_min': 0.0,
        'soc_max': 1.0,
        'nominal_capacity_ah': 1.2,
        'coulombic_efficiency': 0.995,
    }

    cohort_columns = (
        'temperature_c', 'c_rate_charge', 'c_rate_discharge',
        'soc_min', 'soc_max', 'nominal_capacity_ah',
    )

    def __init__(
        self,
        reference_data: Union[str, Path, pd.DataFrame],
        weights_dir: Optional[Union[str, Path]] = None,
        use_ai: bool = True,
    ):
        if isinstance(reference_data, pd.DataFrame):
            self.reference_path = None
            self.reference = reference_data.copy()
        else:
            self.reference_path = Path(reference_data)
            if not self.reference_path.is_file():
                raise FileNotFoundError(f'RWTH 特征表不存在: {self.reference_path}')
            self.reference = pd.read_parquet(self.reference_path)
        self._validate_reference()
        self.labels = build_eol_labels(self.reference)
        self.evidence = summarize_eol_labels(self.labels)
        self.profiles = self._build_cell_profiles()
        self.weights_dir = self._resolve_weights_dir(weights_dir) if use_ai else None
        self._ai_model = None
        self._ai_scaler = None
        self._ai_load_error: Optional[str] = None
        self._last_ai_rate_key = None
        self._last_ai_rate_value: Optional[float] = None

    def _validate_reference(self) -> None:
        required = {'cell_id', 'cycle_index', 'soh', *self.cohort_columns}
        missing = sorted(required - set(self.reference.columns))
        if missing:
            raise ValueError(f'RWTH 特征表缺少字段: {missing}')
        if self.reference['cell_id'].nunique() < 3:
            raise ValueError('寿命参考队列至少需要 3 只电芯')

    @staticmethod
    def _resolve_weights_dir(weights_dir: Optional[Union[str, Path]]) -> Optional[Path]:
        if weights_dir:
            candidate = Path(weights_dir)
            return candidate if (candidate / 'xgb_model.pkl').is_file() else None
        env_dir = os.getenv('SODIUM_LIFETIME_MODEL_DIR')
        if env_dir and (Path(env_dir) / 'xgb_model.pkl').is_file():
            return Path(env_dir)
        root = Path(__file__).resolve().parents[1] / 'weights' / 'sodium_ion'
        candidates = [root / 'xgb_model.pkl']
        candidates.extend(root.glob('server_artifacts_*/server_runs/*/models/xgb_model.pkl'))
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            return None

        def validation_score(model_path: Path) -> Tuple[float, float]:
            metrics_path = model_path.parent / 'test_results.json'
            try:
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                rmse = float(metrics['xgb']['RMSE'])
                if not math.isfinite(rmse):
                    rmse = float('inf')
            except (OSError, ValueError, KeyError, TypeError):
                rmse = float('inf')
            return rmse, -model_path.stat().st_mtime

        return min(candidates, key=validation_score).parent

    def _build_cell_profiles(self) -> pd.DataFrame:
        rows = []
        for cell_id, group in self.reference.groupby('cell_id', sort=True):
            ordered = group.sort_values('cycle_index')
            cycle = ordered['cycle_index'].to_numpy(dtype=float)
            soh = np.minimum.accumulate(np.clip(ordered['soh'].to_numpy(dtype=float), 0.0, 1.2))
            span = max(float(cycle[-1] - cycle[0]), 1.0)
            total_rate = max(float(soh[0] - soh[-1]) / span, 1e-8)
            row = {
                'cell_id': str(cell_id),
                'condition': str(ordered['condition'].iloc[-1]) if 'condition' in ordered else 'unknown',
                'cycles': int(len(ordered)),
                'first_soh': float(soh[0]),
                'last_soh': float(soh[-1]),
                'degradation_rate': total_rate,
            }
            for column in self.cohort_columns:
                row[column] = float(pd.to_numeric(ordered[column], errors='coerce').median())
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _stress_components(scenario: Dict[str, float]) -> Dict[str, float]:
        temp = float(np.clip(scenario['temperature_c'], -30.0, 80.0))
        kelvin = temp + 273.15
        hot = math.exp(20000.0 / 8.314 * (1.0 / 298.15 - 1.0 / kelvin))
        cold = 1.0 + 0.08 * max(10.0 - temp, 0.0)
        thermal = max(hot, cold) * (1.0 + 0.03 * max(scenario['temperature_spread_c'] - 3.0, 0.0))
        electrical = 0.5 * (
            max(scenario['c_rate_charge'], 0.05) ** 1.15
            + max(scenario['c_rate_discharge'], 0.05) ** 1.15
        )
        dod = float(np.clip(scenario['soc_max'] - scenario['soc_min'], 0.05, 1.0))
        soc = dod ** 1.2 * (1.0 + 1.2 * max((scenario['soc_min'] + scenario['soc_max']) / 2 - 0.5, 0.0) ** 2)
        return {
            'thermal': float(thermal),
            'electrical': float(electrical),
            'soc_window': float(soc),
            'combined': float(thermal * electrical * soc),
        }

    def _domain_check(self, scenario: Dict[str, float], current_soh: float) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        scores: List[float] = []
        for column in ('temperature_c', 'c_rate_charge', 'c_rate_discharge'):
            values = self.profiles[column].to_numpy(dtype=float)
            lower, upper = float(values.min()), float(values.max())
            value = scenario[column]
            scale = max(float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 0.5)
            exceedance = max(lower - value, value - upper, 0.0) / scale
            if exceedance > 0:
                checks.append({'feature': column, 'value': value, 'training_range': [lower, upper]})
                scores.append(min(exceedance, 3.0) / 3.0)

        training_capacity = float(self.profiles['nominal_capacity_ah'].median())
        capacity_ratio = max(
            scenario['nominal_capacity_ah'] / training_capacity,
            training_capacity / scenario['nominal_capacity_ah'],
        )
        if capacity_ratio > 1.5:
            checks.append({
                'feature': 'nominal_capacity_ah',
                'value': scenario['nominal_capacity_ah'],
                'training_value': training_capacity,
                'ratio': capacity_ratio,
            })
            scores.append(min(math.log(capacity_ratio) / math.log(10.0), 1.0))

        if abs(scenario['soc_min']) > 0.02 or abs(scenario['soc_max'] - 1.0) > 0.02:
            checks.append({
                'feature': 'soc_window',
                'value': [scenario['soc_min'], scenario['soc_max']],
                'training_value': [0.0, 1.0],
            })
            scores.append(min(abs(scenario['soc_min']) + abs(1.0 - scenario['soc_max']), 1.0))

        soh_min, soh_max = float(self.reference['soh'].min()), float(self.reference['soh'].max())
        if current_soh < soh_min or current_soh > soh_max:
            checks.append({'feature': 'soh', 'value': current_soh, 'training_range': [soh_min, soh_max]})
            scores.append(min(abs(current_soh - np.clip(current_soh, soh_min, soh_max)) / 0.1, 1.0))

        score = float(np.clip(max(scores, default=0.0), 0.0, 1.0))
        return {
            'is_ood': bool(checks),
            'ood_score': round(score, 4),
            'out_of_domain_features': checks,
        }

    def _cohort_rates(self, scenario: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        distance_columns = ('temperature_c', 'c_rate_charge', 'c_rate_discharge', 'soc_min', 'soc_max')
        distances = np.zeros(len(self.profiles), dtype=float)
        for column in distance_columns:
            values = self.profiles[column].to_numpy(dtype=float)
            spread = max(float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 0.25)
            distances += ((values - scenario[column]) / spread) ** 2
        nearest_count = min(12, len(self.profiles))
        nearest_index = np.argsort(distances)[:nearest_count]
        nearest = self.profiles.iloc[nearest_index].copy()
        weights = np.exp(-0.5 * np.sqrt(distances[nearest_index]))
        weights = np.maximum(weights, 1e-6)

        reference_stress = np.array([
            self._stress_components({
                **scenario,
                'temperature_c': float(row.temperature_c),
                'temperature_spread_c': 3.0,
                'c_rate_charge': float(row.c_rate_charge),
                'c_rate_discharge': float(row.c_rate_discharge),
                'soc_min': float(row.soc_min),
                'soc_max': float(row.soc_max),
            })['combined']
            for row in nearest.itertuples()
        ])
        requested_stress = self._stress_components(scenario)['combined']
        correction = np.clip(requested_stress / np.maximum(reference_stress, 1e-6), 0.25, 4.0)
        rates = nearest['degradation_rate'].to_numpy(dtype=float) * correction
        return rates, weights, nearest

    @staticmethod
    def _history_rate(history: Sequence[Dict[str, Any]]) -> Optional[float]:
        if len(history) < 3:
            return None
        ordered = sorted(history, key=lambda item: float(item['cycle_index']))
        cycle = np.asarray([item['cycle_index'] for item in ordered], dtype=float)
        soh = np.minimum.accumulate(np.asarray([item['soh'] for item in ordered], dtype=float))
        unique = np.r_[True, np.diff(cycle) > 0]
        cycle, soh = cycle[unique], soh[unique]
        if len(cycle) < 3 or cycle[-1] - cycle[0] < 2:
            return None
        recent_start = max(0, len(cycle) - 100)
        spans = []
        for start in (0, recent_start):
            width = cycle[-1] - cycle[start]
            if width > 0:
                spans.append(max((soh[start] - soh[-1]) / width, 0.0))
        positive = [rate for rate in spans if rate > 0]
        return float(np.median(positive)) if positive else 0.0

    def _load_ai(self) -> bool:
        if self._ai_model is not None and self._ai_scaler is not None:
            return True
        if self._ai_load_error is not None or self.weights_dir is None:
            return False
        try:
            import joblib
            from .models import XGBoostWrapper

            self._ai_model = XGBoostWrapper.load(str(self.weights_dir / 'xgb_model.pkl'))
            scalers = joblib.load(self.weights_dir / 'soh_scalers.pkl')
            self._ai_scaler = scalers['X']
            return True
        except Exception as exc:  # 模型缺失时仍允许队列基线提供降级服务
            self._ai_load_error = str(exc)
            self._ai_model = None
            self._ai_scaler = None
            return False

    def _model_window(
        self,
        history: Sequence[Dict[str, Any]],
        scenario: Dict[str, float],
    ) -> Optional[pd.DataFrame]:
        if len(history) < 8:
            return None
        ordered = sorted(history, key=lambda item: float(item['cycle_index']))
        cycles = np.asarray([float(item['cycle_index']) for item in ordered])
        if cycles[-1] - cycles[0] < 7:
            return None
        target_cycle = np.linspace(max(cycles[-1] - 31, cycles[0]), cycles[-1], 32)
        defaults = {
            column: float(pd.to_numeric(self.reference[column], errors='coerce').median())
            for column in ACTUAL_FEATURE_COLUMNS if column in self.reference.columns
        }
        rows: Dict[str, np.ndarray] = {}
        for column in ACTUAL_FEATURE_COLUMNS:
            values = [item.get(column) for item in ordered]
            valid = np.asarray([value is not None and math.isfinite(float(value)) for value in values])
            if valid.sum() >= 2:
                rows[column] = np.interp(
                    target_cycle,
                    cycles[valid],
                    np.asarray(values, dtype=object)[valid].astype(float),
                )
            else:
                rows[column] = np.full(32, defaults.get(column, 0.0), dtype=float)

        rows['soh'] = np.interp(target_cycle, cycles, [float(item['soh']) for item in ordered])
        rows['nominal_capacity_ah'][:] = scenario['nominal_capacity_ah']
        for column in ('temperature_c', 'c_rate_charge', 'c_rate_discharge', 'rest_time_h', 'soc_min', 'soc_max'):
            if all(item.get(column) is None for item in ordered):
                rows[column][:] = scenario[column]
        rows['soc_mean'] = (rows['soc_min'] + rows['soc_max']) / 2.0
        if all(item.get('cumulative_ah_throughput') is None for item in ordered):
            rows['cumulative_ah_throughput'] = (
                target_cycle * scenario['nominal_capacity_ah'] * (scenario['soc_max'] - scenario['soc_min'])
            )

        soh = rows['soh']
        for name, lag in (('soh_diff_1', 1), ('soh_diff_3', 3), ('soh_diff_5', 5)):
            result = np.zeros(32, dtype=float)
            result[lag:] = soh[lag:] - soh[:-lag]
            result[:lag] = result[lag]
            rows[name] = result
        decay = np.zeros(32, dtype=float)
        for index in range(2, 32):
            recent = np.clip(soh[max(0, index - 10):index + 1], 1e-6, None)
            decay[index] = max(-np.polyfit(np.arange(len(recent)), np.log(recent), 1)[0], 0.0)
        rows['soh_decay_rate'] = decay
        acceleration = np.zeros(32, dtype=float)
        diff = rows['soh_diff_1']
        acceleration[3:] = diff[3:] - diff[:-3]
        rows['capacity_fade_acceleration'] = acceleration
        if 'ce_trend' in rows:
            rows['ce_trend'] = pd.Series(rows['coulombic_efficiency']).rolling(5, min_periods=1).mean().to_numpy()
        return pd.DataFrame({column: rows[column] for column in ACTUAL_FEATURE_COLUMNS})

    def _ai_rate(
        self,
        history: Sequence[Dict[str, Any]],
        scenario: Dict[str, float],
    ) -> Optional[float]:
        cache_key = (
            tuple((int(item['cycle_index']), round(float(item['soh']), 8)) for item in history),
            tuple((name, round(float(scenario[name]), 8)) for name in sorted(scenario)),
        )
        if cache_key == self._last_ai_rate_key:
            return self._last_ai_rate_value
        window = self._model_window(history, scenario)
        if window is None or not self._load_ai():
            return None
        try:
            from .evaluate import CovariateRolloutConfig, ModelEvaluator, RolloutScenario

            rollout_scenario = RolloutScenario(
                temperature_c=scenario['temperature_c'],
                c_rate_charge=scenario['c_rate_charge'],
                c_rate_discharge=scenario['c_rate_discharge'],
                rest_time_h=scenario['rest_time_h'],
                soc_min=scenario['soc_min'],
                soc_max=scenario['soc_max'],
                soc_mean=(scenario['soc_min'] + scenario['soc_max']) / 2.0,
            )
            config = CovariateRolloutConfig(
                scenario=rollout_scenario,
                nominal_capacity_ah=scenario['nominal_capacity_ah'],
                delta_ah_per_step=scenario['nominal_capacity_ah'] * (scenario['soc_max'] - scenario['soc_min']),
                enforce_monotonic=True,
            )
            trajectory = ModelEvaluator().rollout_sequence(
                self._ai_model,
                window[ACTUAL_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
                MODEL_VALIDATED_HORIZON,
                rollout_cfg=config,
                x_scaler=self._ai_scaler,
            )
            start_soh = float(window['soh'].iloc[-1])
            value = max((start_soh - float(trajectory[-1])) / MODEL_VALIDATED_HORIZON, 0.0)
            self._last_ai_rate_key = cache_key
            self._last_ai_rate_value = value
            return value
        except Exception as exc:
            self._ai_load_error = str(exc)
            return None

    def _model_metrics(self) -> Dict[str, Any]:
        metrics = None
        if self.weights_dir and (self.weights_dir / 'test_results.json').is_file():
            try:
                metrics = json.loads((self.weights_dir / 'test_results.json').read_text(encoding='utf-8')).get('xgb')
            except (OSError, ValueError):
                metrics = None
        return {
            'name': 'RWTH residual XGBoost + censored cohort',
            'ai_artifact_available': bool(self.weights_dir),
            'ai_loaded': self._ai_model is not None and self._ai_load_error is None,
            'validated_prediction_horizon_cycles': MODEL_VALIDATED_HORIZON,
            'rul_ai_activation_rule': f'current_soh <= {RUL_AI_MAX_SOH:.2f}',
            'one_step_test_metrics': metrics,
            'load_error': self._ai_load_error,
        }

    def evidence_summary(self) -> Dict[str, Any]:
        return {
            **self.evidence,
            'dataset': 'RWTH commercial sodium-ion aging, DOD100 subset',
            'rows': int(len(self.reference)),
            'conditions': int(self.reference['condition'].nunique()) if 'condition' in self.reference else None,
            'model': self._model_metrics(),
        }

    def predict(
        self,
        current_soh: float,
        current_cycle: int,
        eol_threshold: float = 0.80,
        scenario: Optional[Dict[str, Any]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        cycles_per_day: float = 1.0,
        max_horizon_cycles: int = 5000,
    ) -> Dict[str, Any]:
        current_soh = float(current_soh)
        eol_threshold = float(eol_threshold)
        if not 0 < current_soh <= 1.1:
            raise ValueError('当前 SOH 必须位于 (0, 1.1]')
        if not 0 < eol_threshold <= 1.0:
            raise ValueError('EOL SOH 阈值必须位于 (0, 1.0]')
        if current_cycle < 0 or cycles_per_day <= 0 or max_horizon_cycles <= 0:
            raise ValueError('循环数、每日循环数和预测范围必须为正值')

        normalized_scenario = dict(self.scenario_defaults)
        for key, value in (scenario or {}).items():
            if key in normalized_scenario and value is not None:
                normalized_scenario[key] = _as_float(value, normalized_scenario[key])
        if not 0 <= normalized_scenario['soc_min'] < normalized_scenario['soc_max'] <= 1:
            raise ValueError('SOC 窗口必须满足 0 <= soc_min < soc_max <= 1')
        if normalized_scenario['c_rate_charge'] <= 0 or normalized_scenario['c_rate_discharge'] <= 0:
            raise ValueError('充放电 C 倍率必须大于 0')
        if normalized_scenario['nominal_capacity_ah'] <= 0:
            raise ValueError('额定容量必须大于 0')
        if not -30 <= normalized_scenario['temperature_c'] <= 80:
            raise ValueError('温度必须位于 -30 到 80 摄氏度')
        normalized_history = [dict(item) for item in (history or [])]
        domain = self._domain_check(normalized_scenario, current_soh)
        stress = self._stress_components(normalized_scenario)

        evidence_labels = build_eol_labels(self.reference, (eol_threshold,))
        event_count = int(evidence_labels['event_observed'].sum())
        cell_count = int(len(evidence_labels))
        event_fraction = event_count / cell_count if cell_count else 0.0

        if current_soh <= eol_threshold:
            return {
                'status': 'eol_reached',
                'prediction': {
                    'current_soh': current_soh,
                    'eol_soh_threshold': eol_threshold,
                    'rul_cycles': 0,
                    'predicted_eol_cycle': int(current_cycle),
                    'rul_days': 0.0,
                    'within_requested_horizon': True,
                    'degradation_rate_per_cycle': 0.0,
                },
                'uncertainty': {'confidence_score': 1.0, 'confidence_level': 'observed', 'rul_interval_cycles': [0, 0]},
                'domain_check': domain,
                'stress': stress,
                'evidence': {'cells': cell_count, 'events': event_count, 'right_censored': cell_count - event_count},
                'model': {**self._model_metrics(), 'ai_used_for_prediction': False},
                'trajectory': [{'cycle': int(current_cycle), 'soh': current_soh}],
                'warnings': [],
            }

        cohort_rates, cohort_weights, nearest = self._cohort_rates(normalized_scenario)
        cohort_rate = _weighted_quantile(cohort_rates, cohort_weights, 0.50)
        history_rate = self._history_rate(normalized_history)
        ai_rate = (
            self._ai_rate(normalized_history, normalized_scenario)
            if current_soh <= RUL_AI_MAX_SOH else None
        )
        physical_stress_multiplier = float(np.clip(stress['combined'], 0.25, 4.0))

        sources = [('rwth_cohort', cohort_rate, 0.60)]
        if history_rate is not None and history_rate > 0:
            history_weight = min(0.55, 0.15 + len(normalized_history) / 80.0)
            sources.append((
                'cell_history_stress_adjusted',
                history_rate * physical_stress_multiplier,
                history_weight,
            ))
        if ai_rate is not None and ai_rate > 0:
            plausible_low = max(_weighted_quantile(cohort_rates, cohort_weights, 0.10) * 0.1, 1e-8)
            plausible_high = max(_weighted_quantile(cohort_rates, cohort_weights, 0.90) * 10.0, plausible_low)
            if plausible_low <= ai_rate <= plausible_high:
                sources.append(('xgboost_128_cycle', ai_rate, min(0.25, len(normalized_history) / 128.0)))

        ai_used_for_prediction = any(name == 'xgboost_128_cycle' for name, _, _ in sources)

        total_weight = sum(weight for _, _, weight in sources)
        degradation_rate = sum(rate * weight for _, rate, weight in sources) / total_weight
        degradation_rate = max(float(degradation_rate), 1e-8)
        remaining_soh = current_soh - eol_threshold
        rul_cycles = int(math.ceil(remaining_soh / degradation_rate))

        rate_low = max(_weighted_quantile(cohort_rates, cohort_weights, 0.10), degradation_rate * 0.25, 1e-8)
        rate_high = max(_weighted_quantile(cohort_rates, cohort_weights, 0.90), degradation_rate * 1.75)
        widening = 1.0 + 1.5 * domain['ood_score'] + (0.7 if event_count < 10 else 0.0)
        rate_low /= widening
        rate_high *= widening
        interval = [
            int(max(0, math.floor(remaining_soh / rate_high))),
            int(math.ceil(remaining_soh / rate_low)),
        ]

        history_coverage = min(len(normalized_history) / 32.0, 1.0)
        confidence = (
            0.18
            + 0.28 * min(event_fraction / 0.5, 1.0)
            + 0.24 * (1.0 - domain['ood_score'])
            + 0.20 * history_coverage
            + 0.10 * float(ai_rate is not None)
        )
        confidence_cap = 0.80 if event_count >= 20 else 0.65 if event_count >= 10 else 0.45
        confidence_cap = min(confidence_cap, 0.80 - 0.35 * domain['ood_score'])
        confidence = float(np.clip(min(confidence, confidence_cap), 0.05, 0.95))
        level = 'high' if confidence >= 0.75 else 'medium' if confidence >= 0.50 else 'low'

        warnings = []
        if event_count < 10:
            warnings.append(f'{eol_threshold:.0%} EOL 仅有 {event_count}/{cell_count} 个实测事件，结果以外推为主')
        elif event_count < 20:
            warnings.append(f'{eol_threshold:.0%} EOL 实测事件有限（{event_count}/{cell_count}），区间已加宽')
        if domain['is_ood']:
            warnings.append('输入工况超出 RWTH 训练域，不能视为同规格电芯的标定结果')
        if len(normalized_history) < 8:
            warnings.append('历史循环少于 8 个点，本次未启用 XGBoost 短期趋势校准')
        elif current_soh > RUL_AI_MAX_SOH:
            warnings.append('当前 SOH 高于 98%，按验证集门禁不启用 AI 远期 RUL 融合')
        if self.weights_dir is None:
            warnings.append('未找到钠电 XGBoost 权重，本次使用删失感知队列基线')
        if rul_cycles > MODEL_VALIDATED_HORIZON:
            warnings.append('RUL 超过模型已验证的 128 循环预测跨度，远期部分属于工程外推')

        trajectory_end = min(rul_cycles, int(max_horizon_cycles))
        sample_count = min(51, trajectory_end + 1)
        offsets = np.unique(np.linspace(0, trajectory_end, sample_count, dtype=int))
        trajectory = [
            {
                'cycle': int(current_cycle + offset),
                'soh': round(max(current_soh - degradation_rate * offset, eol_threshold), 6),
            }
            for offset in offsets
        ]

        return {
            'status': 'ok',
            'prediction': {
                'current_soh': current_soh,
                'eol_soh_threshold': eol_threshold,
                'rul_cycles': rul_cycles,
                'predicted_eol_cycle': int(current_cycle + rul_cycles),
                'rul_days': round(rul_cycles / cycles_per_day, 2),
                'within_requested_horizon': rul_cycles <= max_horizon_cycles,
                'degradation_rate_per_cycle': degradation_rate,
            },
            'uncertainty': {
                'confidence_score': round(confidence, 4),
                'confidence_level': level,
                'rul_interval_cycles': interval,
                'interval_basis': 'RWTH matched-cohort weighted 10%-90% rates, OOD/censoring widened',
            },
            'domain_check': domain,
            'stress': {
                **{key: round(value, 5) for key, value in stress.items()},
                'history_rate_multiplier': round(physical_stress_multiplier, 5),
            },
            'evidence': {
                'dataset': 'RWTH commercial sodium-ion aging, DOD100 subset',
                'cells': cell_count,
                'events': event_count,
                'right_censored': cell_count - event_count,
                'event_fraction': round(event_fraction, 4),
                'matched_cells': nearest['cell_id'].tolist(),
            },
            'rate_sources': [
                {'source': name, 'rate_per_cycle': float(rate), 'fusion_weight': round(weight / total_weight, 4)}
                for name, rate, weight in sources
            ],
            'model': {**self._model_metrics(), 'ai_used_for_prediction': ai_used_for_prediction},
            'trajectory': trajectory,
            'warnings': warnings,
        }
