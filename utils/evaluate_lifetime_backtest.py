"""钠电寿命引擎的按电芯留一回测。

每一折从参考队列中彻底移除待测电芯，再在该电芯 25%/50%/75%
寿命观测点预测 RUL。达到 EOL 的样本计算误差和区间覆盖率；未达到 EOL
的右删失样本只检查预测寿命是否超过已知下界，不伪造真实 RUL。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.soh_ai.lifetime import SodiumLifetimePredictor, build_eol_labels


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    observed = [row for row in rows if row['event_observed']]
    censored = [row for row in rows if not row['event_observed']]
    result: Dict[str, Any] = {
        'samples': len(rows),
        'observed_samples': len(observed),
        'right_censored_samples': len(censored),
        'censored_lower_bound_consistency': (
            float(np.mean([row['predicted_rul'] >= row['rul_lower_bound'] for row in censored]))
            if censored else None
        ),
    }
    if not observed:
        result.update({
            'mae_cycles': None,
            'rmse_cycles': None,
            'median_absolute_error_cycles': None,
            'mape_pct': None,
            'interval_coverage': None,
            'within_20pct_accuracy': None,
            'bias_cycles': None,
        })
        return result

    actual = np.asarray([row['actual_rul'] for row in observed], dtype=float)
    predicted = np.asarray([row['predicted_rul'] for row in observed], dtype=float)
    error = predicted - actual
    interval_hit = np.asarray([
        row['rul_interval'][0] <= row['actual_rul'] <= row['rul_interval'][1]
        for row in observed
    ])
    result.update({
        'mae_cycles': float(np.mean(np.abs(error))),
        'rmse_cycles': float(np.sqrt(np.mean(error ** 2))),
        'median_absolute_error_cycles': float(np.median(np.abs(error))),
        'mape_pct': float(np.mean(np.abs(error) / np.maximum(actual, 1.0)) * 100.0),
        'interval_coverage': float(interval_hit.mean()),
        'within_20pct_accuracy': float((np.abs(error) <= np.maximum(actual * 0.20, 20.0)).mean()),
        'bias_cycles': float(error.mean()),
    })
    return result


def evaluate_leave_one_cell_out(
    frame: pd.DataFrame,
    thresholds: Iterable[float] = (0.90, 0.85, 0.80),
    observation_fractions: Sequence[float] = (0.25, 0.50, 0.75),
    use_ai: bool = False,
    weights_dir: str | Path | None = None,
    progress_every: int = 5,
    test_cells: Sequence[str] | None = None,
    exclude_all_test_cells_from_reference: bool = False,
) -> Dict[str, Any]:
    thresholds = tuple(float(value) for value in thresholds)
    outcomes = build_eol_labels(frame, thresholds).set_index(['cell_id', 'eol_soh_threshold'])
    all_cells = sorted(frame['cell_id'].astype(str).unique())
    cells = sorted(str(cell) for cell in test_cells) if test_cells is not None else all_cells
    unknown = sorted(set(cells) - set(all_cells))
    if unknown:
        raise ValueError(f'测试电芯不在特征表中: {unknown}')
    fixed_reference = (
        frame[~frame['cell_id'].astype(str).isin(cells)].copy()
        if exclude_all_test_cells_from_reference else None
    )
    if fixed_reference is not None and fixed_reference['cell_id'].nunique() < 3:
        raise ValueError('固定参考队列至少需要 3 只非测试电芯')
    rows: List[Dict[str, Any]] = []

    for fold_index, cell_id in enumerate(cells, start=1):
        train = (
            fixed_reference
            if fixed_reference is not None
            else frame[frame['cell_id'].astype(str) != cell_id].copy()
        )
        test = frame[frame['cell_id'].astype(str) == cell_id].sort_values('cycle_index').copy()
        predictor = SodiumLifetimePredictor(train, weights_dir=weights_dir, use_ai=use_ai)

        for fraction in observation_fractions:
            position = min(max(int(np.ceil(len(test) * fraction)) - 1, 0), len(test) - 1)
            observed_history = test.iloc[:position + 1]
            current_cycle = int(observed_history['cycle_index'].iloc[-1])
            current_soh = float(observed_history['soh'].iloc[-1])
            history = observed_history[['cycle_index', 'soh']].tail(32).to_dict('records')
            scenario = {
                'temperature_c': float(observed_history['temperature_c'].median()),
                'c_rate_charge': float(observed_history['c_rate_charge'].median()),
                'c_rate_discharge': float(observed_history['c_rate_discharge'].median()),
                'rest_time_h': float(observed_history['rest_time_h'].median()),
                'soc_min': float(observed_history['soc_min'].median()),
                'soc_max': float(observed_history['soc_max'].median()),
                'nominal_capacity_ah': float(observed_history['nominal_capacity_ah'].median()),
            }

            for threshold in thresholds:
                outcome = outcomes.loc[(cell_id, threshold)]
                event_observed = bool(outcome['event_observed'])
                event_cycle = int(outcome['event_cycle']) if event_observed else None
                if event_observed and current_cycle >= event_cycle:
                    continue
                result = predictor.predict(
                    current_soh=current_soh,
                    current_cycle=current_cycle,
                    eol_threshold=threshold,
                    scenario=scenario,
                    history=history,
                    max_horizon_cycles=10000,
                )
                rows.append({
                    'fold': fold_index,
                    'cell_id': cell_id,
                    'condition': str(test['condition'].iloc[-1]),
                    'observation_fraction': float(fraction),
                    'observation_cycle': current_cycle,
                    'observation_soh': current_soh,
                    'eol_soh_threshold': threshold,
                    'event_observed': event_observed,
                    'event_cycle': event_cycle,
                    'actual_rul': event_cycle - current_cycle if event_observed else None,
                    'rul_lower_bound': int(outcome['last_observed_cycle'] - current_cycle),
                    'predicted_rul': int(result['prediction']['rul_cycles']),
                    'rul_interval': result['uncertainty']['rul_interval_cycles'],
                    'confidence_score': result['uncertainty']['confidence_score'],
                    'ood_score': result['domain_check']['ood_score'],
                    'ai_used': bool(result['model'].get('ai_used_for_prediction', False)),
                    'rate_sources': result.get('rate_sources', []),
                    'warnings': result['warnings'],
                })

        if progress_every and (fold_index % progress_every == 0 or fold_index == len(cells)):
            print(
                f'PROGRESS {fold_index}/{len(cells)} cells, samples={len(rows)}',
                flush=True,
            )

    per_threshold = {
        str(threshold): _metrics([
            row for row in rows if row['eol_soh_threshold'] == threshold
        ])
        for threshold in thresholds
    }
    per_fraction = {
        str(fraction): _metrics([
            row for row in rows if row['observation_fraction'] == fraction
        ])
        for fraction in observation_fractions
    }
    return {
        'protocol': {
            'name': (
                'fixed cell holdout lifetime backtest'
                if exclude_all_test_cells_from_reference
                else 'leave-one-cell-out lifetime backtest'
            ),
            'leakage_control': (
                'all fixed test cells excluded from cohort, EOL labels and rate estimation'
                if exclude_all_test_cells_from_reference
                else 'held-out cell excluded from cohort, EOL labels and rate estimation'
            ),
            'cells': len(cells),
            'reference_cells': int(
                fixed_reference['cell_id'].nunique()
                if fixed_reference is not None else frame['cell_id'].nunique() - 1
            ),
            'thresholds': list(thresholds),
            'observation_fractions': list(observation_fractions),
            'right_censoring': 'no synthetic RUL; lower-bound consistency only',
            'ai_enabled': bool(use_ai),
        },
        'summary': {
            'overall': _metrics(rows),
            'per_threshold': per_threshold,
            'per_observation_fraction': per_fraction,
            'mean_confidence_score': float(np.mean([row['confidence_score'] for row in rows])),
            'mean_ood_score': float(np.mean([row['ood_score'] for row in rows])),
            'ai_used_samples': int(sum(row['ai_used'] for row in rows)),
        },
        'predictions': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='钠电寿命引擎按电芯留一回测')
    parser.add_argument(
        '--feature-table',
        default='models/data/processed/sodium_ion/feature_table.parquet',
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.90, 0.85, 0.80])
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument('--use-ai', action='store_true')
    parser.add_argument('--weights-dir')
    parser.add_argument(
        '--test-split',
        help='冻结测试集 Parquet；提供后从参考队列中排除其中全部电芯',
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.feature_table)
    test_cells = None
    if args.test_split:
        test_frame = pd.read_parquet(args.test_split)
        test_cells = sorted(test_frame['cell_id'].astype(str).unique())
    result = evaluate_leave_one_cell_out(
        frame,
        thresholds=args.thresholds,
        observation_fractions=args.fractions,
        use_ai=args.use_ai,
        weights_dir=args.weights_dir,
        test_cells=test_cells,
        exclude_all_test_cells_from_reference=bool(args.test_split),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2), flush=True)
    print(f'结果已保存: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
