"""从已审计的 RWTH 特征表生成 EOL 与早期 RUL 标签。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.soh_ai.lifetime import (
    DEFAULT_EOL_THRESHOLDS,
    build_eol_labels,
    build_landmark_labels,
    summarize_eol_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='构建 RWTH 钠电删失感知寿命标签')
    parser.add_argument(
        '--feature-table',
        default='models/data/processed/sodium_ion/feature_table.parquet',
    )
    parser.add_argument(
        '--output-dir',
        default='models/data/processed/sodium_ion/lifetime',
    )
    args = parser.parse_args()

    source = Path(args.feature_table)
    if not source.is_file():
        raise FileNotFoundError(f'特征表不存在: {source}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source)
    labels = build_eol_labels(frame, DEFAULT_EOL_THRESHOLDS)
    landmarks = build_landmark_labels(frame, DEFAULT_EOL_THRESHOLDS)
    summary = summarize_eol_labels(labels)
    summary.update({
        'source': str(source),
        'feature_rows': int(len(frame)),
        'landmark_samples': int(len(landmarks)),
        'label_policy': {
            'soh_curve': 'causal-smoothed input followed by cumulative-minimum crossing',
            'censoring': 'cells not crossing a threshold are right-censored',
            'landmarks': [0.25, 0.50, 0.75],
        },
    })

    labels.to_parquet(output_dir / 'eol_labels.parquet', index=False)
    landmarks.to_parquet(output_dir / 'rul_landmarks.parquet', index=False)
    (output_dir / 'lifetime_audit.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
