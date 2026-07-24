"""评估不同预测跨度下的 XGBoost 与持久性基线。

该脚本采用按电芯留一交叉验证，直接预测未来第 N 个循环的 SOH。
持久性基线定义为 ``SOH(t + N) = SOH(t)``，用于判断模型是否真正
优于“健康状态短期不变”的朴素假设。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from models.soh_ai.config import FEATURE_CFG
from models.soh_ai.data_pipeline import CycleFeatureExtractor, DataCleaner, SequenceBuilder
from models.soh_ai.models import XGBoostWrapper
from models.soh_ai.trainer import EnsembleTrainer
from utils.sodium_dataset_loader import SodiumDatasetLoader


def _feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "cell_id", "chemistry", "dataset_id", "condition", "cycle_index",
        "soh_raw", "soh_jump_flag",
    }
    return [col for col in df.columns if col not in exclude and df[col].dtype.kind in "fiu"]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_horizons(
    df: pd.DataFrame,
    horizons: Iterable[int],
    lookback: int = 32,
) -> dict:
    """按电芯留一评估多个直接预测跨度。"""
    feature_cols = _feature_columns(df)
    scale_cols = [col for col in feature_cols if col != FEATURE_CFG.target_col]
    soh_index = feature_cols.index(FEATURE_CFG.target_col)
    groups = df["cell_id"].to_numpy()
    cells = df["cell_id"].dropna().unique().tolist()
    splitter = GroupKFold(n_splits=len(cells))
    builder = SequenceBuilder()
    results = {
        "lookback": lookback,
        "feature_columns": feature_cols,
        "cells": [str(cell) for cell in cells],
        "horizons": {},
    }

    fold_indices = list(splitter.split(df, groups=groups))
    for horizon in horizons:
        fold_results = []
        for fold, (train_val_idx, test_idx) in enumerate(fold_indices, start=1):
            train_val = df.iloc[train_val_idx].copy()
            test = df.iloc[test_idx].copy()
            train_val_cells = train_val["cell_id"].dropna().unique().tolist()
            val_cells = train_val_cells[:1]
            train_cells = train_val_cells[1:]
            train = train_val[train_val["cell_id"].isin(train_cells)].copy()
            val = train_val[train_val["cell_id"].isin(val_cells)].copy()

            scaler = RobustScaler(quantile_range=(5, 95))
            scaler.fit(train[scale_cols].values)

            def scale(frame: pd.DataFrame) -> pd.DataFrame:
                output = frame.copy()
                output.loc[:, scale_cols] = scaler.transform(output[scale_cols].values)
                return output

            X_train, y_train, _, _ = builder.build_sequences(
                scale(train), lookback=lookback, horizon=horizon,
                feature_cols=feature_cols,
            )
            X_val, y_val, _, _ = builder.build_sequences(
                scale(val), lookback=lookback, horizon=horizon,
                feature_cols=feature_cols,
            )
            X_test, y_test, _, _ = builder.build_sequences(
                scale(test), lookback=lookback, horizon=horizon,
                feature_cols=feature_cols,
            )

            y_train_final = y_train[:, -1:]
            y_val_final = y_val[:, -1:]
            y_test_final = y_test[:, -1]
            xgb = XGBoostWrapper()
            xgb.fit(
                EnsembleTrainer._extract_tabular_features(X_train), y_train_final,
                EnsembleTrainer._extract_tabular_features(X_val), y_val_final,
            )
            xgb_pred = xgb.predict(
                EnsembleTrainer._extract_tabular_features(X_test)
            ).ravel()
            persistence_pred = X_test[:, -1, soh_index]
            xgb_metrics = _metrics(y_test_final, xgb_pred)
            persistence_metrics = _metrics(y_test_final, persistence_pred)
            improvement = 1.0 - xgb_metrics["rmse"] / persistence_metrics["rmse"]
            fold_results.append({
                "fold": fold,
                "train_cells": [str(cell) for cell in train_cells],
                "val_cells": [str(cell) for cell in val_cells],
                "test_cells": [str(cell) for cell in test["cell_id"].unique()],
                "samples": int(len(y_test_final)),
                "xgb": xgb_metrics,
                "persistence": persistence_metrics,
                "rmse_improvement_ratio": float(improvement),
                "beats_persistence": bool(improvement > 0),
            })

        xgb_rmse = np.array([fold["xgb"]["rmse"] for fold in fold_results])
        base_rmse = np.array([fold["persistence"]["rmse"] for fold in fold_results])
        results["horizons"][str(horizon)] = {
            "folds": fold_results,
            "summary": {
                "xgb_rmse_mean": float(xgb_rmse.mean()),
                "xgb_rmse_std": float(xgb_rmse.std(ddof=1)),
                "persistence_rmse_mean": float(base_rmse.mean()),
                "persistence_rmse_std": float(base_rmse.std(ddof=1)),
                "mean_rmse_improvement_ratio": float(1.0 - xgb_rmse.mean() / base_rmse.mean()),
                "folds_beating_persistence": int((xgb_rmse < base_rmse).sum()),
                "total_folds": int(len(fold_results)),
            },
        }
    return results


def evaluate_fixed_split_horizons(
    splits: dict,
    horizons: Iterable[int],
    lookback: int = 32,
) -> dict:
    """在冻结的电芯级 train/val/test 划分上评估多跨度基线。"""
    train, val, test = (splits[name].copy() for name in ("train", "val", "test"))
    feature_cols = _feature_columns(train)
    scale_cols = [col for col in feature_cols if col != FEATURE_CFG.target_col]
    scaler = RobustScaler(quantile_range=(5, 95))
    scaler.fit(train[scale_cols].values)

    def scale(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output.loc[:, scale_cols] = scaler.transform(output[scale_cols].values)
        return output

    scaled = {name: scale(frame) for name, frame in splits.items()}
    soh_index = feature_cols.index(FEATURE_CFG.target_col)
    builder = SequenceBuilder()
    results = {
        "lookback": lookback,
        "feature_columns": feature_cols,
        "split_cells": {
            name: sorted(frame["cell_id"].astype(str).unique().tolist())
            for name, frame in splits.items()
        },
        "horizons": {},
    }

    for horizon in horizons:
        sequences = {}
        for name, frame in scaled.items():
            sequences[name] = builder.build_sequences(
                frame, lookback=lookback, horizon=horizon,
                feature_cols=feature_cols,
            )
        X_train, y_train = sequences["train"][:2]
        X_val, y_val = sequences["val"][:2]
        X_test, y_test, test_cells = sequences["test"][:3]
        if not len(X_train) or not len(X_val) or not len(X_test):
            raise ValueError(f"horizon={horizon} 时存在空序列划分")

        xgb = XGBoostWrapper()
        xgb.fit(
            EnsembleTrainer._extract_tabular_features(X_train), y_train[:, -1:],
            EnsembleTrainer._extract_tabular_features(X_val), y_val[:, -1:],
        )
        y_true = y_test[:, -1]
        xgb_pred = xgb.predict(
            EnsembleTrainer._extract_tabular_features(X_test)
        ).ravel()
        persistence_pred = X_test[:, -1, soh_index]
        xgb_metrics = _metrics(y_true, xgb_pred)
        persistence_metrics = _metrics(y_true, persistence_pred)

        per_cell = {}
        for cell_id in np.unique(test_cells):
            mask = test_cells == cell_id
            per_cell[str(cell_id)] = {
                "samples": int(mask.sum()),
                "xgb": _metrics(y_true[mask], xgb_pred[mask]),
                "persistence": _metrics(y_true[mask], persistence_pred[mask]),
            }

        results["horizons"][str(horizon)] = {
            "samples": int(len(y_true)),
            "xgb": xgb_metrics,
            "persistence": persistence_metrics,
            "rmse_improvement_ratio": float(
                1.0 - xgb_metrics["rmse"] / persistence_metrics["rmse"]
            ),
            "beats_persistence": bool(xgb_metrics["rmse"] < persistence_metrics["rmse"]),
            "per_cell": per_cell,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="钠电多跨度预测与持久性基线评估")
    parser.add_argument("--datasets", nargs="+", choices=["wenzhou", "mendeley-nfm"],
                        default=["wenzhou", "mendeley-nfm"])
    parser.add_argument("--processed-dir",
                        help="直接使用已审计的 train/val/test Parquet 目录")
    parser.add_argument("--horizons", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--lookback", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.processed_dir:
        processed_dir = Path(args.processed_dir)
        splits = {
            name: pd.read_parquet(processed_dir / f"{name}.parquet")
            for name in ("train", "val", "test")
        }
        result = evaluate_fixed_split_horizons(
            splits, args.horizons, lookback=args.lookback
        )
        result["processed_dir"] = str(processed_dir.resolve())
    else:
        cells = SodiumDatasetLoader().load_all(include=args.datasets)
        extractor = CycleFeatureExtractor()
        raw_features = pd.concat([extractor.extract(cell) for cell in cells], ignore_index=True)
        clean_df = DataCleaner().clean(raw_features)
        result = evaluate_horizons(clean_df, args.horizons, lookback=args.lookback)
        result["datasets"] = args.datasets

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for horizon, item in result["horizons"].items():
        if "summary" in item:
            summary = item["summary"]
            print(
                f"horizon={horizon}: XGB RMSE={summary['xgb_rmse_mean']:.6f}, "
                f"persistence RMSE={summary['persistence_rmse_mean']:.6f}, "
                f"improvement={summary['mean_rmse_improvement_ratio']:.2%}, "
                f"wins={summary['folds_beating_persistence']}/{summary['total_folds']}"
            )
        else:
            print(
                f"horizon={horizon}: XGB RMSE={item['xgb']['rmse']:.6f}, "
                f"persistence RMSE={item['persistence']['rmse']:.6f}, "
                f"improvement={item['rmse_improvement_ratio']:.2%}, "
                f"samples={item['samples']}"
            )
    print(f"结果已保存: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
