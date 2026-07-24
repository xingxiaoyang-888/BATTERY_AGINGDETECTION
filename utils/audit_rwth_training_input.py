"""审计 RWTH 原始循环数据与最终训练输入。"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from models.soh_ai.config import FEATURE_CFG, PROCESSED_DATA_DIR, SODIUM_RAW_DATA_DIR
from models.soh_ai.data_pipeline import SequenceBuilder
from utils.sodium_dataset_loader import RWTHCommercialAgingLoader


def audit_raw_rwth(base_dir: Path) -> dict:
    started = time.time()
    cells = RWTHCommercialAgingLoader.load(str(base_dir))
    details = {}
    issues = []
    for cell in cells:
        capacities = np.asarray(
            [cycle.discharge_capacity_ah for cycle in cell.cycles], dtype=float
        )
        detail = {
            "condition": cell.metadata.get("condition"),
            "cycles": len(cell.cycles),
            "capacity_min_ah": float(capacities.min()),
            "capacity_median_ah": float(np.median(capacities)),
            "capacity_max_ah": float(capacities.max()),
            "reference_capacity_checks": len(
                cell.metadata.get("reference_capacity_checks", [])
            ),
        }
        details[cell.cell_id] = detail
        if len(capacities) < 30:
            issues.append(f"{cell.cell_id}: 少于 30 个完整 DOD100 训练循环")
        if not np.isfinite(capacities).all() or (capacities <= 0).any():
            issues.append(f"{cell.cell_id}: 容量包含非有限值或非正值")

    condition_counts = Counter(cell.metadata.get("condition") for cell in cells)
    return {
        "passed": len(cells) == 47 and not issues,
        "source_cells": 67,
        "training_cells": len(cells),
        "held_out_sparse_or_partial_dod_cells": 20,
        "cycles_total": sum(len(cell.cycles) for cell in cells),
        "conditions": dict(sorted(condition_counts.items())),
        "issues": issues,
        "elapsed_seconds": time.time() - started,
        "cell_details": details,
    }


def audit_processed_input(processed_dir: Path) -> dict:
    feature_path = processed_dir / "feature_table.parquet"
    if not feature_path.exists():
        return {"passed": False, "issues": [f"缺少 {feature_path}"]}

    df = pd.read_parquet(feature_path)
    issues = []
    if set(df["chemistry"].unique()) != {"sodium-ion"}:
        issues.append("训练输入混入非钠离子电池")
    if df["cell_id"].nunique() != 47:
        issues.append(f"清洗后应为 47 只 DOD100 电芯，实际 {df['cell_id'].nunique()} 只")
    if int(df.isna().sum().sum()) != 0:
        issues.append("训练输入仍含 NaN")
    if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
        issues.append("训练输入仍含无穷值")

    split_cells = {}
    split_rows = {}
    split_frames = {}
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            issues.append(f"缺少 {path.name}")
            continue
        frame = pd.read_parquet(path)
        split_frames[name] = frame
        split_cells[name] = sorted(frame["cell_id"].unique().tolist())
        split_rows[name] = len(frame)

    if len(split_cells) == 3:
        sets = {name: set(cells) for name, cells in split_cells.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
            issues.append("训练/验证/测试存在电芯重叠")

    exclude = {
        "cell_id", "chemistry", "dataset_id", "condition", "cycle_index",
        "soh_raw", "soh_jump_flag",
    }
    feature_cols = [
        column for column in df.columns
        if column not in exclude and df[column].dtype.kind in "fiu"
    ]
    sequence_counts = {}
    builder = SequenceBuilder()
    for horizon in (1, 16, 32, 64, 128):
        sequence_counts[str(horizon)] = {}
        for name, frame in split_frames.items():
            _, targets, _, _ = builder.build_sequences(
                frame, lookback=FEATURE_CFG.lookback_window,
                horizon=horizon, feature_cols=feature_cols,
            )
            sequence_counts[str(horizon)][name] = int(len(targets))
            if len(targets) == 0:
                issues.append(f"{name} 在 horizon={horizon} 时没有序列样本")

    soh = pd.to_numeric(df["soh"], errors="coerce")
    return {
        "passed": not issues,
        "rows": len(df),
        "cells": int(df["cell_id"].nunique()),
        "conditions": int(df["condition"].nunique()) if "condition" in df else 0,
        "soh_range": [float(soh.min()), float(soh.max())],
        "split_rows": split_rows,
        "split_cells": split_cells,
        "sequence_counts": sequence_counts,
        "feature_columns": feature_cols,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 RWTH 训练输入")
    parser.add_argument(
        "--raw-dir", default=str(Path(SODIUM_RAW_DATA_DIR) / "rwth_67_cells")
    )
    parser.add_argument("--processed-dir", default=PROCESSED_DATA_DIR)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = {
        "raw": None if args.skip_raw else audit_raw_rwth(Path(args.raw_dir)),
        "processed": audit_processed_input(Path(args.processed_dir)),
    }
    report["passed"] = report["processed"]["passed"] and (
        report["raw"] is None or report["raw"]["passed"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "raw": None if report["raw"] is None else {
            key: value for key, value in report["raw"].items()
            if key != "cell_details"
        },
        "processed": {
            key: value for key, value in report["processed"].items()
            if key != "split_cells"
        },
    }, ensure_ascii=False, indent=2))
    print(f"审计报告: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
