"""钠离子电池训练前的数据审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from models.soh_ai.config import PROCESSED_DATA_DIR, SODIUM_RAW_DATA_DIR


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> Dict[str, dict]:
    """记录原始数据文件大小和 SHA256，确保本地与服务器数据一致。"""
    manifest = {}
    if not root.exists():
        return manifest
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        manifest[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _hash_file(path),
        }
    return manifest


def audit_feature_table(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"available": False, "passed": False, "errors": [f"缺少处理数据: {path}"]}

    df = pd.read_parquet(path)
    errors = []
    warnings = []
    chemistry = set(df.get("chemistry", pd.Series(dtype=str)).dropna().unique())
    if chemistry != {"sodium-ion"}:
        errors.append(f"处理数据不是纯钠电，发现化学体系: {sorted(chemistry)}")
    if df.get("cell_id", pd.Series(dtype=str)).nunique() < 3:
        errors.append("独立数据实体少于 3 个，无法形成训练/验证/测试划分")

    range_rules = {
        "soh": (0.5, 1.1),
        "coulombic_efficiency": (0.0, 1.2),
        "c_rate_charge": (0.0, 10.0),
        "c_rate_discharge": (0.0, 10.0),
        "temperature_c": (-50.0, 80.0),
        "nominal_capacity_ah": (0.01, 500.0),
        "internal_resistance": (0.0, 10.0),
    }
    ranges = {}
    for col, (lower, upper) in range_rules.items():
        if col not in df.columns:
            warnings.append(f"缺少可选特征列: {col}")
            continue
        values = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if values.empty:
            warnings.append(f"特征列无有效数据: {col}")
            continue
        ranges[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
        }
        invalid = int((~values.between(lower, upper)).sum())
        if invalid:
            errors.append(f"{col} 有 {invalid} 个值超出 [{lower}, {upper}]")

    return {
        "available": True,
        "rows": int(len(df)),
        "cells": int(df["cell_id"].nunique()),
        "datasets": (
            df.groupby("dataset_id")["cell_id"].nunique().to_dict()
            if "dataset_id" in df.columns else {}
        ),
        "ranges": ranges,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计钠电原始数据与处理产物")
    parser.add_argument("--raw-dir", default=SODIUM_RAW_DATA_DIR)
    parser.add_argument("--processed-dir", default=PROCESSED_DATA_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    report = {
        "raw_root": str(raw_dir.resolve()),
        "processed_root": str(processed_dir.resolve()),
        "raw_manifest": build_manifest(raw_dir),
        "feature_audit": audit_feature_table(processed_dir / "feature_table.parquet"),
    }

    output = Path(args.output) if args.output else processed_dir / "data_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["feature_audit"], ensure_ascii=False, indent=2))
    print(f"审计报告: {output}")
    return 0 if report["feature_audit"].get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
