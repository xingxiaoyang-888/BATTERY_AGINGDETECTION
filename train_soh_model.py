#!/usr/bin/env python
# train_soh_model.py
"""
SOH AI 代理模型 — 训练入口脚本
=====================================

用法:
    # 完整训练流程（数据管线 + 模型训练）
    python train_soh_model.py --pipeline --train

    # 仅数据管线（跳过训练）
    python train_soh_model.py --pipeline

    # 使用已有处理数据直接训练
    python train_soh_model.py --train

    # 使用合成数据快速验证
    python train_soh_model.py --synthetic --train

训练流程:
    Phase 1: 数据加载 → 特征提取 → 清洗 → 序列构建 → Parquet 存储
    Phase 2: 模型训练 → XGBoost 基线 → BiLSTM 序列模型 → 验证评估
    Phase 3: 模型导出 → 权重保存 → 推理测试
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline(use_synthetic: bool = False, transfer: bool = False,
                      datasets=None):
    """Phase 1: 数据管线"""
    from models.soh_ai.data_pipeline import (
        SOHDataPipeline, create_synthetic_test_data
    )
    from utils.download_wenzhou_data import check_data_integrity

    logger.info("=" * 60)
    logger.info("Phase 1: 数据管线")
    logger.info("=" * 60)

    # 确定数据来源
    integrity = check_data_integrity()

    if use_synthetic:
        logger.info(f"生成合成测试数据 (8 cells × 500 cycles)...")
        cells = create_synthetic_test_data(n_cells=8, n_cycles=500, seed=42)
    elif transfer:
        # 迁移学习模式: 分离源域/目标域
        from utils.soh_data_loader import WenzhouDataLoader
        loader = WenzhouDataLoader()
        logger.info("迁移学习模式: 分离加载源域(Li-ion)和目标域(Na-ion)数据...")
        source_cells = []
        for ds in ["cold-curse", "randomized"]:
            try:
                sc = loader.load_dataset(ds)
                logger.info(f"  [源域] {ds}: {len(sc)} 个电芯")
                source_cells.extend(sc)
            except Exception:
                pass
        target_cells = loader.load_dataset("sodium-ion")
        logger.info(f"  [目标域] sodium-ion: {len(target_cells)} 个电芯")
        cells = source_cells + target_cells
        logger.info(f"源域: {len(source_cells)} 电芯 | 目标域: {len(target_cells)} 电芯")
    else:
        from utils.sodium_dataset_loader import SodiumDatasetLoader
        logger.info("加载纯钠离子电池数据...")
        cells = SodiumDatasetLoader().load_all(include=datasets)
        if not cells:
            raise RuntimeError(
                "没有加载到钠电数据。请检查 Wenzhou H 系列或 data/raw/sodium_ion/。"
            )
        logger.info(f"加载了 {len(cells)} 个钠电电芯")

    # 运行管线（迁移学习模式启用 chemistry_aware）
    pipeline = SOHDataPipeline()
    result = pipeline.run(cells, save=True,
                         chemistry_aware=transfer,
                         target_chemistry="sodium-ion")
    logger.info("Phase 1 完成！")
    return result


def train_models(result=None, transfer: bool = False):
    """Phase 2: 模型训练"""
    from models.soh_ai.config import (
        PROCESSED_DATA_DIR, WEIGHTS_DIR,
        XGB_CFG, LSTM_CFG, TRAIN_CFG,
    )

    logger.info("=" * 60)
    logger.info("Phase 2: 模型训练")
    if transfer:
        logger.info("模式: 迁移学习 (源域预训练 → 目标域微调)")
    logger.info("=" * 60)

    # 检查处理后的数据是否存在
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.parquet')
    if not os.path.exists(train_path):
        logger.error("未找到处理后的训练数据。请先运行: python train_soh_model.py --pipeline")
        return None

    # --- 加载已处理数据并训练 ---
    try:
        from models.soh_ai.trainer import EnsembleTrainer
        from models.soh_ai.data_pipeline import SOHDataPipeline
        import pandas as pd
        import joblib
        import json

        # 重建管线产物
        feature_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, 'feature_table.parquet'))
        splits = {}
        for name in ['train', 'val', 'test']:
            path = os.path.join(PROCESSED_DATA_DIR, f'{name}.parquet')
            if os.path.exists(path):
                splits[name] = pd.read_parquet(path)

        scalers = joblib.load(os.path.join(WEIGHTS_DIR, 'soh_scalers.pkl'))

        with open(os.path.join(PROCESSED_DATA_DIR, 'feature_columns.json'), 'r') as f:
            feature_cols = json.load(f)

        # 检查是否为迁移学习模式
        is_cross = False
        if 'chemistry' in feature_df.columns and feature_df['chemistry'].nunique() > 1:
            is_cross = True

        # 构建序列
        pipeline = SOHDataPipeline()
        scaled_splits = pipeline._fit_scale(splits, feature_df)

        sequences = {}
        for split_name, sdf in scaled_splits.items():
            X, y, _, _ = pipeline.seq_builder.build_sequences(sdf, feature_cols=feature_cols)
            sequences[split_name] = (X, y)

        processed = {
            'feature_df': feature_df,
            'splits': splits,
            'sequences': sequences,
            'scalers': scalers,
            'feature_cols': feature_cols,
            'is_cross_chemistry': is_cross,
        }

        # 如果有源域预训练数据，也加载
        pretrain_path = os.path.join(PROCESSED_DATA_DIR, 'pretrain_source.parquet')
        if transfer and is_cross:
            logger.info("迁移学习模式: 加载源域预训练数据...")
            # 重新运行管线以获取正确的跨域划分
            logger.warning("  建议通过 train.py --transfer 运行完整迁移学习流程。")
            logger.warning("  当前 --train 模式从 parquet 文件恢复，可能缺少源域序列。")

        ensemble = EnsembleTrainer()
        if transfer and is_cross:
            results = ensemble.train_transfer(processed)
        else:
            results = ensemble.train_all(processed)
        ensemble.save_all(WEIGHTS_DIR)
        logger.info("Phase 2 完成！")
        return results
    except ImportError as e:
        logger.warning(
            f"模型训练模块导入失败: {e}\n"
            "当前数据管线已就绪，可直接用于训练。\n"
            f"处理后的数据位置: {PROCESSED_DATA_DIR}\n"
            f"标准化器位置: {os.path.join(WEIGHTS_DIR, 'soh_scalers.pkl')}"
        )
        return None


def main():
    parser = argparse.ArgumentParser(
        description='SOH AI 代理模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认：纯钠电数据管线
  python train_soh_model.py --pipeline

  # 指定数据来源
  python train_soh_model.py --pipeline --datasets wenzhou mendeley-nfm rwth

  # 可选对照：完整迁移学习流程
  python train_soh_model.py --transfer --all

  # 仅数据管线（迁移学习模式）
  python train_soh_model.py --transfer --pipeline

  # 使用合成数据快速验证
  python train_soh_model.py --synthetic --all
        """
    )
    parser.add_argument('--pipeline', action='store_true',
                       help='运行数据管线 (Phase 1)')
    parser.add_argument('--train', action='store_true',
                       help='训练模型 (Phase 2)')
    parser.add_argument('--synthetic', action='store_true',
                       help='使用合成数据（开发验证用）')
    parser.add_argument('--transfer', action='store_true',
                       help='启用迁移学习模式：锂电(源域)预训练 → 钠电(目标域)微调')
    parser.add_argument('--datasets', nargs='+',
                       choices=['wenzhou', 'mendeley-nfm', 'rwth'],
                       default=['wenzhou', 'mendeley-nfm', 'rwth'],
                       help='纯钠电模式使用的数据源')
    parser.add_argument('--all', action='store_true',
                       help='运行完整流程 (Pipeline + Train)')

    args = parser.parse_args()

    # 默认：如果没有任何参数，运行数据管线
    if not args.pipeline and not args.train and not args.all:
        args.pipeline = True

    if args.all:
        args.pipeline = True
        args.train = True

    is_transfer = getattr(args, 'transfer', False)

    result = None
    if args.pipeline:
        result = run_data_pipeline(
            use_synthetic=args.synthetic,
            transfer=is_transfer,
            datasets=args.datasets,
        )

    if args.train:
        train_models(result, transfer=is_transfer)


if __name__ == '__main__':
    main()
