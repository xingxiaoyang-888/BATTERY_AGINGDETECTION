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


def run_data_pipeline(use_synthetic: bool = False):
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

    if use_synthetic or not integrity['can_proceed']:
        if not use_synthetic:
            logger.warning(
                "真实数据不可用，回退到合成数据模式。\n"
                "  运行 python utils/download_wenzhou_data.py --method manual 查看下载指引"
            )
        logger.info(f"生成合成测试数据 (8 cells × 500 cycles)...")
        cells = create_synthetic_test_data(n_cells=8, n_cycles=500, seed=42)
    else:
        from utils.soh_data_loader import WenzhouDataLoader
        loader = WenzhouDataLoader()
        logger.info("加载真实 Wenzhou 数据...")
        cells = loader.load_dataset("sodium-ion")
        try:
            cold_curse = loader.load_dataset("cold-curse")
            cells.extend(cold_curse)
        except Exception:
            pass
        logger.info(f"加载了 {len(cells)} 个电芯")

    # 运行管线
    pipeline = SOHDataPipeline()
    result = pipeline.run(cells, save=True)
    logger.info("Phase 1 完成！")
    return result


def train_models(result=None):
    """Phase 2: 模型训练（预留接口，完整实现见 Phase 2）"""
    from models.soh_ai.config import (
        PROCESSED_DATA_DIR, WEIGHTS_DIR,
        XGB_CFG, LSTM_CFG, TRAIN_CFG,
    )

    logger.info("=" * 60)
    logger.info("Phase 2: 模型训练")
    logger.info("=" * 60)

    # 检查处理后的数据是否存在
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.parquet')
    if not os.path.exists(train_path):
        logger.error("未找到处理后的训练数据。请先运行: python train_soh_model.py --pipeline")
        return None

    # --- 导入训练模块 ---
    try:
        from models.soh_ai.trainer import SOHModelTrainer
    except ImportError:
        logger.warning(
            "模型训练模块尚未实现（计划在 Phase 2 完成）。\n"
            "当前数据管线已就绪，可直接用于训练。\n"
            f"处理后的数据位置: {PROCESSED_DATA_DIR}\n"
            f"标准化器位置: {os.path.join(WEIGHTS_DIR, 'soh_scalers.pkl')}"
        )
        return None

    trainer = SOHModelTrainer()
    models = trainer.train_all(processed_dir=PROCESSED_DATA_DIR)
    logger.info("Phase 2 完成！")
    return models


def main():
    parser = argparse.ArgumentParser(
        description='SOH AI 代理模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--pipeline', action='store_true',
                       help='运行数据管线 (Phase 1)')
    parser.add_argument('--train', action='store_true',
                       help='训练模型 (Phase 2)')
    parser.add_argument('--synthetic', action='store_true',
                       help='使用合成数据（开发验证用）')
    parser.add_argument('--all', action='store_true',
                       help='运行完整流程 (Pipeline + Train)')

    args = parser.parse_args()

    # 默认：如果没有任何参数，运行数据管线
    if not args.pipeline and not args.train and not args.all:
        args.pipeline = True

    if args.all:
        args.pipeline = True
        args.train = True

    result = None
    if args.pipeline:
        result = run_data_pipeline(use_synthetic=args.synthetic)

    if args.train:
        train_models(result)


if __name__ == '__main__':
    main()
