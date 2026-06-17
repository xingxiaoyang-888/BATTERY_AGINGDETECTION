#!/usr/bin/env python
# models/soh_ai/run_pipeline.py
"""
SOH AI 数据管线 — 运行入口脚本
====================================
功能:
  1. 检查真实数据可用性
  2. 如有真实数据 → 运行完整管线
  3. 如无真实数据 → 生成合成数据运行管线（用于开发验证）
  4. 输出数据质量报告
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_real_data_available() -> bool:
    """检查是否有可用的真实数据（非 torrent 文件）"""
    from models.soh_ai.config import RAW_DATA_DIR
    from utils.download_wenzhou_data import check_data_integrity

    integrity = check_data_integrity()
    return integrity['can_proceed']


def run_with_real_data():
    """使用真实 Wenzhou 数据运行管线"""
    from utils.soh_data_loader import WenzhouDataLoader
    from models.soh_ai.data_pipeline import SOHDataPipeline

    logger.info("=" * 60)
    logger.info("使用 Wenzhou 真实数据运行 SOH 管线")
    logger.info("=" * 60)

    # 1. 加载真实数据
    loader = WenzhouDataLoader()
    logger.info("\n>>> 加载数据集...")

    # 优先加载钠离子电池数据（核心）
    sodium_cells = loader.load_dataset("sodium-ion")
    logger.info(f"钠离子电芯: {len(sodium_cells)} 个")

    # 如果有冷诅咒数据，一并加载（用于迁移学习）
    try:
        cold_curse_cells = loader.load_dataset("cold-curse")
        logger.info(f"冷诅咒电芯: {len(cold_curse_cells)} 个")
    except Exception:
        cold_curse_cells = []
        logger.warning("冷诅咒数据不可用")

    # 合并所有可用数据
    all_cells = sodium_cells + cold_curse_cells

    if not all_cells:
        logger.error("没有加载到任何电芯数据！")
        return None

    # 2. 运行管线
    logger.info(f"\n>>> 启动数据管线 ({len(all_cells)} 个电芯)...")
    pipeline = SOHDataPipeline()
    result = pipeline.run(all_cells, save=True)

    # 3. 打印质量报告
    print_quality_report(pipeline.quality_report)

    return result


def run_with_synthetic_data():
    """使用合成数据运行管线（开发/测试用）"""
    from models.soh_ai.data_pipeline import (
        SOHDataPipeline, create_synthetic_test_data
    )

    logger.info("=" * 60)
    logger.info("使用合成数据运行 SOH 管线 (DEV MODE)")
    logger.info("=" * 60)

    # 1. 生成合成数据
    logger.info("\n>>> 生成合成测试数据...")
    cells = create_synthetic_test_data(
        n_cells=8,      # 模拟 8 个电芯
        n_cycles=500,   # 每个 500 个循环
        seed=42
    )

    # 2. 运行管线
    logger.info(f"\n>>> 启动数据管线 ({len(cells)} 个电芯)...")
    pipeline = SOHDataPipeline()
    result = pipeline.run(cells, save=True)

    # 3. 打印质量报告
    print_quality_report(pipeline.quality_report)

    return result


def print_quality_report(report: dict):
    """格式化打印数据质量报告"""
    print("\n" + "=" * 60)
    print("  数据质量报告 (Data Quality Report)")
    print("=" * 60)

    print(f"  时间戳:          {report.get('timestamp', 'N/A')}")
    print(f"  原始数据维度:    {report.get('original_shape', 'N/A')}")
    print(f"  清洗后维度:      {report.get('final_shape', 'N/A')}")
    print(f"  覆盖电芯数:      {report.get('cells', 'N/A')}")
    print(f"  丢弃列 (高缺失):  {report.get('dropped_columns', [])}")
    print(f"  拒绝电芯 (循环少): {report.get('rejected_cells', [])}")
    print(f"  异常值替换:      {report.get('outliers_replaced', 0)}")
    print(f"  容量跳变:        {report.get('capacity_jumps', 0)}")
    print(f"  最终 NaN 数:     {report.get('final_nan_count', 0)}")

    print("\n  输出文件:")
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    weights_dir = Path(__file__).resolve().parent / "weights"
    for d in [processed_dir, weights_dir]:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file():
                    size_kb = f.stat().st_size / 1024
                    print(f"    {f.name} ({size_kb:.1f} KB)")

    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SOH AI 数据管线运行脚本')
    parser.add_argument('--synthetic', action='store_true',
                       help='强制使用合成数据（即使有真实数据）')
    parser.add_argument('--force-real', action='store_true',
                       help='强制使用真实数据（即使不全）')

    args = parser.parse_args()

    if args.synthetic:
        result = run_with_synthetic_data()
    elif args.force_real:
        result = run_with_real_data()
    else:
        # 自动检测
        if check_real_data_available():
            result = run_with_real_data()
        else:
            logger.info("真实数据不可用，回退到合成数据模式")
            logger.info("（运行 python utils/download_wenzhou_data.py --method manual 查看下载指引）")
            result = run_with_synthetic_data()

    if result:
        logger.info("\n✓ 管线运行成功！")
        logger.info(f"  训练样本数: {len(result['splits']['train'])}")
        logger.info(f"  验证样本数: {len(result['splits']['val'])}")
        logger.info(f"  测试样本数: {len(result['splits']['test'])}")
        logger.info(f"  序列样本数 (train): {result['sequences']['train'][0].shape[0]}")
        logger.info(f"  特征维度: {result['sequences']['train'][0].shape[2]}")
    else:
        logger.error("✗ 管线运行失败！")


if __name__ == '__main__':
    main()
