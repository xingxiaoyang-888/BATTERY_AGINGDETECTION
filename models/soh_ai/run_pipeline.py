#!/usr/bin/env python
# models/soh_ai/run_pipeline.py
"""Run pipeline."""

import sys
import os
import logging
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_real_data_available() -> bool:
    """Check real data available."""
    from models.soh_ai.config import SODIUM_RAW_DATA_DIR
    from utils.download_wenzhou_data import check_data_integrity

    integrity = check_data_integrity()
    nfm_dir = Path(SODIUM_RAW_DATA_DIR) / 'mendeley_nfm'
    has_nfm = any((nfm_dir / name).exists() for name in ('Dataset 1-1.xlsx', 'Dataset 1-2.xlsx'))
    return integrity['can_proceed'] or has_nfm


def run_with_real_data(transfer: bool = False, datasets=None):
    """Run with real data."""
    from utils.soh_data_loader import WenzhouDataLoader
    from models.soh_ai.data_pipeline import SOHDataPipeline

    logger.info("=" * 60)
    logger.info("使用 Wenzhou 真实数据运行 SOH 管线")
    if transfer:
        logger.info("模式: 迁移学习 (化学体系感知)")
    logger.info("=" * 60)


    loader = WenzhouDataLoader()
    logger.info("\n>>> 加载数据集...")

    if transfer:

        source_cells = []
        for ds in ["cold-curse", "randomized"]:
            try:
                sc = loader.load_dataset(ds)
                logger.info(f"[源域] {ds}: {len(sc)} 个电芯")
                source_cells.extend(sc)
            except Exception:
                pass
        target_cells = loader.load_dataset("sodium-ion")
        logger.info(f"[目标域] sodium-ion: {len(target_cells)} 个电芯")
        all_cells = source_cells + target_cells
        logger.info(f"总计: 源域 {len(source_cells)} + 目标域 {len(target_cells)} = {len(all_cells)} 电芯")
    else:

        from utils.sodium_dataset_loader import SodiumDatasetLoader
        all_cells = SodiumDatasetLoader().load_all(include=datasets)
        logger.info(f"钠离子电芯: {len(all_cells)} 个")

    if not all_cells:
        logger.error("没有加载到任何电芯数据！")
        return None


    logger.info(f"\n>>> 启动数据管线 ({len(all_cells)} 个电芯)...")
    pipeline = SOHDataPipeline()
    result = pipeline.run(all_cells, save=True,
                         chemistry_aware=transfer,
                         target_chemistry="sodium-ion")

    if transfer and result.get('is_cross_chemistry'):
        logger.info(f"  跨化学体系: {result['source_chemistry']} → {result['target_chemistry']}")
        logger.info(f"  源域电芯: {result.get('source_cells', [])}")
        logger.info(f"  目标域电芯: {result.get('target_cells', [])}")


    print_quality_report(pipeline.quality_report)

    return result


def run_with_synthetic_data():
    """Run with synthetic data."""
    from models.soh_ai.data_pipeline import (
        SOHDataPipeline, create_synthetic_test_data
    )

    logger.info("=" * 60)
    logger.info("使用合成数据运行 SOH 管线 (DEV MODE)")
    logger.info("=" * 60)


    logger.info("\n>>> 生成合成测试数据...")
    cells = create_synthetic_test_data(
        n_cells=8,
        n_cycles=500,
        seed=42
    )


    logger.info(f"\n>>> 启动数据管线 ({len(cells)} 个电芯)...")
    pipeline = SOHDataPipeline()
    result = pipeline.run(cells, save=True)


    print_quality_report(pipeline.quality_report)

    return result


def print_quality_report(report: dict):
    """Print quality report."""
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
    from models.soh_ai.config import PROCESSED_DATA_DIR, WEIGHTS_DIR
    processed_dir = Path(PROCESSED_DATA_DIR)
    weights_dir = Path(WEIGHTS_DIR)
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
    parser.add_argument('--transfer', action='store_true',
                       help='启用迁移学习模式：锂电(源域) + 钠电(目标域) 分离加载')
    parser.add_argument('--datasets', nargs='+',
                       choices=['wenzhou', 'mendeley-nfm', 'rwth'],
                       default=['wenzhou', 'mendeley-nfm'],
                       help='纯钠电模式使用的数据源')

    args = parser.parse_args()

    is_transfer = getattr(args, 'transfer', False)

    if args.synthetic:
        result = run_with_synthetic_data()
    elif args.force_real:
        result = run_with_real_data(transfer=is_transfer, datasets=args.datasets)
    else:

        if check_real_data_available():
            result = run_with_real_data(transfer=is_transfer, datasets=args.datasets)
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
