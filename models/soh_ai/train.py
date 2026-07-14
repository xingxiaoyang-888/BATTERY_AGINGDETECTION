#!/usr/bin/env python
# models/soh_ai/train.py
"""
SOH AI 模型训练 — 命令行入口
=================================

用法:
  # 全量训练（所有模型）
  python -m models.soh_ai.train

  # 仅训练指定模型
  python -m models.soh_ai.train --model lstm
  python -m models.soh_ai.train --model xgb --model transformer

  # 自定义超参数
  python -m models.soh_ai.train --model lstm --epochs 300 --batch_size 32 --lr 0.001

  # 使用合成数据（无真实数据时）
  python -m models.soh_ai.train --synthetic --model all

  # 从检查点恢复训练
  python -m models.soh_ai.train --resume models/weights/checkpoints/lstm_attention_best.pt

  # K-Fold 交叉验证
  python -m models.soh_ai.train --kfold 5

设计原则:
  1. 所有参数通过命令行传递，方便远程服务器运行
  2. 训练失败时保留已有模型和日志
  3. 详细的分阶段日志，方便排查问题
"""

import sys
import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# 确保项目根目录在 Python path 中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ═══════════════════════════════════════════════════════════════
# 日志配置
# ═══════════════════════════════════════════════════════════════

def setup_logging(log_dir: str = None, verbose: bool = False):
    """配置双通道日志：控制台 + 文件"""
    log_dir = Path(log_dir or (_project_root / "models" / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'
    datefmt = '%H:%M:%S'

    # 根 logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(console)

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return log_file


# ═══════════════════════════════════════════════════════════════
# 训练主函数
# ═══════════════════════════════════════════════════════════════

def run_training(args: argparse.Namespace) -> int:
    """
    执行训练流程。

    Returns:
        int: 0 = 成功, 非0 = 失败
    """
    logger = logging.getLogger(__name__)

    # ── 1. 配置检查 ──
    logger.info("=" * 70)
    logger.info("  SOH AI 模型训练启动")
    logger.info("=" * 70)
    logger.info(f"  时间: {datetime.now().isoformat()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  项目根目录: {_project_root}")

    # 检查 PyTorch / CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        logger.info(f"  PyTorch: {torch.__version__}, CUDA: {cuda_ok}")
        if cuda_ok:
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("  PyTorch 未安装! 请运行: pip install torch")
        return 1

    # ── 2. 加载数据 ──
    logger.info("\n" + "=" * 70)
    logger.info("  阶段 A: 数据加载")
    logger.info("=" * 70)

    if args.synthetic:
        logger.info("  使用合成数据模式 (--synthetic)")
        from models.soh_ai.data_pipeline import (
            SOHDataPipeline, create_synthetic_test_data
        )
        cells = create_synthetic_test_data(
            n_cells=args.syn_cells,
            n_cycles=args.syn_cycles,
            seed=args.seed,
        )
    else:
        from utils.download_wenzhou_data import check_data_integrity
        integrity = check_data_integrity()
        if not integrity['can_proceed']:
            logger.error("  真实数据不可用! 请先下载数据集:")
            logger.error("    python utils/download_wenzhou_data.py --method manual")
            logger.error("  或使用合成数据: --synthetic")
            return 2

        logger.info(f"  真实数据完整性: {integrity}")
        from utils.soh_data_loader import WenzhouDataLoader
        loader = WenzhouDataLoader()
        all_cells = []
        for ds in ["sodium-ion", "cold-curse"]:
            try:
                cells = loader.load_dataset(ds)
                logger.info(f"  ✓ {ds}: {len(cells)} 个电芯")
                all_cells.extend(cells)
            except Exception as e:
                logger.warning(f"  ✗ {ds}: {e}")

        if not all_cells:
            logger.error("  未能加载任何电芯数据!")
            return 2
        cells = all_cells

    logger.info(f"  总计 {len(cells)} 个电芯用于训练")

    # ── 3. 数据管线 ──
    logger.info("\n" + "=" * 70)
    logger.info("  阶段 B: 数据管线处理")
    logger.info("=" * 70)

    from models.soh_ai.data_pipeline import SOHDataPipeline
    pipeline = SOHDataPipeline()
    try:
        processed = pipeline.run(cells, save=True)
    except Exception as e:
        logger.error(f"  数据管线失败: {e}", exc_info=args.verbose)
        return 3

    # 打印数据摘要
    logger.info(f"  特征表: {processed['feature_df'].shape}")
    for name, (X, y) in processed['sequences'].items():
        logger.info(f"  {name}: X{X.shape}, y{y.shape}")

    # ── 4. 模型训练 ──
    logger.info("\n" + "=" * 70)
    logger.info("  阶段 C: 模型训练")
    logger.info("=" * 70)

    from models.soh_ai.trainer import (
        XGBoostTrainer, LSTMTrainer, TransformerTrainer, EnsembleTrainer,
    )
    from models.soh_ai.config import WEIGHTS_DIR

    models_to_train = args.model if args.model else ['all']
    if 'all' in models_to_train:
        models_to_train = ['xgb', 'lstm', 'transformer']

    logger.info(f"  待训练模型: {models_to_train}")
    skip_xgb = 'xgb' not in models_to_train
    skip_lstm = 'lstm' not in models_to_train
    skip_transformer = 'transformer' not in models_to_train

    # 自定义超参数
    epoch_overrides = {}
    if args.epochs:
        epoch_overrides['lstm'] = args.epochs
        epoch_overrides['transformer'] = args.epochs
    if args.batch_size:
        epoch_overrides['lstm_batch'] = args.batch_size
        epoch_overrides['transformer_batch'] = args.batch_size

    ensemble_trainer = EnsembleTrainer()

    try:
        results = ensemble_trainer.train_all(
            processed,
            epochs=epoch_overrides if epoch_overrides else None,
            skip_xgb=skip_xgb,
            skip_lstm=skip_lstm,
            skip_transformer=skip_transformer,
        )
    except Exception as e:
        logger.error(f"  训练失败: {e}", exc_info=args.verbose)
        return 4

    # ── 5. 保存 ──
    logger.info("\n" + "=" * 70)
    logger.info("  阶段 D: 持久化")
    logger.info("=" * 70)

    output_dir = args.output_dir or str(WEIGHTS_DIR)
    ensemble_trainer.save_all(output_dir)

    # 保存运行配置（用于复现）
    run_config = {
        'timestamp': datetime.now().isoformat(),
        'arguments': vars(args),
        'models_trained': results['available_models'],
        'test_results': results.get('test_results', {}),
    }
    config_path = Path(output_dir) / 'run_config.json'
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  运行配置已保存: {config_path}")

    # ── 6. 摘要 ──
    logger.info("\n" + "=" * 70)
    logger.info("  训练摘要")
    logger.info("=" * 70)
    logger.info(f"  已训练模型: {results['available_models']}")
    if results.get('test_results'):
        for model_name, metrics in results['test_results'].items():
            logger.info(f"  {model_name:12s} → RMSE: {metrics['RMSE']:.6f}, "
                       f"MAE: {metrics['MAE']:.6f}")

    logger.info(f"\n  模型文件位置: {output_dir}")
    logger.info("  训练完成! ✓")
    return 0


# ═══════════════════════════════════════════════════════════════
# 命令行解析
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='SOH AI 模型训练 — 钠离子电池寿命预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全量训练
  python -m models.soh_ai.train

  # 仅训练 LSTM
  python -m models.soh_ai.train --model lstm --epochs 200

  # 合成数据调试
  python -m models.soh_ai.train --synthetic --model all

  # 5-Fold 交叉验证
  python -m models.soh_ai.train --kfold 5 --model lstm
        """
    )

    # ── 数据参数 ──
    data = parser.add_argument_group('数据')
    data.add_argument('--synthetic', action='store_true',
                     help='使用合成数据（无真实数据时自动启用）')
    data.add_argument('--syn-cells', type=int, default=8,
                     help='合成电芯数量 (默认: 8)')
    data.add_argument('--syn-cycles', type=int, default=500,
                     help='每个合成电芯的循环数 (默认: 500)')

    # ── 模型选择 ──
    model = parser.add_argument_group('模型选择')
    model.add_argument('--model', type=str, nargs='+',
                      choices=['all', 'xgb', 'lstm', 'transformer'],
                      default=['all'],
                      help='要训练的模型 (默认: all)')

    # ── 训练超参数 ──
    train = parser.add_argument_group('训练超参数')
    train.add_argument('--epochs', type=int, default=None,
                      help='最大训练轮数 (覆盖配置默认值)')
    train.add_argument('--batch_size', type=int, default=None,
                      help='批次大小 (覆盖配置默认值)')
    train.add_argument('--lr', type=float, default=None,
                      help='学习率 (覆盖配置默认值)')
    train.add_argument('--patience', type=int, default=None,
                      help='早停等待轮数')
    train.add_argument('--seed', type=int, default=42,
                      help='随机种子 (默认: 42)')
    train.add_argument('--kfold', type=int, default=None,
                      help='K-Fold 交叉验证折数 (如: 5)')
    train.add_argument('--resume', type=str, default=None,
                      help='从指定检查点恢复训练')
    train.add_argument('--no-cuda', action='store_true',
                      help='强制使用 CPU 训练')

    # ── 输出参数 ──
    output = parser.add_argument_group('输出')
    output.add_argument('--output-dir', type=str, default=None,
                       help='模型输出目录 (默认: models/weights/)')
    output.add_argument('--log-dir', type=str, default=None,
                       help='日志输出目录 (默认: models/logs/)')
    output.add_argument('-v', '--verbose', action='store_true',
                       help='详细日志输出（DEBUG 级别）')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 日志初始化
    log_file = setup_logging(args.log_dir, args.verbose)
    logger = logging.getLogger(__name__)

    # 随机种子
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 强制 CPU
    if args.no_cuda:
        from models.soh_ai.config import TRAIN_CFG
        TRAIN_CFG.device = "cpu"
        logger.info("  强制使用 CPU 训练 (--no-cuda)")

    # 运行训练
    try:
        exit_code = run_training(args)
    except KeyboardInterrupt:
        logger.info("\n  训练被用户中断 (Ctrl+C)")
        exit_code = 130
    except Exception as e:
        logger.error(f"\n  未捕获异常: {e}", exc_info=True)
        exit_code = 99

    logger.info(f"\n退出码: {exit_code}")
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
