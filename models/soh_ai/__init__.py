# models/soh_ai/__init__.py
"""
SOH AI 代理模型 — 数据驱动电池寿命预测模块
===============================================

本模块提供基于 Wenzhou 真实老化数据训练的 AI 代理模型，
用于:
  1. 在线步长 SOH 损耗预测 (替代/校准 NREL 半经验公式)
  2. 长周期寿命轨迹推演 (RUL 预测)
  3. 拐点检测 (加速衰减预警)

子模块:
  - config:          全局配置与超参数
  - data_pipeline:   数据提取、清洗、特征工程管线
  - models:          神经网络架构 (XGBoost / BiLSTM+Attention / Transformer)
  - trainer:         离线训练 + 验证 + Early Stopping + Checkpoint
  - train:           训练 CLI 入口 (含合成数据模式)
  - evaluate:        模型评估、指标计算、可视化

典型用法:

    # --- 数据管线 ---
    from models.soh_ai.data_pipeline import SOHDataPipeline

    pipeline = SOHDataPipeline()
    processed = pipeline.run(cells_data)

    # --- 训练 ---
    from models.soh_ai.trainer import EnsembleTrainer

    trainer = EnsembleTrainer()
    results = trainer.train_all(processed)

    # --- 评估 ---
    from models.soh_ai.evaluate import ModelEvaluator

    evaluator = ModelEvaluator()
    test_results = evaluator.evaluate_all(
        results['ensemble'],
        processed['sequences']['test'][0],
        processed['sequences']['test'][1],
    )
    evaluator.print_summary(test_results)

    # --- 或直接命令行 ---
    # python -m models.soh_ai.train --synthetic --model all
    # python -m models.soh_ai.evaluate --model_dir models/weights/
"""

# ── 配置 ──
from .config import (
    FeatureConfig, FEATURE_CFG,
    XGBoostConfig, XGB_CFG,
    LSTMAttentionConfig, LSTM_CFG,
    TransformerConfig, TRANSFORMER_CFG,
    EnsembleConfig, ENSEMBLE_CFG,
    TrainingConfig, TRAIN_CFG,
    InferenceConfig, INFER_CFG,
    DataQualityConfig, DQ_CFG,
    PROCESSED_DATA_DIR, SODIUM_RAW_DATA_DIR, WEIGHTS_DIR,
)

# ── 数据管线 ──
from .data_pipeline import (
    SOHDataPipeline,
    CycleFeatureExtractor,
    DataCleaner,
    SequenceBuilder,
    DataSplitter,
    create_synthetic_test_data,
)

# ── 模型架构 ──
from .models import (
    XGBoostWrapper,
    BiLSTMAttention,
    TemporalTransformer,
    EnsembleModel,
    AdditiveAttention,
    PositionalEncoding,
)

# ── 训练器 ──
from .trainer import (
    BaseTrainer,
    XGBoostTrainer,
    LSTMTrainer,
    TransformerTrainer,
    EnsembleTrainer,
    CrossValidator,
    TrainingHistory,
)

# ── 评估 ──
from .evaluate import (
    RegressionMetrics,
    Visualizer,
    ModelEvaluator,
)

__all__ = [
    # Config
    'FeatureConfig', 'FEATURE_CFG',
    'XGBoostConfig', 'XGB_CFG',
    'LSTMAttentionConfig', 'LSTM_CFG',
    'TransformerConfig', 'TRANSFORMER_CFG',
    'EnsembleConfig', 'ENSEMBLE_CFG',
    'TrainingConfig', 'TRAIN_CFG',
    'InferenceConfig', 'INFER_CFG',
    'DataQualityConfig', 'DQ_CFG',
    'PROCESSED_DATA_DIR', 'WEIGHTS_DIR',
    'SODIUM_RAW_DATA_DIR',
    # Data Pipeline
    'SOHDataPipeline',
    'CycleFeatureExtractor',
    'DataCleaner',
    'SequenceBuilder',
    'DataSplitter',
    'create_synthetic_test_data',
    # Models
    'XGBoostWrapper',
    'BiLSTMAttention',
    'TemporalTransformer',
    'EnsembleModel',
    'AdditiveAttention',
    'PositionalEncoding',
    # Trainer
    'BaseTrainer',
    'XGBoostTrainer',
    'LSTMTrainer',
    'TransformerTrainer',
    'EnsembleTrainer',
    'CrossValidator',
    'TrainingHistory',
    # Evaluate
    'RegressionMetrics',
    'Visualizer',
    'ModelEvaluator',
]
