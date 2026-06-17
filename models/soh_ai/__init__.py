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
  - models:          神经网络架构 (LSTM / Transformer / XGBoost)
  - trainer:         离线训练 + 验证 + 超参数搜索
  - predictor:       在线推理接口
  - calibration:     模型校准与不确定性量化
  - transfer_learning: 跨体系/跨工况迁移学习

典型用法:

    # --- 训练 ---
    from models.soh_ai.data_pipeline import SOHDataPipeline
    from models.soh_ai.trainer import SOHModelTrainer
    from models.soh_ai.config import *

    pipeline = SOHDataPipeline()
    processed = pipeline.run(cells_data)
    trainer = SOHModelTrainer()
    models = trainer.train_all(processed)

    # --- 推理 ---
    from models.soh_ai.predictor import SOHPredictor

    predictor = SOHPredictor()
    step_loss = predictor.predict_step_loss(temp_c, c_rate, soc, soh)
    trajectory, rul = predictor.predict_trajectory(
        cycle_history, future_conditions, horizon=500
    )
"""

from .config import (
    FeatureConfig, FEATURE_CFG,
    XGBoostConfig, XGB_CFG,
    LSTMAttentionConfig, LSTM_CFG,
    TransformerConfig, TRANSFORMER_CFG,
    EnsembleConfig, ENSEMBLE_CFG,
    TrainingConfig, TRAIN_CFG,
    InferenceConfig, INFER_CFG,
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
]
