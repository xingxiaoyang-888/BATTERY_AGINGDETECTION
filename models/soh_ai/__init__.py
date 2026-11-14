# models/soh_ai/__init__.py
"""Init."""


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


from .data_pipeline import (
    SOHDataPipeline,
    CycleFeatureExtractor,
    DataCleaner,
    SequenceBuilder,
    DataSplitter,
    create_synthetic_test_data,
)


TORCH_AVAILABLE = False
try:
    from .models import (
        XGBoostWrapper,
        BiLSTMAttention,
        TemporalTransformer,
        EnsembleModel,
        AdditiveAttention,
        PositionalEncoding,
    )
    from .trainer import (
        BaseTrainer,
        XGBoostTrainer,
        LSTMTrainer,
        TransformerTrainer,
        EnsembleTrainer,
        CrossValidator,
        TrainingHistory,
    )
    from .evaluate import RegressionMetrics, Visualizer, ModelEvaluator
    TORCH_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise

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
    'TORCH_AVAILABLE',
    # Data Pipeline
    'SOHDataPipeline',
    'CycleFeatureExtractor',
    'DataCleaner',
    'SequenceBuilder',
    'DataSplitter',
    'create_synthetic_test_data',
]

if TORCH_AVAILABLE:
    __all__ += [
        'XGBoostWrapper', 'BiLSTMAttention', 'TemporalTransformer',
        'EnsembleModel', 'AdditiveAttention', 'PositionalEncoding',
        'BaseTrainer', 'XGBoostTrainer', 'LSTMTrainer',
        'TransformerTrainer', 'EnsembleTrainer', 'CrossValidator',
        'TrainingHistory', 'RegressionMetrics', 'Visualizer', 'ModelEvaluator',
    ]
