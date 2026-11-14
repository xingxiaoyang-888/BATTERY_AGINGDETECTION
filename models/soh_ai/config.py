# models/soh_ai/config.py
"""Configuration and hyperparameters for the legacy SOH experiments."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ============================================================

# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RAW_DATA_DIR = os.path.join(BASE_DIR, "Wenzhou series battery degradation datasets (April_2026)")


SODIUM_RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "sodium_ion")


PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "models", "data", "processed", "sodium_ion")
WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights", "sodium_ion")

# ============================================================

# ============================================================


SODIUM_ION_FILES = {
    "name": "Wenzhou Sodium-ion Battery Degradation Data",
    "torrent": os.path.join(RAW_DATA_DIR, "Wenzhou Sodium-ion Battery Degradation Data.torrent"),
    "cells": [
        {"id": "H-1-2", "xls": "H-1-2.xls", "nda": "H-1-2_127.0.0.1-BTS82-244-7-7-2818579880.nda"},
        {"id": "H-2-1", "xls": "H-2-1.xls", "nda": "H-2-1_127.0.0.1-BTS82-244-8-1-2818579880.nda"},
        {"id": "H-5-1", "xls": "H-5-1.xls", "nda": "H-5-1_127.0.0.1-BTS82..."},
    ],
    "summary": "bs-11.5.xlsx",
}


COLD_CURSE_FILES = {
    "name": "Wenzhou Cold Curse Battery Data",
    "torrent": os.path.join(RAW_DATA_DIR, "Wenzhou Cold Curse Battery Data.torrent"),
    "cell_count": 141,
    "mat_pattern": "Dongzhen-{id:03d}-Detail_Cycle_information.mat",
    "final_mat_pattern": "Dongzhen-{id:03d}-Detail_Final.mat",
}


RANDOMIZED_FILES = {
    "name": "Wenzhou Randomized Battery Data",
    "torrent": os.path.join(RAW_DATA_DIR, "Wenzhou Randomized Battery Data.torrent"),
}


PACK_FILES = {
    "name": "Wenzhou Pack Degradation Data",
    "torrent": os.path.join(RAW_DATA_DIR, "Wenzhou Pack Degradation Data.torrent"),
}

# ============================================================

# ============================================================

@dataclass
class FeatureConfig:
    """Feature Config."""


    lookback_window: int = 32
    prediction_horizon: int = 1
    max_prediction_horizon: int = 128


    target_col: str = "soh"
    aux_targets: List[str] = field(default_factory=lambda: [
        "capacity_ah",
        "internal_resistance_ohm",
    ])



    state_features: List[str] = field(default_factory=lambda: [
        "soh",
        "cycle_index",
        "cumulative_ah_throughput",
        "internal_resistance",
        "coulombic_efficiency",
    ])


    operating_features: List[str] = field(default_factory=lambda: [
        "temperature_c",
        "c_rate_charge",
        "c_rate_discharge",
        "soc_min",
        "soc_max",
        "soc_mean",
        "rest_time_h",
    ])


    trend_features: List[str] = field(default_factory=lambda: [
        "soh_diff_1",
        "soh_diff_3",
        "soh_diff_5",
        "soh_decay_rate",
        "r_diff_1",
        "ce_trend",
        "capacity_fade_acceleration",
    ])


    electrochemical_features: List[str] = field(default_factory=lambda: [
        "dq_dv_peak_shift",
        "dq_dv_peak_height_ratio",
        "mean_charge_voltage",
        "mean_discharge_voltage",
        "voltage_hysteresis",
    ])

    @property
    def all_features(self) -> List[str]:
        """All features."""
        return (
            self.state_features +
            self.operating_features +
            self.trend_features +
            self.electrochemical_features
        )

    @property
    def n_features(self) -> int:
        """N features."""
        return len(ACTUAL_FEATURE_COLUMNS)


FEATURE_CFG = FeatureConfig()


ACTUAL_FEATURE_COLUMNS = [
    "nominal_capacity_ah",
    "soh",
    "cumulative_ah_throughput",
    "coulombic_efficiency",
    "temperature_c",
    "c_rate_charge",
    "c_rate_discharge",
    "rest_time_h",
    "soc_min",
    "soc_max",
    "soc_mean",
    "mean_charge_voltage",
    "mean_discharge_voltage",
    "voltage_hysteresis",
    "soh_diff_1",
    "soh_diff_3",
    "soh_diff_5",
    "soh_decay_rate",
    "capacity_fade_acceleration",
    "ce_trend",
]

# ============================================================

# ============================================================

@dataclass
class XGBoostConfig:
    """XGBoost Config."""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_lambda: float = 1.0
    reg_alpha: float = 0.1
    early_stopping_rounds: int = 50
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"


@dataclass
class LSTMAttentionConfig:
    """LSTMAttention Config."""
    input_dim: int = FEATURE_CFG.n_features
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    attention_dim: int = 64
    bidirectional: bool = True

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 30              # Early stopping
    lr_factor: float = 0.5
    lr_patience: int = 10

    seq_len: int = FEATURE_CFG.lookback_window


@dataclass
class TransformerConfig:
    """Transformer Config."""
    input_dim: int = FEATURE_CFG.n_features
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 128

    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 40
    warmup_steps: int = 1000


@dataclass
class EnsembleConfig:
    """Ensemble Config."""

    xgboost_weight: float = 0.3
    lstm_weight: float = 0.4
    transformer_weight: float = 0.3


    n_bootstrap_samples: int = 100
    confidence_level: float = 0.90


XGB_CFG = XGBoostConfig()
LSTM_CFG = LSTMAttentionConfig()
TRANSFORMER_CFG = TransformerConfig()
ENSEMBLE_CFG = EnsembleConfig()

# ============================================================

# ============================================================

@dataclass
class TrainingConfig:
    """Training Config."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


    n_folds: int = 5
    leave_one_cell_out: bool = True


    device: str = "cuda"            # "cuda" | "cpu"
    num_workers: int = 0


    seed: int = 42


    save_best_only: bool = True
    checkpoint_dir: str = os.path.join(WEIGHTS_DIR, "checkpoints")


    log_interval: int = 10
    use_wandb: bool = False


TRAIN_CFG = TrainingConfig()

# ============================================================

# ============================================================

@dataclass
class InferenceConfig:
    """Inference Config."""

    initial_fusion_weight: float = 0.3
    max_fusion_weight: float = 0.7
    fusion_confidence_threshold: float = 0.6


    eol_soh_threshold: float = 0.80
    rul_early_warning_cycles: int = 100


    knee_soh_window: int = 30
    knee_acceleration_threshold: float = 2.0


    device: str = "cpu"


INFER_CFG = InferenceConfig()

# ============================================================

# ============================================================

@dataclass
class DataQualityConfig:
    """Data Quality Config."""
    max_missing_rate: float = 0.2
    max_outlier_std: float = 5.0
    min_cycles_per_cell: int = 30
    max_capacity_jump: float = 0.05
    soh_smooth_window: int = 5


DQ_CFG = DataQualityConfig()
