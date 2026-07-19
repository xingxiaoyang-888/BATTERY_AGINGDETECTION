# models/soh_ai/config.py
"""
SOH AI 代理模型 — 全局配置与超参数定义
============================================
统一管理：数据路径、特征定义、模型超参数、训练配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据集原始目录（Wenzhou 系列）
RAW_DATA_DIR = os.path.join(BASE_DIR, "Wenzhou series battery degradation datasets (April_2026)")

# 已提取/处理的中间数据目录
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "models", "data", "processed")

# 模型权重保存目录
WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")

# ============================================================
# 2. 数据集结构描述
# ============================================================

# 钠离子电池老化数据（核心训练集）
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

# 冷诅咒电池数据（迁移学习源域，141 个电芯）
COLD_CURSE_FILES = {
    "name": "Wenzhou Cold Curse Battery Data",
    "torrent": os.path.join(RAW_DATA_DIR, "Wenzhou Cold Curse Battery Data.torrent"),
    "cell_count": 141,
    "mat_pattern": "Dongzhen-{id:03d}-Detail_Cycle_information.mat",
    "final_mat_pattern": "Dongzhen-{id:03d}-Detail_Final.mat",
}

# 随机工况电池数据（泛化增强）
RANDOMIZED_FILES = {
    "name": "Wenzhou Randomized Battery Data",
    "torrent": os.path.join(RAW_DATA_DIR, "Wenzhou Randomized Battery Data.torrent"),
}

# Pack 级老化数据（Pack 级验证）
PACK_FILES = {
    "name": "Wenzhou Pack Degradation Data",
    "torrent": os.path.join(RAW_DATA_DIR, "Wenzhou Pack Degradation Data.torrent"),
}

# ============================================================
# 3. 特征工程配置
# ============================================================

@dataclass
class FeatureConfig:
    """
    循环级特征定义
    ===============
    每个充放电循环提取的特征向量维度与含义
    """

    # ---- 滑动窗口参数 ----
    lookback_window: int = 32        # 输入序列长度（用前 32 个循环预测未来）
    prediction_horizon: int = 1      # 默认预测步长（1 步 = 1 个循环）
    max_prediction_horizon: int = 128  # 最大轨迹推演长度

    # ---- 目标变量 ----
    target_col: str = "soh"          # 主目标：健康状态 SOH (0~1)
    aux_targets: List[str] = field(default_factory=lambda: [
        "capacity_ah",               # 实际放电容量 (Ah)
        "internal_resistance_ohm",   # 直流内阻 (Ω)
    ])

    # ---- 输入特征分组 ----
    # A. 当前状态特征（标量，每循环 1 个值）
    state_features: List[str] = field(default_factory=lambda: [
        "soh",                       # 当前 SOH
        "cycle_index",               # 循环序数（归一化到 0~1）
        "cumulative_ah_throughput",  # 累计安时吞吐量 (Ah)
        "internal_resistance",       # 当前内阻 (Ω)
        "coulombic_efficiency",      # 库仑效率
    ])

    # B. 工况特征（标量）
    operating_features: List[str] = field(default_factory=lambda: [
        "temperature_c",             # 环境/电芯温度 (°C)
        "c_rate_charge",             # 充电倍率 (C)
        "c_rate_discharge",          # 放电倍率 (C)
        "soc_min",                   # SOC 窗口下限
        "soc_max",                   # SOC 窗口上限
        "soc_mean",                  # 平均 SOC
        "rest_time_h",               # 静置时间 (小时)
    ])

    # C. 趋势特征（从历史序列中提取）
    trend_features: List[str] = field(default_factory=lambda: [
        "soh_diff_1",                # SOH 一阶差分 (最近 1 步)
        "soh_diff_3",                # SOH 三阶差分 (最近 3 步)
        "soh_diff_5",                # SOH 五阶差分 (最近 5 步)
        "soh_decay_rate",            # 指数衰减率拟合
        "r_diff_1",                  # 内阻一阶差分
        "ce_trend",                  # 库仑效率趋势 (最近 5 步均值)
        "capacity_fade_acceleration", # 容量衰减加速度（二阶差分）
    ])

    # D. 电化学特征（从充放电曲线中提取，需要原始 V-Q 数据）
    electrochemical_features: List[str] = field(default_factory=lambda: [
        "dq_dv_peak_shift",          # dQ/dV 峰位偏移 (mV)
        "dq_dv_peak_height_ratio",   # dQ/dV 峰高比值
        "mean_charge_voltage",       # 平均充电电压 (V)
        "mean_discharge_voltage",    # 平均放电电压 (V)
        "voltage_hysteresis",        # 电压滞回 (V)
    ])

    @property
    def all_features(self) -> List[str]:
        """返回所有输入特征的完整列表"""
        return (
            self.state_features +
            self.operating_features +
            self.trend_features +
            self.electrochemical_features
        )

    @property
    def n_features(self) -> int:
        """输入特征总数。

        训练管线当前稳定产出 11 维特征，直接返回真实维度，避免
        “概念特征集”与实际输入列数漂移。
        """
        return 11


# 全局特征配置实例
FEATURE_CFG = FeatureConfig()

# 当前训练管线稳定使用的实际输入特征列
ACTUAL_FEATURE_COLUMNS = [
    "soh",
    "cumulative_ah_throughput",
    "internal_resistance",
    "coulombic_efficiency",
    "temperature_c",
    "c_rate_charge",
    "c_rate_discharge",
    "rest_time_h",
    "soc_min",
    "soc_max",
    "soc_mean",
]

# ============================================================
# 4. 模型超参数配置
# ============================================================

@dataclass
class XGBoostConfig:
    """XGBoost 基线模型超参数"""
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
    """BiLSTM + Attention 序列模型超参数"""
    input_dim: int = FEATURE_CFG.n_features  # 输入特征维度
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    attention_dim: int = 64
    bidirectional: bool = True
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 30              # Early stopping
    lr_factor: float = 0.5          # ReduceLROnPlateau 衰减因子
    lr_patience: int = 10           # ReduceLROnPlateau 等待轮数
    # 数据
    seq_len: int = FEATURE_CFG.lookback_window


@dataclass
class TransformerConfig:
    """Temporal Transformer 模型超参数"""
    input_dim: int = FEATURE_CFG.n_features
    d_model: int = 128              # 嵌入维度
    nhead: int = 8                  # 多头注意力头数
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 128          # 最大序列长度（含位置编码）
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 40
    warmup_steps: int = 1000


@dataclass
class EnsembleConfig:
    """模型集成配置"""
    # 三个模型的融合权重（可动态调整）
    xgboost_weight: float = 0.3     # 鲁棒基线
    lstm_weight: float = 0.4        # 序列主力
    transformer_weight: float = 0.3 # 长程依赖

    # 不确定性量化
    n_bootstrap_samples: int = 100  # Bootstrap 采样数
    confidence_level: float = 0.90  # 置信水平


# 全局模型配置实例
XGB_CFG = XGBoostConfig()
LSTM_CFG = LSTMAttentionConfig()
TRANSFORMER_CFG = TransformerConfig()
ENSEMBLE_CFG = EnsembleConfig()

# ============================================================
# 5. 训练配置
# ============================================================

@dataclass
class TrainingConfig:
    """离线训练全局配置"""
    # 数据划分
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 交叉验证（针对钠电数据量少的场景）
    n_folds: int = 5                # K-Fold 折数
    leave_one_cell_out: bool = True # 是否使用留一电芯交叉验证

    # 设备
    device: str = "cuda"            # "cuda" | "cpu"
    num_workers: int = 0            # DataLoader 工作进程数 (0=主进程, 避免多进程复制数据OOM)

    # 随机种子
    seed: int = 42

    # 模型保存
    save_best_only: bool = True
    checkpoint_dir: str = os.path.join(WEIGHTS_DIR, "checkpoints")

    # 日志
    log_interval: int = 10          # 每 N 个 epoch 记录一次
    use_wandb: bool = False         # 是否使用 Weights & Biases


TRAIN_CFG = TrainingConfig()

# ============================================================
# 6. 推理与部署配置
# ============================================================

@dataclass
class InferenceConfig:
    """在线推理配置"""
    # 物理+AI 融合
    initial_fusion_weight: float = 0.3   # AI 初始权重（初期以物理为主）
    max_fusion_weight: float = 0.7       # AI 最大权重
    fusion_confidence_threshold: float = 0.6  # 置信度阈值

    # RUL 预测
    eol_soh_threshold: float = 0.80      # 寿命终止 SOH 阈值
    rul_early_warning_cycles: int = 100  # 提前 N 个循环预警

    # 拐点检测
    knee_soh_window: int = 30            # 拐点检测窗口（最近 N 个循环）
    knee_acceleration_threshold: float = 2.0  # 衰减加速度阈值（× 基线）

    # 设备
    device: str = "cpu"                  # 推理设备（服务器通常用 CPU）


INFER_CFG = InferenceConfig()

# ============================================================
# 7. 数据质量阈值
# ============================================================

@dataclass
class DataQualityConfig:
    """数据质量检查阈值"""
    max_missing_rate: float = 0.2        # 单列最大缺失率（超过则丢弃该特征）
    max_outlier_std: float = 5.0         # 异常值判定（偏离均值 N 个标准差）
    min_cycles_per_cell: int = 30        # 单个电芯最少循环数（不足则丢弃）
    max_capacity_jump: float = 0.05      # 容量跳变阈值（可能数据错误，>5% 标记异常）
    soh_smooth_window: int = 5           # SOH 平滑窗口大小


DQ_CFG = DataQualityConfig()
