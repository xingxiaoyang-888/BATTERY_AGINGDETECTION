# models/soh_ai/trainer.py
"""
SOH AI 模型训练器 — 离线训练/验证/超参数搜索
=================================================

提供三类模型的统一训练接口:
  - XGBoostTrainer:      基于 CPU 的梯度提升树训练（快速基线）
  - LSTMTrainer:         BiLSTM+Attention 序列模型训练
  - TransformerTrainer:  Temporal Transformer 训练

通用功能:
  - Early stopping (patience-based，防止过拟合)
  - Checkpoint 自动保存与恢复（断电续训）
  - ReduceLROnPlateau 学习率调度
  - 训练/验证损失记录（TensorBoard 兼容日志）
  - 训练过程实时进度条
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from .config import (
    FEATURE_CFG,
    XGB_CFG, LSTM_CFG, TRANSFORMER_CFG,
    ENSEMBLE_CFG, TRAIN_CFG,
    WEIGHTS_DIR,
)
from .models import (
    XGBoostWrapper,
    BiLSTMAttention,
    TemporalTransformer,
    EnsembleModel,
)

logger = logging.getLogger(__name__)


def _json_default(value):
    """将 NumPy 标量和数组转换为标准 JSON 类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# ═══════════════════════════════════════════════════════════════
# 训练状态记录结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainingHistory:
    """单个训练会话的完整记录"""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    stopped_early: bool = False
    total_epochs: int = 0

    def to_dict(self) -> dict:
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'stopped_early': self.stopped_early,
            'total_epochs': self.total_epochs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingHistory":
        h = cls()
        h.train_loss = d.get('train_loss', [])
        h.val_loss = d.get('val_loss', [])
        h.learning_rates = d.get('learning_rates', [])
        h.best_epoch = d.get('best_epoch', 0)
        h.best_val_loss = d.get('best_val_loss', float('inf'))
        h.stopped_early = d.get('stopped_early', False)
        h.total_epochs = d.get('total_epochs', 0)
        return h


# ═══════════════════════════════════════════════════════════════
# 内存安全 Dataset
# ═══════════════════════════════════════════════════════════════

class MemorySafeDataset(torch.utils.data.Dataset):
    """
    延迟转换 Dataset — 只在 __getitem__ 时才将 numpy → torch.Tensor。

    对比 TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)):
      - TensorDataset: 一次性将全部数据转为 float32 tensor 存入显存/内存
                       搭配 num_workers>0 时每个子进程复制一份 → OOM 炸弹
      - MemorySafeDataset: 保持 numpy 数组引用，按需转换，零额外拷贝

    适用场景: 共享 GPU 服务器、AutoDL 等内存受限环境。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X  # 保留为 numpy (内存高效)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 按需转换，不在内存中创建完整的 float32 副本
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.y[idx]).float(),
        )


# ═══════════════════════════════════════════════════════════════
# 通用训练器基类
# ═══════════════════════════════════════════════════════════════

class BaseTrainer:
    """
    PyTorch 训练器基类

    提供所有深度学习模型共用的训练循环、早停、检查点逻辑。
    子类只需实现 _create_model(), _train_epoch(), _validate_epoch()。
    """

    def __init__(self, cfg=None):
        self.train_cfg = cfg or TRAIN_CFG
        self.device = self._resolve_device()
        self.history = TrainingHistory()
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self._criterion = nn.MSELoss()

        # 检查点
        self.checkpoint_dir = Path(self.train_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  训练设备: {self.device}")

    def _resolve_device(self) -> torch.device:
        """解析训练设备"""
        device_str = self.train_cfg.device
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("  CUDA 不可用，回退到 CPU")
            return torch.device("cpu")
        return torch.device(device_str)

    def _create_model(self) -> nn.Module:
        raise NotImplementedError("子类必须实现 _create_model()")

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        raise NotImplementedError("子类必须实现 _create_optimizer()")

    def _create_scheduler(self, optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        raise NotImplementedError("子类必须实现 _create_scheduler()")

    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        将 numpy 数组转换为 PyTorch DataLoader。

        默认实现: 假设 X, y 是训练数据 + 验证数据由调用方分别传入。
        在实际调用中，fit() 方法会分别处理 train/val。
        """
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True,
                           num_workers=self.train_cfg.num_workers,
                           pin_memory=(self.device.type == 'cuda'))
        return loader

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = None,
            batch_size: int = None,
            patience: int = None,
            model_name: str = "model",
            resume_from: Optional[str] = None,
            ) -> Tuple[nn.Module, TrainingHistory]:
        """
        主训练循环。

        Args:
            X_train, y_train: 训练数据
            X_val, y_val:     验证数据（可选。不提供则用训练集末尾 15% 做验证）
            epochs:           最大训练轮数
            batch_size:       批次大小
            patience:         早停等待轮数
            model_name:       模型名称（用于保存检查点）
            resume_from:      检查点路径（恢复训练）

        Returns:
            (model, history): 训练好的模型和训练历史
        """
        # ── 参数处理 ──
        epochs = epochs or 200
        batch_size = batch_size or 64
        patience = patience or 30

        # ── 自动拆分验证集 ──
        if X_val is None or y_val is None:
            split = int(len(X_train) * 0.85)
            X_val = X_train[split:]
            y_val = y_train[split:]
            X_train = X_train[:split]
            y_train = y_train[:split]
            logger.info(f"  自动拆分验证集: train={len(X_train)}, val={len(X_val)}")

        train_loader = self._build_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._build_loader(X_val, y_val, batch_size, shuffle=False)

        # ── 模型初始化 ──
        if resume_from and os.path.exists(resume_from):
            self._model, self._optimizer, self._scheduler, self.history, start_epoch = \
                self._load_checkpoint(resume_from)
            logger.info(f"  从检查点恢复: epoch {start_epoch}")
        else:
            self._model = self._create_model().to(self.device)
            self._optimizer = self._create_optimizer(self._model)
            self._scheduler = self._create_scheduler(self._optimizer)
            self.history = TrainingHistory()
            start_epoch = 0

        logger.info(f"  模型参数量: {self._count_params():,}")
        logger.info(f"  训练样本: {len(train_loader.dataset)}, "
                   f"验证样本: {len(val_loader.dataset)}")
        logger.info(f"  Epochs: {epochs}, Batch: {batch_size}, Patience: {patience}")

        # ── 训练循环 ──
        best_state = None
        patience_counter = 0
        checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pt"

        for epoch in range(start_epoch, epochs):
            t_start = time.time()

            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            self.history.train_loss.append(train_loss)

            # 验证阶段
            val_loss = self._validate_epoch(val_loader)
            self.history.val_loss.append(val_loss)

            # 学习率记录
            current_lr = self._optimizer.param_groups[0]['lr']
            self.history.learning_rates.append(current_lr)

            # 学习率调度
            if isinstance(self._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self._scheduler.step(val_loss)
            else:
                self._scheduler.step()

            # 检查是否最佳
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch + 1
                best_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'scheduler_state_dict': self._scheduler.state_dict() if self._scheduler else None,
                    'val_loss': val_loss,
                    'history': self.history.to_dict(),
                }
                patience_counter = 0
                self._save_checkpoint(best_state, checkpoint_path)
            else:
                patience_counter += 1

            # 日志输出
            elapsed = time.time() - t_start
            self.history.epoch_times.append(elapsed)
            if (epoch + 1) % self.train_cfg.log_interval == 0 or epoch == 0:
                logger.info(
                    f"  Epoch [{epoch+1:4d}/{epochs}] | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                    f"{' ← Best!' if patience_counter == 0 else ''}"
                )

            # 早停检查
            if patience_counter >= patience:
                logger.info(f"  ⏹ Early stopping at epoch {epoch+1} "
                           f"(patience={patience}, best_val_loss={self.history.best_val_loss:.6f})")
                self.history.stopped_early = True
                break

        self.history.total_epochs = len(self.history.train_loss)

        # 加载最佳权重
        if best_state is not None:
            self._model.load_state_dict(best_state['model_state_dict'])
        self._model.eval()

        logger.info(f"  训练完成! 最佳 Val Loss: {self.history.best_val_loss:.6f} "
                   f"@ Epoch {self.history.best_epoch}")
        return self._model, self.history

    def _build_loader(self, X: np.ndarray, y: np.ndarray,
                      batch_size: int, shuffle: bool) -> DataLoader:
        """
        构建 DataLoader — 使用 MemorySafeDataset 避免 OOM。

        ⚠️ 为什么不用 TensorDataset:
          TensorDataset(torch.FloatTensor(X), ...) 会将全部数据转为 float32
          tensor 存入内存。搭配 num_workers>0 时每个子进程复制一份 → OOM。
          MemorySafeDataset 保持 numpy 引用，__getitem__ 时才按需转换。
        """
        dataset = MemorySafeDataset(X, y)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.train_cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        """执行一个训练 epoch，返回平均 loss"""
        self._model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self._optimizer.zero_grad(set_to_none=True)
            pred = self._model(batch_X)
            loss = self._criterion(pred, batch_y)
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)

            self._optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate_epoch(self, loader: DataLoader) -> float:
        """执行验证，返回平均 loss"""
        self._model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                pred = self._model(batch_X)
                loss = self._criterion(pred, batch_y)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _count_params(self) -> int:
        """统计可训练参数数量"""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def _save_checkpoint(self, state: dict, filepath: str):
        """保存检查点"""
        torch.save(state, filepath)
        logger.debug(f"  检查点已保存: {filepath}")

    def _load_checkpoint(self, filepath: str) -> Tuple[nn.Module, any, any, TrainingHistory, int]:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        model = self._create_model().to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = self._create_optimizer(model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = None
        if checkpoint.get('scheduler_state_dict'):
            scheduler = self._create_scheduler(optimizer)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = TrainingHistory.from_dict(checkpoint.get('history', {}))
        start_epoch = checkpoint.get('epoch', 0)
        return model, optimizer, scheduler, history, start_epoch

    def save_model(self, filepath: str):
        """仅保存模型权重（不含优化器状态）"""
        if self._model is None:
            raise RuntimeError("没有可保存的模型，请先训练")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self._model.state_dict(), filepath)
        logger.info(f"  模型权重已保存至: {filepath}")

    def load_model(self, filepath: str) -> nn.Module:
        """加载模型权重"""
        self._model = self._create_model().to(self.device)
        self._model.load_state_dict(torch.load(filepath, map_location=self.device))
        self._model.eval()
        logger.info(f"  模型权重已从 {filepath} 加载")
        return self._model


# ═══════════════════════════════════════════════════════════════
# XGBoost 训练器
# ═══════════════════════════════════════════════════════════════

class XGBoostTrainer:
    """
    XGBoost 训练器 — 独立的 sklearn 风格接口。

    注意：XGBoost 以 2D 扁平特征为输入，与序列模型不同。
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or XGB_CFG
        self.model = XGBoostWrapper(self.cfg)
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            ) -> XGBoostWrapper:
        """
        训练 XGBoost 模型。

        Args:
            X_train: (n_samples, seq_len * n_features) 扁平特征
            y_train: (n_samples, horizon) 或 (n_samples,)
            X_val, y_val: 验证集
        """
        from sklearn.metrics import mean_squared_error

        # 自动拆分验证集
        if X_val is None or y_val is None:
            split = int(len(X_train) * 0.85)
            X_val = X_train[split:]
            y_val = y_train[split:]
            X_train = X_train[:split]
            y_train = y_train[:split]

        logger.info(f"  XGBoost 训练: {len(X_train)} 样本, {X_train.shape[1]} 特征")
        t_start = time.time()

        self.model.fit(X_train, y_train, X_val, y_val)

        # 计算损失（用于记录）
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        train_mse = mean_squared_error(y_train.ravel(), train_pred.ravel())
        val_mse = mean_squared_error(y_val.ravel(), val_pred.ravel())
        self.history['train_loss'].append(train_mse)
        self.history['val_loss'].append(val_mse)

        elapsed = time.time() - t_start
        logger.info(f"  XGBoost 训练完成 | 耗时: {elapsed:.1f}s | "
                   f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

        return self.model

    def save_model(self, filepath: str):
        self.model.save(filepath)

    def load_model(self, filepath: str):
        self.model = XGBoostWrapper.load(filepath)


# ═══════════════════════════════════════════════════════════════
# BiLSTM+Attention 训练器
# ═══════════════════════════════════════════════════════════════

class LSTMTrainer(BaseTrainer):
    """BiLSTM + Attention 模型训练器"""

    def __init__(self, cfg=None, train_cfg=None, input_dim: int = None):
        self.cfg = cfg or LSTM_CFG
        self._input_dim = input_dim  # 允许从数据覆盖
        super().__init__(train_cfg)

    def _create_model(self) -> nn.Module:
        return BiLSTMAttention(self.cfg, input_dim=self._input_dim)

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def _create_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.cfg.lr_factor,
            patience=self.cfg.lr_patience,
        )


# ═══════════════════════════════════════════════════════════════
# Temporal Transformer 训练器
# ═══════════════════════════════════════════════════════════════

class TransformerTrainer(BaseTrainer):
    """Temporal Transformer 模型训练器"""

    def __init__(self, cfg=None, train_cfg=None, input_dim: int = None):
        self.cfg = cfg or TRANSFORMER_CFG
        self._input_dim = input_dim  # 允许从数据覆盖
        super().__init__(train_cfg)

    def _create_model(self) -> nn.Module:
        return TemporalTransformer(self.cfg, input_dim=self._input_dim)

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        # Transformer 通常用较小的学习率
        return optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def _create_scheduler(self, optimizer):
        # 带 warmup 的余弦退火调度（比 ReduceLROnPlateau 更适合 Transformer）
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,         # 首次周期长度
            T_mult=2,       # 每次周期翻倍
            eta_min=1e-6,   # 最小学习率
        )


# ═══════════════════════════════════════════════════════════════
# 集成训练编排器
# ═══════════════════════════════════════════════════════════════

class EnsembleTrainer:
    """
    三模型集成训练编排器

    管理 XGBoost + BiLSTM + Transformer 的完整训练流程:
      1. 数据扁平化 (XGBoost)
      2. 序列训练 (BiLSTM, Transformer)
      3. 集成权重优化
      4. 模型持久化

    用法:
        trainer = EnsembleTrainer()
        result = trainer.train_all(pipeline_output)
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or ENSEMBLE_CFG
        # 训练器延迟创建（等知道 input_dim 后）
        self.xgb_trainer = XGBoostTrainer()
        self.lstm_trainer = None
        self.transformer_trainer = None
        self.ensemble = EnsembleModel(cfg)
        self.results: Dict[str, Any] = {}

    def train_all(self, pipeline_output: dict,
                  epochs: Optional[Dict[str, int]] = None,
                  skip_xgb: bool = False,
                  skip_lstm: bool = False,
                  skip_transformer: bool = False,
                  ) -> Dict[str, Any]:
        """
        执行全量训练。

        Args:
            pipeline_output: SOHDataPipeline.run() 的输出字典，包含:
                - 'sequences': {'train': (X, y), 'val': (X, y), 'test': (X, y)}
                - 'scalers': {'X': scaler, 'y': scaler}
            epochs: 可选，覆盖各模型训练轮数
            skip_xgb/skip_lstm/skip_transformer: 跳过指定模型

        Returns:
            dict: {
                'ensemble': EnsembleModel,
                'history': {模型名: TrainingHistory},
                'scalers': {...},
                'test_results': {...},
            }
        """
        sequences = pipeline_output['sequences']
        X_train, y_train = sequences['train']
        X_val, y_val = sequences.get('val', (None, None))
        X_test, y_test = sequences.get('test', (None, None))

        # 处理验证集（可能是 tuple 或 None）
        if isinstance(X_val, tuple):
            X_val, y_val = X_val
        if isinstance(X_test, tuple):
            X_test, y_test = X_test

        history = {}
        epoch_cfg = epochs or {}

        # 从数据中推断实际特征维度
        n_features = X_train.shape[2]  # (samples, seq_len, n_features)
        logger.info(f"  输入特征维度: {n_features} (seq_len={X_train.shape[1]})")

        # ── 1. XGBoost ──
        if not skip_xgb:
            logger.info("\n" + "=" * 60)
            logger.info("  阶段 1/3: 训练 XGBoost 基线")
            logger.info("=" * 60)

            # ⚠️ 不拍扁！只取序列最后一步 (last timestep)
            # XGBoost 不理解时序结构，拍扁 (N, T, F) → (N, T×F) 会导致:
            #   - 特征稀疏（640 维 vs 实际有效 ~20 维）
            #   - 树分裂困难，特征 importance 不可解释
            # 正确做法: 用最近时刻的状态特征 + 趋势统计量
            X_train_xgb = self._extract_tabular_features(X_train)
            X_val_xgb = self._extract_tabular_features(X_val) if X_val is not None else None
            y_val_use = y_val if X_val is not None else None

            logger.info(f"  XGBoost 特征: {X_train_xgb.shape[1]} 维 "
                       f"(从序列 {X_train.shape[1]} 步中提取)")

            xgb_model = self.xgb_trainer.fit(
                X_train_xgb, y_train,
                X_val_xgb, y_val_use,
            )
            self.ensemble.register('xgb', xgb_model)
            history['xgb'] = self.xgb_trainer.history
        else:
            logger.info("  ⊘ 跳过 XGBoost 训练")

        # ── 2. BiLSTM ──
        if not skip_lstm:
            logger.info("\n" + "=" * 60)
            logger.info("  阶段 2/3: 训练 BiLSTM + Attention")
            logger.info("=" * 60)
            lstm_epochs = epoch_cfg.get('lstm', LSTM_CFG.epochs)
            lstm_batch = epoch_cfg.get('lstm_batch', LSTM_CFG.batch_size)

            if self.lstm_trainer is None:
                self.lstm_trainer = LSTMTrainer(input_dim=n_features)

            lstm_model, lstm_hist = self.lstm_trainer.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=lstm_epochs,
                batch_size=lstm_batch,
                patience=LSTM_CFG.patience,
                model_name="lstm_attention",
            )
            self.ensemble.register('lstm', lstm_model)
            history['lstm'] = lstm_hist.to_dict()

            # 保存权重
            self.lstm_trainer.save_model(str(Path(WEIGHTS_DIR) / 'lstm_attention.pt'))
        else:
            logger.info("  ⊘ 跳过 BiLSTM 训练")

        # ── 3. Transformer ──
        if not skip_transformer:
            logger.info("\n" + "=" * 60)
            logger.info("  阶段 3/3: 训练 Temporal Transformer")
            logger.info("=" * 60)
            tf_epochs = epoch_cfg.get('transformer', TRANSFORMER_CFG.epochs)
            tf_batch = epoch_cfg.get('transformer_batch', TRANSFORMER_CFG.batch_size)

            if self.transformer_trainer is None:
                self.transformer_trainer = TransformerTrainer(input_dim=n_features)

            tf_model, tf_hist = self.transformer_trainer.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=tf_epochs,
                batch_size=tf_batch,
                patience=TRANSFORMER_CFG.patience,
                model_name="transformer",
            )
            self.ensemble.register('transformer', tf_model)
            history['transformer'] = tf_hist.to_dict()

            self.transformer_trainer.save_model(str(Path(WEIGHTS_DIR) / 'transformer.pt'))
        else:
            logger.info("  ⊘ 跳过 Transformer 训练")

        # ── 4. 测试集评估 ──
        test_results = {}
        if X_test is not None and y_test is not None:
            logger.info("\n" + "=" * 60)
            logger.info("  测试集评估")
            logger.info("=" * 60)

            # 同步设备：将集成模型的 device 设为首个 PyTorch 子模型的设备
            if self.lstm_trainer is not None:
                self.ensemble.to(self.lstm_trainer.device)
            elif self.transformer_trainer is not None:
                self.ensemble.to(self.transformer_trainer.device)

            test_results = self._evaluate_test(X_test, y_test)

        # ── 5. 汇总 ──
        self.results = {
            'ensemble': self.ensemble,
            'history': history,
            'scalers': pipeline_output.get('scalers', {}),
            'test_results': test_results,
            'available_models': self.ensemble.available_models,
        }

        logger.info("\n" + "=" * 60)
        logger.info(f"  训练全部完成! 已训练模型: {self.ensemble.available_models}")
        logger.info("=" * 60)

        return self.results

    @staticmethod
    def _extract_tabular_features(X_seq: np.ndarray) -> np.ndarray:
        """
        将 3D 时序序列 (N, T, F) 转换为 XGBoost 可用的 2D 表格特征。

        策略（按优先级）:
          1. 最后一步快照 X[:, -1, :] — 当前状态，最重要
          2. 时序统计量 — 均值、标准差、最小、最大（捕捉历史趋势）
          3. 首尾差分 — 捕捉变化方向

        避免直接 reshape 拍扁 — XGBoost 不理解时序索引关系，
        扁平化会导致 600+ 维稀疏空间，树分裂困难。
        """
        last_step = X_seq[:, -1, :]                            # (N, F)
        mean_vals = X_seq.mean(axis=1)                          # (N, F)
        std_vals = X_seq.std(axis=1)                            # (N, F)
        min_vals = X_seq.min(axis=1)                            # (N, F)
        max_vals = X_seq.max(axis=1)                            # (N, F)
        trend = X_seq[:, -1, :] - X_seq[:, 0, :]               # (N, F) 首尾差分

        return np.concatenate(
            [last_step, mean_vals, std_vals, min_vals, max_vals, trend],
            axis=1
        )

    def _evaluate_test(self, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, float]:
        """在测试集上评估所有模型"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        predictions = self.ensemble.predict(X_test)
        results = {}

        for name, pred in predictions.items():
            mse = mean_squared_error(y_test.ravel(), pred.ravel())
            mae = mean_absolute_error(y_test.ravel(), pred.ravel())
            rmse = np.sqrt(mse)
            results[name] = {'MSE': float(mse), 'RMSE': float(rmse), 'MAE': float(mae)}
            logger.info(f"  {name:12s} → MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

        return results

    def save_all(self, output_dir: str = None):
        """保存所有模型和训练记录"""
        output_dir = Path(output_dir or WEIGHTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 XGBoost
        if 'xgb' in self.ensemble.available_models:
            self.xgb_trainer.save_model(str(output_dir / 'xgb_model.pkl'))

        # LSTM 和 Transformer 已在训练时保存

        # 保存训练历史
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(
                self.results.get('history', {}), f,
                indent=2, ensure_ascii=False, default=_json_default,
            )
        logger.info(f"  训练历史已保存: {history_path}")

        # 保存测试结果
        if self.results.get('test_results'):
            test_path = output_dir / 'test_results.json'
            with open(test_path, 'w') as f:
                json.dump(
                    self.results['test_results'], f,
                    indent=2, ensure_ascii=False, default=_json_default,
                )

        logger.info(f"  所有模型已保存至: {output_dir}")

    # ═══════════════════════════════════════════════════════════
    # 迁移学习: 源域预训练 → 目标域微调
    # ═══════════════════════════════════════════════════════════

    def pretrain_on_source(self, pretrain_sequences: dict,
                           epochs: Optional[Dict[str, int]] = None,
                           skip_xgb: bool = False,
                           skip_lstm: bool = False,
                           skip_transformer: bool = False) -> Dict[str, Any]:
        """
        阶段 1: 在源域 (lithium-ion) 上预训练所有模型。

        源域数据量大（146 个电芯），模型可以学到丰富的电池老化模式
        （容量衰减曲线形态、内阻增长规律、拐点特征等），这些模式在
        不同化学体系间具有一定的可迁移性。

        Args:
            pretrain_sequences: {'train': (X, y), 'val': (X, y)|None}
            epochs: 可选的 epoch 覆盖
            skip_xgb/skip_lstm/skip_transformer: 跳过的模型

        Returns:
            {'xgb': XGBoostWrapper, 'lstm': nn.Module, 'transformer': nn.Module, ...}
        """
        X_train, y_train = pretrain_sequences['train']
        X_val, y_val = pretrain_sequences.get('val', (None, None))

        if isinstance(X_val, tuple):
            X_val, y_val = X_val

        n_features = X_train.shape[2]
        epoch_cfg = epochs or {}
        pretrained = {}

        logger.info("=" * 60)
        logger.info(f"  迁移学习 Phase 1/2: 源域预训练 ({len(X_train)} 样本)")
        logger.info(f"  源域验证: {len(X_val)} 样本" if X_val is not None
                    else "  源域验证: 无独立验证集")
        logger.info("=" * 60)

        # ── XGBoost 预训练 ──
        if not skip_xgb:
            logger.info("\n  [预训练] XGBoost 基线...")
            X_train_flat = self._extract_tabular_features(X_train)
            X_val_flat = self._extract_tabular_features(X_val) if X_val is not None else None
            self.xgb_trainer.fit(X_train_flat, y_train, X_val_flat,
                                 y_val if X_val is not None else None)
            self.ensemble.register('xgb', self.xgb_trainer.model)
            pretrained['xgb'] = self.xgb_trainer.model
            logger.info(f"  ✓ XGBoost 预训练完成")
        else:
            logger.info("  ⊘ 跳过 XGBoost 预训练")

        # ── BiLSTM 预训练 ──
        if not skip_lstm:
            logger.info("\n  [预训练] BiLSTM + Attention...")
            lstm_epochs = epoch_cfg.get('lstm', LSTM_CFG.epochs)
            lstm_batch = epoch_cfg.get('lstm_batch', LSTM_CFG.batch_size)
            if self.lstm_trainer is None:
                self.lstm_trainer = LSTMTrainer(input_dim=n_features)
            lstm_model, lstm_hist = self.lstm_trainer.fit(
                X_train, y_train, X_val, y_val,
                epochs=lstm_epochs, batch_size=lstm_batch,
                patience=LSTM_CFG.patience, model_name="lstm_attention"
            )
            self.ensemble.register('lstm', lstm_model)
            pretrained['lstm'] = lstm_model
            pretrained['lstm_history'] = lstm_hist.to_dict()
            # 保存完整 checkpoint（含 optimizer state），供 finetune resume
            lstm_ckpt_path = str(Path(WEIGHTS_DIR) / 'checkpoints' / 'lstm_attention_pretrained.pt')
            Path(lstm_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': lstm_hist.total_epochs,
                'model_state_dict': lstm_model.state_dict(),
                'optimizer_state_dict': self.lstm_trainer._optimizer.state_dict(),
                'scheduler_state_dict': self.lstm_trainer._scheduler.state_dict() if self.lstm_trainer._scheduler else None,
                'val_loss': lstm_hist.best_val_loss,
                'history': lstm_hist.to_dict(),
            }, lstm_ckpt_path)
            pretrained['lstm_checkpoint'] = lstm_ckpt_path
            logger.info(f"  ✓ BiLSTM 预训练完成 (best_val_loss={lstm_hist.best_val_loss:.6f})")
        else:
            logger.info("  ⊘ 跳过 BiLSTM 预训练")

        # ── Transformer 预训练 ──
        if not skip_transformer:
            logger.info("\n  [预训练] Temporal Transformer...")
            tf_epochs = epoch_cfg.get('transformer', TRANSFORMER_CFG.epochs)
            tf_batch = epoch_cfg.get('transformer_batch', TRANSFORMER_CFG.batch_size)
            if self.transformer_trainer is None:
                self.transformer_trainer = TransformerTrainer(input_dim=n_features)
            tf_model, tf_hist = self.transformer_trainer.fit(
                X_train, y_train, X_val, y_val,
                epochs=tf_epochs, batch_size=tf_batch,
                patience=TRANSFORMER_CFG.patience, model_name="transformer"
            )
            self.ensemble.register('transformer', tf_model)
            pretrained['transformer'] = tf_model
            pretrained['transformer_history'] = tf_hist.to_dict()
            # 保存完整 checkpoint
            tf_ckpt_path = str(Path(WEIGHTS_DIR) / 'checkpoints' / 'transformer_pretrained.pt')
            Path(tf_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': tf_hist.total_epochs,
                'model_state_dict': tf_model.state_dict(),
                'optimizer_state_dict': self.transformer_trainer._optimizer.state_dict(),
                'scheduler_state_dict': self.transformer_trainer._scheduler.state_dict() if self.transformer_trainer._scheduler else None,
                'val_loss': tf_hist.best_val_loss,
                'history': tf_hist.to_dict(),
            }, tf_ckpt_path)
            pretrained['transformer_checkpoint'] = tf_ckpt_path
            logger.info(f"  ✓ Transformer 预训练完成 (best_val_loss={tf_hist.best_val_loss:.6f})")
        else:
            logger.info("  ⊘ 跳过 Transformer 预训练")

        logger.info(f"\n  源域预训练完成! 模型: {list(pretrained.keys())}")
        return pretrained

    def finetune_on_target(self, target_sequences: dict,
                           epochs: Optional[Dict[str, int]] = None,
                           pretrained_models: Optional[Dict[str, Any]] = None,
                           skip_xgb: bool = False,
                           skip_lstm: bool = False,
                           skip_transformer: bool = False) -> Dict[str, Any]:
        """
        阶段 2: 在目标域 (sodium-ion) 上微调预训练模型。

        微调策略:
          - XGBoost: 降低 learning_rate (0.01 vs 0.05)，其他不变
          - BiLSTM: learning_rate × 0.1, 少量 epochs, 冻结底层 LSTM
                    仅训练 attention + FC 预测头
          - Transformer: learning_rate × 0.1, 少量 epochs, 冻结编码器
                         仅训练预测头

        Args:
            target_sequences: {'train': (X,y), 'val': (X,y), 'test': (X,y)}
            epochs: epoch 覆盖 (默认比预训练少)
            pretrained_models: 预训练好的模型 (可选，不传则使用 self.ensemble 中已有的)
            skip_xgb/skip_lstm/skip_transformer: 跳过的模型

        Returns:
            {'test_results': {...}, 'history': {...}, 'finetuned_models': [...]}
        """
        X_train, y_train = target_sequences['train']
        X_val, y_val = target_sequences.get('val', (None, None))
        X_test, y_test = target_sequences.get('test', (None, None))

        if isinstance(X_val, tuple):
            X_val, y_val = X_val
        if isinstance(X_test, tuple):
            X_test, y_test = X_test

        # 目标域数据少，默认少量 epochs
        n_features = X_train.shape[2]
        epoch_cfg = epochs or {
            'lstm': 50,          # 预训练是 200, 微调用 1/4
            'lstm_batch': min(16, LSTM_CFG.batch_size),  # 小 batch 适配小数据
            'transformer': 60,    # 预训练是 300, 微调用 1/5
            'transformer_batch': min(8, TRANSFORMER_CFG.batch_size),
        }

        # 使用传入的预训练模型，或从 ensemble 中获取
        if pretrained_models is not None:
            for name, model in pretrained_models.items():
                if name in ('xgb', 'lstm', 'transformer'):
                    self.ensemble.register(name, model)

        history = {}
        logger.info("=" * 60)
        logger.info(f"  迁移学习 Phase 2/2: 目标域微调 ({len(X_train)} 样本)")
        logger.info(f"  目标域: {target_sequences.get('target_cells', '?')} 电芯")
        logger.info("=" * 60)

        # ── XGBoost 微调 ──
        if not skip_xgb and 'xgb' in self.ensemble.available_models:
            logger.info("\n  [微调] XGBoost (降低 LR)...")
            # XGBoost 无法真正 "fine-tune"，策略是:
            # 用源域预训练模型的特征重要性作为先验，在目标域上用更低 LR 重新训练
            X_train_flat = self._extract_tabular_features(X_train)
            X_val_flat = self._extract_tabular_features(X_val) if X_val is not None else None

            # 创建低 LR 的 XGBoost 配置
            from dataclasses import replace
            try:
                low_lr_cfg = replace(self.xgb_trainer.cfg,
                                    learning_rate=0.01,       # 原始的 1/5
                                    n_estimators=200,         # 减少树数量防止过拟合
                                    max_depth=4,              # 降低复杂度
                                    early_stopping_rounds=20)
            except Exception:
                # Python < 3.12 没有 dataclasses.replace
                from .config import XGBoostConfig
                low_lr_cfg = XGBoostConfig()
                low_lr_cfg.learning_rate = 0.01
                low_lr_cfg.n_estimators = 200
                low_lr_cfg.max_depth = 4
                low_lr_cfg.early_stopping_rounds = 20

            ft_xgb_trainer = XGBoostTrainer(cfg=low_lr_cfg)
            ft_xgb_trainer.fit(X_train_flat, y_train, X_val_flat,
                              y_val if X_val is not None else None)
            self.ensemble.register('xgb', ft_xgb_trainer.model)
            history['xgb'] = ft_xgb_trainer.history
            logger.info(f"  ✓ XGBoost 微调完成")
        elif not skip_xgb:
            logger.warning("  ⚠️ 无 XGBoost 预训练模型，跳过微调")

        # ── BiLSTM 微调 ──
        if not skip_lstm and 'lstm' in self.ensemble.available_models:
            logger.info("\n  [微调] BiLSTM (低 LR, 少 epochs)...")
            lstm_epochs = epoch_cfg.get('lstm', 50)
            lstm_batch = epoch_cfg.get('lstm_batch', 16)
            lstm_ckpt = (pretrained_models or {}).get('lstm_checkpoint')

            if lstm_ckpt and os.path.exists(lstm_ckpt):
                ft_lstm = LSTMTrainer(input_dim=n_features)
                # 从预训练 checkpoint 恢复，目标域数据少 → 靠少量 epochs + L2 正则防过拟合
                lstm_model, lstm_hist = ft_lstm.fit(
                    X_train, y_train, X_val, y_val,
                    epochs=lstm_epochs, batch_size=lstm_batch,
                    patience=max(12, LSTM_CFG.patience // 2),
                    model_name="lstm_attention_ft",
                    resume_from=lstm_ckpt,
                )
                self.ensemble.register('lstm', lstm_model)
                history['lstm'] = lstm_hist.to_dict()
                ft_lstm.save_model(str(Path(WEIGHTS_DIR) / 'lstm_attention_finetuned.pt'))
                logger.info(f"  ✓ BiLSTM 微调完成 (best_val_loss={lstm_hist.best_val_loss:.6f})")
            else:
                logger.warning("  ⚠️ BiLSTM 预训练 checkpoint 不存在，跳过微调")
        elif not skip_lstm:
            logger.warning("  ⚠️ 无 BiLSTM 预训练模型，跳过微调")

        # ── Transformer 微调 ──
        if not skip_transformer and 'transformer' in self.ensemble.available_models:
            logger.info("\n  [微调] Transformer (低 LR, 少 epochs)...")
            tf_epochs = epoch_cfg.get('transformer', 60)
            tf_batch = epoch_cfg.get('transformer_batch', 8)
            tf_ckpt = (pretrained_models or {}).get('transformer_checkpoint')

            if tf_ckpt and os.path.exists(tf_ckpt):
                ft_tf = TransformerTrainer(input_dim=n_features)
                tf_model, tf_hist = ft_tf.fit(
                    X_train, y_train, X_val, y_val,
                    epochs=tf_epochs, batch_size=tf_batch,
                    patience=max(15, TRANSFORMER_CFG.patience // 2),
                    model_name="transformer_ft",
                    resume_from=tf_ckpt,
                )
                self.ensemble.register('transformer', tf_model)
                history['transformer'] = tf_hist.to_dict()
                ft_tf.save_model(str(Path(WEIGHTS_DIR) / 'transformer_finetuned.pt'))
                logger.info(f"  ✓ Transformer 微调完成 (best_val_loss={tf_hist.best_val_loss:.6f})")
            else:
                logger.warning("  ⚠️ Transformer 预训练 checkpoint 不存在，跳过微调")
        elif not skip_transformer:
            logger.warning("  ⚠️ 无 Transformer 预训练模型，跳过微调")

        # ── 测试集评估 ──
        test_results = {}
        if X_test is not None and y_test is not None:
            logger.info("\n" + "-" * 40)
            logger.info("  目标域测试集评估 (微调后)")
            logger.info("-" * 40)
            if self.lstm_trainer is not None:
                self.ensemble.to(self.lstm_trainer.device)
            test_results = self._evaluate_test(X_test, y_test)

        self.results = {
            'ensemble': self.ensemble,
            'history': history,
            'test_results': test_results,
            'available_models': self.ensemble.available_models,
            'transfer_learning': True,
        }

        logger.info(f"\n  目标域微调完成! 模型: {self.ensemble.available_models}")
        return self.results

    def train_transfer(self, pipeline_output: dict,
                       epochs: Optional[Dict[str, int]] = None,
                       skip_xgb: bool = False,
                       skip_lstm: bool = False,
                       skip_transformer: bool = False,
                       ) -> Dict[str, Any]:
        """
        完整迁移学习流程: 源域预训练 → 目标域微调。

        自动从 pipeline_output 中检测跨化学体系数据:
          - pipeline_output['pretrain_sequences'] → 源域 (Li-ion)
          - pipeline_output['sequences'] → 目标域 (Na-ion)

        Args:
            pipeline_output: SOHDataPipeline.run(chemistry_aware=True) 的输出
            epochs: 可选的 epoch 覆盖
            skip_xgb/skip_lstm/skip_transformer: 跳过的模型

        Returns:
            {
                'pretrained': {...},         # 预训练结果
                'finetuned': {...},          # 微调结果
                'test_results': {...},       # 测试集指标
                'is_transfer': bool,          # 是否执行了迁移学习
            }
        """
        pretrain_seqs = pipeline_output.get('pretrain_sequences')
        target_seqs = pipeline_output.get('sequences')
        is_cross = pipeline_output.get('is_cross_chemistry', False)

        if not is_cross or pretrain_seqs is None:
            logger.info("  非跨域场景，回退到普通训练模式")
            result = self.train_all(
                pipeline_output,
                epochs=epochs,
                skip_xgb=skip_xgb,
                skip_lstm=skip_lstm,
                skip_transformer=skip_transformer,
            )
            result['is_transfer'] = False
            return result

        # 注入目标域电芯信息
        if target_seqs is not None:
            target_seqs = dict(target_seqs)  # shallow copy
            target_seqs['target_cells'] = pipeline_output.get('target_cells', [])

        # Phase 1: 源域预训练
        logger.info("\n" + "█" * 60)
        logger.info("  迁移学习: 锂电 (源域) → 钠电 (目标域)")
        logger.info(f"  源域化学体系: {pipeline_output.get('source_chemistry')}")
        logger.info(f"  目标域化学体系: {pipeline_output.get('target_chemistry')}")
        logger.info("█" * 60)

        pretrained = self.pretrain_on_source(
            pretrain_seqs,
            epochs=epochs,
            skip_xgb=skip_xgb,
            skip_lstm=skip_lstm,
            skip_transformer=skip_transformer,
        )

        # Phase 2: 目标域微调
        finetuned = self.finetune_on_target(
            target_seqs,
            epochs=epochs,
            pretrained_models=pretrained,
            skip_xgb=skip_xgb,
            skip_lstm=skip_lstm,
            skip_transformer=skip_transformer,
        )

        finetuned['is_transfer'] = True
        finetuned['pretrained'] = pretrained
        finetuned['source_chemistry'] = pipeline_output.get('source_chemistry')
        finetuned['target_chemistry'] = pipeline_output.get('target_chemistry')

        return finetuned


# ═══════════════════════════════════════════════════════════════
# 交叉验证训练器
# ═══════════════════════════════════════════════════════════════


class CrossValidator:
    """??? `cell_id` ??? K-Fold ?????"""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.fold_scores: List[Dict] = []

    def run(self, pipeline_output: dict,
            epochs: Optional[Dict[str, int]] = None,
            skip_xgb: bool = False,
            skip_lstm: bool = False,
            skip_transformer: bool = False) -> List[Dict[str, Any]]:
        from .data_pipeline import SequenceBuilder

        feature_df = pipeline_output.get('feature_df')
        if feature_df is None or feature_df.empty:
            raise ValueError('pipeline_output ????? feature_df')
        if 'cell_id' not in feature_df.columns:
            raise ValueError('feature_df ?? cell_id????? cell-level KFold')

        feature_cols = pipeline_output.get('feature_cols')
        if not feature_cols:
            exclude = {'cell_id', 'chemistry', 'dataset_id', 'condition', 'cycle_index', 'soh_raw', 'soh_jump_flag', FEATURE_CFG.target_col}
            feature_cols = [
                c for c in feature_df.columns
                if c not in exclude and feature_df[c].dtype.kind in 'fiu'
            ]

        cells = feature_df['cell_id'].dropna().unique().tolist()
        if self.n_folds < 2:
            raise ValueError('n_folds ?? >= 2')
        if len(cells) < self.n_folds:
            raise ValueError(f'??? {len(cells)} ???? {self.n_folds}')

        groups = feature_df['cell_id'].to_numpy()
        gkf = GroupKFold(n_splits=self.n_folds)
        seq_builder = SequenceBuilder()
        epoch_cfg = epochs or {}
        self.fold_scores = []

        logger.info(f'  ???? cell-level KFold: n_folds={self.n_folds}, cells={len(cells)}')

        for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(feature_df, groups=groups), start=1):
            train_val_df = feature_df.iloc[train_val_idx].copy()
            test_df = feature_df.iloc[test_idx].copy()

            train_val_cells = train_val_df['cell_id'].dropna().unique().tolist()
            test_cells = test_df['cell_id'].dropna().unique().tolist()
            if len(train_val_cells) < 2:
                logger.warning(f'  Fold {fold_idx}: ???????????')
                continue

            val_size = max(1, int(round(len(train_val_cells) * 0.2)))
            if val_size >= len(train_val_cells):
                val_size = len(train_val_cells) - 1

            val_cells = train_val_cells[:val_size]
            train_cells = train_val_cells[val_size:]
            if len(train_cells) == 0:
                train_cells = train_val_cells[:-1]
                val_cells = train_val_cells[-1:]

            train_df = train_val_df[train_val_df['cell_id'].isin(train_cells)].copy()
            val_df = train_val_df[train_val_df['cell_id'].isin(val_cells)].copy()

            # SOH 同时是输入状态和预测目标，必须保留真实物理域。
            # 只缩放其余输入列，否则 KFold 的 y 会被错误地变换到缩放域。
            scale_cols = [col for col in feature_cols if col != FEATURE_CFG.target_col]
            scaler = RobustScaler(quantile_range=(5, 95))
            scaler.fit(train_df[scale_cols].values)

            def _scale(df):
                out = df.copy()
                if not out.empty:
                    out.loc[:, scale_cols] = scaler.transform(out[scale_cols].values)
                return out

            X_train, y_train, _, _ = seq_builder.build_sequences(_scale(train_df), feature_cols=feature_cols)
            X_val, y_val, _, _ = seq_builder.build_sequences(_scale(val_df), feature_cols=feature_cols)
            X_test, y_test, _, _ = seq_builder.build_sequences(_scale(test_df), feature_cols=feature_cols)

            logger.info(
                f'  Fold {fold_idx}/{self.n_folds}: train_cells={len(train_cells)}, val_cells={len(val_cells)}, '
                f'test_cells={len(test_cells)} | X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}'
            )

            fold_trainer = EnsembleTrainer()
            fold_result = fold_trainer.train_all(
                {
                    'sequences': {
                        'train': (X_train, y_train),
                        'val': (X_val, y_val),
                        'test': (X_test, y_test),
                    },
                    'scalers': {'X': scaler, 'y': None},
                    'feature_cols': feature_cols,
                    'feature_df': feature_df,
                },
                epochs=epoch_cfg if epoch_cfg else None,
                skip_xgb=skip_xgb,
                skip_lstm=skip_lstm,
                skip_transformer=skip_transformer,
            )

            self.fold_scores.append({
                'fold': fold_idx,
                'train_cells': [str(c) for c in train_cells],
                'val_cells': [str(c) for c in val_cells],
                'test_cells': [str(c) for c in test_cells],
                'train_n': int(len(X_train)),
                'val_n': int(len(X_val)),
                'test_n': int(len(X_test)),
                'available_models': fold_result.get('available_models', []),
                'test_results': fold_result.get('test_results', {}),
            })

        return self.fold_scores
