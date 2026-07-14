# models/soh_ai/models.py
"""
SOH AI 模型定义 — XGBoost / BiLSTM+Attention / Temporal Transformer
====================================================================

本模块定义三类模型的神经网络架构和 XGBoost 封装。
所有模型遵循统一接口:
  - forward(x) → Tensor  (PyTorch 模型)
  - predict(X) → np.ndarray  (推理接口)

架构设计原则:
  1. 输入/输出维度由全局 config 统一管理，避免魔数分散
  2. 每个模型独立可训练、可保存、可加载
  3. PyTorch 模型支持 dynamic sequence length（推理时可变）
"""

import logging
import numpy as np
import warnings
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    FEATURE_CFG,
    XGB_CFG,
    LSTM_CFG,
    TRANSFORMER_CFG,
    ENSEMBLE_CFG,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 通用工具层
# ═══════════════════════════════════════════════════════════════

def _init_weights(module: nn.Module, method: str = "xavier"):
    """统一的权重初始化，支持 Xavier / Kaiming / 正交"""
    if isinstance(module, nn.Linear):
        if method == "xavier":
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        elif method == "kaiming":
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码，用于 Transformer 注入时序信息。

    与原始 "Attention Is All You Need" 实现一致:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════
# 模型 1: XGBoost 基线 (scikit-learn 兼容封装)
# ═══════════════════════════════════════════════════════════════

class XGBoostWrapper:
    """
    XGBoost 回归器的轻量封装，提供与 PyTorch 模型一致的训练/保存/加载接口。

    输入:  (n_samples, seq_len * n_features) — 2D 扁平特征
    输出:  (n_samples, horizon) — SOH 预测值

    为什么用 XGBoost？
      - 小数据场景下表现稳定，不易过拟合
      - 对特征共线性不敏感
      - 训练极快，可作为快速基线
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or XGB_CFG
        self._model = None
        self._is_fitted = False
        self._multi_output = False  # 始终为 False（仅保留向后兼容）

    def _build(self):
        """延迟构建 XGBoost 模型（避免未安装时导入失败）"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost 未安装。请运行: pip install xgboost"
            )
        self._model = xgb.XGBRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            min_child_weight=self.cfg.min_child_weight,
            reg_lambda=self.cfg.reg_lambda,
            reg_alpha=self.cfg.reg_alpha,
            objective=self.cfg.objective,
            eval_metric=self.cfg.eval_metric,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "XGBoostWrapper":
        """
        训练 XGBoost 模型（仅支持单步预测 horizon=1）。

        ⚠️ 架构决策：电池 RUL 预测不使用 Direct Multi-step 策略。
        如需多步推演，请在推理时使用自回归滚动 (autoregressive rollout):
          for step in range(horizon):
              pred[step] = model.predict(X_last)
              X_last = slide_window(X_last, pred[step])  # 拼接新预测

        Args:
            X: 训练特征, shape (n_samples, n_features) — 已处理为 2D
            y: 训练目标, shape (n_samples, 1) — 原始 SOH 值
            X_val: 验证特征
            y_val: 验证目标
        """
        if self._model is None:
            self._build()

        # 防御性检查：禁止多步预测
        y_flat = y.ravel() if y.ndim > 1 else y
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(
                f"XGBoost 仅支持 horizon=1 单步预测，收到 y.shape={y.shape}。"
                f"多步推演请在推理时使用自回归滚动策略。"
            )

        y_val_flat = None
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_flat = y_val.ravel() if y_val.ndim > 1 else y_val
            eval_set = [(X_val, y_val_flat)]
        else:
            # 无验证集时禁用 early stopping，否则 XGBoost 会报错
            self._model.set_params(early_stopping_rounds=None)

        self._model.fit(X, y_flat, eval_set=eval_set, verbose=False)
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 SOH 值，返回 (n_samples, 1)"""
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        return self._model.predict(X).reshape(-1, 1)

    def save(self, filepath: str):
        """保存模型到磁盘"""
        import joblib
        state = {
            'model': self._model,
            'config': self.cfg,
        }
        joblib.dump(state, filepath)
        logger.info(f"  XGBoost 模型已保存至: {filepath}")

    @classmethod
    def load(cls, filepath: str, cfg=None) -> "XGBoostWrapper":
        """从磁盘加载模型"""
        import joblib
        state = joblib.load(filepath)
        wrapper = cls(cfg=state.get('config', cfg))
        # 兼容旧格式（multi_output 残留）
        if 'models' in state and 'multi_output' in state:
            wrapper._model = state['models'] if not state['multi_output'] else state['models']  # fallback
        else:
            wrapper._model = state.get('model', state.get('models'))
        wrapper._is_fitted = True
        logger.info(f"  XGBoost 模型已从 {filepath} 加载")
        return wrapper

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ═══════════════════════════════════════════════════════════════
# 模型 2: BiLSTM + Attention
# ═══════════════════════════════════════════════════════════════

class AdditiveAttention(nn.Module):
    """
    加性注意力机制 (Bahdanau-style)

    对 LSTM 输出的每个时间步计算注意力权重，加权求和得到上下文向量。

    Args:
        hidden_dim: LSTM 隐藏状态维度 (双向 = hidden_dim * 2)
        attention_dim: 注意力投影维度
    """

    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim) — LSTM 全序列输出
            mask:     (batch, seq_len) — True=有效, False=填充

        Returns:
            context:   (batch, hidden_dim) — 注意力加权上下文向量
            attn_w:    (batch, seq_len)    — 注意力权重（可用于可视化）
        """
        # 计算每个时间步的注意力分数
        scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_w = F.softmax(scores, dim=-1)  # (batch, seq_len)
        context = torch.bmm(attn_w.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        return context, attn_w


class BiLSTMAttention(nn.Module):
    """
    双向 LSTM + 多头加性注意力 + 预测头

    架构:
      Input ──→ BiLSTM ──→ Attention Pooling ──→ FC Block ──→ Output
                (2层双向)    (加性注意力)       (Dropout+ReLU)

    输入:  (batch, seq_len, n_features)
    输出:  (batch, horizon)

    配置来源: LSTMAttentionConfig (config.py)
    """

    def __init__(self, cfg=None, input_dim: int = None):
        super().__init__()
        self.cfg = cfg or LSTM_CFG

        # 允许动态覆盖 input_dim（适配数据管线实际特征数）
        _input_dim = input_dim or self.cfg.input_dim
        hidden_dim = self.cfg.hidden_dim
        num_layers = self.cfg.num_layers
        dropout = self.cfg.dropout
        self.bidirectional = self.cfg.bidirectional
        self.horizon = FEATURE_CFG.prediction_horizon

        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        # 注意力池化
        self.attention = AdditiveAttention(
            hidden_dim=lstm_out_dim,
            attention_dim=self.cfg.attention_dim,
        )

        # 预测头
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.horizon),
        )

        self.apply(lambda m: _init_weights(m, method="xavier"))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, n_features) 输入序列
            mask: (batch, seq_len) 填充掩码（可选）

        Returns:
            y: (batch, horizon) SOH 预测值
        """
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, L, D)
        context, attn_w = self.attention(lstm_out, mask)
        out = self.dropout(context)
        out = self.fc(out)
        return out


# ═══════════════════════════════════════════════════════════════
# 模型 3: Temporal Transformer
# ═══════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    """
    时序 Transformer 编码器 + 预测头

    架构:
      Input ──→ Linear Projection + Positional Encoding
            ──→ Transformer Encoder × N_layers
            ──→ Mean Pooling (可替换为 CLS token)
            ──→ FC Block ──→ Output

    输入:  (batch, seq_len, n_features)
    输出:  (batch, horizon)

    配置来源: TransformerConfig (config.py)
    """

    def __init__(self, cfg=None, input_dim: int = None):
        super().__init__()
        self.cfg = cfg or TRANSFORMER_CFG

        # 允许动态覆盖 input_dim（适配数据管线实际特征数）
        _input_dim = input_dim or self.cfg.input_dim
        d_model = self.cfg.d_model
        nhead = self.cfg.nhead
        num_layers = self.cfg.num_encoder_layers
        dim_feedforward = self.cfg.dim_feedforward
        dropout = self.cfg.dropout
        self.horizon = FEATURE_CFG.prediction_horizon

        # 输入投影：将 n_features 映射到 d_model
        self.input_proj = nn.Linear(_input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=self.cfg.max_seq_len,
            dropout=dropout,
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=self.cfg.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.horizon),
        )

        self.apply(lambda m: _init_weights(m, method="kaiming"))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, n_features)
            mask: (batch, seq_len) True=padding, False=valid
                  Transformer 期望 True=ignore, 所以需要反转为 src_key_padding_mask

        Returns:
            y: (batch, horizon)
        """
        # 投影 + 位置编码
        x = self.input_proj(x)  # (B, L, d_model)
        x = self.pos_encoder(x)

        # Transformer 编码
        # src_key_padding_mask: True = 忽略该位置
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # True=padding, False=valid

        encoded = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, L, d_model)

        # 均值池化
        if mask is not None:
            # 仅在有效位置上求均值
            valid_mask = mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        # 预测头
        out = self.pred_head(pooled)  # (B, horizon)
        return out


# ═══════════════════════════════════════════════════════════════
# 集成模型 (Ensemble)
# ═══════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    三模型加权集成预测器

    集成策略:
      最终预测 = w₁ × XGBoost + w₂ × BiLSTM + w₃ × Transformer

    权重可固定（配置中的默认值）或动态调整（基于各模型验证性能）。

    支持 Bootstrap 不确定性量化:
      对每个模型进行 N 次采样，输出均值 ± 置信区间。
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or ENSEMBLE_CFG
        self.xgb_model: Optional[XGBoostWrapper] = None
        self.lstm_model: Optional[BiLSTMAttention] = None
        self.transformer_model: Optional[TemporalTransformer] = None
        self.weights = {
            'xgb': self.cfg.xgboost_weight,
            'lstm': self.cfg.lstm_weight,
            'transformer': self.cfg.transformer_weight,
        }
        self._device = torch.device("cpu")

    def register(self, name: str, model):
        """注册子模型"""
        if name == 'xgb':
            self.xgb_model = model
        elif name == 'lstm':
            self.lstm_model = model
        elif name == 'transformer':
            self.transformer_model = model
        else:
            raise ValueError(f"未知模型名称: {name}，可选: xgb, lstm, transformer")

    def to(self, device: torch.device):
        """将所有 PyTorch 子模型移至指定设备"""
        self._device = device
        if self.lstm_model is not None:
            self.lstm_model.to(device)
        if self.transformer_model is not None:
            self.transformer_model.to(device)
        return self

    def set_weights(self, xgb: float = None, lstm: float = None, transformer: float = None):
        """动态调整集成权重（自动归一化）"""
        w = {
            'xgb': xgb if xgb is not None else self.weights['xgb'],
            'lstm': lstm if lstm is not None else self.weights['lstm'],
            'transformer': transformer if transformer is not None else self.weights['transformer'],
        }
        total = sum(w.values())
        if total > 0:
            self.weights = {k: v / total for k, v in w.items()}
        logger.info(f"  集成权重已更新: {self.weights}")

    def predict(self,
                X_seq: np.ndarray,
                return_std: bool = False) -> Dict[str, np.ndarray]:
        """
        集成预测。

        Args:
            X_seq: (n_samples, seq_len, n_features) 序列输入
            return_std: 是否返回不确定性估计

        Returns:
            {'ensemble': np.ndarray (n, horizon),
             'xgb': ..., 'lstm': ..., 'transformer': ...,
             'std': ... (if return_std)}
        """
        results = {}

        # XGBoost: 从序列提取表格特征（同 EnsembleTrainer._extract_tabular_features）
        if self.xgb_model is not None:
            last_step = X_seq[:, -1, :]
            mean_vals = X_seq.mean(axis=1)
            std_vals = X_seq.std(axis=1)
            min_vals = X_seq.min(axis=1)
            max_vals = X_seq.max(axis=1)
            trend = X_seq[:, -1, :] - X_seq[:, 0, :]
            X_tab = np.concatenate(
                [last_step, mean_vals, std_vals, min_vals, max_vals, trend],
                axis=1
            )
            results['xgb'] = self.xgb_model.predict(X_tab)

        # BiLSTM
        if self.lstm_model is not None:
            self.lstm_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_seq).to(self._device)
                results['lstm'] = self.lstm_model(X_t).cpu().numpy()

        # Transformer
        if self.transformer_model is not None:
            self.transformer_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_seq).to(self._device)
                results['transformer'] = self.transformer_model(X_t).cpu().numpy()

        # 加权集成
        ensemble_pred = None
        total_w = 0.0
        for name, w_key in [('xgb', 'xgb'), ('lstm', 'lstm'), ('transformer', 'transformer')]:
            if name in results:
                w = self.weights[w_key]
                if ensemble_pred is None:
                    ensemble_pred = w * results[name]
                else:
                    ensemble_pred += w * results[name]
                total_w += w

        if total_w > 0 and ensemble_pred is not None:
            results['ensemble'] = ensemble_pred / total_w

        return results

    @property
    def available_models(self) -> list:
        """返回已注册的模型名称"""
        models = []
        if self.xgb_model is not None:
            models.append('xgb')
        if self.lstm_model is not None:
            models.append('lstm')
        if self.transformer_model is not None:
            models.append('transformer')
        return models
