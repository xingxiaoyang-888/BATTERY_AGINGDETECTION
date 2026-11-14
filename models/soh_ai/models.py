# models/soh_ai/models.py
"""Legacy XGBoost, recurrent, Transformer, and ensemble architectures."""

import logging
import numpy as np
import warnings
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    FEATURE_CFG,
    ACTUAL_FEATURE_COLUMNS,
    XGB_CFG,
    LSTM_CFG,
    TRANSFORMER_CFG,
    ENSEMBLE_CFG,
)

logger = logging.getLogger(__name__)


def _default_soh_feature_index() -> int:
    """Locate SOH in the canonical feature ordering."""
    return ACTUAL_FEATURE_COLUMNS.index(FEATURE_CFG.target_col)


# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════

def _init_weights(module: nn.Module, method: str = "xavier"):
    """Initialize linear and recurrent layers using the selected weight strategy."""
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
    """Add sinusoidal temporal positions to Transformer input embeddings."""

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

# ═══════════════════════════════════════════════════════════════

class XGBoostWrapper:
    """Residual XGBoost baseline with single-step prediction and persistence anchoring."""

    def __init__(self, cfg=None, residual_learning: bool = True,
                 soh_feature_index: Optional[int] = None):
        self.cfg = cfg or XGB_CFG
        self.residual_learning = residual_learning
        self.soh_feature_index = (
            _default_soh_feature_index()
            if soh_feature_index is None else soh_feature_index
        )
        self._model = None
        self._is_fitted = False
        self._multi_output = False

    def _build(self):
        """Construct the optional XGBoost model lazily."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost 未安装。请运行: pip install xgboost"
            )
        params = dict(
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
        if self.residual_learning:
            params['base_score'] = 0.0
        self._model = xgb.XGBRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "XGBoostWrapper":
        """Fit."""
        if self._model is None:
            self._build()


        y_flat = y.ravel() if y.ndim > 1 else y
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(
                f"XGBoost 仅支持 horizon=1 单步预测，收到 y.shape={y.shape}。"
                f"多步推演请在推理时使用自回归滚动策略。"
            )

        if self.soh_feature_index >= X.shape[1]:
            raise ValueError(
                f"SOH 特征索引 {self.soh_feature_index} 超出 X.shape={X.shape}"
            )
        train_target = (
            y_flat - X[:, self.soh_feature_index]
            if self.residual_learning else y_flat
        )

        y_val_flat = None
        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_flat = y_val.ravel() if y_val.ndim > 1 else y_val
            val_target = (
                y_val_flat - X_val[:, self.soh_feature_index]
                if self.residual_learning else y_val_flat
            )
            eval_set = [(X_val, val_target)]
        else:

            self._model.set_params(early_stopping_rounds=None)

        self._model.fit(X, train_target, eval_set=eval_set, verbose=False)
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        prediction = self._model.predict(X)
        if self.residual_learning:
            prediction = prediction + X[:, self.soh_feature_index]
        return prediction.reshape(-1, 1)

    def save(self, filepath: str):
        """Save."""
        import joblib
        state = {
            'model': self._model,
            'config': self.cfg,
            'residual_learning': self.residual_learning,
            'soh_feature_index': self.soh_feature_index,
        }
        joblib.dump(state, filepath)
        logger.info(f"  XGBoost 模型已保存至: {filepath}")

    @classmethod
    def load(cls, filepath: str, cfg=None) -> "XGBoostWrapper":
        """Load."""
        import joblib
        state = joblib.load(filepath)
        wrapper = cls(
            cfg=state.get('config', cfg),
            residual_learning=state.get('residual_learning', False),
            soh_feature_index=state.get('soh_feature_index'),
        )

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

# ═══════════════════════════════════════════════════════════════

class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over sequence representations."""

    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute model outputs from the input tensor using the configured feature order."""

        scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_w = F.softmax(scores, dim=-1)  # (batch, seq_len)
        context = torch.bmm(attn_w.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        return context, attn_w


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM encoder with additive attention and residual SOH prediction."""

    def __init__(self, cfg=None, input_dim: int = None,
                 soh_feature_index: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or LSTM_CFG


        _input_dim = input_dim or self.cfg.input_dim
        self.soh_feature_index = (
            _default_soh_feature_index()
            if soh_feature_index is None else soh_feature_index
        )
        if not 0 <= self.soh_feature_index < _input_dim:
            raise ValueError(
                f"SOH 特征索引 {self.soh_feature_index} 超出 input_dim={_input_dim}"
            )
        hidden_dim = self.cfg.hidden_dim
        num_layers = self.cfg.num_layers
        dropout = self.cfg.dropout
        self.bidirectional = self.cfg.bidirectional
        self.horizon = FEATURE_CFG.prediction_horizon


        self.lstm = nn.LSTM(
            input_size=_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim


        self.attention = AdditiveAttention(
            hidden_dim=lstm_out_dim,
            attention_dim=self.cfg.attention_dim,
        )


        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.horizon),
        )

        self.apply(lambda m: _init_weights(m, method="xavier"))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute model outputs from the input tensor using the configured feature order."""
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, L, D)
        context, attn_w = self.attention(lstm_out, mask)
        out = self.dropout(context)
        residual = self.fc(out)
        current_soh = x[:, -1, self.soh_feature_index].unsqueeze(-1)
        return current_soh + residual

    @torch.no_grad()
    def rollout(self, x: torch.Tensor, steps: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Roll the latest history window forward autoregressively; enforce a nonincreasing SOH trajectory."""
        preds = []
        cur = x
        cur_mask = mask

        for _ in range(steps):
            y = self.forward(cur, cur_mask)
            next_pred = y[:, -1:] if y.ndim == 2 else y.unsqueeze(-1)
            preds.append(next_pred)

            next_step = cur[:, -1:, :].clone()
            next_step[:, :, self.soh_feature_index] = next_pred
            cur = torch.cat([cur[:, 1:, :], next_step], dim=1)

            if cur_mask is not None:
                cur_mask = torch.cat([cur_mask[:, 1:], torch.ones_like(cur_mask[:, :1])], dim=1)

        return torch.cat(preds, dim=1)


# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    """Temporal Transformer encoder with persistence-anchored residual prediction."""

    def __init__(self, cfg=None, input_dim: int = None,
                 soh_feature_index: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or TRANSFORMER_CFG


        _input_dim = input_dim or self.cfg.input_dim
        self.soh_feature_index = (
            _default_soh_feature_index()
            if soh_feature_index is None else soh_feature_index
        )
        if not 0 <= self.soh_feature_index < _input_dim:
            raise ValueError(
                f"SOH 特征索引 {self.soh_feature_index} 超出 input_dim={_input_dim}"
            )
        d_model = self.cfg.d_model
        nhead = self.cfg.nhead
        num_layers = self.cfg.num_encoder_layers
        dim_feedforward = self.cfg.dim_feedforward
        dropout = self.cfg.dropout
        self.horizon = FEATURE_CFG.prediction_horizon


        self.input_proj = nn.Linear(_input_dim, d_model)


        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=self.cfg.max_seq_len,
            dropout=dropout,
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=self.cfg.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.horizon),
        )

        self.apply(lambda m: _init_weights(m, method="kaiming"))
        nn.init.zeros_(self.pred_head[-1].weight)
        nn.init.zeros_(self.pred_head[-1].bias)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute model outputs from the input tensor using the configured feature order."""
        current_soh = x[:, -1, self.soh_feature_index].unsqueeze(-1)


        x = self.input_proj(x)  # (B, L, d_model)
        x = self.pos_encoder(x)



        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()

        encoded = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, L, d_model)


        if mask is not None:

            valid_mask = mask.bool().unsqueeze(-1).float()  # (B, L, 1)
            pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)


        residual = self.pred_head(pooled)  # (B, horizon)
        return current_soh + residual


    @torch.no_grad()
    def rollout(self, x: torch.Tensor, steps: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Roll the latest history window forward autoregressively; enforce a nonincreasing SOH trajectory."""
        preds = []
        cur = x
        cur_mask = mask
        for _ in range(steps):
            y = self.forward(cur, cur_mask)
            next_pred = y[:, -1:] if y.ndim == 2 else y.unsqueeze(-1)
            preds.append(next_pred)
            next_step = cur[:, -1:, :].clone()
            next_step[:, :, self.soh_feature_index] = next_pred
            cur = torch.cat([cur[:, 1:, :], next_step], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat([cur_mask[:, 1:], torch.ones_like(cur_mask[:, :1])], dim=1)
        return torch.cat(preds, dim=1)

# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════

class EnsembleModel:
    """Combine available regressors using normalized validation-derived weights."""

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
        """Register."""
        if name == 'xgb':
            self.xgb_model = model
        elif name == 'lstm':
            self.lstm_model = model
        elif name == 'transformer':
            self.transformer_model = model
        else:
            raise ValueError(f"未知模型名称: {name}，可选: xgb, lstm, transformer")

    def to(self, device: torch.device):
        """To."""
        self._device = device
        if self.lstm_model is not None:
            self.lstm_model.to(device)
        if self.transformer_model is not None:
            self.transformer_model.to(device)
        return self

    def set_weights(self, xgb: float = None, lstm: float = None, transformer: float = None):
        """Normalize and update ensemble weights."""
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
        """Predict."""
        results = {}


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
        """Available models."""
        models = []
        if self.xgb_model is not None:
            models.append('xgb')
        if self.lstm_model is not None:
            models.append('lstm')
        if self.transformer_model is not None:
            models.append('transformer')
        return models
