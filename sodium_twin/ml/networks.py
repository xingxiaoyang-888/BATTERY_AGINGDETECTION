"""Patch-based and convolutional encoders with ordered physical SOH trajectories.

The patch encoder is inspired by PatchTST (Nie et al., ICLR 2023), with a
battery-specific channel fusion and cumulative degradation head. It is not an
exact reproduction of the paper. Outputs use SOH percentage points.
"""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "patchtst"
    features: int = 4
    lookback: int = 64
    horizon: int = 128
    patch_length: int = 8
    stride: int = 4
    width: int = 64
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1

    def __post_init__(self):
        if self.architecture not in {"patchtst", "itransformer", "selective-ssm", "tcn"}:
            raise ValueError("Unknown architecture.")
        if (
            min(
                self.features,
                self.lookback,
                self.horizon,
                self.patch_length,
                self.stride,
                self.width,
                self.heads,
                self.layers,
            )
            <= 0
        ):
            raise ValueError("Dimensions must be positive.")
        if self.patch_length > self.lookback or self.width % self.heads:
            raise ValueError("Patch length or attention dimensions are invalid.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")


class DegradationHead(nn.Module):
    """Guarantee q10 <= q50 <= q90 and nonincreasing trajectories by construction."""

    def __init__(self, width: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.projection = nn.Linear(width, horizon * 3)
        nn.init.normal_(self.projection.weight, std=0.005)
        nn.init.constant_(self.projection.bias, -4.0)

    def forward(self, embedding, current_soh):
        rates = F.softplus(self.projection(embedding).reshape(-1, self.horizon, 3))
        slow = rates[..., 0]
        middle = slow + rates[..., 1]
        fast = middle + rates[..., 2]
        ordered_rates = torch.stack([fast, middle, slow], dim=-1)
        return (current_soh[:, None, None] - ordered_rates.cumsum(dim=1)).clamp(0, 105)


class PatchEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        count = 1 + math.ceil((cfg.lookback - cfg.patch_length) / cfg.stride)
        self.pad = (count - 1) * cfg.stride + cfg.patch_length - cfg.lookback
        self.embed = nn.Linear(cfg.patch_length, cfg.width)
        self.position = nn.Parameter(torch.randn(1, count, cfg.width) * 0.02)
        # Construct each block separately to avoid identical cloned initial weights.
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    cfg.width,
                    cfg.heads,
                    cfg.width * 4,
                    cfg.dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(cfg.layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.width)
        self.fusion = nn.Sequential(nn.Linear(cfg.features * cfg.width, cfg.width), nn.GELU())
        self.reconstruct = nn.Linear(cfg.width, cfg.patch_length)

    def patches(self, x):
        series = x.transpose(1, 2)
        if self.pad:
            series = F.pad(series, (0, self.pad), mode="replicate")
        return series.unfold(-1, self.cfg.patch_length, self.cfg.stride)

    def encode_tokens(self, patches):
        b, c, n, p = patches.shape
        tokens = self.embed(patches.reshape(b * c, n, p)) + self.position
        for layer in self.blocks:
            tokens = layer(tokens)
        return self.norm(tokens).reshape(b, c, n, -1)

    def forward(self, x):
        tokens = self.encode_tokens(self.patches(x))
        return self.fusion(tokens.mean(dim=2).flatten(start_dim=1))

    def masked_loss(self, x, mask_ratio=0.4):
        """Mask raw time values before patching to avoid overlapping-patch leakage."""
        if not 0 < mask_ratio < 1:
            raise ValueError("Mask ratio must be between zero and one.")
        mask = torch.rand(x.shape[:2], device=x.device) < mask_ratio
        mask[:, 0] = True
        masked = x.masked_fill(mask[..., None], 0)
        target = self.patches(x)
        patch_mask = self.patches(mask[..., None].expand_as(x).float()).bool()
        prediction = self.reconstruct(self.encode_tokens(self.patches(masked)))
        return (prediction - target).square()[patch_mask].mean()


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.padding = dilation * 2
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        out = self.conv(F.pad(x, (self.padding, 0)))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(F.gelu(out))


class TemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input = nn.Conv1d(cfg.features, cfg.width, 1)
        self.blocks = nn.Sequential(
            *[CausalBlock(cfg.width, 2**i, cfg.dropout) for i in range(cfg.layers)]
        )

    def forward(self, x):
        return self.blocks(self.input(x.transpose(1, 2)))[..., -1]


class InvertedEncoder(nn.Module):
    """Embed each sensor history as a token and attend across measured variables."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(cfg.lookback, cfg.width)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    cfg.width,
                    cfg.heads,
                    cfg.width * 4,
                    cfg.dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(cfg.layers)
            ]
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(cfg.features * cfg.width),
            nn.Linear(cfg.features * cfg.width, cfg.width),
            nn.GELU(),
        )

    def forward(self, x):
        tokens = self.embed(x.transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens)
        return self.fuse(tokens.flatten(start_dim=1))


class SelectiveBlock(nn.Module):
    """Portable input-selective diagonal state-space scan inspired by Mamba.

    This reference has O(T * D) scan cost, no CUDA extension, and no claim of
    equivalence to the fused Mamba or Mamba-2 implementations.
    """

    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, width * 2)
        self.conv = nn.Conv1d(width, width, 3, groups=width)
        self.select = nn.Linear(width, width * 3)
        self.log_a = nn.Parameter(torch.zeros(width))
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        signal, gate = self.input(self.norm(x)).chunk(2, dim=-1)
        signal = F.silu(self.conv(F.pad(signal.transpose(1, 2), (2, 0))).transpose(1, 2))
        delta, b, c = self.select(signal).chunk(3, dim=-1)
        decay = torch.exp(-F.softplus(delta) * self.log_a.exp())
        state = torch.zeros_like(signal[:, 0])
        outputs = []
        for step in range(x.shape[1]):
            state = (
                decay[:, step] * state
                + (1 - decay[:, step]) * torch.tanh(b[:, step]) * signal[:, step]
            )
            outputs.append(state * torch.tanh(c[:, step]))
        output = torch.stack(outputs, dim=1) * F.silu(gate)
        return x + self.dropout(self.output(output))


class SelectiveEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(cfg.features, cfg.width)
        self.blocks = nn.Sequential(
            *[SelectiveBlock(cfg.width, cfg.dropout) for _ in range(cfg.layers)]
        )
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, x):
        return self.norm(self.blocks(self.embed(x))[:, -1])


class BatteryForecaster(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        encoders = {
            "patchtst": PatchEncoder,
            "itransformer": InvertedEncoder,
            "selective-ssm": SelectiveEncoder,
            "tcn": TemporalEncoder,
        }
        self.encoder = encoders[cfg.architecture](cfg)
        self.head = DegradationHead(cfg.width, cfg.horizon)
        self.hazard = nn.Linear(cfg.width, cfg.horizon)

    def forward(self, x, current_soh):
        if x.ndim != 3 or x.shape[1:] != (self.cfg.lookback, self.cfg.features):
            raise ValueError("Input must have shape [batch, lookback, features].")
        if current_soh.shape != (x.shape[0],):
            raise ValueError("Current SOH must have shape [batch].")
        return self.head(self.encoder(x), current_soh)

    def forward_with_hazard(self, x, current_soh):
        embedding = self.encoder(x)
        return self.head(embedding, current_soh), self.hazard(embedding)

    def metadata(self):
        return {"config": asdict(self.cfg), "parameters": sum(p.numel() for p in self.parameters())}


def pinball_loss(prediction, target):
    quantiles = prediction.new_tensor([0.1, 0.5, 0.9])
    residual = target[..., None] - prediction
    return torch.maximum(quantiles * residual, (quantiles - 1) * residual).mean()


def discrete_survival_nll(logits, event_bin, observed):
    """Right-censored discrete hazard likelihood with inclusive censoring bins.

    event_bin is the last fully observed bin for censored observations; for
    observed events it is the interval containing the event. No RUL is imputed.
    """
    if logits.ndim != 2 or event_bin.shape != logits.shape[:1] or observed.shape != event_bin.shape:
        raise ValueError("Invalid survival tensor shapes.")
    if ((event_bin < 0) | (event_bin >= logits.shape[1])).any():
        raise ValueError("Event/censoring bin is outside the modeled horizon.")
    grid = torch.arange(logits.shape[1], device=logits.device)[None, :]
    survival_mask = grid < event_bin[:, None]
    survival_mask |= (grid == event_bin[:, None]) & ~observed.bool()[:, None]
    survival = (F.logsigmoid(-logits) * survival_mask).sum(dim=1)
    event = F.logsigmoid(logits).gather(1, event_bin.long()[:, None]).squeeze(1)
    return -(survival + event * observed).mean()
