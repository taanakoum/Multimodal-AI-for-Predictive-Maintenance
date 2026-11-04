"""Modality encoders and evidential heads (plan, "Model architecture").

* :class:`ChannelEmbedding`: a channel's vector from its physical description
  (:mod:`pdmfm.sensors`: family, axis, location, site, sampling rate and window length), so
  the same kind of sensor gets the same vector in every dataset.
* :class:`ModalityEncoder`: adapter (LayerNorm then Linear) from the encoder's width to
  d = 256 (an MLP 2 -> 64 -> d for descriptor tokens such as slow channels, where a
  LayerNorm over two numbers would erase them), plus token-role, channel and position
  embeddings (the position of a token within its channel and role: a T token's time slot, a
  photo token's grid cell), a level MLP (F -> 64 -> d for F level features) for level tokens
  and an age embedding for photos; attention pooling with one learned query gives the
  modality summary z_m from that modality's tokens only, so per-modality opinions stay
  independent.
* :class:`PrototypeEvidentialHead`: text-anchored classes. q = normalize(W_q z); logits =
  s q P^T + b over the frozen Qwen prototypes P of the label space; evidence = softplus.
* :class:`NIGHead`: MLP d -> 128 -> 4 with the NIG constraints.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pdmfm import sensors
from pdmfm.losses import nig_parameters
from pdmfm.model.inputs import MAX_POSITIONS, ROLE, ROLES


class ChannelEmbedding(nn.Module):
    """(C, d) channel vectors from descriptions: family, axis, location and site embeddings,
    averaged as sum / 2 (four unit-variance parts keep the unit variance of the single
    positional embedding they replace, so content and metadata keep their balance at
    initialization), plus a linear map of the scale (log10 rate in kHz, log10 window in
    seconds), started at zero. It holds no per-dataset table: any dataset's ``codes`` and
    ``scale`` (:func:`pdmfm.sensors.table`) can be embedded by the same parameters."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.family = nn.Embedding(len(sensors.FAMILIES), d)
        self.axis = nn.Embedding(len(sensors.AXES), d)
        self.location = nn.Embedding(len(sensors.LOCATIONS), d)
        self.site = nn.Embedding(sensors.MAX_SITES, d)
        self.scale = nn.Linear(2, d)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

    def forward(self, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        parts = (
            self.family(codes[:, 0])
            + self.axis(codes[:, 1])
            + self.location(codes[:, 2])
            + self.site(codes[:, 3])
        )
        return parts / 2 + self.scale(scale)


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int = 256,
        levels: bool = False,
        age: bool = False,
        heads: int = 4,
        dropout: float = 0.1,
        descriptor: bool = False,
        level_dim: int = 4,
    ) -> None:
        super().__init__()
        if descriptor:
            self.adapter = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, d))
        else:
            self.adapter = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.role = nn.Embedding(len(ROLES), d)
        self.channel = ChannelEmbedding(d)
        self.position = nn.Embedding(MAX_POSITIONS, d)
        nn.init.normal_(self.position.weight, std=0.02)  # new information: starts small (ViT)
        self.level = (
            nn.Sequential(nn.Linear(level_dim, 64), nn.GELU(), nn.Linear(64, d)) if levels else None
        )
        self.age = nn.Linear(1, d) if age else None
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def embed(
        self,
        tokens: torch.Tensor,  # (B, T, d_in)
        roles: torch.Tensor,  # (T,)
        channels: torch.Tensor,  # (T,) index into the channel table
        channel_codes: torch.Tensor,  # (C, 4) family, axis, location, site
        channel_scale: torch.Tensor,  # (C, 2) log10 rate (kHz), log10 window (s)
        positions: torch.Tensor | None = None,  # (T,) slot within the channel and role
        token_present: torch.Tensor | None = None,  # (B, T) bool
        levels: torch.Tensor | None = None,  # (B, L, level_dim)
        level_channels: torch.Tensor | None = None,  # (L,)
        age: torch.Tensor | None = None,  # (B, T) passes since capture
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(token embeddings (B, T [+ L], d), token present (B, T [+ L]) bool)."""
        table = self.channel(channel_codes, channel_scale)  # (C, d)
        if positions is None:
            positions = torch.zeros_like(roles)
        x = self.adapter(tokens) + self.role(roles) + table[channels] + self.position(positions)
        if self.age is not None and age is not None:
            x = x + self.age(torch.log1p(age.clamp_min(0)).unsqueeze(-1))
        present = (
            token_present
            if token_present is not None
            else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        )
        if self.level is not None and levels is not None and level_channels is not None:
            level_role = torch.full_like(level_channels, ROLE["level"])
            lv = self.level(levels) + self.role(level_role) + table[level_channels]
            lv = lv + self.position(torch.zeros_like(level_channels))
            x = torch.cat([x, lv], 1)
            present = torch.cat([present, present.new_ones(lv.shape[:2])], 1)
        return x, present

    def pool(self, x: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(summary z (B, d), attention weights over the tokens (B, T))."""
        # A row with no token at all would make attention undefined; let it attend
        # everywhere. Its modality is absent, so its output is masked downstream.
        padding = ~present
        padding = padding & ~padding.all(1, keepdim=True)
        pooled, weights = self.attention(
            self.query.expand(x.shape[0], -1, -1), x, x, key_padding_mask=padding
        )
        return self.norm(self.dropout(pooled.squeeze(1))), weights.squeeze(1)

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """(summary z (B, d), attention weights over the tokens (B, T [+ L]))."""
        return self.pool(*self.embed(*args, **kwargs))


class PrototypeEvidentialHead(nn.Module):
    """``text_rank`` > 0 adds the plan's learned text projection W_t (p_c = W_t Qwen(c)) in
    low-rank form, W_t = I + A B with rank ``text_rank``: it starts at the Qwen prototypes
    and lets near-parallel ones (cosine up to 0.99) move apart; weight decay pulls it back
    towards the text."""

    prototypes: torch.Tensor  # (K, text width) buffer, L2-normalized

    def __init__(self, d: int, prototypes: torch.Tensor, text_rank: int = 0) -> None:
        super().__init__()
        self.register_buffer("prototypes", nn.functional.normalize(prototypes.float(), dim=-1))
        self.project = nn.Linear(d, prototypes.shape[1], bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.zeros(prototypes.shape[0]))
        self.text_rank = text_rank
        if text_rank:
            width = prototypes.shape[1]
            self.text_down = nn.Parameter(torch.randn(width, text_rank) / math.sqrt(width))
            self.text_up = nn.Parameter(torch.zeros(text_rank, width))

    def class_vectors(self) -> torch.Tensor:
        """(K, text width) class directions: the prototypes, through W_t if learned."""
        p = self.prototypes
        if self.text_rank:
            p = nn.functional.normalize(p + (p @ self.text_down) @ self.text_up, dim=-1)
        return p

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        q = nn.functional.normalize(self.project(z), dim=-1)
        return self.log_scale.exp() * q @ self.class_vectors().T + self.bias

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Evidence (B, K) >= 0."""
        return nn.functional.softplus(self.logits(z))


class LinearEvidentialHead(nn.Module):
    """Free linear class head (no text prototypes): a diagnostic of what text anchoring costs."""

    def __init__(self, d: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.linear(z))


class NIGHead(nn.Module):
    def __init__(self, d: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 4): gamma, nu > 0, alpha > 1, beta > 0."""
        return nig_parameters(self.mlp(z))
