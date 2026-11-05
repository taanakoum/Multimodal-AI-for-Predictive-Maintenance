"""Plan B, the late-evidential model (plan, Options B; fusion candidate 1).

One expert per modality: a :class:`ModalityEncoder` with its own evidential class head and
NIG head. The class opinions are combined by Dempster's rule (TMC) and the NIGs by MoNIG.
An absent modality contributes zero evidence (a vacuous opinion, which Dempster's rule
leaves out exactly) and is skipped in the NIG fusion. Per-modality trust is 1 - u_m.

Level features and descriptor tokens (slow channels) mix units (log levels near 1, force
means near 100 N), so they are standardized with the mean and standard deviation of the
training rows (``train_rows``), stored as buffers; without training rows they pass as they are.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import nn

from pdmfm import sensors
from pdmfm.losses import ds_combine_all, nig_fuse_all
from pdmfm.model.inputs import TaskInputs
from pdmfm.model.layers import (
    LinearEvidentialHead,
    ModalityEncoder,
    NIGHead,
    PrototypeEvidentialHead,
)


def class_head(head: str, d: int, prototypes: torch.Tensor) -> nn.Module:
    """An evidential class head: Qwen prototypes, prototypes through a low-rank learned W_t
    (``projected``), or a free linear layer (``linear``, no text anchoring)."""
    if head == "linear":
        return LinearEvidentialHead(d, len(prototypes))
    if head in ("prototype", "projected"):
        return PrototypeEvidentialHead(d, prototypes, 16 if head == "projected" else 0)
    raise ValueError(f"head must be prototype, projected or linear, not {head!r}")


def _standardizer(values: np.ndarray, rows, mask: np.ndarray | None = None):
    """(mean, std) over ``rows`` of (N, ...) values, per remaining position; masked entries
    are ignored. Identity when ``rows`` is None; a constant feature keeps std 1."""
    shape = values.shape[1:]
    if rows is None:
        return np.zeros(shape, np.float32), np.ones(shape, np.float32)
    picked = np.asarray(values[rows], dtype=np.float64)
    if mask is not None:
        picked = np.where(mask[rows][..., None], picked, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a position masked in every row
        mean, std = np.nanmean(picked, axis=0), np.nanstd(picked, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return np.nan_to_num(mean).astype(np.float32), std.astype(np.float32)


class LateEvidential(nn.Module):
    channel_codes: torch.Tensor  # (C, 4) buffer: the dataset's channel descriptions
    channel_scale: torch.Tensor  # (C, 2) buffer

    def __init__(
        self,
        inputs: TaskInputs,
        d: int = 256,
        dropout: float = 0.1,
        train_rows=None,
        head: str = "prototype",
    ) -> None:
        super().__init__()
        self.names = list(inputs.modalities)
        self.encoders = nn.ModuleDict()
        self.class_heads = nn.ModuleDict()
        self.reg_heads = nn.ModuleDict()
        prototypes = None if inputs.prototypes is None else torch.from_numpy(inputs.prototypes)
        if inputs.channels is None:
            raise ValueError("inputs.channels is missing: describe every channel (pdmfm.sensors)")
        used = max(
            int(np.max(np.concatenate([m.channels, m.level_channels])))
            if m.level_channels is not None
            else int(m.channels.max())
            for m in inputs.modalities.values()
        )
        if used >= len(inputs.channels):
            raise ValueError(f"a token uses channel {used}; only {len(inputs.channels)} described")
        codes, scale = sensors.table(inputs.channels)
        self.register_buffer("channel_codes", torch.from_numpy(codes))
        self.register_buffer("channel_scale", torch.from_numpy(scale))
        for name, modality in inputs.modalities.items():
            self.encoders[name] = ModalityEncoder(
                modality.dim,
                d=d,
                levels=modality.levels is not None,
                age=modality.age is not None,
                dropout=dropout,
                descriptor=modality.descriptor,
                level_dim=4 if modality.levels is None else int(modality.levels.shape[-1]),
            )
            self.register_buffer(f"roles_{name}", torch.as_tensor(modality.roles, dtype=torch.long))
            self.register_buffer(
                f"channels_{name}", torch.as_tensor(modality.channels, dtype=torch.long)
            )
            positions = (
                np.zeros(len(modality.roles), dtype=np.int64)
                if modality.positions is None
                else modality.positions
            )
            self.register_buffer(f"positions_{name}", torch.as_tensor(positions, dtype=torch.long))
            if modality.level_channels is not None:
                level_channels = torch.as_tensor(modality.level_channels, dtype=torch.long)
                self.register_buffer(f"level_channels_{name}", level_channels)
            if modality.levels is not None:
                self._buffers_from(f"levels_{name}", _standardizer(modality.levels, train_rows))
            if modality.descriptor:
                stats = _standardizer(modality.tokens, train_rows, modality.token_present)
                self._buffers_from(f"tokens_{name}", stats)
            if prototypes is not None:
                self.class_heads[name] = class_head(head, d, prototypes)
            if inputs.y_reg is not None:
                self.reg_heads[name] = NIGHead(d)

    def _buffers_from(self, prefix: str, stats) -> None:
        mean, std = stats
        self.register_buffer(f"{prefix}_mean", torch.from_numpy(mean))
        self.register_buffer(f"{prefix}_std", torch.from_numpy(std))

    def _standardized(self, prefix: str, values):
        if values is None or not hasattr(self, f"{prefix}_mean"):
            return values
        return (values - getattr(self, f"{prefix}_mean")) / getattr(self, f"{prefix}_std")

    def embed(self, name: str, part: dict[str, torch.Tensor]):
        """(token embeddings, token present) of one modality's batch part."""
        encoder = self.encoders[name]
        assert isinstance(encoder, ModalityEncoder)
        return encoder.embed(
            self._standardized(f"tokens_{name}", part["tokens"]),
            getattr(self, f"roles_{name}"),
            getattr(self, f"channels_{name}"),
            self.channel_codes,
            self.channel_scale,
            positions=getattr(self, f"positions_{name}"),
            token_present=part.get("token_present"),
            levels=self._standardized(f"levels_{name}", part.get("levels")),
            level_channels=getattr(self, f"level_channels_{name}", None),
            age=part.get("age"),
            hide=part.get("hide"),
            level_hide=part.get("level_hide"),
            level_present=part.get("level_present"),
        )

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """``batch[name]`` holds tokens, present and optionally token_present, levels, age."""
        evidence, nig, present, attention, summaries = [], [], [], {}, []
        for name in self.names:
            part = batch[name]
            encoder = self.encoders[name]
            assert isinstance(encoder, ModalityEncoder)
            z, weights = encoder.pool(*self.embed(name, part))
            summaries.append(z)
            here = part["present"]
            present.append(here)
            attention[name] = weights
            if name in self.class_heads:
                evidence.append(self.class_heads[name](z) * here.unsqueeze(-1))
            if name in self.reg_heads:
                nig.append(self.reg_heads[name](z))
        out = {"present": torch.stack(present, 1), "attention": attention}
        out["z"] = torch.stack(summaries, 1)  # (B, M, d) the modality summaries
        if evidence:
            out["evidence"] = torch.stack(evidence, 1)  # (B, M, K)
            out["combined"], out["conflict"] = ds_combine_all(out["evidence"])
        if nig:
            out["nig"] = torch.stack(nig, 1)  # (B, M, 4)
            out["fused_nig"] = nig_fuse_all(out["nig"], out["present"])
        return out
