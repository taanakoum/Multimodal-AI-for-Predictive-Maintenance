"""Plan B, the late-evidential model (plan, Options B; fusion candidate 1).

One expert per modality: a :class:`ModalityEncoder` with its own evidential class head and
NIG head. The class opinions are combined by Dempster's rule (TMC) and the NIGs by MoNIG.
An absent modality contributes zero evidence (a vacuous opinion, which Dempster's rule
leaves out exactly) and is skipped in the NIG fusion. Per-modality trust is 1 - u_m.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pdmfm.losses import ds_combine_all, nig_fuse_all
from pdmfm.model.inputs import TaskInputs
from pdmfm.model.layers import ModalityEncoder, NIGHead, PrototypeEvidentialHead


class LateEvidential(nn.Module):
    def __init__(self, inputs: TaskInputs, d: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.names = list(inputs.modalities)
        self.encoders = nn.ModuleDict()
        self.class_heads = nn.ModuleDict()
        self.reg_heads = nn.ModuleDict()
        prototypes = None if inputs.prototypes is None else torch.from_numpy(inputs.prototypes)
        n_channels = 1 + max(
            int(np.max(np.concatenate([m.channels, m.level_channels])))
            if m.level_channels is not None
            else int(m.channels.max())
            for m in inputs.modalities.values()
        )
        for name, modality in inputs.modalities.items():
            self.encoders[name] = ModalityEncoder(
                modality.dim,
                d=d,
                n_channels=n_channels,
                levels=modality.levels is not None,
                age=modality.age is not None,
                dropout=dropout,
                descriptor=modality.descriptor,
            )
            self.register_buffer(f"roles_{name}", torch.as_tensor(modality.roles, dtype=torch.long))
            self.register_buffer(
                f"channels_{name}", torch.as_tensor(modality.channels, dtype=torch.long)
            )
            if modality.level_channels is not None:
                level_channels = torch.as_tensor(modality.level_channels, dtype=torch.long)
                self.register_buffer(f"level_channels_{name}", level_channels)
            if prototypes is not None:
                self.class_heads[name] = PrototypeEvidentialHead(d, prototypes)
            if inputs.y_reg is not None:
                self.reg_heads[name] = NIGHead(d)

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """``batch[name]`` holds tokens, present and optionally token_present, levels, age."""
        evidence, nig, present, attention = [], [], [], {}
        for name in self.names:
            part = batch[name]
            z, weights = self.encoders[name](
                part["tokens"],
                getattr(self, f"roles_{name}"),
                getattr(self, f"channels_{name}"),
                token_present=part.get("token_present"),
                levels=part.get("levels"),
                level_channels=getattr(self, f"level_channels_{name}", None),
                age=part.get("age"),
            )
            here = part["present"]
            present.append(here)
            attention[name] = weights
            if name in self.class_heads:
                evidence.append(self.class_heads[name](z) * here.unsqueeze(-1))
            if name in self.reg_heads:
                nig.append(self.reg_heads[name](z))
        out = {"present": torch.stack(present, 1), "attention": attention}
        if evidence:
            out["evidence"] = torch.stack(evidence, 1)  # (B, M, K)
            out["combined"], out["conflict"] = ds_combine_all(out["evidence"])
        if nig:
            out["nig"] = torch.stack(nig, 1)  # (B, M, 4)
            out["fused_nig"] = nig_fuse_all(out["nig"], out["present"])
        return out
