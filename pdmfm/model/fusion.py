"""Gate-2 fusion candidates (plan, "Fusion search candidates").

Every candidate keeps Plan B's per-modality experts (embedding, attention pooling, an
evidential class head and a NIG head per modality) and adds a fusion stage over the tokens
of all modalities (``parts``: per modality, its embedded tokens, token mask, presence and
pooled vector). The fused representation gets its own class and NIG heads, and the fused
opinion is combined with the per-modality opinions by Dempster's rule (classes) and MoNIG
(regression), as the plan's architecture draws it ("fused evidential head, combined with the
per-modality evidence"). An absent modality's tokens are masked out of the fusion and its
opinion is vacuous. Each fusion stage stays within 3 M parameters at d = 256:

* ``concat``: each modality's pooled vector (its expert's attention pooling, zero when the
  modality is absent), concatenated with presence flags, MLP.
* ``perceiver``: 32 learned latents; 2 cross-attention layers over all tokens, then 4
  self-attention layers; the latents' mean.
* ``mbt``: multimodal bottleneck transformer: 4 bottleneck tokens; 4 layers, each running
  one shared transformer layer over every modality's tokens plus the bottleneck, the
  bottleneck averaged over the present modalities (weights shared across modalities, with
  the modality told apart by its role embeddings, to stay within the budget).
* ``c2f``: coarse-to-fine hub (after OmniFuser's C2F, rewritten): the sensor modalities'
  tokens form the hub; every other modality is summarized by 4 learned landmarks, the hub
  cross-attends to each modality's landmarks through a per-token gate, then 2 self-attention
  layers over the hub; its masked mean.

``late`` (Plan B) has no fusion stage and is :class:`pdmfm.model.late.LateEvidential`.
"""

from __future__ import annotations

import torch
from torch import nn

from pdmfm.losses import ds_combine_all, nig_fuse_all
from pdmfm.model.inputs import TaskInputs
from pdmfm.model.late import LateEvidential, class_head
from pdmfm.model.layers import ModalityEncoder, NIGHead

FUSIONS = ("late", "concat", "perceiver", "mbt", "c2f")
NON_HUB = ("photo", "slow")  # modalities that are not sensor hubs in c2f


def _layer(d: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d, nhead=4, dim_feedforward=d, dropout=dropout, batch_first=True, norm_first=True
    )


def _masked_mean(x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    weight = present.unsqueeze(-1).float()
    return (x * weight).sum(1) / weight.sum(1).clamp_min(1.0)


class ConcatFusion(nn.Module):
    def __init__(self, names: list[str], d: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(names) * (d + 1), d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, parts):
        pooled = [
            torch.cat([z * here.unsqueeze(-1), here.unsqueeze(-1).float()], -1)
            for _, _, here, z in parts
        ]
        return self.norm(self.mlp(torch.cat(pooled, -1)))


class PerceiverFusion(nn.Module):
    def __init__(self, names: list[str], d: int, dropout: float, latents: int = 32) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, latents, d) * 0.02)
        self.cross = nn.ModuleList(
            nn.MultiheadAttention(d, 4, dropout=dropout, batch_first=True) for _ in range(2)
        )
        self.cross_ff = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(2)
        )
        self.cross_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(2))
        self.self_layers = nn.ModuleList(_layer(d, dropout) for _ in range(4))
        self.norm = nn.LayerNorm(d)

    def forward(self, parts):
        tokens = torch.cat([x for x, _, _, _ in parts], 1)
        present = torch.cat([tp & here.unsqueeze(-1) for _, tp, here, _ in parts], 1)
        padding = ~present & ~(~present).all(1, keepdim=True)
        z = self.latents.expand(tokens.shape[0], -1, -1)
        for attention, ff, norm in zip(self.cross, self.cross_ff, self.cross_norm, strict=True):
            update, _ = attention(norm(z), tokens, tokens, key_padding_mask=padding)
            z = z + update
            z = z + ff(z)
        for layer in self.self_layers:
            z = layer(z)
        return self.norm(z.mean(1))


class MBTFusion(nn.Module):
    def __init__(self, names: list[str], d: int, dropout: float, bottleneck: int = 4) -> None:
        super().__init__()
        self.bottleneck = nn.Parameter(torch.randn(1, bottleneck, d) * 0.02)
        self.layers = nn.ModuleList(_layer(d, dropout) for _ in range(4))
        self.norm = nn.LayerNorm(d)
        self.k = bottleneck

    def forward(self, parts):
        b = self.bottleneck.expand(parts[0][0].shape[0], -1, -1)
        for layer in self.layers:
            updates, weights = [], []
            new_parts = []
            for x, tp, here, z in parts:
                seq = torch.cat([x, b], 1)
                keep = torch.cat([tp, tp.new_ones(tp.shape[0], self.k)], 1)
                out = layer(seq, src_key_padding_mask=~keep)
                new_parts.append((out[:, : x.shape[1]], tp, here, z))
                updates.append(out[:, x.shape[1] :] * here.view(-1, 1, 1))
                weights.append(here.float().view(-1, 1, 1))
            total = torch.stack(weights).sum(0)
            b = torch.where(total > 0, torch.stack(updates).sum(0) / total.clamp_min(1.0), b)
            parts = new_parts
        return self.norm(b.mean(1))


class C2FFusion(nn.Module):
    def __init__(self, names: list[str], d: int, dropout: float, landmarks: int = 4) -> None:
        super().__init__()
        self.hub = [i for i, n in enumerate(names) if n not in NON_HUB] or [0]
        self.others = [i for i in range(len(names)) if i not in self.hub]
        self.queries = nn.ParameterList(
            nn.Parameter(torch.randn(1, landmarks, d) * 0.02) for _ in self.others
        )
        self.summarize = nn.ModuleList(
            nn.MultiheadAttention(d, 4, dropout=dropout, batch_first=True) for _ in self.others
        )
        self.attend = nn.ModuleList(
            nn.MultiheadAttention(d, 4, dropout=dropout, batch_first=True) for _ in self.others
        )
        self.gates = nn.ModuleList(nn.Linear(2 * d, d) for _ in self.others)
        self.layers = nn.ModuleList(_layer(d, dropout) for _ in range(2))
        self.norm = nn.LayerNorm(d)

    def forward(self, parts):
        hub = torch.cat([parts[i][0] for i in self.hub], 1)
        hub_present = torch.cat([parts[i][1] & parts[i][2].unsqueeze(-1) for i in self.hub], 1)
        for j, i in enumerate(self.others):
            x, tp, here, _ = parts[i]
            padding = ~tp & ~(~tp).all(1, keepdim=True)
            query = self.queries[j].expand(x.shape[0], -1, -1)
            landmarks, _ = self.summarize[j](query, x, x, key_padding_mask=padding)
            delta, _ = self.attend[j](hub, landmarks, landmarks)
            gate = torch.sigmoid(self.gates[j](torch.cat([hub, delta], -1)))
            hub = hub + gate * delta * here.view(-1, 1, 1)  # an absent modality adds nothing
        padding = ~hub_present & ~(~hub_present).all(1, keepdim=True)
        for layer in self.layers:
            hub = layer(hub, src_key_padding_mask=padding)
        return self.norm(_masked_mean(hub, hub_present))


STAGES = {"concat": ConcatFusion, "perceiver": PerceiverFusion, "mbt": MBTFusion, "c2f": C2FFusion}


class FusionModel(LateEvidential):
    """Plan B's experts plus a fusion stage whose opinion joins the combination."""

    def __init__(
        self,
        inputs: TaskInputs,
        fusion: str,
        d: int = 256,
        dropout: float = 0.1,
        train_rows=None,
        head: str = "prototype",
    ) -> None:
        super().__init__(inputs, d=d, dropout=dropout, train_rows=train_rows, head=head)
        if fusion not in STAGES:
            raise ValueError(f"fusion must be one of {FUSIONS}, not {fusion!r}")
        self.fusion_name = fusion
        self.stage = STAGES[fusion](self.names, d, dropout)
        self.fused_class_head = (
            None
            if inputs.prototypes is None
            else class_head(head, d, torch.from_numpy(inputs.prototypes))
        )
        self.fused_reg_head = None if inputs.y_reg is None else NIGHead(d)

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        evidence, nig, present, attention, parts = [], [], [], {}, []
        for name in self.names:
            part = batch[name]
            encoder = self.encoders[name]
            assert isinstance(encoder, ModalityEncoder)
            x, token_present = encoder.embed(
                self._standardized(f"tokens_{name}", part["tokens"]),
                getattr(self, f"roles_{name}"),
                getattr(self, f"channels_{name}"),
                token_present=part.get("token_present"),
                levels=self._standardized(f"levels_{name}", part.get("levels")),
                level_channels=getattr(self, f"level_channels_{name}", None),
                age=part.get("age"),
            )
            z, weights = encoder.pool(x, token_present)
            here = part["present"]
            present.append(here)
            attention[name] = weights
            parts.append((x, token_present, here, z))
            if name in self.class_heads:
                evidence.append(self.class_heads[name](z) * here.unsqueeze(-1))
            if name in self.reg_heads:
                nig.append(self.reg_heads[name](z))
        present_all = torch.stack(present, 1)
        anything = present_all.any(1)
        fused_z = self.stage(parts)
        out = {"present": present_all, "attention": attention}
        if evidence:
            out["evidence"] = torch.stack(evidence, 1)
            assert self.fused_class_head is not None
            fused = self.fused_class_head(fused_z) * anything.unsqueeze(-1)
            out["fused_evidence"] = fused
            stacked = torch.cat([fused.unsqueeze(1), out["evidence"]], 1)
            out["combined"], out["conflict"] = ds_combine_all(stacked)
        if nig:
            out["nig"] = torch.stack(nig, 1)
            assert self.fused_reg_head is not None
            out["fused_own_nig"] = self.fused_reg_head(fused_z)
            stacked = torch.cat([out["fused_own_nig"].unsqueeze(1), out["nig"]], 1)
            flags = torch.cat([anything.unsqueeze(1), present_all], 1)
            out["fused_nig"] = nig_fuse_all(stacked, flags)
        return out


def build_model(inputs: TaskInputs, fusion: str, d: int, dropout: float, train_rows, head: str):
    """The model of a fusion candidate: Plan B for ``late``, :class:`FusionModel` otherwise."""
    if fusion == "late":
        return LateEvidential(inputs, d=d, dropout=dropout, train_rows=train_rows, head=head)
    return FusionModel(inputs, fusion, d=d, dropout=dropout, train_rows=train_rows, head=head)
