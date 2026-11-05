"""Stage 6: pretraining the fusion stage (or Plan B's experts) across the source datasets
(plan, "Self-supervised pretraining and modality alignment", level 2). Encoders stay frozen:
everything runs on the cached tokens.

**One backbone, shared.** For outer fold k, every source dataset gets a Plan B model (or a
Gate-2 candidate) whose expert for each sensor family (vibration, force, current, acoustic,
slow channels, photos), whose class heads' text projection per family and, for a fusion
candidate, whose fusion stage are the same modules for all datasets. A dataset keeps what is
its own: its token layout and channel descriptions (:mod:`pdmfm.sensors`), its
standardization of level and slow descriptors, its class prototypes and biases, its
regression heads. The channel descriptions are what make the shared experts read every
dataset's tokens alike.

**Data.** Every source dataset's rows that :func:`pdmfm.protocols.pretraining_mask` allows
for fold k (none of fold k's test or inner-validation units, no target dataset), labelled or
not; fold k's inner-validation units (CWRU: its 3 hp windows) choose the checkpoint. A step
draws one dataset with probability proportional to the square root of its rows, then a batch
whose units are balanced (a unit uniformly, then one of its rows).

**Losses**, L = w_mask L_mask + w_nce L_nce + w_sup L_sup (plan: 1, 0.5 and 1):

* ``L_mask``: each present modality is dropped with p = 0.3 (at least one kept) and 40 % of
  the kept tokens are hidden: their content becomes the learned [MASK], their role, channel
  and position stay. A decoder, queried with the hidden tokens' identities and reading only
  the kept modalities' summaries (and the fused representation), predicts each hidden
  token's frozen, layer-normed encoder feature (1 - cosine) and the standardized level and
  slow descriptors (MSE). A dropped modality must be predicted from the others.
* ``L_nce``: two views of the same window (:data:`VIEWS`: CWRU drive end / fan end, XJTU-SY
  horizontal / vertical, Paderborn vibration / currents, Nonastreda and MATWI force /
  photos), each pooled (the fused representation, or the mean of the view's modality
  summaries), projected and compared by a symmetric InfoNCE with a learnable temperature
  (from 0.07). Windows of the same unit are never negatives.
* ``L_sup``: Plan B's supervised losses (evidential classes against the text prototypes, NIG
  regression) on the labelled rows, unmasked: the plan's text anchoring.

**Recipe** (plan): AdamW 3e-4, weight decay 0.05, cosine schedule after a 5 % warm-up, 100
epochs, batch 256, bf16 on a GPU. The checkpoint is the epoch with the best mean
inner-validation selection score over the datasets (bag macro-F1 and 1 - RMSE / std); a
collapse monitor records the per-dimension standard deviation of the summaries.

**Output**: ``runs/pretrain/<run-id>/fold<k>-seed<s>/``: ``best.pt`` (the backbone and each
dataset's heads; ``python -m pdmfm.planb --init`` fine-tunes from it, refusing another fold's
backbone, whose pretraining saw this fold's test units), ``last.pt`` (resumes a job stopped
at its time limit), ``history.csv``, ``summary.json``; then ``summary.csv`` and
``summary.md`` over folds and seeds.

    python -m pdmfm.pretrain --encoder mantis --arms F E --run-id pre-v1 --device cuda
    python -m pdmfm.pretrain --encoder mantis --arms F E --run-id pre-cpu --folds 1 --seeds 0 \\
        --epochs 2 --batch-size 32 --steps-per-epoch 20       # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pdmfm import protocols
from pdmfm.model.fusion import FUSIONS, STAGES, FusionModel, build_model
from pdmfm.model.inputs import CHANNEL_EMBEDDING, TaskInputs, assemble
from pdmfm.model.late import LateEvidential
from pdmfm.model.layers import ModalityEncoder, PrototypeEvidentialHead
from pdmfm.paths import REPO_ROOT
from pdmfm.train import TrainConfig, _losses, batch, predict, selection_score
from pdmfm.utils import stable_hash, write_json_atomic
from pdmfm.windows import WindowStore

SOURCES = ("cwru", "xjtu_sy", "nonastreda", "matwi", "paderborn")
FAMILIES = ("vibration", "force", "current", "acoustic", "ae", "slow", "photo")
# Two views of a window per dataset: family -> channel names (None: all of its channels).
VIEWS: dict[str, tuple[dict, dict]] = {
    "cwru": ({"vibration": ("de",)}, {"vibration": ("fe",)}),
    "xjtu_sy": ({"vibration": ("vib_h",)}, {"vibration": ("vib_v",)}),
    "paderborn": ({"vibration": None}, {"current": None}),
    "nonastreda": ({"force": None}, {"photo": None}),
    "matwi": ({"force": None}, {"photo": None}),
}
VERSION = 1


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: float = 0.05
    d: int = 256
    dropout: float = 0.1
    head: str = "prototype"
    fusion: str = "late"  # Plan B's experts, or the Gate-2 winner's fusion stage
    mask_ratio: float = 0.4
    drop_modality: float = 0.3
    w_mask: float = 1.0
    w_nce: float = 0.5
    w_sup: float = 1.0
    temperature: float = 0.07
    steps_per_epoch: int = 0  # 0: the training rows of all datasets / batch size
    kl_epochs: int = 10
    device: str = "cpu"

    def as_dict(self) -> dict:
        return asdict(self)

    def train_config(self) -> TrainConfig:
        """The supervised configuration Plan B's model and losses read."""
        return TrainConfig(
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, d=self.d,
            dropout=self.dropout, kl_epochs=self.kl_epochs, head=self.head,
            fusion=self.fusion, device=self.device,
        )  # fmt: skip


@dataclass
class Source:
    """One source dataset of fold k: its inputs, rows, units and model."""

    name: str
    inputs: TaskInputs
    train_rows: np.ndarray
    val_rows: np.ndarray
    channel_names: list[str]  # by channel index: high-rate, slow, then "image"
    y_mean: float = 0.0
    y_std: float = 1.0
    model: nn.Module | None = None

    def __post_init__(self) -> None:
        units = self.inputs.manifest["unit"].astype(str).to_numpy()
        self.units = units
        groups = pd.Series(self.train_rows).groupby(units[self.train_rows]).indices
        self.unit_rows = [self.train_rows[i] for i in groups.values()]


# ----------------------------------------------------------------------------- data


def validation_mask(dataset: str, manifest: pd.DataFrame, fold: int) -> np.ndarray:
    if dataset == "cwru":
        return protocols.cwru_gate1(manifest).val
    return protocols.SOURCE_FOLDS[dataset](manifest, fold).val


def test_mask(dataset: str, manifest: pd.DataFrame, fold: int) -> np.ndarray:
    if dataset == "cwru":  # CWRU has no test units; its 3 hp windows are validation
        return np.zeros(len(manifest), dtype=bool)
    return protocols.SOURCE_FOLDS[dataset](manifest, fold).test


def load_sources(
    datasets: tuple[str, ...],
    encoder: str,
    arm_list: tuple[str, ...],
    fold: int,
    photos: bool = True,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
) -> list[Source]:
    """The source datasets that have a window store and the encoder's features; a missing one
    is skipped with a message (Paderborn before its download)."""
    sources = []
    for dataset in datasets:
        if dataset in protocols.TARGETS:
            raise ValueError(f"{dataset} is a held-out target; pretraining never reads it")
        try:
            manifest = WindowStore.open(dataset, window_root).manifest
            allowed = protocols.pretraining_mask(dataset, manifest, fold)
            train_units = sorted(set(manifest.loc[allowed, "unit"].astype(str)))
            inputs = assemble(
                dataset, encoder.replace("{fold}", str(fold)), arm_list, photos=photos,
                window_root=window_root, feature_root=feature_root, train_units=train_units,
            )  # fmt: skip
        except (FileNotFoundError, ValueError) as error:
            log(f"[pretrain] {dataset}: skipped ({error})")
            continue
        present = np.any([m.present for m in inputs.modalities.values()], axis=0)
        labelled = np.zeros(len(manifest), dtype=bool)
        if inputs.y_class is not None:
            labelled |= inputs.y_class >= 0
        if inputs.y_reg is not None:
            labelled |= np.isfinite(inputs.y_reg)
        train = allowed & present
        val = validation_mask(dataset, manifest, fold) & present & labelled
        assert not (train & test_mask(dataset, manifest, fold)).any()
        assert not (train & val).any()
        store = WindowStore.open(dataset, window_root)
        names = [*store.channels, *store.slow_channels, "image"]
        source = Source(dataset, inputs, np.flatnonzero(train), np.flatnonzero(val), names)
        if inputs.y_reg is not None:
            y = inputs.y_reg[source.train_rows]
            y = y[np.isfinite(y)]
            if len(y):
                source.y_mean, source.y_std = float(y.mean()), float(y.std()) or 1.0
        log(
            f"[pretrain] {dataset}: {len(source.train_rows)} training windows of "
            f"{len(source.unit_rows)} units, {len(source.val_rows)} validation; "
            f"modalities {list(inputs.modalities)}"
        )
        sources.append(source)
    if not sources:
        raise FileNotFoundError("no source dataset could be loaded")
    return sources


def dataset_weights(sources: list[Source]) -> np.ndarray:
    """Sampling probability of each dataset, proportional to the square root of its rows."""
    weights = np.sqrt([len(s.train_rows) for s in sources]).astype(float)
    return weights / weights.sum()


def sample_rows(source: Source, size: int, rng: np.random.Generator) -> np.ndarray:
    """A batch with units balanced: a unit uniformly, then one of its rows."""
    units = rng.integers(0, len(source.unit_rows), size)
    return np.array([source.unit_rows[u][rng.integers(0, len(source.unit_rows[u]))]
                     for u in units])  # fmt: skip


# ----------------------------------------------------------------------------- the model


class MaskDecoder(nn.Module):
    """Queries (the hidden tokens' identities) attend to the context (the kept modalities'
    summaries, and the fused representation); one output map per family and target kind."""

    def __init__(self, d: int, token_dims: dict[str, int], level_dims: dict[str, int]) -> None:
        super().__init__()
        self.attend = nn.MultiheadAttention(d, 4, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.tokens = nn.ModuleDict({f: nn.Linear(d, dim) for f, dim in token_dims.items()})
        self.levels = nn.ModuleDict({f: nn.Linear(d, dim) for f, dim in level_dims.items()})

    def forward(self, queries, context, context_present) -> torch.Tensor:
        padding = ~context_present
        padding = padding & ~padding.all(1, keepdim=True)  # a row with no context: masked out
        h, _ = self.attend(queries, context, context, key_padding_mask=padding)
        h = self.norm(queries + h)
        return h + self.mlp(h)


class Pretrainer(nn.Module):
    """The shared backbone and every source dataset's model built around it."""

    def __init__(self, sources: list[Source], cfg: PretrainConfig) -> None:
        super().__init__()
        if cfg.fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        train_cfg = cfg.train_config()
        present = {name for s in sources for name in s.inputs.modalities}
        self.families = [f for f in FAMILIES if f in present]
        self.experts = nn.ModuleDict()
        self.projections = nn.ModuleDict()  # each family's text projection W_q
        self.scales = nn.ParameterDict()
        self.models = nn.ModuleDict()
        self.stage = (
            None if cfg.fusion == "late" else STAGES[cfg.fusion](self.families, cfg.d, cfg.dropout)
        )
        token_dims, level_dims = {}, {}
        self.expert_shapes: dict[str, tuple] = {}
        for source in sources:
            model = build_model(
                source.inputs, cfg.fusion, cfg.d, cfg.dropout, source.train_rows, train_cfg.head
            )
            for name in model.names:
                modality = source.inputs.modalities[name]
                shape = (modality.dim, modality.levels is not None, modality.descriptor,
                         0 if modality.levels is None else modality.levels.shape[-1])  # fmt: skip
                if name in self.experts:
                    if self.expert_shapes[name] != shape:
                        raise ValueError(
                            f"{source.name}: {name} tokens differ from other datasets'"
                        )
                    model.encoders[name] = self.experts[name]
                else:
                    self.experts[name] = model.encoders[name]
                    self.expert_shapes[name] = shape
                token_dims[name] = modality.dim
                if modality.levels is not None:
                    level_dims[name] = int(modality.levels.shape[-1])
                head = model.class_heads[name] if name in model.class_heads else None
                if isinstance(head, PrototypeEvidentialHead):
                    if name in self.projections:
                        projection = self.projections[name]
                        assert isinstance(projection, nn.Linear)
                        head.project = projection
                        head.log_scale = self.scales[name]
                    else:
                        self.projections[name] = head.project
                        self.scales[name] = head.log_scale
            if isinstance(model, FusionModel):
                assert self.stage is not None
                model.use_stage(self.stage, self.families)
            source.model = model
            self.models[source.name] = model
        self.decoder = MaskDecoder(cfg.d, token_dims, level_dims)
        self.projector = nn.Sequential(nn.Linear(cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, 128))
        self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.temperature)))

    def backbone(self) -> dict:
        """What every dataset shares, plus each dataset's full model (for ``--init``)."""
        return {
            "families": self.families,
            "experts": {f: e.state_dict() for f, e in self.experts.items()},
            "projections": {f: p.state_dict() for f, p in self.projections.items()},
            "scales": {f: s.detach().cpu() for f, s in self.scales.items()},
            "stage": None if self.stage is None else self.stage.state_dict(),
            "models": {name: m.state_dict() for name, m in self.models.items()},
        }


# ----------------------------------------------------------------------------- losses


def _copy(parts: dict) -> dict:
    return {name: dict(part) for name, part in parts.items()}


def mask_parts(parts: dict, ratio: float, drop: float, rng: torch.Generator) -> dict:
    """Drop modalities (p = ``drop``, never all of a row's) and hide ``ratio`` of the kept
    tokens. Returns name -> (hidden tokens (B, T), hidden levels (B, L) or None, kept (B,)),
    and sets ``present``, ``hide`` and ``level_hide`` in ``parts``."""
    names = list(parts)
    present = torch.stack([parts[n]["present"] for n in names], 1)
    device = present.device
    dropped = (torch.rand(present.shape, generator=rng).to(device) < drop) & present
    lost = present.any(1) & ~(present & ~dropped).any(1)
    if lost.any():  # keep one present modality of such rows
        keep = torch.multinomial(present[lost].float().cpu(), 1, generator=rng).squeeze(1)
        dropped[lost] = dropped[lost] & ~nn.functional.one_hot(keep, len(names)).bool().to(device)
    hidden = {}
    for m, name in enumerate(names):
        part = parts[name]
        kept = present[:, m] & ~dropped[:, m]
        b, t = part["tokens"].shape[:2]
        tp = part.get("token_present", torch.ones(b, t, dtype=torch.bool, device=device))
        hide = (torch.rand(b, t, generator=rng).to(device) < ratio) & tp & kept[:, None]
        hidden_tokens = hide | (tp & dropped[:, m : m + 1])
        part["present"] = kept
        part["hide"] = hide
        hidden_levels = None
        if "levels" in part:
            nl = part["levels"].shape[1]
            level_hide = (torch.rand(b, nl, generator=rng).to(device) < ratio) & kept[:, None]
            part["level_hide"] = level_hide
            hidden_levels = level_hide | dropped[:, m : m + 1].expand(b, nl)
        hidden[name] = (hidden_tokens, hidden_levels, kept)
    return hidden


def mask_loss(pre: Pretrainer, model, parts: dict, out: dict, hidden: dict) -> torch.Tensor:
    """1 - cosine to the hidden tokens' layer-normed features; MSE on hidden level and slow
    descriptors; the mean of the kinds present."""
    context = out["z"]
    context_present = torch.stack([hidden[n][2] for n in model.names], 1)
    if "fused_z" in out:
        context = torch.cat([context, out["fused_z"].unsqueeze(1)], 1)
        context_present = torch.cat([context_present, context_present.any(1, keepdim=True)], 1)
    terms: dict[str, list[torch.Tensor]] = {"tokens": [], "levels": [], "slow": []}
    for name in model.names:
        part, (hidden_tokens, hidden_levels, _) = parts[name], hidden[name]
        encoder = model.encoders[name]
        assert isinstance(encoder, ModalityEncoder)
        codes, scale = model.channel_codes, model.channel_scale
        if hidden_tokens.any():
            identity = encoder.identity(
                getattr(model, f"roles_{name}"), getattr(model, f"channels_{name}"), codes,
                scale, getattr(model, f"positions_{name}"),
            )  # fmt: skip
            queries = identity.unsqueeze(0).expand(len(hidden_tokens), -1, -1)
            predicted = pre.decoder.tokens[name](pre.decoder(queries, context, context_present))
            tokens = part["tokens"].float()
            if source_descriptor(model, name):
                target = model._standardized(f"tokens_{name}", tokens)
                error = (predicted - target)[hidden_tokens] ** 2
                terms["slow"].append(error.mean())
            else:
                target = nn.functional.layer_norm(tokens, tokens.shape[-1:])
                cosine = nn.functional.cosine_similarity(predicted, target, dim=-1)
                terms["tokens"].append((1 - cosine)[hidden_tokens].mean())
        if hidden_levels is not None and hidden_levels.any():
            identity = encoder.level_identity(
                getattr(model, f"level_channels_{name}"), codes, scale
            )
            queries = identity.unsqueeze(0).expand(len(hidden_levels), -1, -1)
            predicted = pre.decoder.levels[name](pre.decoder(queries, context, context_present))
            target = model._standardized(f"levels_{name}", part["levels"].float())
            keep = hidden_levels.unsqueeze(-1) & torch.isfinite(target)
            if keep.any():
                terms["levels"].append(((predicted - torch.nan_to_num(target)) ** 2)[keep].mean())
    means = [torch.stack(v).mean() for v in terms.values() if v]
    return torch.stack(means).mean() if means else context.new_zeros(())


def source_descriptor(model, name: str) -> bool:
    return hasattr(model, f"tokens_{name}_mean")  # descriptor tokens are standardized


def view_parts(parts: dict, spec: dict, source: Source, model) -> dict:
    """The batch restricted to one view: modalities outside ``spec`` absent; within a listed
    family, only the named channels' tokens and level tokens."""
    out = _copy(parts)
    for name, part in out.items():
        if name not in spec:
            part["present"] = torch.zeros_like(part["present"])
            continue
        channels = spec[name]
        if channels is None:
            continue
        allowed = torch.tensor([source.channel_names.index(c) for c in channels],
                               device=part["present"].device)  # fmt: skip
        token_channels = getattr(model, f"channels_{name}")
        b, t = part["tokens"].shape[:2]
        tp = part.get("token_present", torch.ones(b, t, dtype=torch.bool, device=allowed.device))
        part["token_present"] = tp & torch.isin(token_channels, allowed)[None, :]
        if "levels" in part:
            level_channels = getattr(model, f"level_channels_{name}")
            keep = torch.isin(level_channels, allowed)[None, :]
            part["level_present"] = keep.expand(b, -1)
        part["present"] = part["present"] & part["token_present"].any(1)
    return out


def pooled(out: dict) -> torch.Tensor:
    """(B, d) a view's representation: the fused one, or the mean of present summaries."""
    if "fused_z" in out:
        return out["fused_z"]
    weight = out["present"].float().unsqueeze(-1)
    return (out["z"] * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def info_nce(pre: Pretrainer, a: torch.Tensor, b: torch.Tensor, units: np.ndarray) -> torch.Tensor:
    """Symmetric InfoNCE; another window of the same unit is never a negative."""
    za = nn.functional.normalize(pre.projector(a), dim=-1)
    zb = nn.functional.normalize(pre.projector(b), dim=-1)
    tau = pre.log_tau.exp().clamp_min(0.01)
    logits = za @ zb.T / tau
    same = torch.from_numpy(units[:, None] == units[None, :]).to(logits.device)
    eye = torch.eye(len(units), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(units), device=logits.device)
    return (nn.functional.cross_entropy(logits, target)
            + nn.functional.cross_entropy(logits.T, target)) / 2  # fmt: skip


def step_losses(
    pre: Pretrainer,
    source: Source,
    rows: np.ndarray,
    cfg: PretrainConfig,
    annealing: float,
    generator: torch.Generator,
    train_cfg: TrainConfig,
) -> dict[str, torch.Tensor]:
    """The three losses on one batch of one dataset."""
    model = source.model
    assert model is not None
    device = torch.device(cfg.device)
    clean = batch(source.inputs, rows, device)
    losses: dict[str, torch.Tensor] = {}
    if cfg.w_sup:
        out = model(clean)
        losses["sup"] = _losses(out, source.inputs, rows, annealing, train_cfg, source.y_mean,
                                source.y_std, device)  # fmt: skip
    if cfg.w_mask:
        parts = _copy(clean)
        hidden = mask_parts(parts, cfg.mask_ratio, cfg.drop_modality, generator)
        out = model(parts)
        losses["mask"] = mask_loss(pre, model, parts, out, hidden)
    if cfg.w_nce and source.name in VIEWS:
        spec_a, spec_b = VIEWS[source.name]
        out_a = model(view_parts(clean, spec_a, source, model))
        out_b = model(view_parts(clean, spec_b, source, model))
        valid = (out_a["present"].any(1) & out_b["present"].any(1)).cpu().numpy()
        if valid.sum() >= 2:
            keep = torch.from_numpy(valid).to(device)
            losses["nce"] = info_nce(pre, pooled(out_a)[keep], pooled(out_b)[keep],
                                     source.units[rows][valid])  # fmt: skip
    return losses


# ----------------------------------------------------------------------------- training


@torch.no_grad()
def evaluate(pre: Pretrainer, sources: list[Source], train_cfg: TrainConfig) -> dict:
    """Mean inner-validation selection score over the datasets, and the collapse monitor
    (per-dimension std of the summaries over validation windows)."""
    pre.eval()
    scores, stds = {}, []
    device = torch.device(train_cfg.device)
    for source in sources:
        model = source.model
        if not len(source.val_rows) or model is None:
            continue
        pred = predict(model, source.inputs, source.val_rows, train_cfg)  # pyright: ignore[reportArgumentType]
        score = selection_score(source.inputs, source.val_rows, pred, source.y_mean, source.y_std)
        if np.isfinite(score):
            scores[source.name] = float(score)
        rows = source.val_rows[:512]
        out = model(batch(source.inputs, rows, device))
        here = out["present"]
        for m in range(here.shape[1]):
            z = out["z"][here[:, m], m]
            if len(z) > 1:
                stds.append(z.float().std(0).mean().item())
    pre.train()
    return {
        "score": float(np.mean(list(scores.values()))) if scores else float("nan"),
        "scores": scores,
        "z_std": float(np.mean(stds)) if stds else float("nan"),
    }


def _schedule(cfg: PretrainConfig, steps: int):
    total = cfg.epochs * steps
    warmup = max(1, int(cfg.warmup * total))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return factor


def pretrain_one(
    sources: list[Source],
    cfg: PretrainConfig,
    fold: int,
    seed: int,
    out: Path,
    meta: dict,
    log=print,
) -> dict:
    """Pretrain on ``sources`` for fold ``fold``; resumes from ``last.pt``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    pre = Pretrainer(sources, cfg).to(device)
    train_cfg = cfg.train_config()
    optimizer = torch.optim.AdamW(pre.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_rows = sum(len(s.train_rows) for s in sources)
    steps = cfg.steps_per_epoch or max(1, total_rows // cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(cfg, steps))
    weights = dataset_weights(sources)
    out.mkdir(parents=True, exist_ok=True)
    settings_key = stable_hash(meta)[:12]
    start, best, history = 0, -np.inf, []
    last_path = out / "last.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["settings_key"] != settings_key:
            raise ValueError(f"{out} was started with other settings or data; use another run id")
        pre.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["generator"])
        start, best, history = state["epoch"], state["best"], state["history"]
        log(f"[pretrain] fold {fold} seed {seed}: resuming after epoch {start}")
    amp = device.type == "cuda"
    started = time.perf_counter()
    for epoch in range(start, cfg.epochs):
        pre.train()
        annealing = min(1.0, epoch / cfg.kl_epochs) if cfg.kl_epochs else 1.0
        sums: dict[str, list[float]] = {"total": [], "sup": [], "mask": [], "nce": []}
        tick = time.perf_counter()
        for _ in range(steps):
            source = sources[int(rng.choice(len(sources), p=weights))]
            rows = sample_rows(source, cfg.batch_size, rng)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                losses = step_losses(pre, source, rows, cfg, annealing, generator, train_cfg)
            weight = {"sup": cfg.w_sup, "mask": cfg.w_mask, "nce": cfg.w_nce}
            total = sum(weight[k] * v.float() for k, v in losses.items())
            if not isinstance(total, torch.Tensor):
                continue
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(pre.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sums["total"].append(float(total.detach()))
            for k, v in losses.items():
                sums[k].append(float(v.detach()))
        evaluation = evaluate(pre, sources, train_cfg)
        record = {
            "epoch": epoch + 1,
            **{f"loss_{k}": float(np.mean(v)) if v else float("nan") for k, v in sums.items()},
            "val_score": evaluation["score"],
            "z_std": evaluation["z_std"],
            "tau": float(pre.log_tau.detach().exp()),
            "seconds": round(time.perf_counter() - tick, 1),
            **{f"val_{k}": v for k, v in evaluation["scores"].items()},
        }
        history.append(record)
        if evaluation["score"] > best:
            best = evaluation["score"]
            torch.save({"meta": {**meta, "epoch": epoch + 1, "val_score": best},
                        "backbone": pre.backbone()}, out / "best.tmp")  # fmt: skip
            (out / "best.tmp").replace(out / "best.pt")
        torch.save(
            {
                "settings_key": settings_key, "model": pre.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng, "torch_rng": torch.get_rng_state(), "generator": generator.get_state(),
                "epoch": epoch + 1, "best": best, "history": history,
            },
            out / "last.tmp",
        )  # fmt: skip
        (out / "last.tmp").replace(last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        write_json_atomic(out / "progress.json", {"fold": fold, "seed": seed, "epoch": epoch + 1,
                                                  "epochs": cfg.epochs, "best": best})  # fmt: skip
        log(
            f"[pretrain] fold {fold} seed {seed} epoch {epoch + 1}/{cfg.epochs}: "
            f"loss {record['loss_total']:.4f} (sup {record['loss_sup']:.3f}, mask "
            f"{record['loss_mask']:.3f}, nce {record['loss_nce']:.3f}), validation "
            f"{record['val_score']:.3f}, z std {record['z_std']:.3f}, {record['seconds']:.0f} s"
        )
    summary = {
        **meta,
        "best_val_score": best,
        "best_epoch": int(np.nanargmax([h["val_score"] for h in history]) + 1) if history else 0,
        "final": history[-1] if history else {},
        "seconds": round(time.perf_counter() - started, 1),
        "steps_per_epoch": steps,
        "dataset_weights": dict(zip([s.name for s in sources], weights.tolist(), strict=True)),
    }
    write_json_atomic(out / "summary.json", summary)
    return summary


def run(
    encoder: str,
    arm_list: tuple[str, ...],
    run_id: str,
    folds: list[int],
    seeds: list[int],
    cfg: PretrainConfig,
    datasets: tuple[str, ...] = SOURCES,
    photos: bool = True,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
) -> Path:
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "pretrain") / run_id
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for fold in folds:
        sources = load_sources(datasets, encoder, arm_list, fold, photos, window_root,
                               feature_root, log)  # fmt: skip
        for seed in seeds:
            meta = {
                "version": VERSION, "fold": fold, "seed": seed, "encoder": encoder,
                "arms": list(arm_list), "photos": photos, "config": cfg.as_dict(),
                "channel_embedding": CHANNEL_EMBEDDING,
                "datasets": [s.name for s in sources],
                "windows": {s.name: int(len(s.train_rows)) for s in sources},
            }  # fmt: skip
            folder = out / f"fold{fold}-seed{seed}"
            summary_path = folder / "summary.json"
            if summary_path.is_file():
                done = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    done.get("config") == meta["config"]
                    and done.get("datasets") == meta["datasets"]
                ):
                    log(f"[pretrain] fold {fold} seed {seed}: done earlier, skipped")
                    records.append(done)
                    continue
                raise ValueError(f"{folder} holds a run with other settings; use another run id")
            records.append(pretrain_one(sources, cfg, fold, seed, folder, meta, log))
    summarize(out, records)
    return out


def summarize(out: Path, records: list[dict]) -> None:
    rows = [
        {"fold": r["fold"], "seed": r["seed"], "best_val_score": r["best_val_score"],
         "best_epoch": r["best_epoch"], **{k: v for k, v in r.get("final", {}).items()
                                           if k.startswith(("loss_", "z_std", "tau"))}}
        for r in records
    ]  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    lines = [
        f"# Pretraining: {out.name}",
        "",
        "| Fold | Seed | Best validation (epoch) | Final loss: total / sup / mask / nce "
        "| z std | τ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['seed']} | {r['best_val_score']:.3f} ({r['best_epoch']}) | "
            f"{r.get('loss_total', np.nan):.3f} / {r.get('loss_sup', np.nan):.3f} / "
            f"{r.get('loss_mask', np.nan):.3f} / {r.get('loss_nce', np.nan):.3f} | "
            f"{r.get('z_std', np.nan):.3f} | {r.get('tau', np.nan):.3f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- fine-tuning


def initialize(model: nn.Module, path: str | Path, dataset: str, fold: int | None,
               encoder: str | None = None, log=print) -> dict:  # fmt: skip
    """Load a pretrained backbone into a Plan B (or Gate-2) model before fine-tuning.

    A source dataset's model takes its pretrained parameters (shared and its own heads; the
    standardization buffers of the fine-tuning rows stay); any other dataset takes the shared
    experts, text projections and stage. A source dataset must use its own fold's backbone:
    another fold's pretraining saw this fold's test units."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    meta, backbone = checkpoint["meta"], checkpoint["backbone"]
    if dataset in meta["datasets"] and fold is not None and meta["fold"] != fold:
        raise ValueError(
            f"{path} was pretrained for fold {meta['fold']}; fold {fold} must use its own "
            "backbone (the other folds' pretraining saw its test units)"
        )
    if encoder is not None and meta["encoder"].replace("{fold}", str(meta["fold"])) != encoder:
        raise ValueError(f"{path} was pretrained on {meta['encoder']} features, not {encoder}")
    if meta.get("channel_embedding") != CHANNEL_EMBEDDING:
        raise ValueError(f"{path} uses channel embedding {meta.get('channel_embedding')}")
    cfg = meta["config"]
    if isinstance(model, FusionModel):
        if cfg["fusion"] != model.fusion_name:
            raise ValueError(f"{path} pretrained a {cfg['fusion']} stage, not {model.fusion_name}")
        model.use_stage(STAGES[cfg["fusion"]](backbone["families"], cfg["d"], cfg["dropout"]),
                        backbone["families"])  # fmt: skip
    elif cfg["fusion"] != "late":
        raise ValueError(f"{path} pretrained a {cfg['fusion']} stage; this model has none")
    own = dict(model.named_parameters())
    loaded, mismatched = 0, []
    if dataset in backbone["models"]:
        for name, value in backbone["models"][dataset].items():
            if name in own:
                if own[name].shape == value.shape:
                    own[name].data.copy_(value)
                    loaded += 1
                else:
                    mismatched.append(name)
    else:
        assert isinstance(model, LateEvidential)
        for family, state in backbone["experts"].items():
            if family in model.encoders:
                model.encoders[family].load_state_dict(state)
                loaded += len(state)
        for family, state in backbone["projections"].items():
            head = model.class_heads[family] if family in model.class_heads else None
            if isinstance(head, PrototypeEvidentialHead):
                head.project.load_state_dict(state)
                head.log_scale.data.copy_(backbone["scales"][family])
                loaded += 2
        if isinstance(model, FusionModel) and backbone["stage"] is not None:
            model.stage.load_state_dict(backbone["stage"])
            loaded += len(backbone["stage"])
    if mismatched:
        raise ValueError(f"{path}: shapes differ for {mismatched[:5]}")
    log(f"[pretrain] {dataset}: {loaded} pretrained tensors loaded from {path}")
    return {"loaded": loaded, "fold": meta["fold"], "epoch": meta.get("epoch")}


# ----------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.pretrain", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument(
        "--encoder", required=True, help='cached encoder, e.g. mantis or "patchtst-fold{fold}"'
    )
    parser.add_argument("--arms", nargs="+", default=["F", "E"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=SOURCES)
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(range(1, protocols.N_FOLDS + 1))
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    defaults = PretrainConfig()
    for name in ("epochs", "batch_size", "steps_per_epoch", "d"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    for name in ("lr", "weight_decay", "mask_ratio", "drop_modality", "w_mask", "w_nce", "w_sup"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    parser.add_argument(
        "--fusion", default="late", choices=FUSIONS, help="late (Plan B) or the Gate-2 winner"
    )
    parser.add_argument("--no-photos", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="default: runs/pretrain/")
    parser.add_argument("--force", action="store_true", help="run on a busy machine (CPU only)")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        # The cached tokens of the five sources take well under 1 GB (as Plan B).
        claim_resources("pretrain", min_idle_cpus=2.0, min_free_ram_gb=1.0, force=args.force)
    cfg = replace(
        PretrainConfig(), epochs=args.epochs, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch, d=args.d, lr=args.lr,
        weight_decay=args.weight_decay, mask_ratio=args.mask_ratio,
        drop_modality=args.drop_modality, w_mask=args.w_mask, w_nce=args.w_nce,
        w_sup=args.w_sup, fusion=args.fusion, device=args.device,
    )  # fmt: skip
    out = run(
        args.encoder, tuple(args.arms), args.run_id, args.folds, args.seeds, cfg,
        tuple(args.datasets), photos=not args.no_photos, out_root=args.out,
    )  # fmt: skip
    print(f"[pretrain] done -> {out}")


if __name__ == "__main__":
    main()
