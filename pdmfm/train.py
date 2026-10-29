"""Supervised training of evidential models on cached features (plan, stage 4 onwards).

* Loss (plan, "Losses on labelled windows"): TMC on every present modality's opinion and on
  the combined opinion, KL weight min(1, epoch / 10); NIG loss (lambda = 0.01) on every
  modality's NIG and on the fused NIG, for rows with a regression target, standardized with
  the training mean and std.
* AdamW, learning rate 1e-3, 50 epochs. The checkpoint kept is the epoch with the best
  inner-validation score: the mean of bag-level macro-F1 and 1 - RMSE / std of the target.
* Bag aggregation (plan): a pass, cut or recording gets the mean evidence (classes) and the
  mean NIG parameters (regression) of its windows.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from pdmfm import losses
from pdmfm import metrics as M
from pdmfm.model.inputs import TaskInputs
from pdmfm.model.late import LateEvidential


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    d: int = 256
    dropout: float = 0.1
    kl_epochs: int = 10  # KL weight min(1, epoch / kl_epochs)
    nig_regularizer: float = 0.01
    device: str = "cpu"

    def as_dict(self) -> dict:
        return asdict(self)


def batch(
    inputs: TaskInputs, rows: np.ndarray, device: torch.device, drop: tuple[str, ...] = ()
) -> dict:
    """The model input of ``rows``; modalities in ``drop`` are marked absent."""
    out = {}
    for name, m in inputs.modalities.items():
        part = {
            "tokens": torch.from_numpy(np.ascontiguousarray(m.tokens[rows])).to(device),
            "present": torch.from_numpy(m.present[rows] & (name not in drop)).to(device),
        }
        if m.token_present is not None:
            part["token_present"] = torch.from_numpy(m.token_present[rows]).to(device)
        if m.levels is not None:
            part["levels"] = torch.from_numpy(np.ascontiguousarray(m.levels[rows])).to(device)
        if m.age is not None:
            part["age"] = torch.from_numpy(m.age[rows]).to(device)
        out[name] = part
    return out


@dataclass
class Fitted:
    model: LateEvidential
    y_mean: float
    y_std: float
    best_epoch: int
    history: list
    seconds: float


def _losses(
    out: dict,
    inputs: TaskInputs,
    rows: np.ndarray,
    annealing: float,
    cfg: TrainConfig,
    y_mean: float,
    y_std: float,
    device,
):
    total = torch.zeros((), device=device)
    present = out["present"]  # (B, M)
    if "evidence" in out and inputs.y_class is not None:
        y = torch.from_numpy(inputs.y_class[rows]).to(device)
        labelled = y >= 0
        if labelled.any():
            yl = y[labelled]
            for m in range(out["evidence"].shape[1]):
                here = present[labelled, m]
                if here.any():
                    per_row = losses.tmc_loss(
                        out["evidence"][labelled, m][here], yl[here], annealing
                    )
                    total = total + per_row.mean()
            total = total + losses.tmc_loss(out["combined"][labelled], yl, annealing).mean()
    if "nig" in out and inputs.y_reg is not None:
        y = torch.from_numpy(((inputs.y_reg[rows] - y_mean) / y_std).astype(np.float32)).to(device)
        labelled = torch.isfinite(y)
        if labelled.any():
            yl = y[labelled]
            for m in range(out["nig"].shape[1]):
                here = present[labelled, m]
                if here.any():
                    nig = out["nig"][labelled, m][here]
                    total = total + losses.nig_loss(nig, yl[here], cfg.nig_regularizer).mean()
            total = (
                total + losses.nig_loss(out["fused_nig"][labelled], yl, cfg.nig_regularizer).mean()
            )
    return total


@torch.no_grad()
def predict(
    model: LateEvidential,
    inputs: TaskInputs,
    rows: np.ndarray,
    cfg: TrainConfig,
    drop: tuple[str, ...] = (),
) -> dict:
    """Numpy outputs for ``rows``: evidence, combined, conflict, nig, fused_nig, present."""
    model.eval()
    device = torch.device(cfg.device)
    collected: dict[str, list] = {}
    for start in range(0, len(rows), 1024):
        out = model(batch(inputs, rows[start : start + 1024], device, drop))
        for key in ("evidence", "combined", "conflict", "nig", "fused_nig", "present"):
            if key in out:
                collected.setdefault(key, []).append(out[key].cpu().numpy())
    return {k: np.concatenate(v) for k, v in collected.items()}


def bag_mean(values: np.ndarray, bags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(bag keys in first-appearance order, mean of ``values`` per bag)."""
    keys, inverse = np.unique(bags, return_inverse=True)
    first = np.full(len(keys), len(bags))
    np.minimum.at(first, inverse, np.arange(len(bags)))  # each bag's first row, in O(n)
    order = np.argsort(first)
    sums = np.zeros((len(keys),) + values.shape[1:], dtype=float)
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=len(keys)).reshape((-1,) + (1,) * (values.ndim - 1))
    return keys[order], (sums / counts)[order]


def selection_score(
    inputs: TaskInputs, rows: np.ndarray, pred: dict, y_mean: float, y_std: float
) -> float:
    """Inner-validation score: mean of bag macro-F1 and 1 - RMSE / std (targets present)."""
    parts = []
    bags = inputs.manifest["bag"].astype(str).to_numpy()[rows]
    if "combined" in pred and inputs.y_class is not None:
        labelled = inputs.y_class[rows] >= 0
        if labelled.any():
            keys, evidence = bag_mean(pred["combined"][labelled], bags[labelled])
            _, y = bag_mean(inputs.y_class[rows][labelled].astype(float), bags[labelled])
            probs = losses.expected_probability(torch.from_numpy(evidence)).numpy()
            parts.append(
                M.classification(y.round().astype(int), probs, len(inputs.classes))["macro_f1"]
            )
    if "fused_nig" in pred and inputs.y_reg is not None:
        y = inputs.y_reg[rows]
        labelled = np.isfinite(y)
        if labelled.any():
            mean = pred["fused_nig"][labelled, 0] * y_std + y_mean
            rmse = float(np.sqrt(np.mean((mean - y[labelled]) ** 2)))
            parts.append(1.0 - rmse / y_std)
    return float(np.mean(parts)) if parts else float("nan")


def fit(
    inputs: TaskInputs,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    cfg: TrainConfig,
    seed: int,
    log=print,
) -> Fitted:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = LateEvidential(inputs, d=cfg.d, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_mean, y_std = 0.0, 1.0
    if inputs.y_reg is not None:
        y = inputs.y_reg[train_rows]
        y = y[np.isfinite(y)]
        if len(y):
            y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
    best, best_state, best_epoch, history = -np.inf, None, 0, []
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        annealing = min(1.0, epoch / cfg.kl_epochs) if cfg.kl_epochs else 1.0
        order = rng.permutation(train_rows)
        running = 0.0
        for start in range(0, len(order), cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            out = model(batch(inputs, rows, device))
            loss = _losses(out, inputs, rows, annealing, cfg, y_mean, y_std, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(rows)
        score = selection_score(
            inputs, val_rows, predict(model, inputs, val_rows, cfg), y_mean, y_std
        )
        history.append({"epoch": epoch + 1, "loss": running / len(order), "val_score": score})
        if score > best:
            best, best_epoch, best_state = score, epoch + 1, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    log(f"[train] {cfg.epochs} epochs in {seconds:.0f} s; best epoch {best_epoch} ({best:.3f})")
    return Fitted(model, y_mean, y_std, best_epoch, history, seconds)
