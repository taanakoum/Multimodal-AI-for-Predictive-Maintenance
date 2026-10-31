"""RUL from a history of snapshots (plan: RUL on XJTU-SY; PROJECT_LOG week 5: one snapshot
alone lacks the temporal context a prognosis needs).

Each snapshot (an XJTU-SY recording, one per minute) is summarized by the mean over its
windows of the cached sensor features: one vector per (channel, arm) token group and the
baseline-relative level descriptors. Like the level tokens, the whole vector is taken
relative to the bearing's healthy start: minus the mean of its first 10 snapshots, of those
seen so far (causal). Absolute embeddings identify the bearing, and a sequence model then
memorizes the training bearings' paths (the first real run peaked on validation after one
epoch; ``PROJECT_LOG.md``, week 6). The features are standardized with the training
snapshots. A causal GRU reads the last ``history`` snapshots of the same bearing (fewer early
in life), each with log(1 + minutes since the previous snapshot) and, as an ablation,
log(1 + minutes since the bearing's first snapshot); a NIG head on its last state gives the
normalized RUL and its uncertainty. Only the current and earlier snapshots are read.
``history = 1`` is the single-snapshot control inside the same model.

Training and scoring follow Plan B (:mod:`pdmfm.planb`): AdamW, NIG loss (lambda 0.01) on
the standardized target, the epoch with the best inner-validation score (1 - RMSE / std),
one interval scale fitted on the validation bearings. Per snapshot: RMSE, MAE, Student-t
NLL, 90 % coverage and width, PHM 2012 score, also after the first prediction time; then
pooled out-of-fold estimates with unit-bootstrap intervals (``summary.md``, ``pooled.csv``)
and, with ``--compare``, the paired difference to a Plan B run on the same folds and seeds.

    python -m pdmfm.rul --encoder mantis --arms F E --run-id rul-xjtu --history 10 \\
        --compare runs/planb/planb-xjtu-mantis-FE
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from pdmfm import losses, protocols
from pdmfm import metrics as M
from pdmfm import stats as S
from pdmfm.model.inputs import TaskInputs, assemble
from pdmfm.model.layers import NIGHead
from pdmfm.paths import REPO_ROOT
from pdmfm.planb import BOOTSTRAP_REPLICATES, _schedule, score_regression, student_t, summarize
from pdmfm.train import bag_mean
from pdmfm.utils import stable_hash, write_json_atomic
from pdmfm.windows import WindowStore

DATASETS = {"xjtu_sy": protocols.xjtu_fold}
TIME_SCALE = 8.0  # log(1 + minutes) / 8: XJTU-SY lives reach 2,538 minutes (log ~ 7.8)
BASELINE_SNAPSHOTS = 10  # the healthy start, as for XJTU-SY's level tokens


@dataclass(frozen=True)
class RULConfig:
    history: int = 10  # snapshots read, the current one included
    relative: bool = True  # features relative to the bearing's healthy start (causal)
    elapsed: bool = False  # also read the minutes since the bearing's first snapshot
    epochs: int = 50  # as Plan B
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    d: int = 64
    dropout: float = 0.1
    nig_regularizer: float = 0.01
    device: str = "cpu"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Snapshots:
    """One row per snapshot, in time order within each unit."""

    table: pd.DataFrame  # bag, unit, minute, y (normalized RUL)
    features: np.ndarray  # (S, F) float32, not standardized
    position: np.ndarray  # (S,) index of the snapshot within its unit


def snapshots(inputs: TaskInputs) -> Snapshots:
    """Mean features of every snapshot's windows: one vector per (channel, role) token group
    of every modality, then the level descriptors."""
    manifest = inputs.manifest
    n = len(manifest)
    parts = []
    for m in inputs.modalities.values():
        tokens = m.tokens * m.present[:, None, None]
        groups = sorted(set(zip(m.channels.tolist(), m.roles.tolist(), strict=True)))
        for channel, role in groups:
            pick = (m.channels == channel) & (m.roles == role)
            parts.append(tokens[:, pick].mean(1))
        if m.levels is not None:
            parts.append(m.levels.reshape(n, -1))
    x = np.concatenate(parts, 1).astype(np.float32)
    bags = manifest["bag"].astype(str).to_numpy()
    keys, mean = bag_mean(x, bags)
    first = manifest.groupby("bag", sort=False).head(1).set_index("bag").loc[keys]
    if inputs.y_reg is None:
        raise ValueError(f"{inputs.dataset} has no regression target")
    _, y = bag_mean(inputs.y_reg[:, None], bags)
    table = pd.DataFrame(
        {
            "bag": keys,
            "unit": first["unit"].astype(str).to_numpy(),
            "minute": first["meta_minute"].to_numpy(dtype=float),
            "y": y[:, 0],
        }
    )
    order = np.lexsort((table["minute"].to_numpy(), table["unit"].to_numpy()))
    table = table.iloc[order].reset_index(drop=True)
    position = table.groupby("unit").cumcount().to_numpy()
    return Snapshots(table, mean[order].astype(np.float32), position)


def relative_to_start(snaps: Snapshots, count: int = BASELINE_SNAPSHOTS) -> np.ndarray:
    """Each snapshot's features minus the mean of its unit's first ``count`` snapshots among
    those up to it: causal, and 0 for a unit's first snapshot."""
    x = snaps.features.astype(np.float64)
    out = np.empty_like(x)
    units = snaps.table["unit"].to_numpy()
    for unit in pd.unique(units):
        rows = np.flatnonzero(units == unit)  # contiguous, in time order
        head = x[rows[:count]]
        running = np.cumsum(head, 0) / np.arange(1, len(head) + 1)[:, None]
        base = np.vstack([running, np.repeat(running[-1:], len(rows) - len(head), 0)])
        out[rows] = x[rows] - base
    return out.astype(np.float32)


def histories(snaps: Snapshots, rows: np.ndarray, history: int) -> tuple[np.ndarray, np.ndarray]:
    """(index (B, history) of each row's snapshots, oldest first, padded with -1 at the
    end; lengths (B,)). Snapshots are read only from the row's own unit, up to the row."""
    length = np.minimum(snaps.position[rows] + 1, history)
    steps = np.arange(history)
    index = rows[:, None] - (length[:, None] - 1) + steps[None, :]
    index = np.where(steps[None, :] < length[:, None], index, -1)
    return index, length


class SnapshotGRU(nn.Module):
    def __init__(self, n_features: int, cfg: RULConfig) -> None:
        super().__init__()
        n_time = 2 if cfg.elapsed else 1
        self.embed = nn.Sequential(
            nn.Linear(n_features + n_time, cfg.d), nn.LayerNorm(cfg.d), nn.GELU(),
            nn.Dropout(cfg.dropout),
        )  # fmt: skip
        self.gru = nn.GRU(cfg.d, cfg.d, batch_first=True)
        self.head = NIGHead(cfg.d)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.embed(x), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, last = self.gru(packed)
        return self.head(last[-1])


class Sequences:
    """Standardized snapshot features and time inputs, gathered into padded histories."""

    def __init__(self, snaps: Snapshots, train: np.ndarray, cfg: RULConfig) -> None:
        features = relative_to_start(snaps) if cfg.relative else snaps.features
        mean = features[train].mean(0)
        std = features[train].std(0)
        self.x = (features - mean) / np.where(std > 1e-6, std, 1.0)
        minutes = snaps.table["minute"].to_numpy()
        start = snaps.table.groupby("unit")["minute"].transform("min").to_numpy()
        self.elapsed = np.log1p(np.maximum(minutes - start, 0)) / TIME_SCALE
        self.minutes = minutes
        self.snaps, self.cfg = snaps, cfg

    def batch(self, rows: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
        index, length = histories(self.snaps, rows, self.cfg.history)
        safe = np.maximum(index, 0)
        valid = (index >= 0)[..., None]
        x = self.x[safe] * valid
        gap = np.zeros(index.shape)
        gap[:, 1:] = self.minutes[safe[:, 1:]] - self.minutes[safe[:, :-1]]
        time_inputs = [np.log1p(np.maximum(gap, 0)) / TIME_SCALE]
        if self.cfg.elapsed:
            time_inputs.append(self.elapsed[safe])
        t = np.stack(time_inputs, -1) * valid
        out = np.concatenate([x, t], -1).astype(np.float32)
        return torch.from_numpy(out).to(device), torch.from_numpy(length)


@dataclass
class FittedRUL:
    model: SnapshotGRU
    sequences: Sequences
    y_mean: float
    y_std: float
    best_epoch: int
    history: list
    seconds: float


@torch.no_grad()
def predict(fitted: FittedRUL, rows: np.ndarray) -> np.ndarray:
    """(len(rows), 4) NIG parameters, standardized target."""
    fitted.model.eval()
    device = torch.device(fitted.sequences.cfg.device)
    out = [
        fitted.model(*fitted.sequences.batch(rows[i : i + 1024], device)).cpu().numpy()
        for i in range(0, len(rows), 1024)
    ]
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)


def _score(target: np.ndarray, nig: np.ndarray) -> float:
    """1 - RMSE / std: the target is standardized with the training mean and std."""
    return 1.0 - float(np.sqrt(np.mean((nig[:, 0] - target) ** 2)))


def fit(
    snaps: Snapshots, train: np.ndarray, val: np.ndarray, cfg: RULConfig, seed: int, log=print
) -> FittedRUL:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(cfg.device)
    y = snaps.table["y"].to_numpy(dtype=np.float32)
    y_mean, y_std = float(y[train].mean()), float(y[train].std()) or 1.0
    target = (y - y_mean) / y_std
    sequences = Sequences(snaps, train, cfg)
    model = SnapshotGRU(snaps.features.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fitted = FittedRUL(model, sequences, y_mean, y_std, 0, [], 0.0)
    best, best_state = -np.inf, None
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(train)
        running = 0.0
        for start in range(0, len(order), cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            nig = model(*sequences.batch(rows, device))
            yt = torch.from_numpy(target[rows]).to(device)
            loss = losses.nig_loss(nig, yt, cfg.nig_regularizer).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(rows)
        score = _score(target[val], predict(fitted, val))
        fitted.history.append(
            {"epoch": epoch + 1, "loss": running / len(order), "val_score": score}
        )
        if score > best:
            best, fitted.best_epoch = score, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    fitted.seconds = time.perf_counter() - started
    log(f"[rul] {cfg.epochs} epochs in {fitted.seconds:.0f} s; best epoch {fitted.best_epoch}")
    return fitted


def _interval(fitted: FittedRUL, y: np.ndarray, nig: np.ndarray) -> float:
    mean, dof, scale = student_t(nig, fitted.y_mean, fitted.y_std)
    return M.interval_scale(y, mean, dof, scale)


def run(
    encoder: str,
    arm_list: tuple[str, ...],
    run_id: str,
    folds: list[int],
    seeds: list[int],
    cfg: RULConfig,
    dataset: str = "xjtu_sy",
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"RUL runs on {sorted(DATASETS)}")
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "rul") / run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = WindowStore.open(dataset, window_root).manifest
    inputs = assemble(
        dataset, encoder, arm_list, photos=False, window_root=window_root,
        feature_root=feature_root,
    )  # fmt: skip
    snaps = snapshots(inputs)
    settings = {"inputs": inputs.description, "train": cfg.as_dict(), "model": "snapshot-gru"}
    key = stable_hash(settings)[:12]
    unit = snaps.table["unit"].to_numpy()
    y = snaps.table["y"].to_numpy()
    for fold in folds:
        split = DATASETS[dataset](manifest, fold)
        split.assert_unit_disjoint(manifest)
        parts = {p: np.flatnonzero(np.isin(unit, list(split.units(manifest, p)))) for p in
                 ("train", "val", "test")}  # fmt: skip
        for seed in seeds:
            path = out / f"fold{fold}-seed{seed}.json"
            if path.is_file():
                if json.loads(path.read_text(encoding="utf-8")).get("settings_key") == key:
                    log(f"[rul] fold {fold} seed {seed}: done earlier, skipped")
                    continue
                raise ValueError(f"{path} holds a run with other settings; use another --run-id")
            fitted = fit(snaps, parts["train"], parts["val"], cfg, seed, log=log)
            val_nig = predict(fitted, parts["val"])
            interval = _interval(fitted, y[parts["val"]], val_nig)
            test_nig, train_nig = predict(fitted, parts["test"]), predict(fitted, parts["train"])
            ym, ys = fitted.y_mean, fitted.y_std
            test = {
                "combined": {"bag": score_regression(y[parts["test"]], test_nig, ym, ys, rul=True)},
                "combined_calibrated": {
                    "bag": score_regression(y[parts["test"]], test_nig, ym, ys, interval, True)
                },
            }
            train_mean = float(y[parts["train"]].mean())
            error = y[parts["test"]] - train_mean
            # The training bearings' mean RUL at the same snapshot number (elapsed time only).
            schedule = _schedule(
                snaps.position[parts["train"]], y[parts["train"]],
                snaps.position[parts["test"]], train_mean, np.mean,
            )  # fmt: skip
            schedule_error = y[parts["test"]] - schedule
            result = {
                "dataset": dataset,
                "fold": fold,
                "seed": seed,
                "settings_key": key,
                "settings": settings,
                "split": split.name,
                "best_epoch": fitted.best_epoch,
                "seconds": round(fitted.seconds, 1),
                "temperature": None,
                "interval_scale": interval,
                "n_params": sum(p.numel() for p in fitted.model.parameters()),
                "test": {"regression": test, "trust": {}},
                "train": {
                    "regression": {
                        "combined": {
                            "bag": score_regression(y[parts["train"]], train_nig, ym, ys, rul=True)
                        }
                    }
                },
                "baselines": {
                    "train_mean": {
                        "rmse": float(np.sqrt(np.mean(error**2))),
                        "mae": float(np.mean(np.abs(error))),
                    },
                    "schedule_regression": {
                        "rmse": float(np.sqrt(np.mean(schedule_error**2))),
                        "mae": float(np.mean(np.abs(schedule_error))),
                    },
                },
                "history": fitted.history,
            }
            bags = snaps.table.iloc[parts["test"]][["bag", "unit"]].copy()
            bags.insert(0, "fold", fold)
            bags["y_reg"] = y[parts["test"]]
            bags["train_mean"] = train_mean
            bags["schedule_reg"] = schedule
            bags["pred_reg"] = test_nig[:, 0] * ys + ym
            bags.to_csv(out / f"fold{fold}-seed{seed}-bags.csv", index=False)
            write_json_atomic(path, result)
    summarize(out)
    return out


def compare(out: Path, reference: Path, n: int | None = None) -> pd.DataFrame:
    """Paired RMSE difference (this run - ``reference``), overall and after FPT, on the bags
    both runs scored for the same folds and seeds; units resampled as in
    :func:`pdmfm.planb.pooled_summary`. Written to ``comparison.csv``."""
    by_seed: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(out.glob("fold*-seed*-bags.csv")):
        other = reference / path.name
        if not other.is_file():
            continue
        mine = pd.read_csv(path)
        theirs = pd.read_csv(other).loc[:, ["bag", "pred_reg"]]
        theirs.columns = ["bag", "ref_reg"]
        seed = path.stem.split("-seed")[1].split("-")[0]
        by_seed.setdefault(seed, []).append(mine.merge(theirs, on="bag", how="inner"))
    if not by_seed:
        raise ValueError(f"no fold and seed in common between {out} and {reference}")
    pooled = [pd.concat(parts, ignore_index=True) for parts in by_seed.values()]

    def rmse_gap(after_fpt: bool):
        def statistic(f):
            keep = f["y_reg"] < 1 - 1e-6 if after_fpt else slice(None)
            return S.rmse(f["y_reg"][keep], f["pred_reg"][keep]) - S.rmse(
                f["y_reg"][keep], f["ref_reg"][keep]
            )

        return statistic

    records = []
    for name, after in (("RMSE, this - reference", False), ("RMSE after FPT, this - reference",
                                                              True)):  # fmt: skip
        estimate, low, high = S.unit_bootstrap(
            pooled,
            rmse_gap(after),  # pyright: ignore[reportArgumentType]
            n=n or BOOTSTRAP_REPLICATES,
        )
        records.append({"quantity": name, "estimate": estimate, "low": low, "high": high})
    table = pd.DataFrame(records)
    table["reference"] = str(reference)
    table["bags"] = sum(len(t) for t in pooled)
    table.to_csv(out / "comparison.csv", index=False)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.rul", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--dataset", default="xjtu_sy", choices=sorted(DATASETS))
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--arms", nargs="+", default=["F", "E"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(range(1, protocols.N_FOLDS + 1))
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    defaults = RULConfig()
    parser.add_argument("--history", type=int, default=defaults.history)
    parser.add_argument(
        "--elapsed", action="store_true", help="also read the minutes since the start (ablation)"
    )
    parser.add_argument(
        "--absolute", action="store_true", help="features not relative to the healthy start"
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--d", type=int, default=defaults.d)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--compare", type=Path, help="a Plan B run on the same folds and seeds")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true", help="run on a busy machine (CPU only)")
    args = parser.parse_args(argv)
    if args.history < 1:
        parser.error("--history must be at least 1")
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        claim_resources("rul", min_idle_cpus=2.0, min_free_ram_gb=1.0, force=args.force)
    cfg = RULConfig(
        history=args.history, relative=not args.absolute, elapsed=args.elapsed,
        epochs=args.epochs, lr=args.lr,
        d=args.d, dropout=args.dropout, device=args.device,
    )  # fmt: skip
    out = run(
        args.encoder, tuple(args.arms), args.run_id, args.folds, args.seeds, cfg, args.dataset,
        out_root=args.out,
    )  # fmt: skip
    if args.compare:
        print(compare(out, args.compare).to_string(index=False))
    print(f"[rul] -> {out}")


if __name__ == "__main__":
    main()
