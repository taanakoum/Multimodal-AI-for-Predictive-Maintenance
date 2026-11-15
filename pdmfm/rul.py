"""Causal RUL from cached recording histories, with calibrated NIG uncertainty.

XJTU-SY retains normalized post-FPT RUL. Nonastreda predicts operating seconds to confirmed
dulled wear, and MATWI predicts remaining recorded operating life, a proxy endpoint. Milling
lifecycle manifests retain missing-signal passes and attach only pre-pass photographs.
Training-only standardization and past-only reference subtraction precede a small GRU.
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
from pdmfm.compute import autocast
from pdmfm.model.inputs import TaskInputs, assemble
from pdmfm.model.layers import NIGHead
from pdmfm.paths import REPO_ROOT
from pdmfm.planb import BOOTSTRAP_REPLICATES, _schedule, score_regression, student_t, summarize
from pdmfm.train import bag_mean
from pdmfm.utils import stable_hash, write_json_atomic
from pdmfm.windows import WindowStore

DATASETS = {
    "xjtu_sy": protocols.xjtu_fold,
    "nonastreda": protocols.nonastreda_fold,
    "matwi": protocols.matwi_fold,
}
TIME_SCALE = 8.0  # log(1 + minutes) / 8: XJTU-SY lives reach 2,538 minutes (log ~ 7.8)
BASELINE_SNAPSHOTS = 10  # the healthy start, as for XJTU-SY's level tokens


@dataclass(frozen=True)
class RULConfig:
    precision: str = "auto"
    predict_batch_size: int = 1024
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
            parts.append(m.levels.reshape(n, -1) * m.present[:, None])
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
            "minute": (
                first["meta_minute"].to_numpy(dtype=float)
                if "meta_minute" in first
                else first["meta_t_start"].to_numpy(float) / 60
            ),
            "y": y[:, 0],
        }
    )
    order = np.lexsort((table["minute"].to_numpy(), table["unit"].to_numpy()))
    table = table.iloc[order].reset_index(drop=True)
    position = table.groupby("unit").cumcount().to_numpy()
    return Snapshots(table, mean[order].astype(np.float32), position)


def milling_snapshots(inputs: TaskInputs, lifecycle: pd.DataFrame, feature_root=None) -> Snapshots:
    """One causal feature vector per physical pass, including signal-missing passes.

    Cached signals are reused only when available. Photos are looked up on the complete
    lifecycle, so the current pass's inspection never enters its prediction.
    """
    from pdmfm.paths import interim_dir

    sensor = snapshots(inputs)
    lookup = {bag: i for i, bag in enumerate(sensor.table.bag)}
    table = lifecycle.copy().reset_index(drop=True)
    x = np.zeros((len(table), sensor.features.shape[1]), np.float32)
    for i, row in enumerate(table.to_dict("records")):
        if row["force_present"] and row["source_bag"] in lookup:
            x[i] = sensor.features[lookup[row["source_bag"]]]
    blocks = [x, table.force_present.to_numpy(np.float32)[:, None]]
    root = Path(feature_root) if feature_root else interim_dir("features")
    kinds = ("tool",) if inputs.dataset == "nonastreda" else ("insert",)
    # A photo bank is optional; absence is represented explicitly, independently of force.
    for kind in kinds:
        folders = sorted((root / "images").glob(f"{inputs.dataset}-{kind}-*/meta.json"))
        folders = [
            p.parent
            for p in folders
            if json.loads(p.read_text(encoding="utf-8")).get("complete")
            and not json.loads(p.read_text(encoding="utf-8"))["signature"].get("views")
        ]
        if not folders:
            continue
        folder = max(folders, key=lambda p: (p / "meta.json").stat().st_mtime_ns)
        paths = pd.read_csv(folder / "paths.csv")["path"].astype(str)
        bank = np.load(folder / "features.npy", mmap_mode="r")
        photos = {Path(p).stem: j for j, p in enumerate(paths)}
        values = np.zeros((len(table), bank.shape[-1]), np.float32)
        present = np.zeros((len(table), 1), np.float32)
        if inputs.dataset == "nonastreda":
            for _, group in table.groupby("unit", sort=False):
                previous = None
                for index in group.index.to_numpy(dtype=np.int64):
                    row = table.iloc[int(index)]
                    if previous in photos:
                        values[index] = bank[photos[previous]].mean(0)
                        present[index] = 1
                    previous = row["bag"]
        else:
            for index, row in enumerate(table.to_dict("records")):
                stem = Path(str(row.get("previous_photo", ""))).stem
                if stem in photos:
                    values[index] = bank[photos[stem]].mean(0)
                    present[index] = 1
        blocks.extend([values, present])
    table["minute"] = table.operation_end_s / 60
    table["y"] = table.rul_s
    table["target_kind"] = "rul_operating_seconds_proxy"
    position = table.groupby("unit", sort=False).cumcount().to_numpy()
    return Snapshots(table, np.concatenate(blocks, 1), position)


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


@torch.inference_mode()
def predict(fitted: FittedRUL, rows: np.ndarray) -> np.ndarray:
    """(len(rows), 4) NIG parameters, standardized target."""
    fitted.model.eval()
    device = torch.device(fitted.sequences.cfg.device)
    out = []
    for i in range(0, len(rows), fitted.sequences.cfg.predict_batch_size):
        with autocast(device, fitted.sequences.cfg.precision):
            pred = fitted.model(
                *fitted.sequences.batch(
                    rows[i : i + fitted.sequences.cfg.predict_batch_size], device
                )
            )
        out.append(pred.float().cpu().numpy())
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
            with autocast(device, cfg.precision):
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
    lifecycle: pd.DataFrame | None = None,
    clock: str = "gap",
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
    life_signature = None
    if dataset != "xjtu_sy":
        if lifecycle is None:
            from pdmfm.lifecycle import build

            lifecycle, _ = build(dataset, clock)
        assert lifecycle is not None
        life_signature = stable_hash(lifecycle.to_dict("records"))
        snaps = milling_snapshots(inputs, lifecycle, feature_root)
    settings = {
        "inputs": inputs.description,
        "train": cfg.as_dict(),
        "model": "snapshot-gru-v2",
        "lifecycle": life_signature,
        "target_kind": "normalized_rul" if dataset == "xjtu_sy" else "rul_operating_seconds_proxy",
    }
    key = stable_hash(settings)[:12]
    unit = snaps.table["unit"].to_numpy()
    y = snaps.table["y"].to_numpy()
    for fold in folds:
        split = DATASETS[dataset](manifest, fold)
        split.assert_unit_disjoint(manifest)
        parts = {p: np.flatnonzero(np.isin(unit, list(split.units(manifest, p))) & np.isfinite(y))
                 for p in
                 ("train", "val", "test")}  # fmt: skip
        if any(len(v) == 0 for v in parts.values()):
            raise ValueError(f"{dataset} fold {fold}: no uncensored targets in a required split")
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
            is_normalized = dataset == "xjtu_sy"
            test = {
                "combined": {
                    "bag": score_regression(y[parts["test"]], test_nig, ym, ys, rul=is_normalized)
                },
                "combined_calibrated": {
                    "bag": score_regression(
                        y[parts["test"]], test_nig, ym, ys, interval, is_normalized
                    )
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
                            "bag": score_regression(
                                y[parts["train"]], train_nig, ym, ys, rul=is_normalized
                            )
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
            if dataset != "xjtu_sy":
                assert lifecycle is not None
                train_units = set(snaps.table.iloc[parts["train"]].unit)
                lives = (
                    lifecycle.loc[lifecycle.unit.isin(train_units)]
                    .groupby("unit")
                    .endpoint_s.first()
                )
                mean_life = float(lives.dropna().mean())
                elapsed = snaps.table.iloc[parts["test"]].operation_end_s.to_numpy()
                life_pred = np.maximum(mean_life - elapsed, 0)
                bags["mean_life_reg"] = life_pred
                delta = life_pred - y[parts["test"]]
                result["baselines"]["mean_life_minus_elapsed"] = {
                    "mae": float(np.abs(delta).mean()),
                    "rmse": float(np.sqrt((delta**2).mean())),
                }
            bags["pred_reg"] = test_nig[:, 0] * ys + ym
            bags["target_kind"] = settings["target_kind"]
            if dataset != "xjtu_sy":
                near = y[parts["test"]] <= 1500
                if near.any():
                    test["last_1500_seconds"] = {
                        "bag": score_regression(
                            y[parts["test"]][near], test_nig[near], ym, ys, interval
                        )
                    }
            bags.to_csv(out / f"fold{fold}-seed{seed}-bags.csv", index=False)
            write_json_atomic(path, result)
    summarize(out, title="RUL from past snapshots")
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
        theirs = pd.read_csv(other).loc[:, ["bag", "pred_reg", "y_reg"]]
        theirs.columns = ["bag", "ref_reg", "ref_y"]
        seed = path.stem.split("-seed")[1].split("-")[0]
        paired = mine.merge(theirs, on="bag", how="inner", validate="one_to_one")
        if not len(paired):
            continue
        if not np.allclose(paired.y_reg, paired.ref_y, equal_nan=True):
            raise ValueError(
                "RUL comparison targets differ; do not compare wear or another endpoint"
            )
        by_seed.setdefault(seed, []).append(paired)
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
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--clock", choices=("gap", "recorded"), default="gap")
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
        precision=args.precision,
    )  # fmt: skip
    out = run(
        args.encoder, tuple(args.arms), args.run_id, args.folds, args.seeds, cfg, args.dataset,
        out_root=args.out, clock=args.clock,
    )  # fmt: skip
    if args.compare:
        print(compare(out, args.compare).to_string(index=False))
    print(f"[rul] -> {out}")


if __name__ == "__main__":
    main()
