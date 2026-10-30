"""Hyperparameter search on the inner-validation units (plan: everything is chosen on
validation units; test units are scored once, with fixed settings).

For every configuration of a small grid and every outer fold, the model is fitted on the
fold's training units (one seed) and keeps its best inner-validation score, the training
loop's selection score: the mean of bag macro-F1 and 1 - RMSE / std. The configuration with
the best mean over folds wins; near-ties (a gap within one standard error of the paired
fold-wise differences) go to the cheaper configuration (smaller width, fewer options). Test
units are never read. The search resumes: finished (configuration, fold) pairs are kept in
``results.csv``.

    python -m pdmfm.tune --dataset nonastreda --encoder mantis --arms F --run-id tune-nona \\
        --grid lr=3e-4,1e-3 dropout=0.1,0.3 d=128,256
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm import protocols
from pdmfm.model.inputs import FOLD_BASELINE
from pdmfm.paths import REPO_ROOT
from pdmfm.planb import FOLDS, prepare_inputs
from pdmfm.train import TrainConfig, fit
from pdmfm.utils import write_json_atomic
from pdmfm.windows import WindowStore

TUNABLE = {
    f.name: f.type
    for f in fields(TrainConfig)
    if f.name in ("lr", "d", "dropout", "weight_decay", "modality_dropout", "photo_lags", "epochs")
}
CAST = {"lr": float, "dropout": float, "weight_decay": float, "modality_dropout": float}


def parse_grid(items: list[str]) -> dict[str, list]:
    """``["lr=3e-4,1e-3", "d=128,256"]`` -> {"lr": [3e-4, 1e-3], "d": [128, 256]}."""
    grid = {}
    for item in items:
        name, _, values = item.partition("=")
        if name not in TUNABLE or not values:
            raise ValueError(f"grid entries look like name=v1,v2 with name in {sorted(TUNABLE)}")
        cast = CAST.get(name, int)
        grid[name] = [cast(v) for v in values.split(",")]
    return grid


def _cost(params: dict) -> tuple:
    """Cheaper first: smaller width, no modality dropout, fewer photo lags, fewer epochs."""
    return (
        params.get("d", 0),
        params.get("modality_dropout", 0.0),
        params.get("photo_lags", 1),
        params.get("epochs", 0),
    )


def search(
    dataset: str,
    encoder: str,
    arm_list: tuple[str, ...],
    run_id: str,
    grid: dict[str, list],
    base: TrainConfig,
    folds: list[int],
    seed: int = 0,
    photos: bool = True,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
) -> dict:
    """Score every configuration on every fold's inner-validation units; write
    ``results.csv`` and ``best.json``; return the best configuration."""
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "tune") / run_id
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.csv"
    done = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    records = done.to_dict("records")
    manifest = WindowStore.open(dataset, window_root).manifest
    options = {
        "photos": photos,
        "without": (),
        "levels": "relative",
        "force_mean": False,
        "window_root": window_root,
        "feature_root": feature_root,
    }
    names = sorted(grid)
    configurations = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[n] for n in names))
    ]
    prepared = None
    for fold in folds:
        split = FOLDS[dataset](manifest, fold)
        if prepared is None or "{fold}" in encoder or dataset in FOLD_BASELINE:
            train_units = sorted(split.units(manifest, "train"))
            prepared = prepare_inputs(
                dataset, encoder.replace("{fold}", str(fold)), arm_list, base, train_units,
                options, log,
            )  # fmt: skip
        inputs, _, _, any_present = prepared
        labelled = any_present.copy()
        if inputs.y_class is not None:
            labelled &= inputs.y_class >= 0
        elif inputs.y_reg is not None:
            labelled &= np.isfinite(inputs.y_reg)
        train_rows = np.flatnonzero(split.train & labelled)
        val_rows = np.flatnonzero(split.val & labelled)
        for params in configurations:
            key = json.dumps(params, sort_keys=True)
            if any(r.get("config") == key and r.get("fold") == fold for r in records):
                continue
            cfg = replace(base, **params)
            fitted = fit(inputs, train_rows, val_rows, cfg, seed, log=log)
            best = max(fitted.history, key=lambda h: h["val_score"])
            records.append(
                {
                    "config": key,
                    "fold": fold,
                    "val_score": best["val_score"],
                    "train_score": best.get("train_score", np.nan),
                    "best_epoch": fitted.best_epoch,
                    "seconds": round(fitted.seconds, 1),
                    **params,
                }
            )
            pd.DataFrame(records).to_csv(results_path, index=False)
            log(f"[tune] fold {fold} {key}: validation {best['val_score']:.3f}")
    table = pd.DataFrame(records)
    ranking = (
        table.groupby("config")
        .agg(mean=("val_score", "mean"), std=("val_score", "std"), folds=("fold", "nunique"))
        .reset_index()
    )
    top = ranking.loc[ranking["mean"].idxmax()]
    # Configurations share the folds, so a near-tie is judged on the paired fold-wise
    # differences to the best one: a gap within one standard error of those differences.
    scores = table.pivot_table(index="fold", columns="config", values="val_score")
    gaps = scores[top["config"]].to_numpy()[:, None] - scores.to_numpy()
    se = np.nanstd(gaps, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(gaps), axis=0))
    gap = dict(zip(scores.columns, np.nanmean(gaps, axis=0), strict=True))
    gap_se = dict(zip(scores.columns, np.nan_to_num(se), strict=True))
    ranking["gap"] = [gap[c] for c in ranking["config"]]
    ranking["gap_se"] = [gap_se[c] for c in ranking["config"]]
    close = ranking[ranking["gap"] <= ranking["gap_se"]]
    chosen = min(close["config"], key=lambda c: _cost(json.loads(c)))
    params = json.loads(chosen)
    report = {
        "run_id": run_id,
        "dataset": dataset,
        "encoder": encoder,
        "arms": list(arm_list),
        "chosen": params,
        "best_mean": float(top["mean"]),
        "chosen_mean": float(ranking.set_index("config").loc[chosen, "mean"]),
        "ranking": ranking.sort_values("mean", ascending=False).to_dict("records"),
        "flags": " ".join(f"--{k.replace('_', '-')} {v}" for k, v in params.items()),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(out / "best.json", report)
    log(f"[tune] chosen {params} (mean validation {report['chosen_mean']:.3f}) -> {out}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.tune", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--dataset", required=True, choices=sorted(FOLDS))
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--arms", nargs="+", default=["F"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--grid", nargs="+", required=True, help="name=v1,v2 ...")
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(range(1, protocols.N_FOLDS + 1))
    )
    parser.add_argument("--no-photos", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true", help="run on a busy machine (CPU only)")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        claim_resources("tune", min_idle_cpus=2.0, min_free_ram_gb=1.0, force=args.force)
    search(
        args.dataset, args.encoder, tuple(args.arms), args.run_id, parse_grid(args.grid),
        TrainConfig(device=args.device), args.folds, photos=not args.no_photos,
        out_root=args.out,
    )  # fmt: skip


if __name__ == "__main__":
    main()
