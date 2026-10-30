"""Stage 4: the Plan B baseline (plan: per-modality evidential heads, combined by
Dempster-Shafer and MoNIG; AdamW 1e-3; 50 epochs; 5 outer folds x 3 seeds).

For each fold and seed: fit on the fold's training units; pick the epoch and fit one
temperature on its inner-validation units; score its test units:

* the combined prediction, per window and per bag (pass, snapshot), before and after
  calibration on the inner-validation units: temperature scaling for the classes, one
  interval scale for wear and RUL (decision D). Accuracy, balanced accuracy, macro-F1, ECE,
  Brier, NLL; RMSE, MAE, Student-t NLL, 90 % interval coverage and width, PHM 2012 score;
* every modality alone, and the combination without each modality (missing-modality test);
* baselines: majority class (training prior), persistence (Nonastreda: the previous pass's
  label and wear), and the training mean for regression.

Results: ``runs/planb/<run-id>/fold<k>-seed<s>.json`` with the test predictions in ``.npz``
beside it (written atomically; a rerun skips finished folds and seeds), then
``summary.csv`` and ``summary.md``. Paired tests across fold x seed come later (report stage).

Inputs are assembled once, or once per fold when they depend on it: ``{fold}`` in the
encoder name (``patchtst-fold{fold}``: fold k's own PatchTST) or a level baseline made of the
fold's training units (Paderborn).

    python -m pdmfm.planb --dataset nonastreda --encoder mantis --arms F --run-id planb-nona
    python -m pdmfm.planb --dataset xjtu_sy --encoder "patchtst-fold{fold}" --arms F E --run-id x
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pdmfm import losses, protocols
from pdmfm import metrics as M
from pdmfm.model.inputs import FOLD_BASELINE, LEVEL_MODES, TaskInputs, assemble
from pdmfm.paths import REPO_ROOT
from pdmfm.train import TrainConfig, bag_mean, fit, predict
from pdmfm.utils import stable_hash, write_json_atomic
from pdmfm.windows import WindowStore

FOLDS = {
    "nonastreda": protocols.nonastreda_fold,
    "xjtu_sy": protocols.xjtu_fold,
    "paderborn": protocols.paderborn_fold,
}


# ----------------------------------------------------------------------------- scoring


def _probs(evidence: np.ndarray) -> np.ndarray:
    return losses.expected_probability(
        torch.from_numpy(np.asarray(evidence, dtype=np.float32))
    ).numpy()


def _recombine(pred: dict, keep: list[int]) -> dict:
    """Combined opinion and fused NIG from the modalities in ``keep`` only."""
    out = {}
    if "evidence" in pred:
        out["combined"] = losses.ds_combine_all(torch.from_numpy(pred["evidence"][:, keep]))[
            0
        ].numpy()
    if "nig" in pred:
        present = torch.from_numpy(pred["present"][:, keep])
        out["fused_nig"] = losses.nig_fuse_all(
            torch.from_numpy(pred["nig"][:, keep]), present
        ).numpy()
    return out


def _classes(inputs: TaskInputs, rows, evidence, temperature: float | None = None) -> dict:
    """Window- and bag-level class metrics of ``evidence`` for the labelled ``rows``."""
    assert inputs.y_class is not None
    y = inputs.y_class[rows]
    keep = y >= 0
    if not keep.any():
        return {}
    bags = inputs.manifest["bag"].astype(str).to_numpy()[rows][keep]
    window = _probs(evidence[keep])
    _, bag_evidence = bag_mean(evidence[keep], bags)
    _, bag_y = bag_mean(y[keep].astype(float), bags)
    bag = _probs(bag_evidence)
    if temperature is not None:
        window, bag = (
            M.apply_temperature(window, temperature),
            M.apply_temperature(bag, temperature),
        )
    return {
        "window": M.classification(y[keep], window),
        "bag": M.classification(bag_y.round().astype(int), bag),
    }


def student_t(nig: np.ndarray, y_mean: float, y_std: float):
    """(mean, degrees of freedom, scale) of the NIG predictive, in target units."""
    params = torch.from_numpy(np.asarray(nig, dtype=np.float32))
    dof, scale = (t.numpy() for t in losses.nig_predictive_scale(params))
    return np.asarray(nig)[:, 0] * y_std + y_mean, dof, scale * y_std


def _regression(
    inputs: TaskInputs, rows, nig: np.ndarray, y_mean: float, y_std: float, interval: float = 1.0
) -> dict:
    """Window- and bag-level regression metrics; ``interval`` multiplies the predictive scale."""
    assert inputs.y_reg is not None
    y = inputs.y_reg[rows]
    keep = np.isfinite(y)
    if not keep.any():
        return {}
    bags = inputs.manifest["bag"].astype(str).to_numpy()[rows][keep]

    def scored(params, target):
        mean, dof, scale = student_t(params, y_mean, y_std)
        out = M.regression(target, mean, dof, scale * interval)
        if inputs.regression_target == "rul":
            out["phm2012"] = M.phm2012_score(target, np.clip(mean, 0.0, 1.0))
        return out

    _, bag_nig = bag_mean(nig[keep], bags)
    _, bag_y = bag_mean(y[keep], bags)
    return {
        "window": scored(nig[keep].astype(np.float32), y[keep]),
        "bag": scored(bag_nig.astype(np.float32), bag_y),
    }


def _trust(pred: dict, names: list[str]) -> dict:
    """Mean uncertainty of the combined opinion, mean conflict, and mean trust (1 - u_m)
    of every modality where present."""
    if "evidence" not in pred:
        return {}
    _, u = losses.opinion(torch.from_numpy(pred["combined"]))
    out = {"uncertainty": float(u.mean()), "conflict": float(pred["conflict"].sum(-1).mean())}
    for m, name in enumerate(names):
        present = pred["present"][:, m]
        if present.any():
            _, um = losses.opinion(torch.from_numpy(pred["evidence"][present, m]))
            out[f"trust_{name}"] = float((1 - um).mean())
    return out


def evaluate(
    inputs: TaskInputs,
    rows: np.ndarray,
    pred: dict,
    y_mean: float,
    y_std: float,
    temperature: float | None,
    interval_scale: float | None = None,
) -> dict:
    names = list(inputs.modalities)
    result: dict = {"trust": _trust(pred, names)}
    if "combined" in pred:
        result["combined"] = _classes(inputs, rows, pred["combined"])
        if temperature is not None:
            result["combined_calibrated"] = _classes(inputs, rows, pred["combined"], temperature)
        for m, name in enumerate(names):
            present = pred["present"][:, m]
            if present.any():
                result[f"only_{name}"] = _classes(
                    inputs, rows[present], pred["evidence"][present, m]
                )
    if "fused_nig" in pred:
        result.setdefault("regression", {})["combined"] = _regression(
            inputs, rows, pred["fused_nig"], y_mean, y_std
        )
        if interval_scale is not None:
            result["regression"]["combined_calibrated"] = _regression(
                inputs, rows, pred["fused_nig"], y_mean, y_std, interval_scale
            )
        for m, name in enumerate(names):
            present = pred["present"][:, m]
            if present.any():
                result["regression"][f"only_{name}"] = _regression(
                    inputs, rows[present], pred["nig"][present, m], y_mean, y_std
                )
    if len(names) > 1:  # missing-modality robustness: the same model without one modality
        for m, name in enumerate(names):
            again = _recombine(pred, [j for j in range(len(names)) if j != m])
            if "combined" in again:
                result[f"without_{name}"] = _classes(inputs, rows, again["combined"])
            if "fused_nig" in again:
                result["regression"][f"without_{name}"] = _regression(
                    inputs, rows, again["fused_nig"], y_mean, y_std
                )
    return result


def baselines(inputs: TaskInputs, train_rows: np.ndarray, test_rows: np.ndarray) -> dict:
    """Majority class and training mean; persistence (previous bag of the same unit, in
    time order) where bags follow each other (Nonastreda passes, XJTU-SY snapshots)."""
    manifest = inputs.manifest
    out: dict = {}
    first = manifest.iloc[test_rows].groupby("bag", sort=False).head(1)
    time_key = (
        "meta_t_start"
        if "meta_t_start" in manifest
        else "meta_minute"
        if "meta_minute" in manifest
        else None
    )
    ordered = first.sort_values(["unit", time_key]) if time_key else None
    if inputs.y_class is not None:
        y_train = inputs.y_class[train_rows]
        y_train = y_train[y_train >= 0]
        k = len(inputs.classes)
        prior = np.bincount(y_train, minlength=k) / max(len(y_train), 1)
        labels = inputs.y_class[first.index.to_numpy()]
        keep = labels >= 0
        out["majority"] = M.classification(labels[keep], np.tile(prior, (int(keep.sum()), 1)))
        if ordered is not None:
            y = inputs.y_class[ordered.index.to_numpy()]
            previous = ordered.groupby("unit")["class_key"].shift(1)
            index = {c: i for i, c in enumerate(inputs.classes)}
            guess = np.array(
                [
                    index.get(p, int(prior.argmax())) if isinstance(p, str) else int(prior.argmax())
                    for p in previous
                ]
            )
            keep = y >= 0
            out["persistence"] = M.classification(y[keep], np.eye(k)[guess[keep]])
    if inputs.y_reg is not None:
        y_train = inputs.y_reg[train_rows]
        mean = float(np.nanmean(y_train))
        y = inputs.y_reg[first.index.to_numpy()]
        keep = np.isfinite(y)
        out["train_mean"] = {
            "rmse": float(np.sqrt(np.mean((y[keep] - mean) ** 2))),
            "mae": float(np.mean(np.abs(y[keep] - mean))),
        }
        if ordered is not None and inputs.regression_target == "wear_um":
            y = inputs.y_reg[ordered.index.to_numpy()]
            previous = (
                pd.Series(y, index=ordered.index)
                .groupby(ordered["unit"].to_numpy())
                .shift(1)
                .fillna(mean)
                .to_numpy()
            )
            keep = np.isfinite(y)
            out["persistence_regression"] = {
                "rmse": float(np.sqrt(np.mean((y[keep] - previous[keep]) ** 2))),
                "mae": float(np.mean(np.abs(y[keep] - previous[keep]))),
            }
    return out


# ----------------------------------------------------------------------------- runs


def _calibrate(inputs: TaskInputs, val_rows, val_pred: dict, y_mean: float, y_std: float):
    """(temperature, interval scale) fitted on the inner-validation windows; None where the
    task or its labels are absent."""
    temperature = interval = None
    if "combined" in val_pred and inputs.y_class is not None:
        y = inputs.y_class[val_rows]
        keep = y >= 0
        if keep.any():
            temperature = M.temperature_scale(y[keep], _probs(val_pred["combined"][keep]))
    if "fused_nig" in val_pred and inputs.y_reg is not None:
        y = inputs.y_reg[val_rows]
        keep = np.isfinite(y)
        if keep.any():
            mean, dof, scale = student_t(val_pred["fused_nig"][keep], y_mean, y_std)
            interval = M.interval_scale(y[keep], mean, dof, scale)
    return temperature, interval


def _prepare(dataset, encoder, arm_list, cfg, train_units, options: dict, log):
    """(inputs, settings, settings key, rows with any modality present) for one encoder.
    ``options`` holds :func:`run`'s photos, without, levels, force_mean, window_root and
    feature_root;
    modalities named in ``without`` are left out of the model."""
    photos, without = options["photos"], options["without"]
    inputs = assemble(
        dataset,
        encoder,
        arm_list,
        photos=photos,
        window_root=options["window_root"],
        feature_root=options["feature_root"],
        train_units=train_units,
        levels=options["levels"],
        force_mean=options["force_mean"],
    )
    unknown = set(without) - set(inputs.modalities)
    if unknown:
        raise ValueError(f"no modality {sorted(unknown)}; {dataset} has {list(inputs.modalities)}")
    for name in without:
        del inputs.modalities[name]
        del inputs.description["modalities"][name]
    if not inputs.modalities:
        raise ValueError("--without left no modality")
    settings = {"inputs": inputs.description, "train": cfg.as_dict(), "photos": photos}
    if without:
        settings["without"] = sorted(without)
    key = stable_hash(settings)[:12]
    names = list(inputs.modalities)
    log(f"[planb] {dataset} {encoder} {'+'.join(arm_list)}: modalities {names}; {key}")
    any_present = np.any([m.present for m in inputs.modalities.values()], axis=0)
    return inputs, settings, key, any_present


def run(
    dataset: str,
    encoder: str,
    arm_list: tuple[str, ...],
    run_id: str,
    folds: list[int],
    seeds: list[int],
    cfg: TrainConfig,
    photos: bool = True,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
    without: tuple[str, ...] = (),
    levels: str = "relative",
    force_mean: bool = False,
) -> Path:
    if dataset not in FOLDS:
        raise ValueError(f"Plan B runs on the source datasets with outer folds: {sorted(FOLDS)}")
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "planb") / run_id
    out.mkdir(parents=True, exist_ok=True)
    manifest = WindowStore.open(dataset, window_root).manifest
    per_fold = "{fold}" in encoder or dataset in FOLD_BASELINE
    options = {
        "photos": photos,
        "without": without,
        "levels": levels,
        "force_mean": force_mean,
        "window_root": window_root,
        "feature_root": feature_root,
    }
    prepared = None
    for fold in folds:
        split = FOLDS[dataset](manifest, fold)
        split.assert_unit_disjoint(manifest)
        if prepared is None or per_fold:
            train_units = sorted(split.units(manifest, "train"))
            encoder_k = encoder.replace("{fold}", str(fold))
            prepared = _prepare(dataset, encoder_k, arm_list, cfg, train_units, options, log)
        inputs, settings, key, any_present = prepared
        labelled = any_present.copy()
        if inputs.y_class is not None:
            labelled &= inputs.y_class >= 0
        elif inputs.y_reg is not None:
            labelled &= np.isfinite(inputs.y_reg)
        train_rows = np.flatnonzero(split.train & labelled)
        val_rows = np.flatnonzero(split.val & labelled)
        test_rows = np.flatnonzero(split.test & labelled)
        for seed in seeds:
            path = out / f"fold{fold}-seed{seed}.json"
            if path.is_file():
                if json.loads(path.read_text(encoding="utf-8")).get("settings_key") == key:
                    log(f"[planb] fold {fold} seed {seed}: done earlier, skipped")
                    continue
                raise ValueError(f"{path} holds a run with other settings; use another --run-id")
            sizes = f"train {len(train_rows)}, validation {len(val_rows)}, test {len(test_rows)}"
            log(f"[planb] fold {fold} seed {seed}: {sizes} windows")
            fitted = fit(inputs, train_rows, val_rows, cfg, seed, log=log)
            val_pred = predict(fitted.model, inputs, val_rows, cfg)
            y_mean, y_std = fitted.y_mean, fitted.y_std
            temperature, interval = _calibrate(inputs, val_rows, val_pred, y_mean, y_std)
            test_pred = predict(fitted.model, inputs, test_rows, cfg)
            trainable = sum(p.numel() for p in fitted.model.parameters() if p.requires_grad)
            result = {
                "dataset": dataset,
                "fold": fold,
                "seed": seed,
                "settings_key": key,
                "settings": settings,
                "split": split.name,
                "best_epoch": fitted.best_epoch,
                "seconds": round(fitted.seconds, 1),
                "temperature": temperature,
                "interval_scale": interval,
                "n_params": int(trainable),
                "validation": evaluate(inputs, val_rows, val_pred, y_mean, y_std, None),
                "test": evaluate(
                    inputs, test_rows, test_pred, y_mean, y_std, temperature, interval
                ),
                "baselines": baselines(inputs, train_rows, test_rows),
                "history": fitted.history,
            }
            np.savez_compressed(out / f"fold{fold}-seed{seed}.npz", rows=test_rows, **test_pred)
            write_json_atomic(path, result)
    summarize(out)
    return out


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        out[prefix] = float(obj)


def summarize(out: Path) -> pd.DataFrame:
    """Mean and std over fold x seed of every metric; ``summary.md`` shows the main ones."""
    rows = []
    for path in sorted(out.glob("fold*-seed*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        flat: dict = {}
        _flatten(
            "",
            {
                "test": result["test"],
                "baselines": result["baselines"],
                "temperature": result["temperature"],
                "interval_scale": result.get("interval_scale"),
            },
            flat,
        )
        rows.append(flat)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    summary = pd.DataFrame({"mean": table.mean(), "std": table.std(ddof=0), "runs": table.count()})
    summary.to_csv(out / "summary.csv")
    (out / "summary.md").write_text(_markdown(out.name, summary, len(rows)), encoding="utf-8")
    return summary


def _cell(summary: pd.DataFrame, key: str, digits: int = 3) -> str:
    if key not in summary.index:
        return "-"
    mean, std = summary.loc[key, "mean"], summary.loc[key, "std"]
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _row(summary: pd.DataFrame, label: str, keys: list[str | None]) -> str:
    return "| " + " | ".join([label] + [_cell(summary, k) if k else "-" for k in keys]) + " |"


def _markdown(run_id: str, summary: pd.DataFrame, runs: int) -> str:
    keys = sorted(
        {k.split(".")[1] for k in summary.index if k.startswith("test.") and k.count(".") >= 3}
    )
    lines = [
        f"# Plan B: {run_id}",
        "",
        f"{runs} runs (outer folds x seeds); test units; mean ± std.",
        "",
    ]
    fitted = [
        f"{name} {_cell(summary, key)}"
        for name, key in (("temperature", "temperature"), ("interval scale", "interval_scale"))
        if key in summary.index
    ]
    if fitted:
        lines += [f"Calibration fitted on the inner-validation units: {'; '.join(fitted)}.", ""]
    class_rows = [k for k in keys if f"test.{k}.bag.macro_f1" in summary.index]
    if class_rows:
        lines += [
            "## Classes (bag level unless stated)",
            "",
            "| Variant | macro-F1 | balanced accuracy | accuracy | ECE (window) | NLL (window) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for k in class_rows:
            bag, window = f"test.{k}.bag", f"test.{k}.window"
            metrics = [f"{bag}.macro_f1", f"{bag}.balanced_accuracy", f"{bag}.accuracy"]
            lines.append(_row(summary, k, [*metrics, f"{window}.ece", f"{window}.nll"]))
        for name in ("majority", "persistence"):
            base = f"baselines.{name}"
            if f"{base}.macro_f1" in summary.index:
                metrics = [f"{base}.macro_f1", f"{base}.balanced_accuracy", f"{base}.accuracy"]
                lines.append(_row(summary, f"baseline: {name}", [*metrics, None, None]))
        lines.append("")
    reg_rows = sorted(
        {
            k.split(".")[2]
            for k in summary.index
            if k.startswith("test.regression.") and k.endswith(".bag.rmse")
        }
    )
    if reg_rows:
        lines += [
            "## Regression (bag level)",
            "",
            "| Variant | RMSE | MAE | NLL | 90 % coverage | 90 % width | PHM 2012 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for k in reg_rows:
            base = f"test.regression.{k}.bag"
            names = ("rmse", "mae", "nll", "coverage_90", "width_90", "phm2012")
            lines.append(_row(summary, k, [f"{base}.{n}" for n in names]))
        for name in ("train_mean", "persistence_regression"):
            base = f"baselines.{name}"
            if f"{base}.rmse" in summary.index:
                lines.append(
                    _row(
                        summary,
                        f"baseline: {name}",
                        [f"{base}.rmse", f"{base}.mae", None, None, None, None],
                    )
                )
        lines.append("")
    trust = [k for k in summary.index if k.startswith("test.trust.")]
    if trust:
        lines += ["## Trust (test windows)", ""] + [
            f"- {k.removeprefix('test.trust.')}: {_cell(summary, k)}" for k in trust
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.planb", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--dataset", required=True, choices=sorted(FOLDS))
    parser.add_argument(
        "--encoder",
        required=True,
        help='sensor encoder whose features are cached (mantis, or "patchtst-fold{fold}")',
    )
    parser.add_argument("--arms", nargs="+", default=["T", "F"], help="sensor arms, e.g. T F E")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(range(1, protocols.N_FOLDS + 1))
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-photos", action="store_true", help="sensor modalities only")
    parser.add_argument(
        "--without",
        nargs="+",
        default=[],
        metavar="MODALITY",
        help="leave modalities out of the model, e.g. slow (a confounded temperature)",
    )
    parser.add_argument(
        "--levels",
        default="relative",
        choices=LEVEL_MODES,
        help="level tokens: baseline-relative (default, the plan), absolute, or both",
    )
    parser.add_argument(
        "--force-mean", action="store_true", help="add force channels' window mean (ablation)"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="results folder (default: runs/planb)")
    parser.add_argument("--force", action="store_true", help="run on a busy machine (CPU only)")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        # A run peaks at about 0.6 GB (Nonastreda with photos, d = 256).
        claim_resources("planb", min_idle_cpus=2.0, min_free_ram_gb=1.0, force=args.force)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    started = time.perf_counter()
    out = run(
        args.dataset,
        args.encoder,
        tuple(args.arms),
        args.run_id,
        args.folds,
        args.seeds,
        cfg,
        photos=not args.no_photos,
        out_root=args.out,
        without=tuple(args.without),
        levels=args.levels,
        force_mean=args.force_mean,
    )
    print(f"[planb] done in {time.perf_counter() - started:.0f} s -> {out}")


if __name__ == "__main__":
    main()
