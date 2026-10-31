"""Stage 4: the Plan B baseline (plan: per-modality evidential heads, combined by
Dempster-Shafer and MoNIG; AdamW 1e-3; 50 epochs; 5 outer folds x 3 seeds).

For each fold and seed: fit on the fold's training units; pick the epoch and fit one
temperature on its inner-validation units; score its test units:

* the combined prediction, per window and per bag (pass, snapshot), before and after
  calibration on the inner-validation units: temperature scaling for the classes, one
  interval scale for wear and RUL (decision D). Accuracy, balanced accuracy, macro-F1, ECE,
  Brier, NLL; RMSE, MAE, Student-t NLL, 90 % interval coverage and width, PHM 2012 score;
* every modality alone, and the model without each modality (missing-modality test: the
  modality marked absent at prediction time, so a fusion stage loses it too);
* baselines: majority class (training prior), persistence (Nonastreda: the previous pass's
  label and wear), and the training mean for regression.

Results: ``runs/planb/<run-id>/fold<k>-seed<s>.json`` with the test predictions in ``.npz``
beside it (written atomically; a rerun skips finished folds and seeds), then
``summary.csv`` and ``summary.md``. Paired tests across fold x seed come later (report stage).

``--fusion`` trains a Gate-2 candidate (:mod:`pdmfm.model.fusion`) instead; its fused head
is also scored alone (``only_fused``). Inputs are assembled once, or once per fold when they
depend on it: ``{fold}`` in the
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
from pdmfm import stats as S
from pdmfm.model.fusion import FUSIONS
from pdmfm.model.inputs import FOLD_BASELINE, LEVEL_MODES, MAX_PHOTO_LAG, TaskInputs, assemble
from pdmfm.paths import REPO_ROOT
from pdmfm.train import TrainConfig, bag_mean, fit, predict
from pdmfm.utils import stable_hash, write_json_atomic
from pdmfm.windows import WindowStore

# Ordered label spaces (wear classes sharp < used < dulled): ordinal metrics are added.
ORDINAL = ("nonastreda", "matwi")
FOLDS = {
    "nonastreda": protocols.nonastreda_fold,
    "matwi": protocols.matwi_fold,
    "xjtu_sy": protocols.xjtu_fold,
    "paderborn": protocols.paderborn_fold,
}


# ----------------------------------------------------------------------------- scoring


def _probs(evidence: np.ndarray) -> np.ndarray:
    return losses.expected_probability(
        torch.from_numpy(np.asarray(evidence, dtype=np.float32))
    ).numpy()


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
    ordinal = inputs.dataset in ORDINAL
    return {
        "window": M.classification(y[keep], window, ordinal),
        "bag": M.classification(bag_y.round().astype(int), bag, ordinal),
    }


def student_t(nig: np.ndarray, y_mean: float, y_std: float):
    """(mean, degrees of freedom, scale) of the NIG predictive, in target units."""
    params = torch.from_numpy(np.asarray(nig, dtype=np.float32))
    dof, scale = (t.numpy() for t in losses.nig_predictive_scale(params))
    return np.asarray(nig)[:, 0] * y_std + y_mean, dof, scale * y_std


def score_regression(
    target: np.ndarray,
    nig: np.ndarray,
    y_mean: float,
    y_std: float,
    interval: float = 1.0,
    rul: bool = False,
) -> dict:
    """Regression metrics of NIG outputs (standardized) against ``target``; ``interval``
    multiplies the predictive scale. RUL adds the PHM 2012 score and the same metrics after
    the first prediction time."""
    mean, dof, scale = student_t(np.asarray(nig, dtype=np.float32), y_mean, y_std)
    out = M.regression(target, mean, dof, scale * interval)
    if rul:
        out["phm2012"] = M.phm2012_score(target, np.clip(mean, 0.0, 1.0))
        # Normalized RUL is 1 until the first prediction time: the degradation phase is
        # where the prognosis matters, so it is scored on its own as well.
        post = target < 1.0 - 1e-6
        if post.any():
            out["post_fpt"] = M.regression(
                target[post], mean[post], dof[post], scale[post] * interval
            )
            out["post_fpt"]["phm2012"] = M.phm2012_score(
                target[post], np.clip(mean[post], 0.0, 1.0)
            )
    return out


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
    rul = inputs.regression_target == "rul"
    _, bag_nig = bag_mean(nig[keep], bags)
    _, bag_y = bag_mean(y[keep], bags)
    return {
        "window": score_regression(y[keep], nig[keep], y_mean, y_std, interval, rul),
        "bag": score_regression(bag_y, bag_nig, y_mean, y_std, interval, rul),
    }


def _trust(pred: dict, names: list[str]) -> dict:
    """Mean uncertainty of the combined opinion, mean conflict, and mean trust (1 - u_m)
    of every modality where present."""
    if "evidence" not in pred:
        return {}
    _, u = losses.opinion(torch.from_numpy(pred["combined"]))
    out = {"uncertainty": float(u.mean()), "conflict": float(pred["conflict"].sum(-1).mean())}
    if "fused_evidence" in pred:
        anything = pred["present"].any(1)
        _, uf = losses.opinion(torch.from_numpy(pred["fused_evidence"][anything]))
        out["trust_fused"] = float((1 - uf).mean())
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
    stale: dict[int, dict] | None = None,
    dropped: dict[str, dict] | None = None,
) -> dict:
    """Every result variant of one part. ``stale`` maps a photo lag to the predictions made
    with the photos of that many passes back (the staleness test); ``dropped`` maps a
    modality to the predictions made without it (:func:`without_each`)."""
    names = list(inputs.modalities)
    anything = pred["present"].any(1)
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
        if "fused_evidence" in pred:
            result["only_fused"] = _classes(
                inputs, rows[anything], pred["fused_evidence"][anything]
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
        if "fused_own_nig" in pred:
            result["regression"]["only_fused"] = _regression(
                inputs, rows[anything], pred["fused_own_nig"][anything], y_mean, y_std
            )
    for name, again in (dropped or {}).items():  # missing-modality robustness
        # Rows whose only modality was dropped have no input left; they are not scored.
        left = again["present"].any(1)
        if "combined" in again:
            result[f"without_{name}"] = _classes(inputs, rows[left], again["combined"][left])
        if "fused_nig" in again:
            result.setdefault("regression", {})[f"without_{name}"] = _regression(
                inputs, rows[left], again["fused_nig"][left], y_mean, y_std
            )
    for lag, older in (stale or {}).items():
        if "combined" in older:
            result[f"photo_lag{lag}"] = _classes(inputs, rows, older["combined"])
        if "fused_nig" in older:
            result["regression"][f"photo_lag{lag}"] = _regression(
                inputs, rows, older["fused_nig"], y_mean, y_std
            )
    return result


def without_each(model, inputs: TaskInputs, rows: np.ndarray, cfg: TrainConfig) -> dict:
    """The predictions without each modality in turn (marked absent at prediction time).
    For late fusion this equals combining the other experts' opinions; a fusion stage also
    loses the modality's tokens."""
    if len(inputs.modalities) < 2:
        return {}
    return {name: predict(model, inputs, rows, cfg, drop=(name,)) for name in inputs.modalities}


def _fit_summary(inputs: TaskInputs, rows, pred: dict, y_mean: float, y_std: float) -> dict:
    """The combined prediction scored on training rows: does the model fit its data?"""
    out: dict = {}
    if "combined" in pred:
        out["combined"] = _classes(inputs, rows, pred["combined"])
    if "fused_nig" in pred:
        out["regression"] = {
            "combined": _regression(inputs, rows, pred["fused_nig"], y_mean, y_std)
        }
    return out


def _bag_predictions(inputs: TaskInputs, rows, pred: dict, y_mean: float, y_std: float):
    """Per test bag: the combined class (argmax of the mean evidence; for ordered classes
    also the ordinal median) and wear or RUL (mean NIG location); calibration changes
    neither."""
    bags = inputs.manifest["bag"].astype(str).to_numpy()[rows]
    table = pd.DataFrame({"bag": pd.unique(bags)})
    if "combined" in pred:
        keys, evidence = bag_mean(pred["combined"], bags)
        probs = _probs(evidence)
        table["pred_class"] = pd.Series(probs.argmax(1), index=keys).loc[table["bag"]].to_numpy()
        if inputs.dataset in ORDINAL:
            median = pd.Series(M.ordinal_median(probs), index=keys)
            table["pred_class_ordinal"] = median.loc[table["bag"]].to_numpy()
    if "fused_nig" in pred:
        keys, nig = bag_mean(pred["fused_nig"], bags)
        mean = nig[:, 0] * y_std + y_mean
        table["pred_reg"] = pd.Series(mean, index=keys).loc[table["bag"]].to_numpy()
    return table


def baseline_table(
    inputs: TaskInputs, train_rows: np.ndarray, test_rows: np.ndarray
) -> pd.DataFrame:
    """One row per test bag (in time order per unit where bags follow each other: milling
    passes and cuts, XJTU-SY snapshots): its unit and labels, and the majority, persistence,
    schedule and training-mean predictions. Persistence is the previous bag's true label or
    wear of the same unit; a unit's first bag falls back to the training majority or mean.
    The schedule baseline knows only the bag's position in its unit's life (pass, cut or
    snapshot number): the training units' majority class and mean target at that position
    (OmniFuser's run-number shortcut: 71 % on held-out tools, above its model)."""
    manifest = inputs.manifest
    first = manifest.iloc[test_rows].groupby("bag", sort=False).head(1)
    trained = manifest.iloc[train_rows].groupby("bag", sort=False).head(1)
    time_key = next((k for k in ("meta_t_start", "meta_minute") if k in manifest), None)
    if time_key:
        first = first.sort_values(["unit", time_key])
        trained = trained.sort_values(["unit", time_key])
    # Positions in each unit's life; used only where bags are in time order (time_key).
    train_position = trained.groupby("unit").cumcount().to_numpy()
    test_position = first.groupby("unit").cumcount().to_numpy()
    index = first.index.to_numpy()
    table = pd.DataFrame(
        {"bag": first["bag"].astype(str).to_numpy(), "unit": first["unit"].astype(str).to_numpy()}
    )
    if inputs.y_class is not None:
        prior = _prior(inputs, train_rows)
        table["y_class"] = inputs.y_class[index]
        table["majority"] = int(prior.argmax())
        if time_key:
            positions = {c: i for i, c in enumerate(inputs.classes)}
            previous = first.groupby("unit")["class_key"].shift(1)
            table["persistence"] = [
                positions.get(p, int(prior.argmax())) if isinstance(p, str) else int(prior.argmax())
                for p in previous
            ]
            y_train = inputs.y_class[trained.index.to_numpy()]
            labelled = y_train >= 0
            table["schedule"] = _schedule(
                train_position[labelled], y_train[labelled], test_position, int(prior.argmax()),
                lambda v: int(np.bincount(v).argmax()),
            )  # fmt: skip
    if inputs.y_reg is not None:
        mean = float(np.nanmean(inputs.y_reg[train_rows]))
        y = inputs.y_reg[index]
        table["y_reg"] = y
        table["train_mean"] = mean
        if time_key and inputs.regression_target == "wear_um":
            previous = pd.Series(y).groupby(table["unit"].to_numpy()).shift(1)
            table["persistence_reg"] = previous.fillna(mean).to_numpy()
        if time_key:
            y_train = inputs.y_reg[trained.index.to_numpy()]
            known = np.isfinite(y_train)
            table["schedule_reg"] = _schedule(
                train_position[known], y_train[known], test_position, mean, np.mean
            )
    return table


def _schedule(train_position, train_value, test_position, fallback, summary) -> np.ndarray:
    """``summary`` of the training values at each test position; ``fallback`` where no
    training unit reaches that position."""
    by_position = {
        int(p): summary(train_value[train_position == p]) for p in np.unique(train_position)
    }
    return np.array([by_position.get(int(p), fallback) for p in test_position])


def _prior(inputs: TaskInputs, train_rows: np.ndarray) -> np.ndarray:
    assert inputs.y_class is not None
    y = inputs.y_class[train_rows]
    y = y[y >= 0]
    return np.bincount(y, minlength=len(inputs.classes)) / max(len(y), 1)


def baselines(inputs: TaskInputs, train_rows: np.ndarray, test_rows: np.ndarray) -> dict:
    """Majority class, persistence and training mean, scored on the test bags."""
    table = baseline_table(inputs, train_rows, test_rows)
    out: dict = {}
    if "y_class" in table:
        keep = (table["y_class"] >= 0).to_numpy()
        y = table["y_class"].to_numpy()[keep]
        prior = _prior(inputs, train_rows)
        ordinal = inputs.dataset in ORDINAL
        out["majority"] = M.classification(y, np.tile(prior, (len(y), 1)), ordinal)
        for name in ("persistence", "schedule"):
            if name in table:
                guess = table[name].to_numpy()[keep]
                out[name] = M.classification(y, np.eye(len(inputs.classes))[guess], ordinal)
    if "y_reg" in table:
        y = table["y_reg"].to_numpy()
        keep = np.isfinite(y)
        for name, column in (
            ("train_mean", "train_mean"),
            ("persistence_regression", "persistence_reg"),
            ("schedule_regression", "schedule_reg"),
        ):
            if column in table:
                error = y[keep] - table[column].to_numpy()[keep]
                out[name] = {
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
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


def prepare_inputs(dataset, encoder, arm_list, cfg, train_units, options: dict, log):
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
            prepared = prepare_inputs(dataset, encoder_k, arm_list, cfg, train_units, options, log)
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
            photo = inputs.modalities.get("photo")
            stale = (
                {
                    lag: predict(fitted.model, inputs, test_rows, cfg, photo_lag=lag)
                    for lag in range(2, MAX_PHOTO_LAG + 1)
                }
                if photo is not None and photo.views is not None
                else None
            )
            train_pred = predict(fitted.model, inputs, train_rows, cfg)
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
                "validation": evaluate(
                    inputs,
                    val_rows,
                    val_pred,
                    y_mean,
                    y_std,
                    None,
                    dropped=without_each(fitted.model, inputs, val_rows, cfg),
                ),  # fmt: skip
                "test": evaluate(
                    inputs,
                    test_rows,
                    test_pred,
                    y_mean,
                    y_std,
                    temperature,
                    interval,
                    stale,
                    without_each(fitted.model, inputs, test_rows, cfg),
                ),  # fmt: skip
                "train": _fit_summary(inputs, train_rows, train_pred, y_mean, y_std),
                "baselines": baselines(inputs, train_rows, test_rows),
                "history": fitted.history,
            }
            bags = baseline_table(inputs, train_rows, test_rows).merge(
                _bag_predictions(inputs, test_rows, test_pred, y_mean, y_std), on="bag"
            )
            bags.insert(0, "fold", fold)
            bags.to_csv(out / f"fold{fold}-seed{seed}-bags.csv", index=False)
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
                "train": result.get("train", {}),
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
    pooled = pooled_summary(out)
    if len(pooled):
        pooled.to_csv(out / "pooled.csv", index=False)
    (out / "summary.md").write_text(
        _markdown(out.name, summary, len(rows), pooled), encoding="utf-8"
    )
    return summary


BOOTSTRAP_REPLICATES = 2000


def pooled_summary(out: Path, n: int | None = None) -> pd.DataFrame:
    """Pooled out-of-fold bag predictions of each seed, with 95 % confidence intervals from
    resampling test units (:mod:`pdmfm.stats`): the model, the baselines and the
    differences."""
    by_seed: dict[int, list[pd.DataFrame]] = {}
    for path in sorted(out.glob("fold*-seed*-bags.csv")):
        seed = int(path.stem.split("-seed")[1].split("-")[0])
        by_seed.setdefault(seed, []).append(pd.read_csv(path))
    if not by_seed:
        return pd.DataFrame()
    tables = [pd.concat(parts, ignore_index=True) for parts in by_seed.values()]
    columns = set(tables[0].columns)
    quantities: list[tuple[str, object]] = []

    def f1_of(column):
        def statistic(f):
            keep = f["y_class"] >= 0
            return S.macro_f1(f["y_class"][keep], f[column][keep])

        return statistic

    def rmse_of(column, after_fpt=False):
        def statistic(f):
            keep = f["y_reg"] < 1 - 1e-6 if after_fpt else slice(None)
            return S.rmse(f["y_reg"][keep], f[column][keep])

        return statistic

    def difference(a, b):
        return lambda f: a(f) - b(f)

    def changed_accuracy(column):
        """Accuracy on the bags whose class differs from the unit's previous bag (where
        persistence is wrong): what a prognosis adds over copying the last label."""

        def statistic(f):
            keep = (f["y_class"] >= 0) & (f["persistence"] != f["y_class"])
            return float(np.mean(f[column][keep] == f["y_class"][keep])) if keep.any() else np.nan

        return statistic

    if {"y_class", "pred_class"} <= columns:
        f1 = {
            name: f1_of(column)
            for name, column in (
                ("model", "pred_class"),
                ("model, ordinal median", "pred_class_ordinal"),
                ("persistence", "persistence"),
                ("schedule", "schedule"),
                ("majority", "majority"),
            )
            if column in columns
        }
        quantities += [(f"macro-F1, {name}", fn) for name, fn in f1.items()]
        quantities += [
            (f"macro-F1, model - {name}", difference(f1["model"], fn))
            for name, fn in f1.items()
            if not name.startswith("model")
        ]
        if "persistence" in columns:
            quantities += [("accuracy where the class changes, model",
                            changed_accuracy("pred_class"))]  # fmt: skip
            if "schedule" in columns:
                quantities += [("accuracy where the class changes, schedule",
                                changed_accuracy("schedule"))]  # fmt: skip
    if {"y_reg", "pred_reg"} <= columns:
        error = {
            name: rmse_of(column)
            for name, column in (
                ("model", "pred_reg"),
                ("persistence", "persistence_reg"),
                ("schedule", "schedule_reg"),
                ("training mean", "train_mean"),
            )
            if column in columns
        }
        quantities += [(f"RMSE, {name}", fn) for name, fn in error.items()]
        quantities += [
            (f"RMSE, model - {name}", difference(error["model"], fn))
            for name, fn in error.items()
            if name != "model"
        ]
        y = tables[0]["y_reg"].to_numpy(dtype=float)
        if bool(np.any(y < 1 - 1e-6)) and float(np.nanmax(y)) <= 1.0:  # normalized RUL
            quantities += [
                ("RMSE after FPT, model", rmse_of("pred_reg", after_fpt=True)),
                ("RMSE after FPT, training mean", rmse_of("train_mean", after_fpt=True)),
            ]
            if "schedule_reg" in columns:
                quantities += [("RMSE after FPT, schedule", rmse_of("schedule_reg", True))]
    records = []
    for name, statistic in quantities:
        estimate, low, high = S.unit_bootstrap(
            tables,
            statistic,  # pyright: ignore[reportArgumentType]
            n=n or BOOTSTRAP_REPLICATES,
        )
        records.append({"quantity": name, "estimate": estimate, "low": low, "high": high})
    return pd.DataFrame(records)


def _cell(summary: pd.DataFrame, key: str, digits: int = 3) -> str:
    if key not in summary.index:
        return "-"
    mean, std = summary.loc[key, "mean"], summary.loc[key, "std"]
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _row(summary: pd.DataFrame, label: str, keys: list[str | None]) -> str:
    return "| " + " | ".join([label] + [_cell(summary, k) if k else "-" for k in keys]) + " |"


def _markdown(
    run_id: str, summary: pd.DataFrame, runs: int, pooled: pd.DataFrame | None = None
) -> str:
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
    fit_lines = [
        f"{what}: training units {_cell(summary, f'train.{key}')}, test units "
        f"{_cell(summary, f'test.{key}')}"
        for what, key in (
            ("macro-F1", "combined.bag.macro_f1"),
            ("RMSE", "regression.combined.bag.rmse"),
        )
        if f"train.{key}" in summary.index
    ]
    if fit_lines:
        lines += ["Fit (bag level): " + "; ".join(fit_lines) + ".", ""]
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
        for name in ("majority", "persistence", "schedule"):
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
            if (
                k in ("combined", "combined_calibrated")
                and f"{base}.post_fpt.rmse" in summary.index
            ):
                lines.append(
                    _row(summary, f"{k}, after FPT", [f"{base}.post_fpt.{n}" for n in names])
                )
        for name in ("train_mean", "persistence_regression", "schedule_regression"):
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
        lines.append("")
    if pooled is not None and len(pooled):
        lines += [
            "## Pooled out-of-fold, bag level",
            "",
            "Every test unit once per seed; seeds averaged; 95 % interval from resampling "
            "whole test units (tools, bearings) 2,000 times.",
            "",
            "| Quantity | Estimate | 95 % interval |",
            "| --- | --- | --- |",
        ]
        for row in pooled.to_dict("records"):
            lines.append(
                f"| {row['quantity']} | {row['estimate']:.3f} | "
                f"[{row['low']:.3f}, {row['high']:.3f}] |"
            )
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
    defaults = TrainConfig()
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--d", type=int, default=defaults.d, help="model width")
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--modality-dropout", type=float, default=0.0, help="drop modalities from the combination"
    )
    parser.add_argument(
        "--photo-lags", type=int, default=1, help="train on photos up to this many passes back"
    )
    parser.add_argument("--photo-views", action="store_true", help="train on augmented photos")
    parser.add_argument(
        "--head",
        default="prototype",
        choices=("prototype", "projected", "linear"),
        help="class head: Qwen prototypes, prototypes through a learned W_t, or free linear",
    )
    parser.add_argument(
        "--fusion", default="late", choices=FUSIONS, help="late (Plan B) or a Gate-2 candidate"
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.0,
        help="drop whole modalities from the inputs in training (fusion search: 0.3)",
    )
    parser.add_argument("--schedule", default="constant", choices=("constant", "cosine"))
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
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d=args.d,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        modality_dropout=args.modality_dropout,
        photo_lags=args.photo_lags,
        photo_views=args.photo_views,
        head=args.head,
        fusion=args.fusion,
        input_dropout=args.input_dropout,
        schedule=args.schedule,
        device=args.device,
    )
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
