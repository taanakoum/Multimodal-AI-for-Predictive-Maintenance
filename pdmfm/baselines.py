"""Supervised baselines on the outer folds (plan, "Baselines in every table"), run and scored
like Plan B so that :mod:`pdmfm.report` can test them against it bag for bag.

* ``gbm``: gradient boosting (scikit-learn ``HistGradientBoosting``) on handcrafted features:
  the ``stats`` encoder of every channel's F arm (and E for bearings: band means and maxima,
  mean, std, min and max of the log spectrum), the four level descriptors relative to the
  healthy baseline (as Plan B; none for Nonastreda's Fz), the slow channels, and for bearings
  the prominence of the defect lines in the envelope spectrum (BPFO, BPFI, BSF, FTF;
  k = 1..3).
* ``probe``: the Gate-1 linear probe (logistic or ridge regression on mean-pooled frozen
  features of one encoder and its arms) scored on the test units: the plan's "MOMENT and
  Mantis linear probes".
* ``patchtst``: PatchTST trained from scratch, supervised, on the T-arm segments of every
  channel (the architecture of :mod:`pdmfm.patchtst`; segment embeddings averaged per
  channel, channels concatenated, one linear head per task); random segments per window in
  training, all of them in evaluation. A GPU job on Ruche; a CPU run is a smoke test.

Every baseline is sensor-only. Hyperparameters (gbm, probe) or the epoch (patchtst) are
chosen on the fold's inner-validation units; test units are scored once. Predictions are
made per window, then per bag (mean probabilities, mean regression), next to the majority,
persistence and schedule columns of :func:`pdmfm.planb.baseline_table`. Results go to
``runs/baselines/<run-id>/`` in Plan B's layout (``fold<k>-seed<s>.json``, ``-bags.csv``,
``summary.md`` with pooled intervals).

    python -m pdmfm.baselines --method gbm --dataset nonastreda --run-id gbm-nona
    python -m pdmfm.baselines --method probe --encoder mantis --arms F --dataset nonastreda \\
        --run-id probe-nona-mantis-F
    python -m pdmfm.baselines --method patchtst --dataset xjtu_sy --device cuda --run-id pt-xjtu
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm import arms, planb, protocols
from pdmfm import metrics as M
from pdmfm import stats as S
from pdmfm.model.inputs import FOLD_BASELINE, TaskInputs, assemble
from pdmfm.paths import REPO_ROOT
from pdmfm.train import bag_mean
from pdmfm.utils import stable_hash, write_json_atomic
from pdmfm.windows import WindowStore

METHODS = ("gbm", "probe", "patchtst")
GBM_GRID = tuple(
    {"learning_rate": lr, "max_leaf_nodes": leaves} for lr in (0.05, 0.1) for leaves in (15, 31)
)
GBM_ITERATIONS = 200
LINE_KINDS = ("BPFO", "BPFI", "BSF", "FTF")
LINE_HARMONICS = 3
SENSOR_ROLES = ("T", "F", "E", "S")


# ----------------------------------------------------------------------------- features


def line_prominence(spectrum: np.ndarray, order: float, max_order: float = 25.0) -> float:
    """Log amplitude of the strongest bin within ±2 % (at least ±1 bin) of ``order``, minus
    the median within ±1 order: how far a line stands out of the envelope floor."""
    n = len(spectrum)
    width = max_order / n
    centre = order / width
    if centre >= n - 1:
        return float("nan")
    half = max(1.0, 0.02 * centre)
    peak = spectrum[int(max(0, np.floor(centre - half))) : int(np.ceil(centre + half)) + 1].max()
    floor = np.median(spectrum[int(max(0, centre - 1 / width)) : int(centre + 1 / width) + 1])
    return float(peak - floor)


_LINES: dict[str, np.ndarray] = {}


def defect_line_features(store: WindowStore) -> np.ndarray | None:
    """(N, vibration channels x 4 lines x 3 harmonics) prominences in each window's envelope
    order spectrum, or None for a dataset without bearings. Computed once per store and
    process (Paderborn assembles its inputs per fold)."""
    from pdmfm.explain import geometry

    if "E" in arms.unavailable(store):
        return None
    if store.key in _LINES:
        return _LINES[store.key]
    g = geometry(store.dataset)
    from pdmfm import preprocess as pp

    rates = pp.defect_frequencies(g["n_balls"], g["ball_d"], g["pitch_d"], g["contact_deg"], 1.0)
    out = []
    for begin in range(0, len(store), 512):
        rows = np.arange(begin, min(begin + 512, len(store)))
        spectra = arms.arm_input(store, rows, "E")[:, :, 0]  # (n, C, 512)
        block = [
            [line_prominence(s, k * rates[kind]) for kind in LINE_KINDS
             for k in range(1, LINE_HARMONICS + 1)]
            for window in spectra for s in window
        ]  # fmt: skip
        out.append(np.asarray(block, dtype=np.float32).reshape(len(rows), -1))
    _LINES[store.key] = np.concatenate(out)
    return _LINES[store.key]


def handcrafted_features(inputs: TaskInputs, store: WindowStore) -> np.ndarray:
    """(N, F) for ``gbm``: stats tokens, relative levels, slow channels (NaN where absent),
    defect-line prominences."""
    blocks = []
    for modality in inputs.modalities.values():
        n = len(modality.tokens)
        tokens = modality.tokens.reshape(n, -1).astype(np.float32)
        if modality.token_present is not None:
            absent = np.repeat(~modality.token_present, modality.tokens.shape[2], axis=1)
            tokens = np.where(absent, np.nan, tokens)
        blocks.append(tokens)
        if modality.levels is not None:
            blocks.append(modality.levels.reshape(n, -1))
    lines = defect_line_features(store)
    if lines is not None:
        blocks.append(lines)
    return np.concatenate(blocks, axis=1)


def pooled_features(inputs: TaskInputs) -> np.ndarray:
    """(N, F) for ``probe``: each sensor channel's tokens of each arm averaged (Gate 1)."""
    blocks = []
    for modality in inputs.modalities.values():
        if modality.descriptor:
            continue
        for role in sorted(set(modality.roles.tolist())):
            for channel in sorted(set(modality.channels[modality.roles == role].tolist())):
                members = (modality.roles == role) & (modality.channels == channel)
                blocks.append(modality.tokens[:, members].mean(axis=1))
    return np.concatenate(blocks, axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- scoring


def _full_probs(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba with a column for every class (zero for classes absent in training)."""
    probs = np.zeros((len(x), n_classes))
    probs[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return probs


def _bags(inputs: TaskInputs, rows: np.ndarray, values: np.ndarray):
    bags = inputs.manifest["bag"].astype(str).to_numpy()[rows]
    return bag_mean(np.asarray(values, dtype=float), bags)


def bag_macro_f1(inputs: TaskInputs, rows: np.ndarray, probs: np.ndarray) -> float:
    assert inputs.y_class is not None
    _, p = _bags(inputs, rows, probs)
    _, y = _bags(inputs, rows, inputs.y_class[rows].astype(float))
    return S.macro_f1(y.round().astype(int), p.argmax(1))


def bag_rmse(inputs: TaskInputs, rows: np.ndarray, predicted: np.ndarray) -> float:
    assert inputs.y_reg is not None
    _, p = _bags(inputs, rows, predicted)
    _, y = _bags(inputs, rows, inputs.y_reg[rows])
    return S.rmse(y, p)


def point_regression(y: np.ndarray, predicted: np.ndarray, rul: bool) -> dict:
    """RMSE and MAE of point predictions; RUL adds the PHM 2012 score and the same after the
    first prediction time (as :func:`pdmfm.planb.score_regression`)."""
    error = np.asarray(predicted, dtype=float) - np.asarray(y, dtype=float)
    out = {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
           "n": int(len(y))}  # fmt: skip
    if rul:
        out["phm2012"] = M.phm2012_score(y, np.clip(predicted, 0.0, 1.0))
        post = y < 1.0 - 1e-6
        if post.any():
            out["post_fpt"] = point_regression(y[post], predicted[post], False)
            out["post_fpt"]["phm2012"] = M.phm2012_score(y[post], np.clip(predicted[post], 0, 1))
    return out


def evaluate(inputs: TaskInputs, rows: np.ndarray, probs, predicted) -> dict:
    """Plan B's result layout: ``combined`` (classes) and ``regression.combined`` (targets),
    per window and per bag."""
    out: dict = {}
    ordinal = inputs.dataset in planb.ORDINAL
    if probs is not None:
        assert inputs.y_class is not None
        y = inputs.y_class[rows]
        _, bag_probs = _bags(inputs, rows, probs)
        _, bag_y = _bags(inputs, rows, y.astype(float))
        out["combined"] = {
            "window": M.classification(y, probs, ordinal),
            "bag": M.classification(bag_y.round().astype(int), bag_probs, ordinal),
        }
    if predicted is not None:
        assert inputs.y_reg is not None
        y = inputs.y_reg[rows]
        keep = np.isfinite(y)
        rul = inputs.regression_target == "rul"
        _, bag_pred = _bags(inputs, rows[keep], predicted[keep])
        _, bag_y = _bags(inputs, rows[keep], y[keep])
        out["regression"] = {
            "combined": {
                "window": point_regression(y[keep], predicted[keep], rul),
                "bag": point_regression(bag_y, bag_pred, rul),
            }
        }
    return out


def bag_predictions(inputs: TaskInputs, rows: np.ndarray, probs, predicted) -> pd.DataFrame:
    """Per test bag: the class (argmax of the mean probabilities; ordinal median for
    ordered classes) and the mean regression."""
    bags = inputs.manifest["bag"].astype(str).to_numpy()[rows]
    table = pd.DataFrame({"bag": pd.unique(bags)})
    if probs is not None:
        keys, mean = bag_mean(np.asarray(probs, dtype=float), bags)
        table["pred_class"] = pd.Series(mean.argmax(1), index=keys).loc[table["bag"]].to_numpy()
        if inputs.dataset in planb.ORDINAL:
            median = pd.Series(M.ordinal_median(mean), index=keys)
            table["pred_class_ordinal"] = median.loc[table["bag"]].to_numpy()
    if predicted is not None:
        keys, mean = bag_mean(np.asarray(predicted, dtype=float), bags)
        table["pred_reg"] = pd.Series(mean, index=keys).loc[table["bag"]].to_numpy()
    return table


# ----------------------------------------------------------------------------- methods


def _gbm(kind: str, params: dict, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingClassifier if kind == "class" else HistGradientBoostingRegressor
    return cls(max_iter=GBM_ITERATIONS, l2_regularization=1.0, early_stopping=False,  # pyright: ignore[reportArgumentType]
               random_state=seed, **params)  # fmt: skip


def _linear(kind: str, strength: float):
    from pdmfm.probe import _model

    return _model("classification" if kind == "class" else "regression", strength)


def fit_tabular(method: str, inputs: TaskInputs, x: np.ndarray, rows: dict, seed: int) -> dict:
    """Fit gbm or probe models per task, choosing the setting on the validation bags.
    Returns per task: the chosen setting, its validation score and the test predictions."""
    from pdmfm.probe import ALPHA_GRID, C_GRID

    out = {}
    tasks = []
    if inputs.y_class is not None:
        tasks.append("class")
    if inputs.y_reg is not None:
        tasks.append("reg")
    for kind in tasks:
        y = inputs.y_class if kind == "class" else inputs.y_reg
        assert y is not None
        train = rows["train"][
            (y[rows["train"]] >= 0) if kind == "class" else np.isfinite(y[rows["train"]])
        ]
        if method == "gbm":
            grid = [("gbm", p) for p in GBM_GRID]
        else:
            grid = [("linear", s) for s in (C_GRID if kind == "class" else ALPHA_GRID)]
        best = None
        for family, setting in grid:
            model = _gbm(kind, setting, seed) if family == "gbm" else _linear(kind, setting)  # pyright: ignore[reportArgumentType]
            model.fit(x[train], y[train])
            if kind == "class":
                probs = _full_probs(model, x[rows["val"]], len(inputs.classes))
                score = bag_macro_f1(inputs, rows["val"], probs)
            else:
                val = rows["val_reg"]
                score = -bag_rmse(inputs, val, np.asarray(model.predict(x[val]), dtype=float))
            if best is None or score > best[0]:
                best = (score, setting, model)
        assert best is not None
        score, setting, model = best
        if kind == "class":
            test = _full_probs(model, x[rows["test"]], len(inputs.classes))
        else:
            test = model.predict(x[rows["test"]])
        out[kind] = {"setting": setting, "validation": float(score), "test": test}
    return out


# ----------------------------------------------------------------------------- PatchTST


def segment_patchtst(n_channels: int, n_classes: int, regression: bool, overrides=None):
    """PatchTST from scratch over the T-arm segments: (B, C, S, 512) -> segment embeddings
    averaged per channel -> channels concatenated -> (class logits, regression output)."""
    import torch
    from transformers import PatchTSTConfig, PatchTSTModel  # pyright: ignore[reportMissingImports]

    from pdmfm import patchtst

    config = {**patchtst.MODEL, "do_mask_input": False, **(overrides or {})}

    class SegmentPatchTST(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = PatchTSTModel(PatchTSTConfig(**config))
            width = config["d_model"] * n_channels
            self.classes = torch.nn.Linear(width, n_classes) if n_classes else None
            self.target = torch.nn.Linear(width, 1) if regression else None

        def forward(self, x):
            b, c, s, length = x.shape
            hidden = self.encoder(past_values=x.reshape(b * c * s, length, 1)).last_hidden_state
            z = hidden.mean(dim=(1, 2)).reshape(b, c, s, -1).mean(dim=2).reshape(b, -1)
            return (
                None if self.classes is None else self.classes(z),
                None if self.target is None else self.target(z).squeeze(-1),
            )

    return SegmentPatchTST()


def fit_patchtst(inputs: TaskInputs, store: WindowStore, rows: dict, seed: int, cfg: dict,
                 log=print) -> dict:  # fmt: skip
    """Train on the T-arm segments; keep the epoch with the best validation score (bag
    macro-F1 and 1 - RMSE / std, as Plan B's selection score)."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    n_classes = len(inputs.classes) if inputs.y_class is not None else 0
    regression = inputs.y_reg is not None
    model = segment_patchtst(len(store.channels), n_classes, regression).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    y_class = None if inputs.y_class is None else torch.as_tensor(inputs.y_class)
    y_reg = None
    y_mean, y_std = 0.0, 1.0
    if inputs.y_reg is not None:
        known = inputs.y_reg[rows["train"]]
        known = known[np.isfinite(known)]
        y_mean, y_std = float(known.mean()), float(known.std()) or 1.0
        y_reg = torch.as_tensor((inputs.y_reg - y_mean) / y_std, dtype=torch.float32)

    def segments(index: np.ndarray, pick: int | None) -> torch.Tensor:
        x = arms.arm_input(store, index, "T")  # (n, C, S, 512)
        if pick is not None and pick < x.shape[2]:
            chosen = np.stack([rng.choice(x.shape[2], pick, replace=False) for _ in index])
            x = np.take_along_axis(x, chosen[:, None, :, None], axis=2)
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    @torch.no_grad()
    def predict(index: np.ndarray):
        model.eval()
        probs, values = [], []
        for begin in range(0, len(index), cfg["batch_size"]):
            logits, value = model(segments(index[begin : begin + cfg["batch_size"]], None))
            if logits is not None:
                probs.append(torch.softmax(logits, -1).cpu().numpy())
            if value is not None:
                values.append(value.cpu().numpy() * y_std + y_mean)
        return (np.concatenate(probs) if probs else None,
                np.concatenate(values) if values else None)  # fmt: skip

    def score(index: np.ndarray, probs, values) -> float:
        parts = []
        if probs is not None:
            parts.append(bag_macro_f1(inputs, index, probs))
        if values is not None:
            keep = np.isfinite(inputs.y_reg[index])  # pyright: ignore[reportOptionalSubscript]
            parts.append(1.0 - bag_rmse(inputs, index[keep], values[keep]) / y_std)
        return float(np.mean(parts))

    best, best_state, best_epoch, history = -np.inf, None, 0, []
    started = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(rows["train"])
        running = 0.0
        for begin in range(0, len(order), cfg["batch_size"]):
            index = order[begin : begin + cfg["batch_size"]]
            logits, value = model(segments(index, cfg["train_segments"]))
            loss = torch.zeros((), device=device)
            if logits is not None and y_class is not None:
                target = y_class[index].to(device)
                keep = target >= 0
                if keep.any():
                    loss = loss + torch.nn.functional.cross_entropy(logits[keep], target[keep])
            if value is not None and y_reg is not None:
                target = y_reg[index].to(device)
                keep = torch.isfinite(target)
                if keep.any():
                    loss = loss + torch.nn.functional.mse_loss(value[keep], target[keep])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(index)
        val_score = score(rows["val"], *predict(rows["val"]))
        history.append({"epoch": epoch + 1, "loss": running / len(order), "val_score": val_score})
        log(f"[baselines] patchtst epoch {epoch + 1}: loss {history[-1]['loss']:.4f}, "
            f"validation {val_score:.3f}")  # fmt: skip
        if val_score > best:
            best, best_epoch, best_state = val_score, epoch + 1, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    probs, values = predict(rows["test"])
    out: dict = {"history": history, "best_epoch": best_epoch,
                 "seconds": time.perf_counter() - started}  # fmt: skip
    if probs is not None:
        out["class"] = {"setting": {"epoch": best_epoch}, "validation": best, "test": probs}
    if values is not None:
        out["reg"] = {"setting": {"epoch": best_epoch}, "validation": best, "test": values}
    return out


# ----------------------------------------------------------------------------- runs


def run(
    method: str,
    dataset: str,
    run_id: str,
    folds: list[int],
    seeds: list[int],
    encoder: str = "stats",
    arm_list: tuple[str, ...] = ("F", "E"),
    patchtst_cfg: dict | None = None,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if dataset not in planb.FOLDS:
        raise ValueError(f"baselines run on the source datasets: {sorted(planb.FOLDS)}")
    if method == "gbm":
        encoder = "stats"
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "baselines") / run_id
    out.mkdir(parents=True, exist_ok=True)
    store = WindowStore.open(dataset, window_root)
    manifest = store.manifest
    cfg = {"epochs": 30, "batch_size": 64, "lr": 3e-4, "train_segments": 4, "device": "cpu",
           **(patchtst_cfg or {})}  # fmt: skip
    settings: dict = {"method": method, "dataset": dataset, "window_store": store.key}
    if method == "gbm":
        settings.update(grid=list(GBM_GRID), iterations=GBM_ITERATIONS, arms=list(arm_list),
                        lines=list(LINE_KINDS))  # fmt: skip
    elif method == "probe":
        settings.update(encoder=encoder, arms=list(arm_list))
    else:
        settings.update(patchtst={k: v for k, v in cfg.items() if k != "device"})
    key = stable_hash(settings)[:12]
    per_fold = dataset in FOLD_BASELINE or "{fold}" in encoder
    prepared = None
    for fold in folds:
        split = planb.FOLDS[dataset](manifest, fold)
        split.assert_unit_disjoint(manifest)
        if prepared is None or per_fold:
            train_units = sorted(split.units(manifest, "train"))
            inputs = assemble(
                dataset, encoder.replace("{fold}", str(fold)), arm_list, photos=False,
                window_root=window_root, feature_root=feature_root, train_units=train_units,
            )  # fmt: skip
            present = np.any([m.present for m in inputs.modalities.values()], axis=0)
            if method == "gbm":
                x = handcrafted_features(inputs, store)
            elif method == "probe":
                x = pooled_features(inputs)
            else:
                x = None
            prepared = (inputs, present, x)
        inputs, present, x = prepared
        labelled = present.copy()
        if inputs.y_class is not None:
            labelled &= inputs.y_class >= 0
        elif inputs.y_reg is not None:
            labelled &= np.isfinite(inputs.y_reg)
        rows = {
            part: np.flatnonzero(getattr(split, part) & labelled)
            for part in ("train", "val", "test")
        }
        val_reg = rows["val"]
        if inputs.y_reg is not None:
            val_reg = val_reg[np.isfinite(inputs.y_reg[val_reg])]
        rows["val_reg"] = val_reg
        for seed in seeds:
            path = out / f"fold{fold}-seed{seed}.json"
            if path.is_file():
                if json.loads(path.read_text(encoding="utf-8")).get("settings_key") == key:
                    log(f"[baselines] {method} fold {fold} seed {seed}: done earlier, skipped")
                    continue
                raise ValueError(f"{path} holds a run with other settings; use another --run-id")
            started = time.perf_counter()
            if method == "patchtst":
                fitted = fit_patchtst(inputs, store, rows, seed, cfg, log)
            else:
                assert x is not None
                fitted = fit_tabular(method, inputs, x, rows, seed)
            probs = fitted["class"]["test"] if "class" in fitted else None
            predicted = fitted["reg"]["test"] if "reg" in fitted else None
            result = {
                "dataset": dataset, "fold": fold, "seed": seed, "method": method,
                "settings_key": key, "settings": settings, "split": split.name,
                "selected": {k: v["setting"] for k, v in fitted.items() if k in ("class", "reg")},
                "validation_score": {k: v["validation"] for k, v in fitted.items()
                                     if k in ("class", "reg")},
                "seconds": round(time.perf_counter() - started, 1),
                "test": evaluate(inputs, rows["test"], probs, predicted),
                "baselines": planb.baselines(inputs, rows["train"], rows["test"]),
                "temperature": None,
            }  # fmt: skip
            if "history" in fitted:
                result["history"] = fitted["history"]
                result["best_epoch"] = fitted["best_epoch"]
            bags = planb.baseline_table(inputs, rows["train"], rows["test"]).merge(
                bag_predictions(inputs, rows["test"], probs, predicted), on="bag"
            )
            bags.insert(0, "fold", fold)
            bags.to_csv(out / f"fold{fold}-seed{seed}-bags.csv", index=False)
            write_json_atomic(path, result)
            log(f"[baselines] {method} {dataset} fold {fold} seed {seed}: "
                f"{result['selected']} in {result['seconds']:.0f} s")  # fmt: skip
    title = {"gbm": "Gradient boosting on handcrafted features",
             "probe": f"Linear probe on {encoder} {'+'.join(arm_list)}",
             "patchtst": "PatchTST from scratch"}[method]  # fmt: skip
    planb.summarize(out, title=title)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.baselines", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=sorted(planb.FOLDS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--encoder", default="mantis", help="probe only: the cached encoder")
    parser.add_argument("--arms", nargs="+", default=["F", "E"])
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(range(1, protocols.N_FOLDS + 1))
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="default: 0 (gbm, probe), 0 1 2")
    parser.add_argument("--epochs", type=int, default=30, help="patchtst only")
    parser.add_argument("--batch-size", type=int, default=64, help="patchtst only (windows)")
    parser.add_argument("--lr", type=float, default=3e-4, help="patchtst only")
    parser.add_argument("--train-segments", type=int, default=4, help="patchtst: per channel")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="default: runs/baselines/")
    parser.add_argument("--force", action="store_true", help="run on a busy machine (CPU)")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        claim_resources("baselines", min_idle_cpus=2.0, min_free_ram_gb=1.0, force=args.force)
    seeds = args.seeds or ([0, 1, 2] if args.method == "patchtst" else [0])
    cfg = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
           "train_segments": args.train_segments, "device": args.device}  # fmt: skip
    out = run(
        args.method, args.dataset, args.run_id, args.folds, seeds, args.encoder,
        tuple(args.arms), cfg, args.out,
    )  # fmt: skip
    print(f"[baselines] done -> {out}")


if __name__ == "__main__":
    main()
