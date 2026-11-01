"""Two automatic leakage checks of the plan ("Cross-validation and evaluation protocol").

* ``context``: a classifier on the context text alone must stay at chance for the label. The
  context sentence is a function of the operating condition, so the strongest classifier on
  it is a lookup table: per distinct sentence, the training units' majority class (or mean
  target), the training majority (mean) for a sentence never seen in training. It is scored
  on the test bags of the outer folds against the training majority (mean): the pooled
  difference with a 95 % interval from resampling test units. A lower bound above zero (an
  RMSE upper bound below zero) means the context carries label information, and the context
  token must then stay out of the inputs or be ablated in every table that uses it.
* ``dataset``: how well features tell which source dataset a window comes from. Logistic
  regression on each window's features averaged over its channels (datasets with different
  channel counts share one space), 5-fold cross-validation grouped by unit, balanced
  accuracy against chance (1 / datasets). Run on encoder features and on the level
  descriptors, absolute and relative to the healthy baseline. Different machines are
  expected to be identifiable; the numbers matter for pretraining (contrastive negatives come
  from the same dataset, so identity cannot solve the task) and are the reference the
  pretrained representation is compared with later. Held-out targets are never read.

    python -m pdmfm.leakage context --datasets nonastreda matwi xjtu_sy --run-id leak
    python -m pdmfm.leakage dataset --features mantis:F stats:F levels:absolute levels:relative \\
        --datasets cwru xjtu_sy nonastreda matwi --run-id leak
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm import planb, protocols
from pdmfm import stats as S
from pdmfm.features import FeatureStore
from pdmfm.model.inputs import REGRESSION_TARGET, relative_levels
from pdmfm.paths import REPO_ROOT
from pdmfm.utils import write_json_atomic
from pdmfm.windows import WindowStore

SOURCES = ("cwru", "xjtu_sy", "nonastreda", "matwi", "paderborn")
PER_DATASET = 1500
REPLICATES = 2000


# ----------------------------------------------------------------------------- context


def _mode(values) -> str:
    """The most frequent value (the first in sorted order on a tie)."""
    names, counts = np.unique(np.asarray(values, dtype=str), return_counts=True)
    return str(names[int(np.argmax(counts))])


def context_check(dataset: str, window_root: Path | None = None, n: int = REPLICATES) -> dict:
    """The context lookup against the training majority (mean) on the outer folds' test bags."""
    if dataset not in planb.FOLDS:
        raise ValueError(f"the context check runs on the source datasets: {sorted(planb.FOLDS)}")
    manifest = WindowStore.open(dataset, window_root).manifest
    first = manifest.groupby("bag", sort=False).head(1)
    unit = first["unit"].astype(str).to_numpy()
    context = first["context"].astype(str).to_numpy()
    label = first["class_key"].astype(str).to_numpy()
    target = REGRESSION_TARGET.get(dataset)
    value = first[target].to_numpy(dtype=float) if target and target in first else None
    has_class = bool((label != "").any())
    records = []
    for fold in range(1, protocols.N_FOLDS + 1):
        split = planb.FOLDS[dataset](manifest, fold)
        train = np.isin(unit, sorted(split.units(manifest, "train")))
        test = np.flatnonzero(np.isin(unit, sorted(split.units(manifest, "test"))))
        record = {"unit": unit[test], "fold": np.full(len(test), fold)}
        if has_class:
            labelled = train & (label != "")
            majority = _mode(label[labelled])
            lookup = {c: _mode(label[labelled & (context == c)])
                      for c in np.unique(context[labelled])}  # fmt: skip
            record["y_class"] = label[test]
            record["context_class"] = np.array([lookup.get(c, majority) for c in context[test]])
            record["majority"] = np.full(len(test), majority)
        if value is not None:
            known = train & np.isfinite(value)
            mean = float(value[known].mean())
            lookup = {c: float(value[known & (context == c)].mean())
                      for c in np.unique(context[known])}  # fmt: skip
            record["y_reg"] = value[test]
            record["context_reg"] = np.array([lookup.get(c, mean) for c in context[test]])
            record["train_mean"] = np.full(len(test), mean)
        records.append(pd.DataFrame(record))
    table = pd.concat(records, ignore_index=True)
    out: dict = {"dataset": dataset, "contexts": int(len(np.unique(context))),
                 "bags": int(len(table)), "units": len(set(table["unit"]))}  # fmt: skip
    if has_class:
        labelled = table.loc[table["y_class"] != ""].copy()
        classes = {c: i for i, c in enumerate(sorted(set(labelled["y_class"])))}
        for column in ("y_class", "context_class", "majority"):
            labelled[column] = [classes.get(c, -1) for c in labelled[column]]
        out["classes"] = _compare(
            pd.DataFrame(labelled), "y_class", "context_class", "majority", S.macro_f1, n
        )
        out["classes"]["informative"] = bool(out["classes"]["low"] > 0)
    if value is not None:
        known = table.loc[np.isfinite(table["y_reg"].to_numpy(dtype=float))]
        out["regression"] = _compare(
            pd.DataFrame(known), "y_reg", "context_reg", "train_mean", S.rmse, n
        )
        out["regression"]["target"] = target
        out["regression"]["informative"] = bool(out["regression"]["high"] < 0)
    return out


def _compare(table: pd.DataFrame, y: str, context: str, reference: str, metric, n: int) -> dict:
    """Pooled metric of the context lookup and of the reference, and their difference with a
    95 % interval from resampling test units."""
    columns = {c: table[c].to_numpy() for c in table.columns}
    estimate, low, high = S.unit_bootstrap(
        [table], lambda f: metric(f[y], f[context]) - metric(f[y], f[reference]), n=n
    )
    return {
        "context": float(metric(columns[y], columns[context])),
        "reference": float(metric(columns[y], columns[reference])),
        "difference": estimate,
        "low": low,
        "high": high,
    }


# ----------------------------------------------------------------------------- dataset identity


def _rows(store: WindowStore, per_dataset: int, seed: int) -> np.ndarray:
    if len(store) <= per_dataset:
        return np.arange(len(store))
    chosen = protocols.subsample(store.manifest, per_dataset / len(store), seed)
    return np.flatnonzero(chosen)


def window_vectors(store: WindowStore, feature: str, rows: np.ndarray, feature_store) -> np.ndarray:
    """(len(rows), D): the feature averaged over tokens and channels. ``feature`` is
    ``encoder:ARM``, ``levels:absolute`` or ``levels:relative``."""
    kind, _, detail = feature.partition(":")
    if kind == "levels":
        if detail == "absolute":
            levels = np.asarray(store.levels, dtype=np.float32)
        elif detail == "relative":
            units = None
            if store.dataset == "paderborn":  # its baseline is made of training bearings
                units = sorted({b for f in protocols.PADERBORN_SOURCE_FOLDS for b in f})
            levels = relative_levels(store, units)
        else:
            raise ValueError("levels:absolute or levels:relative")
        return np.nanmean(levels[rows], axis=1)
    cached = feature_store.find(store, kind, detail)
    if not cached.done[rows].all():
        raise FileNotFoundError(f"{feature} is not cached for every window of {store.dataset}")
    block = np.asarray(cached.features[rows], dtype=np.float32)  # (n, C, K, D)
    return block.mean(axis=(1, 2))


def dataset_probe(
    datasets: list[str],
    feature: str,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    per_dataset: int = PER_DATASET,
    seed: int = 0,
) -> dict:
    """Balanced accuracy of telling the datasets apart from ``feature``, grouped 5-fold by unit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_store = FeatureStore(feature_root)
    blocks, labels, groups, used, skipped = [], [], [], [], {}
    for dataset in datasets:
        if dataset in protocols.TARGETS:
            raise ValueError(f"{dataset} is a held-out target; it is never read before stage 8")
        try:
            store = WindowStore.open(dataset, window_root)
            rows = _rows(store, per_dataset, seed)
            x = window_vectors(store, feature, rows, feature_store)
        except (FileNotFoundError, ValueError, NotImplementedError) as error:
            skipped[dataset] = str(error)
            continue
        blocks.append(x)
        labels += [dataset] * len(rows)
        groups += [f"{dataset}/{u}" for u in store.manifest["unit"].astype(str).to_numpy()[rows]]
        used.append(dataset)
    out: dict = {"feature": feature, "datasets": used, "skipped": skipped}
    if len(used) < 2:
        out["balanced_accuracy"] = float("nan")
        return out
    x = np.nan_to_num(np.concatenate(blocks))
    y, groups_ = np.array(labels), np.array(groups)
    predicted = np.empty(len(y), dtype=object)
    for train, test in GroupKFold(n_splits=5).split(x, y, groups_):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(x[train], y[train])
        predicted[test] = model.predict(x[test])
    out.update(
        windows=int(len(y)), chance=1.0 / len(used),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted.astype(str))),
    )  # fmt: skip
    return out


# ----------------------------------------------------------------------------- the run


def _markdown(run_id: str, context: list[dict], identity: list[dict]) -> str:
    lines = [f"# Leakage checks: {run_id}", ""]
    if context:
        lines += [
            "## Context text alone (lookup per context sentence, outer-fold test bags)",
            "",
            "| Dataset | Contexts | Classes: context / majority macro-F1, difference [95 %] | "
            "Target: context / training-mean RMSE, difference [95 %] | Verdict |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in context:
            cls, reg = c.get("classes"), c.get("regression")
            cells = [c["dataset"], str(c["contexts"])]
            cells.append(
                f"{cls['context']:.3f} / {cls['reference']:.3f}, "
                f"{cls['difference']:+.3f} [{cls['low']:+.3f}, {cls['high']:+.3f}]"
                if cls else "-"
            )  # fmt: skip
            cells.append(
                f"{reg['context']:.3f} / {reg['reference']:.3f}, "
                f"{reg['difference']:+.3f} [{reg['low']:+.3f}, {reg['high']:+.3f}]"
                if reg else "-"
            )  # fmt: skip
            informative = (cls or {}).get("informative") or (reg or {}).get("informative")
            cells.append("**informative**" if informative else "at chance")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if identity:
        lines += [
            "## Which dataset? (features averaged over channels, 5-fold grouped by unit)",
            "",
            "| Feature | Datasets | Windows | Balanced accuracy | Chance |",
            "| --- | --- | --- | --- | --- |",
        ]
        for d in identity:
            lines.append(
                f"| {d['feature']} | {', '.join(d['datasets'])} | {d.get('windows', 0)} | "
                f"{d['balanced_accuracy']:.3f} | {d.get('chance', float('nan')):.3f} |"
            )
        for d in identity:
            for name, reason in d["skipped"].items():
                lines.append(f"- {d['feature']}: {name} skipped ({reason})")
        lines.append("")
    return "\n".join(lines)


def run(
    run_id: str,
    context_datasets: list[str],
    identity_datasets: list[str],
    features: list[str],
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
) -> dict:
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "leakage") / run_id
    out.mkdir(parents=True, exist_ok=True)
    context = []
    for dataset in context_datasets:
        try:
            context.append(context_check(dataset, window_root))
        except FileNotFoundError as error:
            log(f"[leakage] context {dataset}: skipped ({error})")
    identity = [
        dataset_probe(identity_datasets, f, window_root, feature_root) for f in features
    ] if identity_datasets else []  # fmt: skip
    result = {"context": context, "dataset": identity,
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    write_json_atomic(out / "leakage.json", result)
    text = _markdown(run_id, context, identity)
    (out / "summary.md").write_text(text, encoding="utf-8")
    log(text)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.leakage", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("checks", nargs="+", choices=("context", "dataset"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(SOURCES))
    parser.add_argument(
        "--features", nargs="+", default=["mantis:F", "stats:F", "levels:absolute",
                                          "levels:relative"],
    )  # fmt: skip
    parser.add_argument("--out", type=Path, help="default: runs/leakage/")
    args = parser.parse_args(argv)
    context = [d for d in args.datasets if d in planb.FOLDS] if "context" in args.checks else []
    identity = args.datasets if "dataset" in args.checks else []
    run(args.run_id, context, identity, args.features if identity else [], args.out)


if __name__ == "__main__":
    main()
