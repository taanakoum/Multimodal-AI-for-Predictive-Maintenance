"""Anomaly detection from healthy data only (plan, "Anomaly detection, TimesFM and
AnomalyGPT"; pipeline stage 10).

Only healthy windows of the machine are known (the support set). Every score measures how far
a window is from them; no fault label enters any score.

Scores, on cached features:

* ``knn:<encoder>:<arms>``: mean cosine distance to the k = 5 nearest support windows, on
  the cached features as they are (the plan's score).
* ``rms``: the largest z-score over channels of log10 RMS against the support: the RMS 3σ
  alarm.
* ``iforest``: an isolation forest fit on the support's handcrafted features (the ``stats``
  arms plus the four level descriptors of every channel).

Deferred, because they need later stages: the masked-modelling error (the pretrained fusion
stage), the zero-shot text score (a model in the text space) and MOMENT's reconstruction
error (its cache is built on Ruche).

Protocols:

* ``windows`` (CWRU now; KAIST and Paderborn real damage at stage 8): k_h ∈ {10, 50}
  healthy windows drawn as the support, 5 draws; the test set is every other healthy window
  and every faulty one. AUROC, AUPRC and the false-alarm rate at 95 % detection (the share
  of healthy test windows scoring above the 5th percentile of the faulty ones). A k_h that
  leaves fewer than 10 healthy test windows is skipped.
* ``onset`` (XJTU-SY, in-domain): per bearing, its first 10 snapshots are the support; the
  threshold is the mean + 3 std of the support's leave-one-out scores; an alarm needs 3
  consecutive snapshots above it. Scored against the labelled first prediction time (FPT):
  AUROC of post-FPT against pre-FPT snapshots, alarm minus FPT in snapshots (negative:
  earlier), and the share of pre-FPT snapshots above the threshold (false alarms). The FPT
  labels come from a 3σ rule on the horizontal channel's RMS (kurtosis as fallback), so
  ``rms`` is close to the label definition: read the other scores against it.

KAIST and Paderborn real damage are held-out targets: refused without ``final=True``
(``--final``), which only stage 8 may pass.

    python -m pdmfm.anomaly --dataset cwru --features mantis:F mantis:E stats:F+E --run-id an-cwru
    python -m pdmfm.anomaly --dataset xjtu_sy --features mantis:F stats:F+E --run-id an-xjtu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm import arms, protocols
from pdmfm.features import FeatureStore
from pdmfm.paths import REPO_ROOT
from pdmfm.windows import WindowStore

K_NEIGHBOURS = 5
SUPPORT_SIZES = (10, 50)
DRAWS = 5
MIN_HEALTHY_TEST = 10
ONSET_SUPPORT = 10
SIGMAS = 3.0
CONSECUTIVE = 3
TARGETS = ("kaist", "paderborn")  # held out until stage 8
PROTOCOL = {"cwru": "windows", "kaist": "windows", "paderborn": "windows", "xjtu_sy": "onset"}


# ----------------------------------------------------------------------------- scores


def _scale(support: np.ndarray):
    """(mean, std) of the support, each std floored at the median std."""
    mean = support.mean(axis=0)
    std = support.std(axis=0)
    floor = max(float(np.median(std)), 1e-8)
    return mean, np.maximum(std, floor)


def knn_scores(support: np.ndarray, test: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
    """Mean cosine distance of each test row to its k nearest support rows, on the features
    as they are. Centring on a 10-window support first made every vector nearly orthogonal
    (median XJTU-SY onset AUROC 0.48-0.78 against 0.90-0.996; PROJECT_LOG week 7)."""
    a = support / (np.linalg.norm(support, axis=1, keepdims=True) + 1e-12)
    b = test / (np.linalg.norm(test, axis=1, keepdims=True) + 1e-12)
    distance = 1.0 - b @ a.T
    k = min(k, len(support))
    return np.sort(distance, axis=1)[:, :k].mean(axis=1)


def rms_scores(support: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Largest z-score over channels of log10 RMS (support, test: (n, C))."""
    mean = support.mean(axis=0)
    std = np.maximum(support.std(axis=0, ddof=1) if len(support) > 1 else 0.0, 1e-6)
    return ((test - mean) / std).max(axis=1)


def iforest_scores(support: np.ndarray, test: np.ndarray, seed: int = 0) -> np.ndarray:
    """Isolation-forest anomaly score (higher: more anomalous), fit on the support."""
    from sklearn.ensemble import IsolationForest

    mean, std = _scale(support)
    forest = IsolationForest(n_estimators=200, random_state=seed).fit((support - mean) / std)
    return -forest.score_samples((test - mean) / std)


SCORERS = {"knn": knn_scores, "rms": rms_scores, "iforest": iforest_scores}


def leave_one_out(scorer, support: np.ndarray) -> np.ndarray:
    """Each support row scored against the others (the threshold of the onset protocol)."""
    return np.array(
        [scorer(np.delete(support, i, axis=0), support[i : i + 1])[0] for i in range(len(support))]
    )


# ----------------------------------------------------------------------------- metrics


def detection_metrics(scores: np.ndarray, anomalous: np.ndarray, detect: float = 0.95) -> dict:
    """AUROC, AUPRC (anomalies positive) and the false-alarm rate at ``detect`` detection."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores, anomalous = np.asarray(scores, dtype=float), np.asarray(anomalous, dtype=bool)
    if anomalous.all() or not anomalous.any():
        return {"auroc": np.nan, "auprc": np.nan, "far_at_95": np.nan}
    threshold = np.quantile(scores[anomalous], 1.0 - detect)
    return {
        "auroc": float(roc_auc_score(anomalous, scores)),
        "auprc": float(average_precision_score(anomalous, scores)),
        "far_at_95": float(np.mean(scores[~anomalous] >= threshold)),
    }


def first_alarm(above: np.ndarray, consecutive: int = CONSECUTIVE) -> int | None:
    """Index of the first of ``consecutive`` True values in a row, or None."""
    run = 0
    for i, flag in enumerate(np.asarray(above, dtype=bool)):
        run = run + 1 if flag else 0
        if run == consecutive:
            return i - consecutive + 1
    return None


# ----------------------------------------------------------------------------- features


def parse_feature(text: str) -> tuple[str, tuple[str, ...]]:
    encoder, _, arm_text = text.partition(":")
    if not encoder or not arm_text:
        raise ValueError(f"feature set {text!r} must look like encoder:ARM or encoder:ARM+ARM")
    return encoder, tuple(arm_text.split("+"))


def encoder_features(store: WindowStore, feature_store: FeatureStore, text: str):
    """(features (N, F) of the arms the store has, rows cached) for ``encoder:ARM+ARM``."""
    encoder, arm_list = parse_feature(text)
    kept = [a for a in arm_list if a not in arms.unavailable(store)]
    if not kept:
        raise ValueError(f"{store.dataset} has none of the arms {arm_list}")
    blocks, done = [], np.ones(len(store), dtype=bool)
    for arm in kept:
        cached = feature_store.find(store, encoder, arm)
        done &= cached.done
        blocks.append(cached.pooled(np.arange(len(store))))
    x = np.concatenate(blocks, axis=1)
    return x, done & np.isfinite(x).all(axis=1)


def handcrafted(store: WindowStore, feature_store: FeatureStore):
    """The ``stats`` arms (F, plus E for bearings) and every channel's level descriptors."""
    arm_text = "F+E" if "E" not in arms.unavailable(store) else "F"
    x, done = encoder_features(store, feature_store, f"stats:{arm_text}")
    levels = np.asarray(store.levels, dtype=np.float32).reshape(len(store), -1)
    return np.concatenate([x, levels], axis=1), done


def log_rms(store: WindowStore) -> np.ndarray:
    return np.asarray(store.levels[:, :, 0], dtype=float)  # (N, C) log10 RMS


def score_sets(store: WindowStore, feature_store: FeatureStore, feature_texts: list[str]):
    """name -> (scorer, features, cached rows): knn per feature set, rms, iforest."""
    sets = {}
    for text in feature_texts:
        x, done = encoder_features(store, feature_store, text)
        sets[f"knn:{text}"] = (knn_scores, x, done)
    sets["rms"] = (rms_scores, log_rms(store), np.ones(len(store), dtype=bool))
    x, done = handcrafted(store, feature_store)
    sets["iforest"] = (iforest_scores, x, done)
    return sets


# ----------------------------------------------------------------------------- protocols


def healthy_and_faulty(dataset: str, manifest: pd.DataFrame):
    """(healthy, faulty) row masks of a ``windows`` protocol dataset."""
    key = manifest["class_key"].astype(str)
    if dataset == "cwru":
        return (key == "normal").to_numpy(), (key != "normal").to_numpy()
    if dataset == "kaist":  # the 4 Nm recordings: normal against the 14 fault classes
        at4 = (manifest["load_nm"] == 4).to_numpy()
        return at4 & (key == "normal").to_numpy(), at4 & (key != "normal").to_numpy()
    if dataset == "paderborn":  # target bearings: held-out healthy against real damage
        unit = manifest["unit"].astype(str)
        real = set(protocols.PADERBORN_REAL) - set(protocols.PADERBORN_OUT_OF_SPACE)
        return (
            unit.isin(list(protocols.PADERBORN_TARGET_HEALTHY)).to_numpy(),
            unit.isin(sorted(real)).to_numpy(),
        )
    raise ValueError(f"no window protocol for {dataset}")


def window_protocol(store: WindowStore, sets: dict, seed: int = 0) -> pd.DataFrame:
    """One record per score, support size and draw."""
    healthy, faulty = healthy_and_faulty(store.dataset, store.manifest)
    records = []
    for name, (scorer, x, done) in sets.items():
        pool, bad = np.flatnonzero(healthy & done), np.flatnonzero(faulty & done)
        for k_h in SUPPORT_SIZES:
            if len(pool) - k_h < MIN_HEALTHY_TEST:
                continue
            for draw in range(DRAWS):
                rng = np.random.default_rng((seed, k_h, draw))
                support = rng.choice(pool, k_h, replace=False)
                test = np.concatenate([np.setdiff1d(pool, support), bad])
                anomalous = np.isin(test, bad)
                scores = scorer(x[support], x[test])
                records.append(
                    {
                        "score": name,
                        "k_h": k_h,
                        "draw": draw,
                        "healthy_test": int((~anomalous).sum()),
                        "faulty_test": int(anomalous.sum()),
                        **detection_metrics(scores, anomalous),
                    }
                )
    return pd.DataFrame(records)


def onset_protocol(store: WindowStore, sets: dict) -> pd.DataFrame:
    """One record per score and bearing (XJTU-SY): support = the first 10 snapshots."""
    manifest = store.manifest
    records = []
    for name, (scorer, x, done) in sets.items():
        for unit, group in manifest.loc[done].groupby("unit", sort=True):
            ordered = group.sort_values(by="meta_minute")
            rows = ordered.index.to_numpy()
            support, later = rows[:ONSET_SUPPORT], rows[ONSET_SUPPORT:]
            fpt = ordered["fpt"].iloc[0]
            fpt = None if not np.isfinite(fpt) else int(fpt)
            loo = leave_one_out(scorer, x[support])
            threshold = float(loo.mean() + SIGMAS * loo.std(ddof=1))
            scores = scorer(x[support], x[later])
            above = scores > threshold
            alarm = first_alarm(above)
            alarm_index = None if alarm is None else alarm + ONSET_SUPPORT
            record = {
                "score": name, "unit": unit, "snapshots": len(rows), "fpt": fpt,
                "alarm": alarm_index, "threshold": threshold,
            }  # fmt: skip
            if fpt is not None and fpt > ONSET_SUPPORT:
                post = np.arange(ONSET_SUPPORT, len(rows)) >= fpt
                record.update(detection_metrics(scores, post))
                record["false_alarm_rate"] = float(above[~post].mean())
                record["alarm_minus_fpt"] = None if alarm_index is None else alarm_index - fpt
            records.append(record)
    return pd.DataFrame(records)


def _numbers(table: pd.DataFrame, column: str) -> np.ndarray:
    """The finite values of a column (none if the column is absent)."""
    if column not in table:
        return np.zeros(0)
    values = np.array([float(v) if v is not None else np.nan for v in table[column]])
    return values[np.isfinite(values)]


def summarize(results: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if not len(results):
        return pd.DataFrame()
    if protocol == "windows":
        grouped = results.groupby(["score", "k_h"])[["auroc", "auprc", "far_at_95"]]
        spread = pd.DataFrame(grouped.std()).add_suffix("_std")
        return pd.DataFrame(grouped.mean()).join(spread).reset_index()
    rows = []
    for score, group in results.groupby("score"):
        scored = group.dropna(subset=["auroc"]) if "auroc" in group else group.iloc[:0]
        lead = _numbers(scored, "alarm_minus_fpt")
        auroc = _numbers(scored, "auroc")
        false_alarms = _numbers(scored, "false_alarm_rate")
        rows.append(
            {
                "score": score,
                "bearings": int(len(group)),
                "with_fpt": int(len(scored)),
                "auroc_median": float(np.median(auroc)) if len(auroc) else np.nan,
                "auroc_mean": float(np.mean(auroc)) if len(auroc) else np.nan,
                "alarmed": int(np.count_nonzero(group["alarm"].notna().to_numpy())),
                "alarm_minus_fpt_median": float(np.median(lead)) if len(lead) else np.nan,
                "false_alarm_rate_mean": float(np.mean(false_alarms))
                if len(false_alarms)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _markdown(run_id: str, dataset: str, protocol: str, summary: pd.DataFrame) -> str:
    lines = [f"# Anomaly detection: {run_id}", "", f"{dataset}, {protocol} protocol.", ""]
    if protocol == "windows":
        lines += [
            "Mean ± std over 5 support draws.",
            "",
            "| Score | k_h | AUROC | AUPRC | False alarms at 95 % detection |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in summary.to_dict("records"):
            lines.append(
                f"| {r['score']} | {r['k_h']} | {r['auroc']:.3f} ± {r['auroc_std']:.3f} | "
                f"{r['auprc']:.3f} ± {r['auprc_std']:.3f} | "
                f"{r['far_at_95']:.3f} ± {r['far_at_95_std']:.3f} |"
            )
    else:
        lines += [
            f"Support: each bearing's first {ONSET_SUPPORT} snapshots; alarm at {CONSECUTIVE} "
            f"consecutive snapshots above mean + {SIGMAS:g} std of the support's leave-one-out "
            "scores. The FPT labels come from an RMS 3σ rule on the horizontal channel.",
            "",
            "| Score | Bearings with FPT | AUROC (median / mean) | Alarmed | Alarm - FPT "
            "(median, snapshots) | False-alarm rate before FPT |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in summary.to_dict("records"):
            lines.append(
                f"| {r['score']} | {r['with_fpt']}/{r['bearings']} | {r['auroc_median']:.3f} / "
                f"{r['auroc_mean']:.3f} | {r['alarmed']} | {r['alarm_minus_fpt_median']:.1f} | "
                f"{r['false_alarm_rate_mean']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def run(
    dataset: str,
    feature_texts: list[str],
    run_id: str,
    final: bool = False,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    seed: int = 0,
    log=print,
) -> pd.DataFrame:
    if dataset not in PROTOCOL:
        raise ValueError(f"no anomaly protocol for {dataset}; choose from {sorted(PROTOCOL)}")
    if dataset in TARGETS and not final:
        raise ValueError(f"{dataset} is a held-out target: only stage 8 runs it (--final)")
    store = WindowStore.open(dataset, window_root)
    sets = score_sets(store, FeatureStore(feature_root), feature_texts)
    protocol = PROTOCOL[dataset]
    started = time.perf_counter()
    if protocol == "windows":
        results = window_protocol(store, sets, seed)
    else:
        results = onset_protocol(store, sets)
    summary = summarize(results, protocol)
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "anomaly") / run_id
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(_markdown(run_id, dataset, protocol, summary), encoding="utf-8")
    log(f"[anomaly] {dataset}: {len(results)} records in {time.perf_counter() - started:.0f} s")
    log(summary.to_string(index=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.anomaly", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--dataset", required=True, choices=sorted(PROTOCOL))
    parser.add_argument(
        "--features", nargs="+", default=["mantis:F"], help="encoder:ARM[+ARM] for the knn score"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--final", action="store_true", help="stage 8 only: run a target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default: runs/anomaly/")
    parser.add_argument("--force", action="store_true", help="run on a busy machine")
    args = parser.parse_args(argv)
    from pdmfm.resources import claim_resources

    claim_resources("anomaly", min_idle_cpus=1.0, min_free_ram_gb=1.0, force=args.force)
    run(args.dataset, args.features, args.run_id, args.final, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
