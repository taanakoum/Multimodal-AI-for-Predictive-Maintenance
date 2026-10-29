"""Stage 3 of the pipeline: Gate 1, the encoder probe (plan, "Gate 1").

A candidate is an encoder with one or more input arms (``mantis:T``, ``moment:T+E``).
Its cached features are averaged over tokens and concatenated over channels and arms,
standardized, and fed to a linear model: logistic regression (classes) or ridge regression
(RUL). The regularization strength is chosen by cross-validation grouped by unit inside
each training part; scores come from inner-validation units only. Test units are never
read.

Tasks (5 outer folds unless stated; fit on each fold's training units, score on its
inner-validation units):

* ``cwru``: 10 classes; fit on 0-2 hp, score on 3 hp; balanced accuracy.
* ``paderborn``: 3 classes, by bearing; balanced accuracy.
* ``xjtu_sy``: normalized RUL, by bearing; RMSE.
* ``nonastreda``: 3 wear classes from force only, by tool; balanced accuracy.

Rule: rank the candidates per task; the lowest mean rank wins; a tie goes to the cheaper
encoder. All candidates of a run are scored on the same rows (the windows every one of them
has cached), so a 10 % subsample of one candidate restricts the whole comparison to it.
Tasks whose windows or features are missing are skipped and reported. Arms a dataset lacks
are dropped from a candidate (``T+E`` is ``T`` on Nonastreda, which has no E arm); a
candidate left with no arm gets the worst rank on that task, so it cannot win by skipping
a task. An encoder name with
``{fold}`` (``patchtst-fold{fold}:T``) uses fold k's own model on fold k (fold 1 for CWRU,
whose 3 hp validation recordings no fold pretrains on).

    python -m pdmfm.probe --run-id gate1-cpu --candidates mantis:T mantis:F mantis:E mantis:T+E \\
        --references stats:F stats:E
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pdmfm import arms, protocols
from pdmfm.encoders import cost
from pdmfm.features import FeatureStore
from pdmfm.paths import REPO_ROOT
from pdmfm.windows import WindowStore

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ALPHA_GRID = (1e-1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    kind: str  # classification | regression
    target: str  # manifest column
    group: str  # manifest column that groups the hyperparameter cross-validation
    splits: Callable[[pd.DataFrame], list[protocols.Split]]

    @property
    def higher_is_better(self) -> bool:
        return self.kind == "classification"


TASKS = {
    "cwru": Task(
        "cwru", "cwru", "classification", "class_key", "load_hp",
        lambda m: [protocols.cwru_gate1(m)],
    ),
    "paderborn": Task(
        "paderborn", "paderborn", "classification", "class_key", "unit",
        lambda m: [protocols.paderborn_fold(m, k) for k in range(1, protocols.N_FOLDS + 1)],
    ),
    "xjtu_sy": Task(
        "xjtu_sy", "xjtu_sy", "regression", "rul", "unit",
        lambda m: [protocols.xjtu_fold(m, k) for k in range(1, protocols.N_FOLDS + 1)],
    ),
    "nonastreda": Task(
        "nonastreda", "nonastreda", "classification", "class_key", "unit",
        lambda m: [protocols.nonastreda_fold(m, k) for k in range(1, protocols.N_FOLDS + 1)],
    ),
}  # fmt: skip


def parse_candidate(text: str) -> tuple[str, tuple[str, ...]]:
    """``mantis:T+E`` -> ("mantis", ("T", "E"))."""
    encoder, _, arm_text = text.partition(":")
    if not encoder or not arm_text:
        raise ValueError(f"candidate {text!r} must look like encoder:ARM or encoder:ARM+ARM")
    return encoder, tuple(arm_text.split("+"))


def _model(kind: str, strength: float):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "classification":
        return make_pipeline(StandardScaler(), LogisticRegression(C=strength, max_iter=2000))
    return make_pipeline(StandardScaler(), Ridge(alpha=strength))


def _score(kind: str, y_true, y_pred) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

    if kind == "classification":
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
    y_true, y_pred = np.asarray(y_true, float), np.clip(np.asarray(y_pred, float), 0.0, 1.0)
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def primary(kind: str) -> str:
    return "balanced_accuracy" if kind == "classification" else "rmse"


def select_and_fit(kind: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Choose the strength by grouped cross-validation (up to 3 folds), then refit on all."""
    from sklearn.model_selection import GroupKFold

    grid = C_GRID if kind == "classification" else ALPHA_GRID
    n_groups = len(set(groups))
    if n_groups < 2:
        best = grid[len(grid) // 2]
    else:
        splitter = GroupKFold(n_splits=min(3, n_groups))
        sign = 1.0 if kind == "classification" else -1.0
        scores = []
        for strength in grid:
            fold_scores = []
            for train, val in splitter.split(x, y, groups):
                if kind == "classification" and len(set(y[train])) < 2:
                    continue
                model = _model(kind, strength).fit(x[train], y[train])
                fold_scores.append(_score(kind, y[val], model.predict(x[val]))[primary(kind)])
            scores.append(sign * np.mean(fold_scores) if fold_scores else -np.inf)
        best = grid[int(np.argmax(scores))]
    return _model(kind, best).fit(x, y), best


def _fit_and_score(kind, x_train, y_train, groups, x_val, y_val) -> tuple[float, dict]:
    model, strength = select_and_fit(kind, x_train, y_train, groups)
    return strength, _score(kind, y_val, model.predict(x_val))


def _features(stores: FeatureStore, store: WindowStore, encoder: str, arm_list: tuple[str, ...]):
    """Pooled features of all rows (zeros where not cached) and the cached-rows mask."""
    blocks, done = [], np.ones(len(store), dtype=bool)
    for arm in arm_list:
        cached = stores.find(store, encoder, arm)
        done &= cached.done
        blocks.append(cached.pooled(np.arange(len(store))))
    x = np.concatenate(blocks, axis=1)
    done &= np.isfinite(x).all(axis=1)
    return x, done


class NotApplicable(Exception):
    """None of a candidate's arms exists for this dataset (E without bearings)."""


def applicable_arms(store: WindowStore, arm_list: tuple[str, ...]) -> tuple[str, ...]:
    """The arms a dataset has: E only for bearing datasets (so T+E is T on Nonastreda),
    S only at the 12 kHz model rate (AST hears 16 kHz audio, not 1 kHz force)."""
    return tuple(a for a in arm_list if a not in arms.unavailable(store))


def _cached_features(cache: dict, stores: FeatureStore, store: WindowStore, text: str, fold: int):
    """Features of a candidate for one fold; ``{fold}`` in its encoder name selects the
    per-fold model (own PatchTST). Loaded once per encoder and arms."""
    encoder, arm_list = parse_candidate(text.replace("{fold}", str(fold)))
    arm_list = applicable_arms(store, arm_list)
    if not arm_list:
        raise NotApplicable(text)
    if (encoder, arm_list) not in cache:
        cache[encoder, arm_list] = _features(stores, store, encoder, arm_list)
    return cache[encoder, arm_list]


def run(
    task_names: list[str],
    candidates: list[str],
    references: Sequence[str] = (),
    run_id: str | None = None,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
    jobs: int = 1,
) -> dict:
    """Score every candidate and reference on every available task; write the results.
    ``jobs`` > 1 fits the candidates of a split in parallel processes (one thread each)."""
    from joblib import Parallel, delayed, parallel_config

    run_id = run_id or time.strftime("gate1-%Y%m%d-%H%M%S")
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "gate1") / run_id
    out.mkdir(parents=True, exist_ok=True)
    stores = FeatureStore(feature_root)
    entries = [(c, True) for c in candidates] + [(r, False) for r in references]
    records, skipped = [], {}
    for name in task_names:
        task = TASKS[name]
        try:
            store = WindowStore.open(task.dataset, window_root)
        except (FileNotFoundError, ValueError) as error:
            skipped[name] = str(error)
            log(f"[gate1] {name}: skipped ({error})")
            continue
        manifest = store.manifest
        y = manifest[task.target].to_numpy()
        if task.kind == "classification":
            labelled = (manifest[task.target].astype(str) != "").to_numpy()
        else:
            labelled = np.isfinite(y.astype(float))
        groups = manifest[task.group].astype(str).to_numpy()
        cache: dict = {}
        for fold, split in enumerate(task.splits(manifest), 1):
            if task.dataset != "cwru":  # CWRU splits by load: its units repeat by design
                split.assert_unit_disjoint(manifest)
            loaded, not_applicable, missing = {}, [], None
            for text, is_candidate in entries:
                try:
                    loaded[text] = _cached_features(cache, stores, store, text, fold)
                except NotApplicable:
                    not_applicable.append(text)
                except (FileNotFoundError, ValueError) as error:  # features never extracted
                    if is_candidate:  # all candidates must share the tasks: skip the split
                        missing = error
                        break
                    skipped[f"{name}/{split.name}/{text}"] = f"reference dropped: {error}"
            if missing is not None:
                skipped[f"{name}/{split.name}"] = str(missing)
                log(f"[gate1] {name} {split.name}: skipped ({missing})")
                continue
            split_entries = [(t, c) for t, c in entries if t in loaded or t in not_applicable]
            usable = labelled.copy()
            for _, done in loaded.values():
                usable &= done  # every candidate is scored on the same windows
            train, val = split.train & usable, split.val & usable
            if not train.any() or not val.any():
                skipped[f"{name}/{split.name}"] = "no cached windows in train or validation"
                continue
            metric = primary(task.kind)
            scored = [text for text, _ in entries if text in loaded]
            fits = [
                delayed(_fit_and_score)(
                    task.kind, loaded[t][0][train], y[train], groups[train],
                    loaded[t][0][val], y[val],
                )
                for t in scored
            ]  # fmt: skip
            with parallel_config(backend="loky", inner_max_num_threads=1):
                outcomes = Parallel(n_jobs=jobs)(fits)
            outcome = dict(zip(scored, outcomes, strict=True))
            for text, is_candidate in split_entries:
                encoder, arm_list = parse_candidate(text)
                record = {
                    "task": name, "split": split.name, "candidate": text, "encoder": encoder,
                    "arms": "+".join(applicable_arms(store, arm_list)),
                    "is_candidate": is_candidate, "n_train": int(train.sum()),
                    "n_val": int(val.sum()), metric: np.nan,
                }  # fmt: skip
                if text in outcome:
                    strength, score = cast(tuple[float, dict], outcome[text])
                    record.update(strength=strength, **score)
                    log(f"[gate1] {name} {split.name} {text}: {metric} = {score[metric]:.3f}")
                else:
                    log(f"[gate1] {name} {split.name} {text}: no applicable arm (worst rank)")
                records.append(record)
    results = pd.DataFrame(records)
    summary = summarize(results) if len(results) else pd.DataFrame()
    ranked = (
        summary[summary["is_candidate"] & summary["mean_rank"].notna()] if len(summary) else summary
    )
    winner = ranked.iloc[0]["candidate"] if len(ranked) else None
    gaps = [k for k, v in skipped.items() if not v.startswith("reference dropped")]
    report = {
        "run_id": run_id,
        "tasks": task_names,
        "candidates": list(candidates),
        "references": list(references),
        "skipped": skipped,
        "complete": not gaps,  # every task and fold scored for every candidate
        "winner": winner,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results.to_csv(out / "results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(_markdown(report, summary), encoding="utf-8")
    log(f"[gate1] winner: {winner}  ({out})")
    return report


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean score per task and candidate, rank per task (candidates only), mean rank."""
    rows = []
    for key, group in results.groupby(["task", "candidate"]):
        task, candidate = (str(k) for k in cast(tuple, key))
        kind = TASKS[task].kind
        values = np.asarray(group[primary(kind)], dtype=float)
        finite = values[np.isfinite(values)]  # NaN: no applicable arm
        rows.append(
            {
                "task": task,
                "candidate": candidate,
                "is_candidate": bool(group["is_candidate"].iloc[0]),
                "metric": primary(kind),
                "score": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std()) if len(finite) else np.nan,
                "splits": len(group),
            }
        )
    table = pd.DataFrame(rows)
    table["rank"] = np.nan
    for task, group in table[table["is_candidate"]].groupby("task"):
        sign = -1.0 if TASKS[str(task)].higher_is_better else 1.0
        # A candidate without an applicable arm (NaN score) shares the worst rank.
        values = np.nan_to_num(sign * np.asarray(group["score"], dtype=float), nan=np.inf)
        table.loc[group.index, "rank"] = rankdata(values)
    wide = table.pivot(index="candidate", columns="task", values="score")
    ranks = table.pivot(index="candidate", columns="task", values="rank")
    summary = wide.add_prefix("score_").join(ranks.add_prefix("rank_"))
    summary["mean_rank"] = ranks.mean(axis=1)
    summary["is_candidate"] = table.groupby("candidate")["is_candidate"].first()
    summary["cost"] = [cost(c.split(":")[0]) for c in summary.index]
    summary = summary.reset_index()
    order = np.where(np.asarray(summary["is_candidate"], dtype=bool), 0, 1)  # candidates first
    summary = summary.assign(_order=order).sort_values(["_order", "mean_rank", "cost", "candidate"])
    return summary.drop(columns="_order").reset_index(drop=True)


def _number(value, digits: int) -> str:
    """A table cell: the number, or "-" when there is none."""
    number = float(value) if isinstance(value, int | float | np.number) else float("nan")
    return f"{number:.{digits}f}" if np.isfinite(number) else "-"


def _markdown(report: dict, summary: pd.DataFrame) -> str:
    lines = [f"# Gate 1 probe: {report['run_id']}", "", f"Winner: **{report['winner']}**", ""]
    if not report["complete"]:
        lines += [
            "**Incomplete:** some tasks or folds were skipped (listed at the end); the ranking "
            "covers the rest only and is not the final Gate 1.",
            "",
        ]
    if len(summary):
        score_cols = [c for c in summary.columns if c.startswith("score_")]
        header = ["candidate", *[c.removeprefix("score_") for c in score_cols], "mean rank"]
        lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
        for row in summary.to_dict("records"):
            cells = [str(row["candidate"]) + ("" if row["is_candidate"] else " (reference)")]
            cells += [_number(row[c], 3) for c in score_cols]
            cells.append(_number(row["mean_rank"], 2))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "Scores: mean over inner-validation splits of balanced accuracy (cwru, paderborn, "
            "nonastreda; higher is better) and of the RMSE of normalized RUL (xjtu_sy; lower "
            "is better)."
        )
    for task, reason in report["skipped"].items():
        lines.append(f"- skipped {task}: {reason}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.probe", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--run-id")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=sorted(TASKS))
    parser.add_argument(
        "--candidates", nargs="+", required=True, help="encoder:ARM[+ARM], e.g. mantis:T+E"
    )
    parser.add_argument(
        "--references", nargs="*", default=[], help="scored, never ranked (e.g. stats:F)"
    )
    parser.add_argument("--out", type=Path, help="default: runs/gate1/")
    parser.add_argument("--force", action="store_true", help="run even if the machine is busy")
    parser.add_argument(
        "--jobs", type=int, default=0, help="parallel fits (default: the CPUs granted)"
    )
    args = parser.parse_args(argv)
    from pdmfm.resources import claim_resources

    threads = claim_resources("gate1", min_idle_cpus=2.0, min_free_ram_gb=2.0, force=args.force)
    run(
        args.tasks,
        args.candidates,
        args.references,
        args.run_id,
        args.out,
        jobs=args.jobs or threads,
    )


if __name__ == "__main__":
    main()
