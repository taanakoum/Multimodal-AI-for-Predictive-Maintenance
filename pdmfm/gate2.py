"""Gate 2: the fusion search (plan, "Fusion search candidates"; weeks 6-7).

Candidates (:mod:`pdmfm.model.fusion`) are trained with the plan's search recipe: AdamW
3e-4 with a cosine schedule, whole modalities dropped from the inputs with probability 0.3,
the epochs of the base configuration (50). Plan B with its own recipe (``planb``: AdamW
1e-3, constant) is scored alongside, so late fusion is not handicapped by a recipe made for
the larger candidates.

Scoring is by inner cross-validation (:func:`pdmfm.protocols.inner_folds`): within each
outer fold, validation rotates over the other four folds' test groups and training uses the
rest; the outer test units are never read. 5 outer x 4 inner folds give 20 validation
scores per candidate (one seed), and every non-test unit is validated four times. Per fit:

* ``score``: the training loop's selection score at the kept epoch (mean of bag macro-F1
  and 1 - RMSE / std);
* ``robustness``: the mean selection score over the predictions without each modality, on
  the rows that keep some input (as :func:`pdmfm.planb.evaluate` scores them);
* ``ece``: window ECE of the combined opinion (classes), or ``coverage_error``: |90 %
  interval coverage - 0.9| of the fused NIG (regression only).

Ranking (plan): score; candidates within one standard error of the paired fold-wise
differences to the best are near-ties, decided by robustness, then ECE, then fewer
parameters (each compared at a resolution of 0.01).

Decision rule, fixed before any result (as for the Plan B options): the best fusion
candidate replaces late fusion only if its mean score beats the better of ``planb`` and
``late`` by at least ``MIN_GAIN`` and it wins at least ``MIN_WIN_RATE`` of the paired
folds. Otherwise Plan B stays the model (the plan's Gate-2 fallback). The winner and the
reference are then scored once on the outer test units with :mod:`pdmfm.planb` (the
``flags`` in ``decision.json``).

    python -m pdmfm.gate2 --dataset nonastreda --encoder mantis --arms F --run-id gate2-nona
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pdmfm import losses, protocols
from pdmfm import metrics as M
from pdmfm.model.fusion import FUSIONS
from pdmfm.model.inputs import FOLD_BASELINE
from pdmfm.paths import REPO_ROOT
from pdmfm.planb import FOLDS, prepare_inputs, student_t, without_each
from pdmfm.train import TrainConfig, fit, predict, selection_score
from pdmfm.utils import write_json_atomic
from pdmfm.windows import WindowStore

RECIPE = {"lr": 3e-4, "schedule": "cosine", "input_dropout": 0.3}
REFERENCES = ("planb", "late")
MIN_GAIN = 0.01
MIN_WIN_RATE = 0.6
RESOLUTION = 0.01  # robustness, ECE: differences below this are ties


def candidates(base: TrainConfig, names: tuple[str, ...] = ("planb", *FUSIONS)) -> dict:
    """Name -> training configuration: ``planb`` keeps the base recipe (late fusion); every
    fusion candidate, ``late`` included, gets the search recipe."""
    out = {}
    for name in names:
        if name == "planb":
            out[name] = replace(base, fusion="late")
        elif name in FUSIONS:
            out[name] = replace(base, fusion=name, modality_dropout=0.0, **RECIPE)
        else:
            raise ValueError(f"unknown candidate {name!r}; choose from planb, {FUSIONS}")
    return out


def calibration(inputs, rows, pred, y_mean: float, y_std: float) -> dict:
    """Window ECE of the combined opinion and |90 % coverage - 0.9| of the fused NIG."""
    out = {}
    if "combined" in pred and inputs.y_class is not None:
        y = inputs.y_class[rows]
        keep = y >= 0
        if keep.any():
            probs = losses.expected_probability(torch.from_numpy(pred["combined"][keep]))
            out["ece"] = M.ece(y[keep], probs.numpy())
    if "fused_nig" in pred and inputs.y_reg is not None:
        y = inputs.y_reg[rows]
        keep = np.isfinite(y)
        if keep.any():
            mean, dof, scale = student_t(pred["fused_nig"][keep], y_mean, y_std)
            coverage = M.regression(y[keep], mean, dof, scale)["coverage_90"]
            out["coverage_error"] = abs(coverage - 0.9)
    return out


def robustness(inputs, rows, dropped: dict, y_mean: float, y_std: float) -> float:
    """Mean selection score without each modality, over the rows that keep some input."""
    scores = []
    for again in dropped.values():
        left = again["present"].any(1)
        if left.any():
            kept = {key: value[left] for key, value in again.items()}
            scores.append(selection_score(inputs, rows[left], kept, y_mean, y_std))
    return float(np.nanmean(scores)) if scores else np.nan


def score_fit(inputs, train_rows, val_rows, cfg: TrainConfig, seed: int, log) -> dict:
    """Fit on ``train_rows``; the scores of the kept epoch on ``val_rows``."""
    fitted = fit(inputs, train_rows, val_rows, cfg, seed, log=log)
    y_mean, y_std = fitted.y_mean, fitted.y_std
    pred = predict(fitted.model, inputs, val_rows, cfg)
    dropped = without_each(fitted.model, inputs, val_rows, cfg)
    return {
        "score": selection_score(inputs, val_rows, pred, y_mean, y_std),
        "robustness": robustness(inputs, val_rows, dropped, y_mean, y_std),
        **calibration(inputs, val_rows, pred, y_mean, y_std),
        "n_params": sum(p.numel() for p in fitted.model.parameters() if p.requires_grad),
        "best_epoch": fitted.best_epoch,
        "seconds": round(fitted.seconds, 1),
    }


def search(
    dataset: str,
    encoder: str,
    arm_list: tuple[str, ...],
    run_id: str,
    base: TrainConfig,
    names: tuple[str, ...] = ("planb", *FUSIONS),
    folds: list[int] | None = None,
    seed: int = 0,
    photos: bool = True,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    log=print,
) -> dict:
    """Score every candidate on every inner fold of the outer ``folds``; write
    ``results.csv``, ``ranking.csv``, ``decision.json`` and ``summary.md``. Resumes."""
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "gate2") / run_id
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.csv"
    records = pd.read_csv(results_path).to_dict("records") if results_path.is_file() else []
    manifest = WindowStore.open(dataset, window_root).manifest
    configs = candidates(base, names)
    options = {
        "photos": photos,
        "without": (),
        "levels": "relative",
        "force_mean": False,
        "window_root": window_root,
        "feature_root": feature_root,
    }
    folds = folds or list(range(1, protocols.N_FOLDS + 1))
    per_split = dataset in FOLD_BASELINE  # the level baseline comes from training units
    prepared = None
    for k in folds:
        for inner in protocols.inner_folds(dataset, manifest, k):
            inner.assert_unit_disjoint(manifest)
            todo = [
                n
                for n in configs
                if not any(r["candidate"] == n and r["split"] == inner.name for r in records)
            ]
            if not todo:
                continue
            if prepared is None or per_split or "{fold}" in encoder:
                train_units = sorted(inner.units(manifest, "train"))
                prepared = prepare_inputs(
                    dataset, encoder.replace("{fold}", str(k)), arm_list, base, train_units,
                    options, log,
                )  # fmt: skip
            inputs, _, _, any_present = prepared
            labelled = any_present.copy()
            if inputs.y_class is not None:
                labelled &= inputs.y_class >= 0
            elif inputs.y_reg is not None:
                labelled &= np.isfinite(inputs.y_reg)
            train_rows = np.flatnonzero(inner.train & labelled)
            val_rows = np.flatnonzero(inner.val & labelled)
            for name in todo:
                scores = score_fit(inputs, train_rows, val_rows, configs[name], seed, log)
                records.append({"candidate": name, "outer": k, "split": inner.name, **scores})
                pd.DataFrame(records).to_csv(results_path, index=False)
                log(f"[gate2] {inner.name} {name}: {scores['score']:.3f}")
    report = decide(pd.DataFrame(records), configs)
    report.update(
        run_id=run_id, dataset=dataset, encoder=encoder, arms=list(arm_list), seed=seed,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )  # fmt: skip
    pd.DataFrame(report["ranking"]).to_csv(out / "ranking.csv", index=False)
    write_json_atomic(out / "decision.json", report)
    (out / "summary.md").write_text(_markdown(report), encoding="utf-8")
    log(f"[gate2] {report['verdict']} -> {out}")
    return report


def _paired(table: pd.DataFrame, a: str, b: str) -> tuple[float, float, float, int]:
    """(mean gap a - b, its standard error, share of folds a wins, folds) over shared folds."""
    scores = table.pivot_table(index="split", columns="candidate", values="score")
    both = scores.loc[:, [a, b]].dropna().to_numpy(dtype=float)
    gaps = both[:, 0] - both[:, 1]
    if len(gaps) == 0:
        return np.nan, np.nan, np.nan, 0
    se = float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else np.nan
    return float(gaps.mean()), se, float(np.mean(gaps > 0)), len(gaps)


def decide(table: pd.DataFrame, configs: dict) -> dict:
    """The ranking and the Gate-2 verdict (module docstring)."""
    has_ece = "ece" in table.columns and bool(table["ece"].notna().any())
    calibration_key = "ece" if has_ece else "coverage_error"
    if calibration_key not in table:
        table = table.assign(**{calibration_key: np.nan})
    ranking = pd.DataFrame(
        table.groupby("candidate")
        .agg(
            score=("score", "mean"),
            score_std=("score", "std"),
            robustness=("robustness", "mean"),
            calibration=(calibration_key, "mean"),
            n_params=("n_params", "median"),
            folds=("split", "nunique"),
            seconds=("seconds", "mean"),
        )
        .reset_index()
        .sort_values("score", ascending=False)
    )
    top = str(ranking.iloc[0]["candidate"])
    gaps = {c: _paired(table, top, c) for c in ranking["candidate"]}
    ranking["gap_to_best"] = [gaps[c][0] for c in ranking["candidate"]]
    ranking["gap_se"] = [gaps[c][1] for c in ranking["candidate"]]
    references = [r for r in REFERENCES if r in set(ranking["candidate"])]
    reference = (
        max(references, key=lambda r: float(ranking.set_index("candidate").loc[r, "score"]))
        if references
        else None
    )
    fusion = ranking.loc[~ranking["candidate"].isin(REFERENCES)]
    chosen, verdict, against = None, "no fusion candidate was scored", {}
    if len(fusion):
        best = str(fusion.iloc[0]["candidate"])
        near = []
        for c in fusion["candidate"]:
            gap, se = _paired(table, best, c)[:2]
            near.append(gap <= np.nan_to_num(se))  # the best itself: gap 0
        close = fusion.loc[np.asarray(near, dtype=bool)]

        def key(row):
            return (
                -np.round(np.nan_to_num(row["robustness"], nan=-np.inf) / RESOLUTION),
                np.round(np.nan_to_num(row["calibration"], nan=np.inf) / RESOLUTION),
                row["n_params"],
            )

        chosen = str(min(close.to_dict("records"), key=key)["candidate"])
        if reference is None:
            verdict = f"{chosen} is the best fusion candidate (no late reference scored)"
        else:
            gain, se, wins, n = _paired(table, chosen, reference)
            against = {"reference": reference, "gain": gain, "gain_se": se, "win_rate": wins,
                       "folds": n}  # fmt: skip
            passed = gain >= MIN_GAIN and wins >= MIN_WIN_RATE
            verdict = (
                f"{chosen} beats {reference} ({gain:+.3f}, wins {wins:.0%} of {n} folds): "
                "fusion replaces late fusion"
                if passed
                else f"{chosen} does not beat {reference} by the rule ({gain:+.3f}, wins "
                f"{wins:.0%} of {n} folds; needs +{MIN_GAIN} and {MIN_WIN_RATE:.0%}): "
                "Plan B stays the model"
            )
            if not passed:
                chosen = reference
    model = chosen or reference
    flags = ""
    if model is not None:
        cfg = configs[model]
        flags = f"--fusion {cfg.fusion}"
        if model != "planb":
            flags += f" --lr {cfg.lr} --schedule {cfg.schedule} --input-dropout {cfg.input_dropout}"
    return {
        "ranking": ranking.to_dict("records"),
        "calibration_metric": calibration_key,
        "reference": reference,
        "against_reference": against,
        "model": model,
        "flags": flags,
        "verdict": verdict,
        "rule": {"min_gain": MIN_GAIN, "min_win_rate": MIN_WIN_RATE},
    }


def _markdown(report: dict) -> str:
    lines = [
        f"# Gate 2: {report['run_id']} ({report['dataset']}, {report['encoder']} "
        f"{'+'.join(report['arms'])})",
        "",
        f"**{report['verdict']}.** Model for the outer test: `{report['model']}` "
        f"(`python -m pdmfm.planb ... {report['flags']}`).",
        "",
        "Inner cross-validation, one seed; score = mean of bag macro-F1 and 1 - RMSE / std; "
        f"calibration = {report['calibration_metric']}.",
        "",
        "| Candidate | Score | Std | Gap to best (SE) | Robustness | Calibration | Params | "
        "Folds | s / fit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["ranking"]:
        lines.append(
            f"| {row['candidate']} | {row['score']:.3f} | {row['score_std']:.3f} | "
            f"{row['gap_to_best']:.3f} ({row['gap_se']:.3f}) | {row['robustness']:.3f} | "
            f"{row['calibration']:.3f} | {int(row['n_params']):,} | {int(row['folds'])} | "
            f"{row['seconds']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.gate2", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--dataset", required=True, choices=sorted(FOLDS))
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--arms", nargs="+", default=["F"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--candidates", nargs="+", default=["planb", *FUSIONS], help="planb and fusion names"
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(range(1, protocols.N_FOLDS + 1))
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--no-photos", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true", help="run on a busy machine (CPU only)")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        claim_resources("gate2", min_idle_cpus=2.0, min_free_ram_gb=1.0, force=args.force)
    base = TrainConfig(epochs=args.epochs, d=args.d, device=args.device)
    search(
        args.dataset, args.encoder, tuple(args.arms), args.run_id, base, tuple(args.candidates),
        args.folds, args.seed, photos=not args.no_photos, out_root=args.out,
    )  # fmt: skip


if __name__ == "__main__":
    main()
