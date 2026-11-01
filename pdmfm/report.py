"""Stage 12 (part): paired comparisons of methods on the same test bags (plan, "Statistics").

Every method that writes Plan B's bag tables (``fold<k>-seed<s>-bags.csv``: Plan B, a Gate-2
candidate's test run, the baselines of :mod:`pdmfm.baselines`, the RUL model) is compared
with a reference on the bags they share:

* the pooled out-of-fold estimate of each method (macro-F1, RMSE, RMSE after the first
  prediction time for RUL) with a 95 % interval from resampling whole test units, each
  method's seeds averaged within every replicate;
* the difference to the reference, resampled with the same unit draws (paired);
* a two-sided Wilcoxon signed-rank test over test units of the per-unit metric (macro-F1 over
  the classes present in the unit, or RMSE), seeds averaged per unit;
* Holm's correction over the comparisons of one table (one reference, one metric).

Units, not fold x seed runs, are the paired observations: the windows and bags of one tool or
bearing are not independent, and seeds are repeated trainings on the same data (amendment of
2026-10-30). With 10-17 units per dataset the tests are weak; "no significant difference" is
an absence of evidence, and the intervals say how large a difference could still be.

Methods are ``name=folder`` run folders (their ``pred_class``, or ``pred_class_ordinal`` with
``--ordinal``, and ``pred_reg`` columns), plus the built-in baselines of the reference run's
tables: majority, persistence and schedule; training mean, persistence and schedule.

    python -m pdmfm.report --name nonastreda --reference planb \\
        --run planb=runs/planb/exp-base --run gbm=runs/baselines/gbm-nona
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pdmfm.paths import REPO_ROOT
from pdmfm.utils import write_json_atomic

REPLICATES = 2000
BUILTIN = {
    "class": (("majority", "majority"), ("persistence", "persistence"), ("schedule", "schedule")),
    "reg": (
        ("training mean", "train_mean"),
        ("persistence", "persistence_reg"),
        ("schedule", "schedule_reg"),
    ),
}
METRIC = {"class": "macro-F1", "reg": "RMSE", "reg_post": "RMSE after FPT"}
HIGHER_IS_BETTER = {"class": True, "reg": False, "reg_post": False}


@dataclass
class Method:
    name: str
    tables: list[pd.DataFrame]  # one per seed: fold, bag, unit, y, pred


def load_run(folder: Path) -> dict[int, pd.DataFrame]:
    """Seed -> the run's bag tables of every fold, concatenated."""
    by_seed: dict[int, list[pd.DataFrame]] = {}
    for path in sorted(Path(folder).glob("fold*-seed*-bags.csv")):
        seed = int(path.stem.split("-seed")[1].split("-")[0])
        by_seed.setdefault(seed, []).append(pd.read_csv(path, dtype={"bag": str, "unit": str}))
    if not by_seed:
        raise FileNotFoundError(f"no fold*-seed*-bags.csv in {folder}")
    return {seed: pd.concat(parts, ignore_index=True) for seed, parts in sorted(by_seed.items())}


def _frame(table: pd.DataFrame, y: str, pred: str) -> pd.DataFrame:
    out = pd.DataFrame(table[["fold", "bag", "unit", y, pred]]).copy()
    out.columns = ["fold", "bag", "unit", "y", "pred"]
    return out


def tasks_of(table: pd.DataFrame) -> list[str]:
    """Tasks a bag table holds: class, reg, and reg_post for normalized RUL."""
    tasks = []
    if {"y_class", "pred_class"} <= set(table.columns):
        tasks.append("class")
    if {"y_reg", "pred_reg"} <= set(table.columns):
        tasks.append("reg")
        y = table["y_reg"].to_numpy(dtype=float)
        if bool(np.any(y < 1 - 1e-6)) and float(np.nanmax(y)) <= 1.0:
            tasks.append("reg_post")
    return tasks


def methods_for(
    runs: dict[str, dict[int, pd.DataFrame]],
    task: str,
    reference: str,
    ordinal: bool = False,
    builtins: bool = True,
    builtins_from: str | None = None,
) -> list[Method]:
    """The runs that hold ``task`` (reference first), then the built-in baselines of the
    ``builtins_from`` run's first seed (default: the reference; older runs lack the
    schedule column)."""
    y = "y_class" if task == "class" else "y_reg"
    model = "pred_class" if task == "class" else "pred_reg"
    methods = []
    for name in [reference] + [n for n in runs if n != reference]:
        tables = runs[name]
        first = next(iter(tables.values()))
        column = "pred_class_ordinal" if ordinal and task == "class" else model
        if column not in first or y not in first:
            if name == reference:
                raise ValueError(f"the reference {name} has no {column} for {task}")
            continue
        methods.append(Method(name, [_frame(t, y, column) for t in tables.values()]))
    if builtins:
        first = next(iter(runs[builtins_from or reference].values()))
        for name, column in BUILTIN["class" if task == "class" else "reg"]:
            if column in first:
                methods.append(Method(name, [_frame(first, y, column)]))
    return methods


def align(methods: list[Method], task: str) -> tuple[list[Method], dict]:
    """Every method restricted to the labelled bags all of them hold; their labels must
    agree there (otherwise the runs used different label versions)."""
    keys = None
    for m in methods:
        for t in m.tables:
            labelled = t["y"] >= 0 if task == "class" else np.isfinite(t["y"])
            if task == "reg_post":
                labelled &= t["y"] < 1 - 1e-6
            here = set(zip(t.loc[labelled, "fold"], t.loc[labelled, "bag"], strict=True))
            keys = here if keys is None else keys & here
    assert keys is not None
    counts = {m.name: int(len(m.tables[0])) for m in methods}
    aligned, reference_y = [], None
    for m in methods:
        tables = []
        for t in m.tables:
            keep = [k in keys for k in zip(t["fold"], t["bag"], strict=True)]
            part = t.loc[keep].sort_values(["fold", "bag"]).reset_index(drop=True)
            if reference_y is None:
                reference_y = part["y"].to_numpy()
            elif not np.allclose(part["y"].to_numpy(), reference_y, equal_nan=True):
                raise ValueError(f"{m.name}: labels differ from the reference on shared bags")
            tables.append(part)
        aligned.append(Method(m.name, tables))
    info = {"shared_bags": len(keys), "bags_per_method": counts,
            "units": int(aligned[0].tables[0]["unit"].nunique())}  # fmt: skip
    return aligned, info


# ----------------------------------------------------------------------------- statistics


def unit_statistics(table: pd.DataFrame, task: str, n_classes: int, units: list[str]) -> np.ndarray:
    """Per unit, the sufficient statistics of the metric: a confusion matrix (classes) or
    (sum of squared errors, count) (regression). Shape (U, K, K) or (U, 2)."""
    index = {u: i for i, u in enumerate(units)}
    u = np.array([index[x] for x in table["unit"].astype(str)])
    if task == "class":
        out = np.zeros((len(units), n_classes, n_classes))
        np.add.at(out, (u, table["y"].to_numpy(dtype=int), table["pred"].to_numpy(dtype=int)), 1)
        return out
    error = (table["pred"].to_numpy(dtype=float) - table["y"].to_numpy(dtype=float)) ** 2
    out = np.zeros((len(units), 2))
    np.add.at(out[:, 0], u, error)
    np.add.at(out[:, 1], u, 1)
    return out


def metric_from(statistics: np.ndarray, task: str) -> float:
    """Macro-F1 over the classes present (rows of the confusion with labels), or RMSE."""
    if task == "class":
        present = statistics.sum(axis=1) > 0
        tp = np.diag(statistics)
        f1 = 2 * tp / np.maximum(statistics.sum(axis=1) + statistics.sum(axis=0), 1e-12)
        return float(f1[present].mean()) if present.any() else float("nan")
    return float(np.sqrt(statistics[0] / statistics[1])) if statistics[1] > 0 else float("nan")


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN stays NaN and is not counted)."""
    p = np.asarray(pvalues, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    adjusted = np.full(len(p), np.nan)
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p over paired values (NaN pairs dropped; 1 when every
    difference is zero; NaN with fewer than 2 pairs)."""
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if len(d) < 2:
        return float("nan")
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def compare(methods: list[Method], task: str, n: int = REPLICATES, seed: int = 0) -> pd.DataFrame:
    """One row per method (the first is the reference): the pooled estimate and interval,
    the unit average (every unit weighs the same; the pooled value weighs units by their
    bags, so long-lived bearings dominate it), the paired difference and interval, units
    better, Wilcoxon and Holm p."""
    units = sorted(set(methods[0].tables[0]["unit"].astype(str)))
    n_classes = 0
    if task == "class":
        n_classes = 1 + max(
            int(np.max(t[["y", "pred"]].to_numpy(dtype=float))) for m in methods for t in m.tables
        )
    per = [
        [unit_statistics(t, task, n_classes, units) for t in m.tables] for m in methods
    ]  # method -> seed -> (U, ...)

    def value(stat_list, pick=None) -> float:
        return float(np.mean([metric_from((s if pick is None else s[pick]).sum(axis=0), task)
                              for s in stat_list]))  # fmt: skip

    estimate = np.array([value(s) for s in per])
    rng = np.random.default_rng(seed)
    replicates = np.empty((n, len(methods)))
    for b in range(n):
        pick = rng.integers(0, len(units), len(units))
        replicates[b] = [value(s, pick) for s in per]
    unit_values = np.array(
        [[np.mean([metric_from(s[u], task) for s in stat_list]) for u in range(len(units))]
         for stat_list in per]
    )  # fmt: skip
    sign = 1.0 if HIGHER_IS_BETTER[task] else -1.0
    rows, pvalues = [], []
    for i, m in enumerate(methods):
        low, high = np.nanpercentile(replicates[:, i], [2.5, 97.5])
        row = {"method": m.name, "seeds": len(m.tables), "estimate": estimate[i],
               "low": low, "high": high,
               "unit_average": float(np.nanmean(unit_values[i]))}  # fmt: skip
        if i > 0:
            diff = replicates[:, i] - replicates[:, 0]
            d_low, d_high = np.nanpercentile(diff, [2.5, 97.5])
            better = sign * (unit_values[i] - unit_values[0]) > 1e-12
            row.update(
                difference=estimate[i] - estimate[0], difference_low=d_low,
                difference_high=d_high, units_better=int(better.sum()), units=len(units),
                wilcoxon_p=wilcoxon_p(unit_values[i], unit_values[0]),
            )  # fmt: skip
            pvalues.append(row["wilcoxon_p"])
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(methods) > 1:
        table.loc[1:, "holm_p"] = holm(pvalues)
    return table


# ----------------------------------------------------------------------------- the report


def run(
    name: str,
    runs: dict[str, Path],
    reference: str | None = None,
    ordinal: bool = False,
    builtins: bool = True,
    n: int = REPLICATES,
    out_root: Path | None = None,
    log=print,
    builtins_from: str | None = None,
) -> dict[str, pd.DataFrame]:
    if not runs:
        raise ValueError("give at least one run (name=folder)")
    reference = reference or next(iter(runs))
    for role, chosen in (("reference", reference), ("builtins-from", builtins_from)):
        if chosen is not None and chosen not in runs:
            raise ValueError(f"{role} {chosen!r} is not among the runs {list(runs)}")
    loaded = {k: load_run(v) for k, v in runs.items()}
    tasks = tasks_of(next(iter(loaded[reference].values())))
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "report") / name
    out.mkdir(parents=True, exist_ok=True)
    tables, infos = {}, {}
    for task in tasks:
        methods = methods_for(loaded, "class" if task == "class" else "reg", reference, ordinal,
                              builtins, builtins_from)  # fmt: skip
        if task == "reg_post":
            methods = [m for m in methods if m.name != "persistence"]  # undefined for RUL
        methods, info = align(methods, task)
        tables[task] = compare(methods, task, n)
        infos[task] = info
        tables[task].to_csv(out / f"{task}.csv", index=False)
        log(f"[report] {name} {METRIC[task]}: {info['shared_bags']} bags, {info['units']} units")
    (out / "report.md").write_text(_markdown(name, reference, tables, infos), encoding="utf-8")
    meta = {
        "name": name,
        "reference": reference,
        "runs": {k: str(v) for k, v in runs.items()},
        "ordinal": ordinal,
        "replicates": n,
        "alignment": infos,
        "builtins_from": builtins_from or reference,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(out / "report.json", meta)
    return tables


def _markdown(name: str, reference: str, tables: dict, infos: dict) -> str:
    lines = [
        f"# Paired comparison: {name}",
        "",
        f"Reference: **{reference}**. Pooled out-of-fold bags; 95 % intervals from resampling "
        "whole test units (seeds averaged in every replicate); differences paired on the same "
        "draws; Wilcoxon signed-rank over test units (two-sided), Holm-corrected per table. "
        "The unit average weighs every test unit the same; the pooled estimate weighs units by "
        "their bags.",
        "",
    ]
    for task, table in tables.items():
        info = infos[task]
        better = "higher" if HIGHER_IS_BETTER[task] else "lower"
        lines += [
            f"## {METRIC[task]} ({better} is better; {info['shared_bags']} shared bags, "
            f"{info['units']} units)",
            "",
            "| Method | Seeds | Estimate [95 %] | Unit average | Difference to reference [95 %] "
            "| Units better | Wilcoxon p | Holm p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in table.to_dict("records"):
            cells = [str(r["method"]), str(r["seeds"]),
                     f"{r['estimate']:.3f} [{r['low']:.3f}, {r['high']:.3f}]",
                     f"{r['unit_average']:.3f}"]  # fmt: skip
            if r["method"] == table.iloc[0]["method"]:
                cells += ["(reference)", "-", "-", "-"]
            else:
                cells += [
                    f"{r['difference']:+.3f} [{r['difference_low']:+.3f}, "
                    f"{r['difference_high']:+.3f}]",
                    f"{int(r['units_better'])}/{int(r['units'])}",
                    f"{r['wilcoxon_p']:.3g}",
                    f"{r['holm_p']:.3g}",
                ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _pair(text: str) -> tuple[str, Path]:
    name, _, folder = text.partition("=")
    if not name or not folder:
        raise argparse.ArgumentTypeError(f"--run takes name=folder, not {text!r}")
    return name, Path(folder)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.report", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--name", required=True, help="report folder under runs/report/")
    parser.add_argument("--run", type=_pair, action="append", required=True, help="name=folder")
    parser.add_argument("--reference", help="default: the first --run")
    parser.add_argument("--ordinal", action="store_true", help="ordinal-median class columns")
    parser.add_argument("--no-builtins", action="store_true", help="leave out majority etc.")
    parser.add_argument("--builtins-from", help="run whose tables give the built-in baselines")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--out", type=Path, help="default: runs/report/")
    args = parser.parse_args(argv)
    runs = dict(args.run)
    if len(runs) != len(args.run):
        parser.error("two --run share a name")
    run(args.name, runs, args.reference, args.ordinal, not args.no_builtins, args.replicates,
        args.out, builtins_from=args.builtins_from)  # fmt: skip
    print((Path(args.out or REPO_ROOT / "runs" / "report") / args.name / "report.md").read_text(
        encoding="utf-8"))  # fmt: skip


if __name__ == "__main__":
    main()
