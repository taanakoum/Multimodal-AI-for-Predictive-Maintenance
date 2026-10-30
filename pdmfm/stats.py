"""Uncertainty of pooled out-of-fold results (plan, "Reporting": bootstrap CIs against the
best baseline).

The outer folds' test units are disjoint and cover every unit once per seed, so one seed's
out-of-fold predictions can be pooled into one table of bags. Confidence intervals resample
whole units (tools, bearings) with replacement, because the windows and bags of one unit are
not independent; seeds are averaged inside every replicate, because they are repeated
trainings on the same data, not new data. With 10 tools or 15 bearings the intervals are
wide: that is the honest size of the evidence.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def macro_f1(y: np.ndarray, predicted: np.ndarray) -> float:
    """Macro-F1 over the classes present in ``y`` (the plan's definition)."""
    y, predicted = np.asarray(y), np.asarray(predicted)
    scores = []
    for c in np.unique(y):
        tp = np.sum((predicted == c) & (y == c))
        fp = np.sum((predicted == c) & (y != c))
        fn = np.sum((predicted != c) & (y == c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def rmse(y: np.ndarray, predicted: np.ndarray) -> float:
    y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
    keep = np.isfinite(y) & np.isfinite(predicted)
    return float(np.sqrt(np.mean((y[keep] - predicted[keep]) ** 2))) if keep.any() else np.nan


def unit_bootstrap(
    tables: list[pd.DataFrame],
    statistic: Callable[[dict[str, np.ndarray]], float],
    unit: str = "unit",
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(estimate, 2.5 % and 97.5 % percentiles) of ``statistic`` averaged over ``tables``
    (one per seed), with units resampled with replacement in every replicate.
    ``statistic`` receives the table's numeric columns as a dict of NumPy arrays."""
    numeric = [c for c in tables[0].columns if pd.api.types.is_numeric_dtype(tables[0][c])]
    columns = [{c: t[c].to_numpy() for c in numeric} for t in tables]
    estimate = float(np.mean([statistic(c) for c in columns]))
    units = sorted(set(tables[0][unit].astype(str)))
    groups = [t.groupby(t[unit].astype(str)).indices for t in tables]
    rng = np.random.default_rng(seed)
    replicates = np.empty(n)
    for b in range(n):
        pick = rng.choice(units, size=len(units), replace=True)
        values = []
        for table, index in zip(columns, groups, strict=True):
            rows = np.concatenate([index[u] for u in pick if u in index])
            values.append(statistic({c: v[rows] for c, v in table.items()}))
        replicates[b] = np.mean(values)
    low, high = np.nanpercentile(replicates, [2.5, 97.5])
    return estimate, float(low), float(high)
