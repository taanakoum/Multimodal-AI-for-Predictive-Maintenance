"""Pooled out-of-fold statistics (pdmfm.stats)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from pdmfm import stats as S


def test_macro_f1_matches_sklearn_over_present_classes():
    rng = np.random.default_rng(0)
    y, predicted = rng.integers(0, 3, 200), rng.integers(0, 4, 200)
    expected = f1_score(y, predicted, labels=np.unique(y), average="macro", zero_division=0)
    assert S.macro_f1(y, predicted) == pytest.approx(expected)


def test_rmse_ignores_missing_values():
    assert S.rmse(np.array([1.0, np.nan, 3.0]), np.array([1.0, 5.0, 1.0])) == pytest.approx(
        np.sqrt(2.0)
    )


def _tables(n_units: int, seeds: int, shift: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    tables = []
    for _ in range(seeds):
        units = np.repeat([f"u{i}" for i in range(n_units)], 20)
        y = rng.normal(0, 1, len(units))
        tables.append(
            pd.DataFrame({"unit": units, "y": y, "p": y + shift + rng.normal(0, 1, len(y))})
        )
    return tables


def test_unit_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    tables = _tables(10, 3, shift=0.5)

    def error(f):
        return S.rmse(f["y"], f["p"])

    estimate, low, high = S.unit_bootstrap(tables, error, n=300)
    assert low <= estimate <= high
    assert S.unit_bootstrap(tables, error, n=300) == (estimate, low, high)


def test_unit_bootstrap_resamples_units_not_rows():
    """One unit carries all the error: resampling units gives a far wider interval than
    resampling rows would."""
    units = np.repeat([f"u{i}" for i in range(10)], 50)
    p = np.zeros(len(units))
    p[units == "u0"] = 5.0
    table = pd.DataFrame({"unit": units, "y": np.zeros(len(units)), "p": p})
    _, low, high = S.unit_bootstrap([table], lambda f: float(np.mean(f["p"])), n=500)
    assert low == pytest.approx(0.0) and high > 1.0  # u0 drawn 0 to several times
