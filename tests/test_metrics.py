"""Classification, calibration and regression metrics (pdmfm.metrics)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pdmfm import metrics as M


def test_perfect_one_hot_predictions():
    y = np.array([0, 1, 2, 1])
    result = M.classification(y, np.eye(3)[y], 3)
    assert result["accuracy"] == result["balanced_accuracy"] == result["macro_f1"] == 1.0
    assert result["ece"] == pytest.approx(0.0) and result["brier"] == pytest.approx(0.0)
    assert result["nll"] == pytest.approx(0.0, abs=1e-9) and result["n"] == 4


def test_ece_of_a_constant_overconfident_prediction():
    # Confidence 0.8 everywhere, right 60 % of the time: ECE = |0.6 - 0.8| = 0.2.
    y = np.array([0] * 6 + [1] * 4)
    probs = np.tile([0.8, 0.2], (10, 1))
    assert M.ece(y, probs) == pytest.approx(0.2)


def test_ece_weights_bins_by_their_size():
    # Bin of 0.95 (4 rows, all right): gap 0.05; bin of 0.55 (6 rows, half right): gap 0.05.
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    probs = np.array([[0.95, 0.05]] * 4 + [[0.55, 0.45]] * 6)
    assert M.ece(y, probs) == pytest.approx(0.4 * 0.05 + 0.6 * 0.05)


def test_brier_and_nll_by_hand():
    y = np.array([0, 1])
    probs = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert M.brier(y, probs) == pytest.approx(((0.3**2 + 0.3**2) + (0.4**2 + 0.4**2)) / 2)
    assert M.nll(y, probs) == pytest.approx(-(np.log(0.7) + np.log(0.6)) / 2)


def test_macro_f1_counts_classes_never_predicted():
    y = np.array([0, 1, 2])
    result = M.classification(y, np.eye(3)[[0, 0, 0]], 3)
    assert result["macro_f1"] == pytest.approx((0.5 + 0 + 0) / 3)


def test_temperature_one_is_the_identity_and_large_temperatures_flatten():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(M.apply_temperature(probs, 1.0), probs, atol=1e-9)
    flat = M.apply_temperature(probs, 1e6)
    np.testing.assert_allclose(flat, np.full_like(probs, 1 / 3), atol=1e-5)
    sharp = M.apply_temperature(probs, 0.5)
    np.testing.assert_allclose(sharp.sum(1), 1.0)
    assert np.all(sharp.max(1) > probs.max(1))


def _sharpened(n: int, true_confidence: float, power: float, seed: int = 0):
    """Labels right with probability ``true_confidence``; predictions p ** power renormalized."""
    rng = np.random.default_rng(seed)
    predicted = rng.integers(0, 3, n)
    base = np.full((n, 3), (1 - true_confidence) / 2)
    base[np.arange(n), predicted] = true_confidence
    y = np.array([rng.choice(3, p=row) for row in base])
    return y, M.apply_temperature(base, 1 / power)


def test_temperature_scaling_repairs_overconfidence():
    y, probs = _sharpened(3000, 0.7, power=3.0)
    temperature = M.temperature_scale(y, probs)
    assert temperature == pytest.approx(3.0, rel=0.15)
    calibrated = M.apply_temperature(probs, temperature)
    assert M.nll(y, calibrated) < M.nll(y, probs)
    assert M.ece(y, calibrated) < M.ece(y, probs) / 2


def test_temperature_scaling_leaves_calibrated_predictions_alone():
    y, probs = _sharpened(3000, 0.7, power=1.0, seed=1)
    assert M.temperature_scale(y, probs) == pytest.approx(1.0, abs=0.1)


def test_regression_interval_coverage_and_width():
    rng = np.random.default_rng(0)
    n, dof, scale = 20_000, 6.0, 2.0
    mean = rng.normal(0, 5, n)
    y = mean + scale * rng.standard_t(dof, n)
    result = M.regression(y, mean, np.full(n, dof), np.full(n, scale))
    assert result["coverage_90"] == pytest.approx(0.9, abs=0.01)
    assert result["width_90"] == pytest.approx(2 * stats.t.ppf(0.95, dof) * scale)
    assert result["rmse"] == pytest.approx(scale * np.sqrt(dof / (dof - 2)), rel=0.05)
    assert result["n"] == n


def test_regression_level_names_its_keys():
    result = M.regression(np.zeros(3), np.zeros(3), np.full(3, 4.0), np.ones(3), level=0.5)
    assert "coverage_50" in result and "width_50" in result


def test_phm2012_score_known_values():
    assert M.phm2012_score(np.array([100.0]), np.array([100.0])) == pytest.approx(1.0)
    # 20 % early (Er = 20) and 5 % late (Er = -5) both score 0.5: late is punished harder.
    assert M.phm2012_score(np.array([100.0]), np.array([80.0])) == pytest.approx(0.5)
    assert M.phm2012_score(np.array([100.0]), np.array([105.0])) == pytest.approx(0.5)
    late = M.phm2012_score(np.array([100.0]), np.array([110.0]))
    early = M.phm2012_score(np.array([100.0]), np.array([90.0]))
    assert late < early


def test_phm2012_score_skips_failed_points():
    assert M.phm2012_score(np.array([0.0, 50.0]), np.array([10.0, 50.0])) == pytest.approx(1.0)
    assert np.isnan(M.phm2012_score(np.array([0.0]), np.array([1.0])))
