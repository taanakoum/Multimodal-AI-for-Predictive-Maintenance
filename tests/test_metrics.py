"""Classification, calibration and regression metrics (pdmfm.metrics)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pdmfm import metrics as M


def test_perfect_one_hot_predictions():
    y = np.array([0, 1, 2, 1])
    result = M.classification(y, np.eye(3)[y])
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
    result = M.classification(y, np.eye(3)[[0, 0, 0]])
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


def test_regression_nll_is_the_student_t_log_density():
    y, mean, dof, scale = np.array([0.5, -1.0]), np.array([0.0, 1.0]), np.array([4.0, 9.0]), 2.0
    expected = -np.mean(stats.t.logpdf(y, dof, loc=mean, scale=scale))
    assert M.regression(y, mean, dof, np.full(2, scale))["nll"] == pytest.approx(expected)


def _t_samples(n: int, true_scale: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    dof = np.full(n, 8.0)
    mean = rng.normal(0, 3, n)
    return mean + true_scale * rng.standard_t(8.0, n), mean, dof


def test_interval_scale_repairs_too_narrow_intervals():
    """Intervals 2.5 times too narrow: the fitted factor is about 2.5, and the calibrated 90 %
    interval covers about 90 % of fresh targets."""
    y, mean, dof = _t_samples(4000, true_scale=2.5)
    factor = M.interval_scale(y, mean, dof, np.ones(4000))
    assert factor == pytest.approx(2.5, rel=0.05)
    fresh, mean, dof = _t_samples(20_000, true_scale=2.5, seed=1)
    raw = M.regression(fresh, mean, dof, np.ones(20_000))
    calibrated = M.regression(fresh, mean, dof, np.full(20_000, factor))
    assert raw["coverage_90"] < 0.6
    assert calibrated["coverage_90"] == pytest.approx(0.9, abs=0.015)
    assert calibrated["nll"] < raw["nll"]


def test_interval_scale_leaves_calibrated_intervals_alone():
    y, mean, dof = _t_samples(4000, true_scale=1.0, seed=2)
    assert M.interval_scale(y, mean, dof, np.ones(4000)) == pytest.approx(1.0, abs=0.05)


def test_class_metrics_average_over_the_classes_present():
    """Plan: a fold without a healthy test bearing is not scored on the healthy class."""
    y = np.array([0, 0, 1, 1])  # class 2 absent from this fold
    perfect = M.classification(y, np.eye(3)[y])
    assert perfect["macro_f1"] == 1.0 and perfect["balanced_accuracy"] == 1.0
    # Predicting the absent class still costs the true class: class 0 has F1 2/3, class 1 F1 1.
    wrong = M.classification(y, np.eye(3)[[0, 2, 1, 1]])
    assert wrong["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)
    assert wrong["balanced_accuracy"] == pytest.approx((0.5 + 1) / 2)


def test_ordinal_median_is_the_class_of_the_middle_probability_mass():
    probs = np.array(
        [[0.45, 0.10, 0.45],  # argmax: sharp (first of a tie); the median: used
         [0.60, 0.30, 0.10],
         [0.20, 0.25, 0.55],
         [0.50, 0.00, 0.50]]  # exactly half at or below sharp: sharp
    )  # fmt: skip
    np.testing.assert_array_equal(M.ordinal_median(probs), [1, 0, 2, 0])


def test_ordinal_metrics_count_the_class_distance():
    y = np.array([0, 1, 2, 1])
    probs = np.array([[0.9, 0.05, 0.05], [0.45, 0.1, 0.45], [0.1, 0.2, 0.7], [0.0, 0.1, 0.9]])
    out = M.classification(y, probs, ordinal=True)
    assert out["accuracy"] == pytest.approx(0.5)  # argmax: 0, 0, 2, 2
    assert out["mace"] == pytest.approx((0 + 1 + 0 + 1) / 4)
    assert out["recall"] == {"0": 1.0, "1": 0.0, "2": 1.0}
    assert out["ordinal"]["accuracy"] == pytest.approx(0.75)  # median: 0, 1, 2, 2
    assert out["ordinal"]["mace"] == pytest.approx(0.25)
    assert out["ordinal"]["recall"]["1"] == pytest.approx(0.5)
    assert "ordinal" not in M.classification(y, probs)
