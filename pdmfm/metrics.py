"""Metrics of the plan (section "Trust" and "Cross-validation and evaluation protocol").

* Classes: accuracy, balanced accuracy, macro-F1; calibration as ECE (15 equal-width bins on
  the top-class probability), multi-class Brier score and NLL. ``temperature_scale`` fits
  one temperature on validation probabilities (p ** (1/T), renormalized), the plan's
  post-hoc calibration.
* Regression: RMSE, MAE, coverage and mean width of the 90 % predictive interval; the PHM
  2012 score for RUL on snapshots past the first prediction time.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

EPS = 1e-12


def classification(y: np.ndarray, probs: np.ndarray, n_classes: int | None = None) -> dict:
    """Metrics of predicted class probabilities ``probs`` (n, K) against labels ``y`` (n,)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    k = n_classes or probs.shape[1]
    predicted = probs.argmax(1)
    labels = list(range(k))
    return {
        "accuracy": float((predicted == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(
            f1_score(y, predicted, labels=labels, average="macro", zero_division=0.0)  # pyright: ignore[reportArgumentType]
        ),
        "ece": ece(y, probs),
        "brier": brier(y, probs),
        "nll": nll(y, probs),
        "n": int(len(y)),
    }


def ece(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-class probability, equal-width bins."""
    confidence = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            total += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    return float(total)


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1])[y]
    return float(((probs - one_hot) ** 2).sum(1).mean())


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(y)), y] + EPS).mean())


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.asarray(probs, dtype=float) + EPS) / temperature
    logits -= logits.max(1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(1, keepdims=True)


def temperature_scale(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    """The temperature in [0.05, 20] that minimizes the validation NLL."""
    result = optimize.minimize_scalar(
        lambda log_t: nll(y_val, apply_temperature(probs_val, float(np.exp(log_t)))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def regression(
    y: np.ndarray, mean: np.ndarray, dof: np.ndarray, scale: np.ndarray, level: float = 0.9
) -> dict:
    """RMSE, MAE and the coverage and mean width of the central ``level`` Student-t interval."""
    y, mean = np.asarray(y, dtype=float), np.asarray(mean, dtype=float)
    half = stats.t.ppf(0.5 + level / 2, np.asarray(dof, dtype=float)) * np.asarray(
        scale, dtype=float
    )
    inside = np.abs(y - mean) <= half
    return {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "mae": float(np.mean(np.abs(y - mean))),
        f"coverage_{int(level * 100)}": float(inside.mean()),
        f"width_{int(level * 100)}": float(np.mean(2 * half)),
        "n": int(len(y)),
    }


def phm2012_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean PHM 2012 accuracy score over points with actual RUL > 0: percent error
    Er = 100 (actual - predicted) / actual; A = exp(-ln 0.5 Er/5) if Er <= 0 (late
    predictions punished harder), else exp(ln 0.5 Er/20)."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    keep = actual > 0
    if not keep.any():
        return float("nan")
    error = 100.0 * (actual[keep] - predicted[keep]) / actual[keep]
    score = np.where(
        error <= 0, np.exp(-np.log(0.5) * error / 5.0), np.exp(np.log(0.5) * error / 20.0)
    )
    return float(score.mean())
