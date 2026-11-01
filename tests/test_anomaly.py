"""Anomaly detection from healthy data only (pdmfm.anomaly) on synthetic data.

The synthetic CWRU faults change the spectrum (a tone moves) but not the RMS: the feature
distance must find them and the RMS alarm must not. The synthetic XJTU-SY bearings grow
louder after 60 % of their life: the RMS alarm must fire near the labelled FPT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import cwru_reader, xjtu_reader

from pdmfm import anomaly, windows
from pdmfm.encoders import StatsEncoder
from pdmfm.features import FeatureStore


def quiet(*_):
    pass


# ----------------------------------------------------------------------------- scores


def test_knn_distance_values():
    support = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert anomaly.knn_scores(support, np.array([[2.0, 0.0]]), k=1)[0] == pytest.approx(
        0.0, abs=1e-9
    )
    assert anomaly.knn_scores(support[:1], np.array([[0.0, 3.0]]), k=1)[0] == pytest.approx(1.0)
    # k is capped by the support size; the mean of the k smallest distances.
    two = anomaly.knn_scores(support[:2], np.array([[1.0, 0.0]]), k=5)[0]
    assert two == pytest.approx(0.5)


def test_rms_z_scores_take_the_loudest_channel():
    support = np.array([[0.0, 1.0], [2.0, 1.0], [1.0, 1.0]])  # channel 2 constant
    z = anomaly.rms_scores(support, np.array([[1.0, 1.0], [4.0, 1.0]]))
    assert z[0] == pytest.approx(0.0) and z[1] == pytest.approx(3.0)


def test_isolation_forest_ranks_an_outlier_first():
    rng = np.random.default_rng(0)
    support = rng.normal(0, 1, (50, 4))
    test = np.vstack([rng.normal(0, 1, (5, 4)), np.full((1, 4), 8.0)])
    scores = anomaly.iforest_scores(support, test)
    assert int(np.argmax(scores)) == 5


def test_leave_one_out_scores_every_support_row_against_the_others():
    support = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    loo = anomaly.leave_one_out(lambda s, t: anomaly.knn_scores(s, t, k=1), support)
    np.testing.assert_allclose(loo, [0.0, 0.0, 1.0], atol=1e-9)


def test_detection_metrics():
    perfect = anomaly.detection_metrics(np.array([0.1, 0.2, 0.9, 1.0]), np.array([0, 0, 1, 1]))
    assert perfect == {"auroc": 1.0, "auprc": 1.0, "far_at_95": 0.0}
    reversed_ = anomaly.detection_metrics(np.array([0.9, 1.0, 0.1, 0.2]), np.array([0, 0, 1, 1]))
    assert reversed_["auroc"] == 0.0 and reversed_["far_at_95"] == 1.0
    assert np.isnan(anomaly.detection_metrics(np.ones(3), np.ones(3, dtype=bool))["auroc"])


def test_first_alarm_needs_consecutive_exceedances():
    assert anomaly.first_alarm(np.array([0, 1, 1, 0, 1, 1, 1, 1])) == 4
    assert anomaly.first_alarm(np.array([1, 1, 0, 1])) is None
    assert anomaly.first_alarm(np.array([0, 1]), consecutive=1) == 1


def test_parse_feature_sets():
    assert anomaly.parse_feature("mantis:F+E") == ("mantis", ("F", "E"))
    with pytest.raises(ValueError, match="encoder:ARM"):
        anomaly.parse_feature("mantis")


# ----------------------------------------------------------------------------- protocols


def test_healthy_and_faulty_rows_of_the_targets():
    kaist = pd.DataFrame(
        {"class_key": ["normal", "normal", "outer_race/1.0 mm", "normal"], "load_nm": [4, 4, 4, 0]}
    )
    healthy, faulty = anomaly.healthy_and_faulty("kaist", kaist)
    assert healthy.tolist() == [True, True, False, False]  # 4 Nm only
    assert faulty.tolist() == [False, False, True, False]
    paderborn = pd.DataFrame(
        {"unit": ["K001", "K004", "KA04", "KB27", "KA01"], "class_key": [""] * 5}
    )
    healthy, faulty = anomaly.healthy_and_faulty("paderborn", paderborn)
    assert healthy.tolist() == [False, True, False, False, False]  # held-out healthy K004
    assert faulty.tolist() == [False, False, True, False, False]  # real damage, not KB27
    with pytest.raises(ValueError, match="no window protocol"):
        anomaly.healthy_and_faulty("nonastreda", paderborn)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("anomaly")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        cache = FeatureStore(root / "interim" / "features")
        for name, reader in (("cwru", cwru_reader(seconds=12.5)), ("xjtu_sy", xjtu_reader(40))):
            store = windows.build(name, reader=reader, log=quiet)
            for arm in ("F", "E"):
                cache.extract(store, StatsEncoder(), arm, log=quiet)
        yield root


def _run(root, dataset, **kwargs):
    return anomaly.run(
        dataset, ["stats:F", "stats:F+E"], f"test-{dataset}", out_root=root / "runs",
        window_root=root / "interim" / "windows", feature_root=root / "interim" / "features",
        log=quiet, **kwargs,
    )  # fmt: skip


def test_the_window_protocol_finds_a_spectral_fault_that_rms_misses(root):
    summary = _run(root, "cwru")
    k10 = summary[summary["k_h"] == 10].set_index("score")
    assert set(summary["k_h"]) == {10}  # 48 healthy windows: k_h = 50 leaves too few
    assert k10.loc["knn:stats:F", "auroc"] > 0.95
    assert k10.loc["rms", "auroc"] < 0.8  # the synthetic faults keep the same RMS
    results = pd.read_csv(root / "runs" / "test-cwru" / "results.csv")
    assert (results["healthy_test"] == 48 - 10).all() and set(results["draw"]) == set(range(5))
    assert "AUROC" in (root / "runs" / "test-cwru" / "summary.md").read_text(encoding="utf-8")


def test_the_support_never_enters_the_test_set(root):
    store = windows.WindowStore.open("cwru", root / "interim" / "windows")
    seen = []

    def spy(support, test):
        seen.append((support.copy(), test.copy()))
        return np.zeros(len(test))

    x = np.arange(len(store), dtype=float)[:, None]  # each row's own index as its feature
    anomaly.window_protocol(store, {"spy": (spy, x, np.ones(len(store), dtype=bool))})
    for support, test in seen:
        assert not set(support[:, 0]) & set(test[:, 0])


def test_the_onset_protocol_alarms_near_the_labelled_fpt(root):
    summary = _run(root, "xjtu_sy").set_index("score")
    assert summary.loc["rms", "with_fpt"] == 15 and summary.loc["rms", "alarmed"] == 15
    assert summary.loc["rms", "auroc_median"] > 0.9
    assert abs(summary.loc["rms", "alarm_minus_fpt_median"]) <= 5
    results = pd.read_csv(root / "runs" / "test-xjtu_sy" / "results.csv")
    assert (results["alarm"].dropna() >= anomaly.ONSET_SUPPORT).all()  # never in the support


def test_targets_are_refused_before_stage_8(root):
    with pytest.raises(ValueError, match="held-out target"):
        _run(root, "kaist")
    with pytest.raises(ValueError, match="no anomaly protocol"):
        _run(root, "nonastreda")


def test_features_of_missing_or_uncached_arms_are_refused(root, monkeypatch):
    store = windows.WindowStore.open("cwru", root / "interim" / "windows")
    cache = FeatureStore(root / "interim" / "features")
    with pytest.raises(FileNotFoundError, match="no stats S features"):
        anomaly.encoder_features(store, cache, "stats:S")  # S exists at 12 kHz, never cached
    monkeypatch.setattr(anomaly.arms, "unavailable", lambda s: {"S"})
    with pytest.raises(ValueError, match="none of the arms"):
        anomaly.encoder_features(store, cache, "stats:S")
    x, done = anomaly.encoder_features(store, cache, "stats:F+S")  # S dropped, F kept
    assert x.shape == (len(store), 2 * 36) and done.all()
