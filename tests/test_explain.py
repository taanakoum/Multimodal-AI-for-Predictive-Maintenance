"""Band occlusion, the physics alignment score and the faithfulness checks (pdmfm.explain).

The physics tests use signals whose lines are known: a tone at BPFO in the raw spectrum, and
a resonance amplitude-modulated at BPFO (a line in the envelope spectrum only). A model that
reads only the defect line's bin must score PAS well above 1; the same model with its weights
permuted must not, on average; the deletion test must rank the true model's bands first.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from synthetic import build_planb_root

from pdmfm import arms, explain
from pdmfm import preprocess as pp
from pdmfm.datasets.cwru import GEOMETRY

RATE = pp.MODEL_RATE_HZ
L = pp.WINDOW_SAMPLES
SHAFT = 30.0
BPFO_HZ = pp.defect_frequencies(**GEOMETRY, shaft_hz=SHAFT)["BPFO"]  # 107.5 Hz


def quiet(*_):
    pass


class Identity:
    """Encoder stand-in: the 512-vector itself is the feature."""

    dim = 512

    def encode(self, x):
        return np.asarray(x, dtype=np.float32)


class Linear:
    """Logistic model over the features: P(fault) = sigmoid(coef . x + intercept)."""

    def __init__(self, coef: np.ndarray, intercept: float) -> None:
        self.coef_ = np.asarray(coef, dtype=float)[None, :]
        self.intercept_ = np.array([intercept])
        self.classes_ = np.array(["normal", "outer_race"])

    def predict_proba(self, x):
        p = 1.0 / (1.0 + np.exp(-(np.asarray(x) @ self.coef_[0] + self.intercept_[0])))
        return np.stack([1 - p, p], axis=1)


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(L).astype(np.float32)


def tone_window(seed: int = 0, amplitude: float = 0.5) -> np.ndarray:
    """(1, L): white noise plus a tone at BPFO."""
    t = np.arange(L) / RATE
    return (_noise(seed) + amplitude * np.sin(2 * np.pi * BPFO_HZ * t))[None].astype(np.float32)


def modulated_window(seed: int = 0) -> np.ndarray:
    """(1, L): a 3 kHz resonance amplitude-modulated at BPFO, plus noise: the defect line is
    in the envelope spectrum, not in the raw one."""
    t = np.arange(L) / RATE
    carrier = np.sin(2 * np.pi * 3000.0 * t) * (1.0 + 0.8 * np.sin(2 * np.pi * BPFO_HZ * t))
    return (0.3 * _noise(seed) + carrier)[None].astype(np.float32)


def line_model(arm: str, bin_index: int, base_value: float) -> explain.ProbeModel:
    coef = np.zeros(512)
    coef[bin_index] = 8.0
    return explain.ProbeModel(
        rate=RATE, arm_list=(arm,), channels={arm: (0,)}, encoder=Identity(),
        classifier=Linear(coef, -8.0 * base_value), classes=["normal", "outer_race"],
    )  # fmt: skip


def f_bin(frequency: float) -> int:
    return int(frequency / (pp.COMMON_BAND_HZ / 512))


def e_bin(order: float) -> int:
    return int(order / (explain.MAX_ORDER / 512))


# ----------------------------------------------------------------------------- occlusion


def test_line_removal_removes_a_tone_inside_the_band_only():
    t = np.arange(L) / RATE
    x = 0.1 * _noise(1) + np.sin(2 * np.pi * 1000.0 * t) + np.sin(2 * np.pi * 2000.0 * t) + 3.0
    y = explain.remove_lines(x, RATE, [(960.0, 1040.0)])

    def amplitude(signal, f):
        spectrum = np.abs(np.fft.rfft(signal))
        return spectrum[int(round(f * L / RATE))]

    assert amplitude(y, 1000.0) < 0.05 * amplitude(x, 1000.0)  # > 26 dB down
    assert amplitude(y, 2000.0) == pytest.approx(amplitude(x, 2000.0), rel=1e-6)
    assert y.mean() == pytest.approx(x.mean(), abs=1e-4)  # DC untouched


def test_line_removal_keeps_the_broadband_floor():
    x = _noise(2)
    y = explain.remove_lines(x, RATE, [(1000.0, 1078.0)])
    assert np.std(y) == pytest.approx(np.std(x), rel=0.02)  # one band of 64: floor kept


def test_flattening_bins_removes_a_peak_and_nothing_else():
    spectrum = np.zeros((2, 512), dtype=np.float32)
    spectrum[:, 100] = 5.0
    spectrum[:, 300] = 5.0
    out = explain.flatten_bins(spectrum, [(96, 104)])
    assert out[0, 100] == 0.0 and out[0, 300] == 5.0
    assert spectrum[0, 100] == 5.0  # a copy


# ----------------------------------------------------------------------------- physics


def test_defect_lines_follow_the_bearing_formulas():
    hz = explain.defect_lines(GEOMETRY, "outer_race", SHAFT)
    np.testing.assert_allclose(hz, BPFO_HZ * np.arange(1, 6))
    orders = explain.defect_lines(GEOMETRY, "outer_race")
    np.testing.assert_allclose(orders * SHAFT, hz)
    inner = explain.defect_lines(GEOMETRY, "inner race", SHAFT)  # XJTU-SY spelling
    bpfi = pp.defect_frequencies(**GEOMETRY, shaft_hz=SHAFT)["BPFI"]
    assert len(inner) == 15 and np.isclose(inner, bpfi - SHAFT).any()
    assert np.isclose(inner, 5 * bpfi + SHAFT).any()
    with pytest.raises(ValueError, match="no defect lines"):
        explain.defect_lines(GEOMETRY, "normal")


def test_physics_mask_marks_the_bands_around_each_line():
    edges = explain.band_edges("hz")
    assert len(edges) == explain.N_BANDS + 1 and edges[-1] == pp.COMMON_BAND_HZ
    mask = explain.physics_mask(edges, np.array([900.0, -5.0]))  # ±2 %: 882-918 Hz, one band
    width = edges[1]
    assert np.flatnonzero(mask).tolist() == [int(900.0 // width)]
    # 1000 Hz ± 2 % (980-1020 Hz) crosses the band edge at 1015.6 Hz: both bands count.
    assert np.flatnonzero(explain.physics_mask(edges, np.array([1000.0]))).tolist() == [12, 13]
    # A line at a band edge (±2 % reaches both sides) marks both bands.
    edge_line = edges[20]
    assert np.flatnonzero(explain.physics_mask(edges, np.array([edge_line]))).tolist() == [19, 20]
    with pytest.raises(ValueError, match="axis"):
        explain.band_edges("mel")


def test_pas_values():
    mask = np.zeros(64, dtype=bool)
    mask[:4] = True
    concentrated = np.zeros(64)
    concentrated[2] = 1.0
    assert explain.pas(concentrated, mask) == pytest.approx(16.0)  # 1 / (4 / 64)
    assert explain.pas(np.ones(64), mask) == pytest.approx(1.0)  # uniform: chance
    assert np.isnan(explain.pas(np.zeros(64), mask))
    assert np.isnan(explain.pas(-np.ones(64), mask))  # no positive evidence
    assert np.isnan(explain.pas(np.ones(64), np.zeros(64, dtype=bool)))


# ----------------------------------------------------------------------------- models


def test_a_model_reading_the_bpfo_line_scores_high_on_the_hz_axis():
    window = tone_window()
    level = arms.window_input(window[None], RATE, "F")[0, 0, 0, f_bin(BPFO_HZ)]
    model = line_model("F", f_bin(BPFO_HZ), level - 0.3)
    base, attribution, target = explain.occlusion(model, window, SHAFT, "hz")
    assert target == 1 and base[1] > 0.9
    edges = explain.band_edges("hz")
    assert int(np.argmax(attribution)) == int(BPFO_HZ // edges[1])
    mask = explain.physics_mask(edges, explain.defect_lines(GEOMETRY, "outer_race", SHAFT))
    assert explain.pas(attribution, mask) > 5.0


def test_randomized_weights_score_near_chance_on_average():
    window = tone_window()
    level = arms.window_input(window[None], RATE, "F")[0, 0, 0, f_bin(BPFO_HZ)]
    model = line_model("F", f_bin(BPFO_HZ), level - 0.3)
    edges = explain.band_edges("hz")
    mask = explain.physics_mask(edges, explain.defect_lines(GEOMETRY, "outer_race", SHAFT))
    base_x = model.features(model.inputs(window[None], np.array([SHAFT])))
    band_x = explain.variant_features(model, window, SHAFT, "hz", explain.single_bands())
    scores = []
    for seed in range(30):
        random_model = explain.randomized(model, seed)
        assert not np.array_equal(random_model.classifier.coef_, model.classifier.coef_)  # pyright: ignore[reportAttributeAccessIssue]
        _, attribution, _ = explain.attribution_of(random_model, base_x, band_x)
        scores.append(explain.pas(attribution, mask))
    scores = np.array(scores)
    assert np.nanmean(np.nan_to_num(scores)) < 3.0  # expected 1; the true model scores > 5
    assert np.nanmedian(np.nan_to_num(scores)) < 1.0


def test_an_envelope_line_is_found_on_the_order_axis():
    window = modulated_window()
    order = BPFO_HZ / SHAFT
    e_level = arms.window_input(window[None], RATE, "E", np.array([SHAFT]))[0, 0, 0, e_bin(order)]
    model = line_model("E", e_bin(order), e_level - 0.3)
    base, attribution, target = explain.occlusion(model, window, SHAFT, "order")
    assert target == 1 and base[1] > 0.9
    mask = explain.physics_mask(
        explain.band_edges("order"), explain.defect_lines(GEOMETRY, "outer_race")
    )
    # Every bit of attribution lies in the physical bands: the largest possible score.
    assert explain.pas(attribution, mask) == pytest.approx(1 / mask.mean())
    assert int(np.argmax(attribution)) == int(order / (explain.MAX_ORDER / explain.N_BANDS))
    # The raw spectrum of a modulated resonance holds no line at BPFO: the hz axis misses it.
    f_level = arms.window_input(window[None], RATE, "F")[0, 0, 0, f_bin(BPFO_HZ)]
    raw_model = line_model("F", f_bin(BPFO_HZ), f_level - 0.3)
    assert raw_model.predict(window[None], np.array([SHAFT]))[0, 1] > 0.9  # reads its own bin
    with pytest.raises(ValueError, match="E arm"):
        explain.occlusion(raw_model, window, SHAFT, "order")


def test_the_deletion_test_ranks_a_faithful_attribution_first():
    window = tone_window()
    level = arms.window_input(window[None], RATE, "F")[0, 0, 0, f_bin(BPFO_HZ)]
    model = line_model("F", f_bin(BPFO_HZ), level - 0.3)
    _, attribution, target = explain.occlusion(model, window, SHAFT, "hz")
    curve = explain.deletion_curve(model, window, SHAFT, "hz", np.argsort(-attribution), target)
    assert len(curve) == explain.DELETION_STEPS + 1 and curve[0] > 0.9 and curve[-1] < 0.5
    rng = np.random.default_rng(0)
    random_areas = [
        explain.area(
            explain.deletion_curve(model, window, SHAFT, "hz", rng.permutation(64), target)
        )
        for _ in range(5)
    ]
    assert explain.area(curve) < np.mean(random_areas) - 0.2
    assert explain.area(np.ones(5)) == pytest.approx(1.0)


def test_explain_windows_and_the_summary():
    windows = np.stack([tone_window(s) for s in range(4)])
    level = arms.window_input(windows[:1], RATE, "F")[0, 0, 0, f_bin(BPFO_HZ)]
    model = line_model("F", f_bin(BPFO_HZ), level - 0.3)
    table, attributions = explain.explain_windows(
        model, windows, np.full(4, SHAFT), ["outer_race"] * 4, GEOMETRY, ("hz",), log=quiet
    )
    assert len(table) == 4 and attributions["hz"].shape == (4, explain.N_BANDS)
    summary = explain.summarize(table)["hz"]
    assert summary["pas_median"] > 5.0 and summary["pas_above_1"] == 1.0
    assert summary["deletion_area"] < summary["deletion_area_random"]
    assert set(summary["by_fault"]) == {"outer_race"}


# ----------------------------------------------------------------------------- real pipeline


@pytest.fixture(scope="module")
def cwru_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("explain")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root, datasets=("cwru",))
        yield root


def test_online_features_equal_the_cache(cwru_root):
    model, store, rows, cached = explain.fit_probe(
        "cwru", "stats", ("F", "E"), cwru_root / "interim" / "windows",
        cwru_root / "interim" / "features",
    )  # fmt: skip
    assert set(store.manifest.loc[rows, "load_hp"]) == {3}  # explained on the held-out load
    windows = np.asarray(store.signals[rows], dtype=np.float32)
    shaft = store.manifest["shaft_hz"].to_numpy(dtype=float)[rows]
    online = model.features(model.inputs(windows, shaft))
    np.testing.assert_allclose(online, cached[rows], rtol=2e-3, atol=2e-3)  # float16 cache
    assert model.predict(windows, shaft).shape == (len(rows), len(model.classes))


def test_run_writes_the_tables(cwru_root):
    out = cwru_root / "runs"
    summary = explain.run(
        "cwru",
        "stats",
        ("F", "E"),
        "test-run",
        out_root=out,
        log=quiet,
        window_root=cwru_root / "interim" / "windows",
        feature_root=cwru_root / "interim" / "features",
    )
    folder = out / "test-run"
    assert set(summary["axes"]) == {"hz", "order"}
    table = __import__("pandas").read_csv(folder / "windows.csv")
    assert set(table["fault"]) <= {"inner_race", "outer_race", "ball"}  # faulty windows only
    assert set(table["axis"]) == {"hz", "order"}
    saved = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert saved["online_cached_agreement"] == 1.0
    assert "Band occlusion" in (folder / "summary.md").read_text(encoding="utf-8")
    assert np.load(folder / "attributions.npz")["hz"].shape == (len(table) // 2, explain.N_BANDS)


def test_only_cwru_has_a_physics_probe(cwru_root):
    with pytest.raises(ValueError, match="CWRU"):
        explain.fit_probe("nonastreda", "stats", ("F",))
    with pytest.raises(ValueError, match="geometry"):
        explain.geometry("nonastreda")
    assert explain.geometry("xjtu_sy")["n_balls"] == 8
