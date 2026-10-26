"""The EDA script: its helpers, and two analyses end to end on synthetic files.

The script lives in ``scripts/eda`` (not the package), so it is imported from its path. Its
report copies (``docs/eda``) are switched off for every test.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys

import numpy as np
import pytest
from PIL import Image
from scipy import signal as sps
from scipy.io import savemat

from pdmfm import preprocess as pp
from pdmfm.paths import REPO_ROOT

pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def eda():
    spec = importlib.util.spec_from_file_location("run_eda", REPO_ROOT / "scripts/eda/run_eda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated(eda, tmp_path, monkeypatch):
    """Data root in tmp_path; nothing is copied into docs/."""
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(eda, "COPY_TO_DOCS", False)
    return tmp_path


def tone(freq, rate, seconds, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(int(rate * seconds)) / rate)


# ----------------------------------------------------------------------------- helpers


def test_spectral_peaks_returns_a_line_family(eda):
    rate, line = 1000.0, 66.69  # Nonastreda's Fx line and its harmonics
    rng = np.random.default_rng(0)
    x = sum(tone(k * line, rate, 64, 1 / k) for k in (1, 2, 4, 7))
    x = x + 0.01 * rng.standard_normal(len(x))
    peaks = eda.spectral_peaks(x, rate, 3, 495, n=4, seconds=16)
    assert len(peaks) == 4 and peaks == sorted(peaks)
    np.testing.assert_allclose(peaks, [line, 2 * line, 4 * line, 7 * line], atol=0.1)


def test_line_prominence(eda):
    edges = np.linspace(0, 25, 513)
    flat = np.zeros(512)
    assert eda.line_prominence(flat, edges, [3.58]) == 0.0
    bump = flat.copy()
    centres = 0.5 * (edges[:-1] + edges[1:])
    bump[np.abs(centres - 3.58) < 0.05] = 1.0
    assert eda.line_prominence(bump, edges, [3.58]) > 0.5
    assert eda.line_prominence(bump, edges, [5.42]) == 0.0  # the wrong line finds nothing


def test_band_features_shape(eda):
    window = np.random.default_rng(1).standard_normal(pp.WINDOW_SAMPLES)
    features = eda.band_features(window, pp.MODEL_RATE_HZ)
    assert features.shape == (4 + 16,) and np.isfinite(features).all()


def test_top_band_ratio_and_power_share(eda):
    rng = np.random.default_rng(2)
    white = rng.standard_normal(pp.WINDOW_SAMPLES)
    assert abs(eda.top_band_ratio(white, pp.MODEL_RATE_HZ)) < 0.2
    assert 0.08 < eda.power_share_above(white, pp.MODEL_RATE_HZ) < 0.15  # 750/6000 of the band
    limited = pp.band_limit(rng.standard_normal(pp.WINDOW_SAMPLES * 3), pp.MODEL_RATE_HZ)
    middle = limited[pp.WINDOW_SAMPLES : 2 * pp.WINDOW_SAMPLES]
    assert eda.top_band_ratio(middle, pp.MODEL_RATE_HZ) < -3
    assert eda.power_share_above(middle, pp.MODEL_RATE_HZ) < 1e-7


def test_windows_of_gives_the_model_input(eda):
    x = tone(5800, 48_000, 2.0) + tone(1000, 48_000, 2.0)  # 5.8 kHz is above the common band
    windows = eda.windows_of(x, 48_000)
    assert windows.shape == (1, pp.WINDOW_SAMPLES)
    assert eda.power_share_above(windows[0], pp.MODEL_RATE_HZ) < 1e-7


def test_classify_grouped_never_trains_on_the_held_out_group(eda, monkeypatch):
    groups = np.repeat(["T1", "T2", "T3"], 4)
    x = np.column_stack([np.repeat([1.0, 2.0, 3.0], 4), np.zeros(12)])  # column 0 = group id
    y = np.array(["a", "b"] * 6)
    seen = []

    def fake_fit_predict(train_x, train_y, test_x):
        held_out = set(test_x[:, 0])
        assert held_out.isdisjoint(set(train_x[:, 0]))
        seen.append(held_out)
        return np.full(len(test_x), "a")

    monkeypatch.setattr(eda, "_fit_predict", fake_fit_predict)
    result = eda.classify_grouped(x, y, groups)
    assert sorted(min(s) for s in seen) == [1.0, 2.0, 3.0]
    assert result["folds"] == 3 and result["n"] == 12 and result["balanced_accuracy"] == 0.5


def test_classify_grouped_on_separable_data(eda):
    rng = np.random.default_rng(3)
    y = np.array(["sharp", "dulled"] * 20)
    x = (y == "dulled")[:, None] * 5.0 + rng.standard_normal((40, 2))
    result = eda.classify_grouped(x, y, np.repeat(np.arange(5), 8))
    assert result["balanced_accuracy"] == 1.0 and result["folds"] == 5


def test_default_fpt_variant_is_the_library_default(eda):
    defaults = inspect.signature(pp.first_prediction_time).parameters
    first_name, first_rule = next(iter(eda.FPT_VARIANTS.items()))
    assert first_rule == tuple(
        defaults[k].default for k in ("baseline_count", "sigmas", "consecutive")
    )
    assert first_name == "b10_3sd_c3" and eda.ROBUST_FPT in eda.FPT_VARIANTS


def test_unknown_dataset_names_are_rejected(eda, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eda.py", "cwru", "no_such_set"])
    with pytest.raises(SystemExit) as exit_info:
        eda.main()
    assert exit_info.value.code == 2


def test_nothing_is_available_in_an_empty_data_root(eda, isolated):
    assert not any(eda.available(name) for name in eda.ANALYSES)


# ----------------------------------------------------------------------------- end to end


CWRU_ORDERS = {"inner_race": 5.415, "ball": 2.357, "outer_race": 3.585}
CWRU_NUMBERS = {
    "normal": (97, 98, 99, 100),
    "inner_race": (105, 106, 107, 108),
    "ball": (118, 119, 120, 121),
    "outer_race": (130, 131, 132, 133),
}


def write_cwru(folder):
    """16 files in CWRU's format: normal at 48 kHz (flat to 6 kHz after resampling), faults
    at 12 kHz with a recorder roll-off above 5.2 kHz and impacts at their defect order."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(4)
    files = {}
    recorder = sps.butter(16, 5200, fs=12_000, output="sos")
    for kind, numbers in CWRU_NUMBERS.items():
        for load, number in enumerate(numbers):
            rpm = 1797 - 25 * load
            shaft = rpm / 60
            rate = 48_000 if kind == "normal" else 12_000
            t = np.arange(int(rate * 5.2)) / rate
            x = 0.05 * rng.standard_normal(len(t)) + 0.03 * np.sin(2 * np.pi * shaft * t)
            if kind != "normal":
                impacts = np.sin(2 * np.pi * CWRU_ORDERS[kind] * shaft * t) > 0.95
                x = x + 0.5 * impacts * np.sin(2 * np.pi * 3000 * t)
                x = sps.sosfiltfilt(recorder, x)
            savemat(
                folder / f"{number}.mat",
                {f"X{number:03d}_DE_time": x[:, None], f"X{number:03d}RPM": rpm},
            )
            files[number] = (kind, 0.0 if kind == "normal" else 0.007, load)
    return files


def test_cwru_eda_end_to_end(eda, isolated, monkeypatch):
    files = write_cwru(isolated / "data" / "raw" / "cwru")
    monkeypatch.setattr("pdmfm.datasets.cwru.CWRU_FILES", files)
    out = isolated / "out"
    summary = eda.eda_cwru(out, quick=False)
    json.dumps(summary, default=str)  # the script writes it as JSON
    assert summary["recordings"] == 16
    assert summary["classes"] == {"normal": 4, "inner_race": 4, "ball": 4, "outer_race": 4}
    assert summary["windows_1s_total"] == 16 * 5
    # The tachless estimate recovers every header speed.
    assert summary["rpm_vs_vibration_pct_max_abs"] < 0.5
    # The recorder roll-off separates normal from fault on its own ...
    artifact = summary["top_band_ratio_log10_5p7_6khz_minus_4_5khz"]
    assert artifact["normal_median"] > artifact["fault_median"] + 1
    assert artifact["normal_vs_fault_cross_load_from_this_one_feature"]["balanced_accuracy"] == 1.0
    # ... and the common band limit leaves no power there in either class.
    shares = artifact["power_share_above_5p25khz_median"]
    assert shares["resampled_only"]["normal"] > 10 * shares["resampled_only"]["fault"]
    assert max(shares["band_limited"].values()) < 1e-7
    prominence = summary["envelope_line_prominence"]["outer_race"]
    assert prominence["fault"] > prominence["normal"] + 0.1
    assert summary["cross_load_baseline_handcrafted"]["n_test"] == 4 * 5
    assert (out / "envelope_orders.png").is_file() and (out / "levels_by_class.png").is_file()


def write_najafi(folder):
    """Four conditions, ten frames each, drifting slowly: a frame sequence like the real set."""
    rng = np.random.default_rng(5)
    for name in ("Noload", "A50", "Fan", "Rotor-0"):
        (folder / name).mkdir(parents=True)
        base = rng.uniform(0, 255, (240, 320, 3))
        for frame in range(10):
            image = np.clip(base + 2.0 * frame + rng.normal(0, 1, base.shape), 0, 255)
            Image.fromarray(image.astype(np.uint8)).save(folder / name / f"{frame:03d}.bmp")


def test_najafi_eda_end_to_end(eda, isolated):
    write_najafi(isolated / "data" / "extracted" / "najafi_thermal")
    summary = eda.eda_najafi(isolated / "out", quick=False)
    json.dumps(summary, default=str)
    assert summary["images"] == 40
    assert summary["classes"] == {
        "healthy": 10,
        "stator_short_circuit": 10,
        "cooling_fan_failure": 10,
        "stuck_rotor": 10,
    }
    # Every frame's nearest neighbour is a neighbouring frame of the same folder.
    assert summary["share_of_images_whose_nearest_neighbour_is_in_the_same_folder"] == 1.0
    assert summary["share_whose_same_folder_neighbour_is_within_2_files"] == 1.0
    splits = summary["pixel_baseline_by_split"]
    assert splits["last_30pct_of_each_folder"]["n_test"] == 4 * 3  # frames 7, 8, 9 per folder
    assert splits["random_70_30"]["n_train"] + splits["random_70_30"]["n_test"] == 40
