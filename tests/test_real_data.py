"""Facts verified on the real downloads (PROJECT_LOG.md, week 1), kept as regression tests.

Each test is skipped when its dataset is not under ``PDMFM_DATA_ROOT`` (always in CI).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pdmfm.datasets import get_reader


@pytest.mark.data("cwru")
def test_cwru_inventory_and_quirks():
    reader = get_reader("cwru")
    infos = {i.recording_id: i for i in reader.index()}
    assert len(infos) == 40
    assert Counter(i.label for i in infos.values()) == {
        "normal": 4,
        "inner_race": 12,
        "ball": 12,
        "outer_race": 12,
    }
    rec98, rec99 = reader.load(infos["98"]), reader.load(infos["99"])
    assert len(rec98.channels["de"].signal) == 483_903
    assert len(rec99.channels["de"].signal) == 485_063  # its own X099, not the X098 copy
    assert rec99.info.condition["rpm_source"] == "nominal"


@pytest.mark.data("kaist", extracted=True)
def test_kaist_inventory():
    infos = get_reader("kaist").index()
    assert len(infos) == 45
    assert all(i.meta["electrical_file"] for i in infos)
    per_load = Counter(i.condition["load_nm"] for i in infos)
    assert per_load == {0: 15, 2: 15, 4: 15}
    assert len({i.unit for i in infos}) == 15


@pytest.mark.slow
@pytest.mark.data("kaist", extracted=True)
def test_kaist_device_offsets():
    reader = get_reader("kaist")
    offsets, empty = {}, {}
    for info in reader.index():
        meta = reader.load(info).info.meta
        offsets[info.recording_id] = meta["device_offset_s"]
        empty[info.recording_id] = meta["empty_channels"]
    # Currents 2 and 3 are missing in exactly the 9 outer-race files: a class-dependent gap.
    gaps = {k for k, v in empty.items() if v}
    assert gaps == {k for k in empty if "_BPFO_" in k}
    assert all(v == ["current_2", "current_3"] for k, v in empty.items() if k in gaps)
    typical = np.median(list(offsets.values()))
    assert abs(typical + 203) < 10  # about 3.4 min
    outliers = sorted(k for k, v in offsets.items() if abs(v - typical) > 600)
    assert outliers == [
        "0Nm_BPFI_03",
        "0Nm_BPFI_10",
        "0Nm_BPFO_03",
        "2Nm_BPFO_03",
        "4Nm_BPFO_03",
    ]


@pytest.mark.data("xjtu_sy", extracted=True)
def test_xjtu_sy_lifetimes_match_the_paper():
    infos = get_reader("xjtu_sy").index()
    assert len(infos) == 9216
    lifetimes = {i.unit: i.meta["lifetime_min"] for i in infos}
    assert lifetimes["Bearing3_1"] == 2538 and lifetimes["Bearing2_4"] == 42


@pytest.mark.data("najafi_thermal", extracted=True)
def test_najafi_inventory():
    infos = get_reader("najafi_thermal").index()
    assert len(infos) == 369
    assert Counter(i.label for i in infos)["healthy"] == 25


@pytest.mark.data("nasa_milling", extracted=True)
def test_nasa_milling_inventory():
    infos = get_reader("nasa_milling").index()
    assert len(infos) == 167 and len({i.unit for i in infos}) == 16
    assert sum(i.meta["wear_um"] is None for i in infos) == 21


@pytest.mark.slow
@pytest.mark.data("nonastreda")
def test_nonastreda_inventory():
    pytest.importorskip("omnifuser")
    infos = get_reader("nonastreda").index()
    assert len(infos) == 378
    assert Counter(i.label for i in infos) == {"sharp": 147, "used": 118, "dulled": 113}
    assert len({i.unit for i in infos}) == 10


@pytest.mark.data("cwru")
def test_cwru_recorder_roll_off_and_its_removal():
    """EDA finding: 48 kHz normal files keep content up to 6 kHz after resampling, the
    12 kHz fault files do not, so the top band alone tells the classes apart."""
    from pdmfm import preprocess as pp

    def share_above(x, hz=5250):
        power = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
        return power[np.fft.rfftfreq(len(x), 1 / pp.MODEL_RATE_HZ) > hz].sum() / power.sum()

    reader = get_reader("cwru")
    shares = {"normal": [], "fault": []}
    for info in reader.index():
        de = reader.load(info).channels["de"]
        resampled = pp.resample(de.signal, de.rate_hz, pp.MODEL_RATE_HZ)
        window = resampled[pp.WINDOW_SAMPLES : 2 * pp.WINDOW_SAMPLES]
        shares["normal" if info.label == "normal" else "fault"].append(share_above(window))
        limited = pp.model_input(de.signal, de.rate_hz)[0]
        assert share_above(limited[pp.WINDOW_SAMPLES : 2 * pp.WINDOW_SAMPLES]) < 1e-6
    assert min(shares["normal"]) > 10 * np.median(shares["fault"])


@pytest.mark.slow
@pytest.mark.data("kaist", extracted=True)
def test_kaist_current_fundamental_sits_below_the_shaft_rate():
    """EDA finding (open question): current 1 peaks at 48.5 Hz at every load while the shaft
    turns at 49.9-50.1 Hz, which no induction motor can do; check the device's sample rate."""
    from pdmfm import preprocess as pp

    reader = get_reader("kaist")
    for info in [i for i in reader.index() if i.label == "normal"]:
        rec = reader.load(info)
        vib, cur = rec.channels["vib_ax"], rec.channels["current_1"]
        shaft = pp.dominant_frequency(vib.signal, vib.rate_hz, 40, 60)
        supply = pp.dominant_frequency(cur.signal, cur.rate_hz, 20, 100)
        assert 49.7 < shaft < 50.3 and abs(supply - 48.5) < 0.1, (info.recording_id, shaft, supply)


# ----------------------------------------------------------------------------- acceptance
# Paderborn and PHM 2010 could not be downloaded from the development machine. These tests
# check each reader against the publisher's documentation as soon as the files are placed
# (docs/USER_ACTIONS.md); when they pass, set VERIFIED_ON_REAL_DATA = True in the reader.


@pytest.mark.data("paderborn", extracted=True)
def test_paderborn_acceptance_inventory():
    from pdmfm.datasets.paderborn import DAMAGE
    from pdmfm.sources import PADERBORN_SETTINGS

    infos = get_reader("paderborn").index()
    bearings = Counter(i.unit for i in infos)
    assert sorted(bearings) == sorted(DAMAGE), "all 32 bearings of the plan are expected"
    # KAt documentation: 4 operating settings x 20 measurements per bearing.
    assert set(bearings.values()) == {80}
    settings = Counter((i.unit, i.condition["setting"]) for i in infos)
    assert set(settings.values()) == {20}
    assert {s for _, s in settings} == set(PADERBORN_SETTINGS)
    assert all(i.label is not None for i in infos)


@pytest.mark.data("paderborn", extracted=True)
def test_paderborn_acceptance_channels_rates_and_speed():
    reader = get_reader("paderborn")
    infos = reader.index()
    for setting in sorted({i.condition["setting"] for i in infos}):
        info = next(i for i in infos if i.unit == "K001" and i.condition["setting"] == setting)
        rec = reader.load(info)
        expected = {"vibration", "current_1", "current_2", "temperature", "speed", "torque"}
        assert expected <= set(rec.channels), sorted(rec.channels)
        vib = rec.channels["vibration"]
        assert vib.rate_hz == 64_000.0 and 3.9 < vib.seconds < 4.1  # about 4 s per measurement
        assert rec.channels["current_1"].rate_hz == 64_000.0
        speed = float(np.mean(rec.channels["speed"].signal))
        rpm = info.condition["rpm"]
        assert abs(speed - rpm) < 0.05 * rpm, f"{setting}: speed {speed:.0f} vs {rpm:.0f} rpm"


@pytest.mark.data("phm2010", extracted=True)
def test_phm2010_acceptance_inventory_and_wear():
    from scipy import stats

    from pdmfm.datasets.phm2010 import LABELLED

    infos = get_reader("phm2010").index()
    cutters = Counter(i.unit for i in infos)
    assert set(LABELLED) <= set(cutters), sorted(cutters)
    for cutter in LABELLED:
        assert cutters[cutter] == 315, f"{cutter}: {cutters[cutter]} cuts (challenge: 315)"
        wear = [i.meta["wear_um"] for i in infos if i.unit == cutter]
        assert all(w is not None for w in wear), f"{cutter}: cuts without a wear value"
        assert 20 < min(wear) and max(wear) < 400, (cutter, min(wear), max(wear))  # µm
        assert stats.spearmanr(np.arange(len(wear)), wear)[0] > 0.9  # wear grows cut by cut


@pytest.mark.data("phm2010", extracted=True)
def test_phm2010_acceptance_columns_and_spindle_speed():
    """Settles the plan's open item: 10,400 rpm with a 3-flute cutter puts the tooth-passing
    line at 520 Hz in the cutting force."""
    from pdmfm import preprocess as pp
    from pdmfm.datasets.phm2010 import SPINDLE_RPM

    reader = get_reader("phm2010")
    info = next(i for i in reader.index() if i.unit == "c1" and i.meta["cut"] == 1)
    rec = reader.load(info)
    assert len(rec.channels) == 7 and len(rec.channels["fx"].signal) > 50_000
    fx = rec.channels["fx"]
    line = pp.dominant_frequency(fx.signal, fx.rate_hz, 100, 2000, seconds=1.0)
    tooth = 3 * SPINDLE_RPM / 60
    harmonic = min(abs(line - k * tooth / 3) / (k * tooth / 3) for k in (1, 2, 3, 6))
    assert harmonic < 0.02, (
        f"strongest force line {line:.1f} Hz; expected a multiple of {tooth / 3:.1f} Hz"
    )


# ----------------------------------------------------------------------------- window stores
# Built from the real downloads with `python -m pdmfm.windows build ...`; skipped otherwise.


def _store(dataset):
    from pdmfm.windows import WindowStore

    try:
        return WindowStore.open(dataset)
    except (FileNotFoundError, ValueError):
        pytest.skip(f"no {dataset} window store (python -m pdmfm.windows build {dataset})")


def test_window_counts_match_the_eda():
    expected = {"cwru": 356, "kaist": 4378, "xjtu_sy": 9216, "nonastreda": 4241}
    for dataset, count in expected.items():
        store = _store(dataset)
        assert len(store) == count, dataset


def test_xjtu_rul_labels_follow_the_recommended_fpt_rule():
    store = _store("xjtu_sy")
    first = store.manifest.groupby("unit").first()
    fraction = first["fpt"] / first["meta_lifetime_min"]
    # docs/EDA_REPORT.md section 7.2, rule "10 %, 3 sigma, 10": no alarm in the first 5 % of
    # life, the long Bearing3_1 run late in life, and Bearing1_4 (cage) via kurtosis.
    assert (fraction.dropna() > 0.05).all()
    assert abs(fraction["Bearing3_1"] - 0.924) < 0.01
    assert first.loc["Bearing1_4", "fpt_source"] == "kurtosis"


def test_kaist_and_cwru_contexts_depend_on_the_load_only():
    for dataset, load in (("cwru", "load_hp"), ("kaist", "load_nm")):
        manifest = _store(dataset).manifest
        assert manifest.groupby(load)["context"].nunique().eq(1).all(), dataset


def test_nonastreda_windows_see_only_previous_pass_photos():
    from pathlib import PurePosixPath

    manifest = _store("nonastreda").manifest
    for kind in ("tool", "chip", "workpiece"):
        has_photo = manifest[f"image_{kind}"] != ""
        assert (manifest.loc[has_photo, f"age_{kind}"] > 0).all()
        stems = manifest[f"image_{kind}"].map(lambda p: PurePosixPath(p).stem if p else "")
        assert not (stems == manifest["recording_id"]).any()  # never the pass's own photo
    start = manifest.groupby("unit")["meta_t_start"].transform("min")
    first = manifest["meta_t_start"] == start
    assert (manifest.loc[first, "image_tool"] == "").all()  # a tool's first pass has none
    # Every window of a later pass has the previous pass's photos, its first window included
    # (the photo pause between passes; pdmfm.datasets.nonastreda.PASS_GAP_S).
    assert (manifest.loc[~first, "image_tool"] != "").all()
