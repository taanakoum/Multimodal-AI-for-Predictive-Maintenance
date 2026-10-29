"""Window stores (pdmfm.windows) and encoder input arms (pdmfm.arms) on synthetic readers."""

from __future__ import annotations

import gc
import json
import sys

import numpy as np
import pandas as pd
import pytest
from synthetic import FakeReader, cwru_reader, nonastreda_reader, xjtu_reader

from pdmfm import arms, windows
from pdmfm import preprocess as pp
from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def cwru(root):
    reader = cwru_reader()
    return reader, windows.build("cwru", reader=reader, log=lambda *_: None)


def test_cwru_store_holds_the_model_input(cwru):
    reader, store = cwru
    assert store.signals.shape == (32, 2, pp.WINDOW_SAMPLES)  # 16 recordings x 2 windows
    assert store.rate == 12_000.0 and store.families == ("vibration", "vibration")
    first = reader.load(reader.index()[0])
    expected = pp.model_input(first.channels["fe"].signal, 12_000.0)[0]
    row = store.manifest.index[
        (store.manifest["recording_id"] == first.info.recording_id)
        & (store.manifest["window"] == 1)
    ][0]
    np.testing.assert_array_equal(
        store.signals[row, 1], expected[pp.WINDOW_SAMPLES : 2 * pp.WINDOW_SAMPLES]
    )
    np.testing.assert_allclose(
        store.levels[row, 1], pp.level_descriptors(store.signals[row, 1]), rtol=1e-5
    )
    assert store.slow.shape == (32, 0, 2)


def test_cwru_manifest_labels_contexts_and_speed(cwru):
    _, store = cwru
    m = store.manifest
    assert set(m["class_key"]) == {
        "normal",
        "inner_race/0.007 in",
        "outer_race/0.007 in",
        "ball/0.007 in",
    }
    assert m["load_hp"].dtype.kind == "i" and pd.api.types.is_string_dtype(m["unit"])
    # The context depends on the load only (nominal speed), never on the recording.
    assert m.groupby("load_hp")["context"].nunique().eq(1).all() and m["context"].nunique() == 4
    assert np.allclose(m["shaft_hz"], m["rpm"] / 60, rtol=5e-3) and set(m["shaft_source"]) == {
        "measured"
    }


def test_rebuild_is_atomic_and_deterministic(root):
    reader = cwru_reader()
    store = windows.build("cwru", reader=reader, log=lambda *_: None)
    key, signals = store.key, np.array(store.signals)
    del store  # release the memory map: Windows cannot rename a folder that is mapped
    gc.collect()
    again = windows.build("cwru", reader=reader, log=lambda *_: None)
    assert again.key == key
    np.testing.assert_array_equal(np.asarray(again.signals), signals)
    folder = root / "interim" / "windows"
    assert sorted(p.name for p in folder.iterdir()) == ["cwru"]  # no .partial or .old left
    del again
    gc.collect()
    smaller = windows.build("cwru", reader=reader, limit=2, log=lambda *_: None)
    assert smaller.key != key and len(smaller) == 4


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows locks mapped files")
def test_rebuild_while_open_explains_itself(cwru, root):
    reader, _ = cwru
    with pytest.raises(PermissionError, match="still reads it"):
        windows.build("cwru", reader=reader, log=lambda *_: None)
    assert sorted(p.name for p in (root / "interim" / "windows").iterdir()) == ["cwru"]


def test_signature_records_preprocessing_and_source(cwru, root):
    _, store = cwru
    signature = store.signature
    assert signature["preprocess_version"] == pp.PREPROCESS_VERSION
    assert signature["common_band_hz"] == pp.COMMON_BAND_HZ and signature["source"] is None
    manifest = root / "raw" / "cwru" / "MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"files": {"97.mat": {"sha256": "abc", "fetched": "t1"}}}))
    first = windows.source_fingerprint("cwru")
    manifest.write_text(json.dumps({"files": {"97.mat": {"sha256": "abc", "fetched": "t2"}}}))
    assert windows.source_fingerprint("cwru") == first  # a rewrite alone changes nothing
    manifest.write_text(json.dumps({"files": {"97.mat": {"sha256": "def"}}}))
    assert windows.source_fingerprint("cwru") != first


def test_incomplete_or_missing_store_is_refused(cwru, root):
    _, store = cwru
    meta = json.loads((store.folder / "meta.json").read_text())
    meta["complete"] = False
    (store.folder / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="incomplete"):
        windows.WindowStore.open("cwru")
    with pytest.raises(FileNotFoundError):
        windows.WindowStore.open("kaist")


def test_missing_channel_is_an_error(root):
    reader = cwru_reader()
    recording = next(iter(reader.recordings.values()))
    del recording.channels["fe"]
    with pytest.raises(ValueError, match="missing"):
        windows.build("cwru", reader=reader, log=lambda *_: None)
    assert not (root / "interim" / "windows" / "cwru").exists()


def test_xjtu_windows_and_rul_labels(root):
    store = windows.build("xjtu_sy", reader=xjtu_reader(snapshots=30), log=lambda *_: None)
    m = store.manifest
    assert len(store) == 15 * 30 and set(m["shaft_source"]) == {"nominal"}
    for bearing, group in m.groupby("unit"):
        group = group.sort_values("meta_minute")
        rul = group["rul"].to_numpy()
        fpt = int(group["fpt"].iloc[0])
        assert group["fpt_source"].iloc[0] == "rms"
        assert 15 <= fpt <= 22, (bearing, fpt)  # degradation starts after 60 % of 30 snapshots
        assert (rul[:fpt] == 1).all() and rul[-1] == 0 and (np.diff(rul[fpt:]) < 0).all()


def test_nonastreda_attaches_only_previous_pass_photos(root):
    photos = root / "raw" / "nonastreda" / "photos"
    store = windows.build(
        "nonastreda", reader=nonastreda_reader(photo_root=photos), log=lambda *_: None
    )
    m = store.manifest
    assert store.rate == 1000.0 and store.window == 8192 and len(store) == 10 * 6 * 2
    first_pass = m["meta_block"] == 1
    assert (m.loc[first_pass, "image_tool"] == "").all()  # a tool's first pass has no photo
    later = m[~first_pass]
    for _, row in later.iterrows():
        previous = f"{row['unit']}R1B{int(row['meta_block']) - 1}"
        assert row["image_tool"] == f"raw/nonastreda/photos/tool-{previous}.png"
        assert row["age_tool"] > 0
    assert set(m["class_key"]) == {"sharp", "used", "dulled"} and m["wear_um"].notna().all()
    # At 1 kHz the band limit does nothing: the store holds the raw force.
    reader = nonastreda_reader(photo_root=photos)
    raw = reader.load(reader.index()[0]).channels["fx"].signal
    np.testing.assert_array_equal(store.signals[0, 0], raw[:8192])


def paderborn_reader():
    rng = np.random.default_rng(0)
    recordings = []
    for bearing, label in (("K001", "healthy"), ("KA01", "outer_race"), ("KB27", "combined")):
        n = 4 * 64_000
        channels = {
            name: ChannelData(
                name, family, "none", placement, 64_000.0, rng.standard_normal(n).astype(np.float32)
            )
            for name, family, placement in (
                ("vibration", "vibration", "housing"),
                ("current_1", "current", "motor_phase"),
                ("current_2", "current", "motor_phase"),
            )
        }
        shaft_line = np.sin(2 * np.pi * 25.0 * np.arange(n) / 64_000)  # 1500 rpm
        vibration = (shaft_line + 0.1 * rng.standard_normal(n)).astype(np.float32)
        channels["vibration"] = ChannelData(
            "vibration", "vibration", "none", "housing", 64_000.0, vibration
        )
        channels["temperature"] = ChannelData(
            "temperature",
            "slow",
            "none",
            "bearing_module",
            1.0,
            np.array([40.0, 41.0, 42.0, 43.0], np.float32),
        )
        condition = {"setting": "N15_M07_F10", "rpm": 1500.0, "torque_nm": 0.7, "radial_n": 1000.0}
        info = RecordingInfo(
            "paderborn", f"N15_M07_F10_{bearing}_1", bearing, label, None, condition
        )
        recordings.append(Recording(info, channels))
    return FakeReader(recordings)


def test_paderborn_like_store_slow_channels_and_label_space(root):
    store = windows.build("paderborn", reader=paderborn_reader(), log=lambda *_: None)
    m = store.manifest
    assert len(store) == 3 * 3  # 4 s at 12 kHz holds 3 windows of 1.024 s
    assert store.families == ("vibration", "current", "current")
    assert store.slow.shape == (9, 1, 2)
    np.testing.assert_allclose(
        store.slow[:3, 0, 0], [40.0, 41.0, 42.0]
    )  # one 1 Hz sample per window
    assert m.loc[m["unit"] == "KB27", "class_key"].eq("").all()  # outside the 3-class space
    assert np.allclose(m["shaft_hz"], 25.0, rtol=2e-3)


# ----------------------------------------------------------------------------- arms


def test_t_arm_is_the_window_in_segments(cwru):
    _, store = cwru
    x = arms.arm_input(store, np.array([0, 5]), "T")
    assert x.shape == (2, 2, 24, 512) and arms.n_tokens(store, "T") == 6
    np.testing.assert_array_equal(x[1, 0].ravel(), store.signals[5, 0])


def test_f_and_e_arms_match_preprocessing(cwru):
    _, store = cwru
    f = arms.arm_input(store, np.array([3]), "F")
    np.testing.assert_allclose(
        f[0, 1, 0], pp.log_spectrum(store.signals[3, 1], 12_000.0)[0], rtol=1e-5
    )
    e = arms.arm_input(store, np.array([3]), "E")
    shaft = store.manifest.loc[3, "shaft_hz"]
    expected = pp.envelope_order_spectrum(store.signals[3, 0], 12_000.0, shaft)[0]
    np.testing.assert_allclose(e[0, 0, 0], expected, rtol=1e-5)
    assert arms.arm_input(store, np.array([0]), "S").shape == (1, 2, 1, pp.WINDOW_SAMPLES)


def test_e_arm_needs_a_bearing_dataset(root):
    store = windows.build("nonastreda", reader=nonastreda_reader(), log=lambda *_: None)
    with pytest.raises(ValueError, match="E arm"):
        arms.arm_channels(store, "E")
    assert arms.n_tokens(store, "T") == 4  # 16 segments of the 8.192 s force window
    with pytest.raises(ValueError, match="unknown arm"):
        arms.arm_channels(store, "X")


def test_pool_tokens_averages_groups_of_four():
    embeddings = np.arange(2 * 1 * 8 * 3, dtype=float).reshape(2, 1, 8, 3)
    pooled = arms.pool_tokens(embeddings, "T")
    assert pooled.shape == (2, 1, 2, 3)
    np.testing.assert_allclose(pooled[0, 0, 0], embeddings[0, 0, :4].mean(axis=0))
    assert arms.pool_tokens(embeddings, "F") is embeddings


def test_info_command(cwru, capsys):
    windows.main(["info"])
    out = capsys.readouterr().out
    assert "cwru" in out and "32 windows" in out


def test_unavailable_arms(cwru, root):
    _, store = cwru
    assert arms.unavailable(store) == set()
    nonastreda = windows.build("nonastreda", reader=nonastreda_reader(), log=lambda *_: None)
    assert arms.unavailable(nonastreda) == {"E", "S"}  # no bearings; 1 kHz force is not audio
    with pytest.raises(ValueError, match="no S arm"):
        arms.arm_channels(nonastreda, "S")


def test_a_store_built_by_other_code_is_refused(root):
    store = windows.build("cwru", reader=cwru_reader(), log=lambda *_: None)
    folder = store.folder
    meta = json.loads((folder / "meta.json").read_text())
    del store
    gc.collect()
    meta["signature"]["preprocess_version"] = pp.PREPROCESS_VERSION - 1
    (folder / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="built by other code.*preprocess_version"):
        windows.WindowStore.open("cwru")


def test_a_smoke_build_never_replaces_the_full_store(root):
    full = windows.build("cwru", reader=cwru_reader(), log=lambda *_: None)
    smoke = windows.build("cwru", reader=cwru_reader(), limit=2, log=lambda *_: None)
    assert smoke.folder.name == "cwru-limit2" and smoke.limit == 2 and len(smoke) == 4
    assert windows.WindowStore.open("cwru").key == full.key and len(full) == 32


def test_a_snapshot_too_short_for_a_window_does_not_shift_rul_labels(root):
    reader = xjtu_reader(snapshots=30)
    short = reader.recordings["Bearing1_1/10"]
    for name, channel in short.channels.items():  # 0.5 s instead of 1.08 s
        short.channels[name] = type(channel)(
            **{**channel.__dict__, "signal": channel.signal[:6000]}
        )
    store = windows.build("xjtu_sy", reader=reader, log=lambda *_: None)
    bearing = store.manifest[store.manifest["unit"] == "Bearing1_1"]
    assert len(bearing) == 29 and "Bearing1_1/10" not in set(bearing["recording_id"])
    rul = bearing.sort_values("meta_minute")["rul"].to_numpy()
    assert rul[-1] == 0 and (np.diff(rul) <= 0).all()  # one label per window, in order
    assert store.manifest["rul"].notna().all()
