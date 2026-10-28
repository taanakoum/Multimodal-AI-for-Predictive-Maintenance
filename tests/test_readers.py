"""Readers on synthetic files written in each publisher's real format (no data needed)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from pdmfm.datasets import get_reader
from pdmfm.datasets.kaist import normalize_name, parse_condition
from pdmfm.datasets.najafi import parse_folder

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- CWRU


def test_cwru_reads_own_variables_and_falls_back_to_nominal_rpm(tmp_path, monkeypatch):
    from pdmfm import sources

    monkeypatch.setattr(
        "pdmfm.datasets.cwru.CWRU_FILES",
        {97: ("normal", 0.0, 0), 99: ("normal", 0.0, 2), 105: ("inner_race", 0.007, 0)},
    )
    de97, de98, de99 = (RNG.standard_normal((480, 1)) for _ in range(3))
    savemat(tmp_path / "97.mat", {"X097_DE_time": de97, "X097_FE_time": de97, "X097RPM": 1796})
    # 99.mat carries a copy of file 98, as the real file does.
    savemat(tmp_path / "99.mat", {"X098_DE_time": de98, "X099_DE_time": de99, "X099_FE_time": de99})
    savemat(tmp_path / "105.mat", {"X105_DE_time": de97[:120], "X105RPM": 1797})
    reader = get_reader("cwru", root=tmp_path)
    infos = {i.recording_id: i for i in reader.index()}
    assert infos["105"].unit == "inner_race_0.007in" and infos["105"].severity == "0.007 in"
    rec99 = reader.load(infos["99"])
    np.testing.assert_allclose(rec99.channels["de"].signal, de99.ravel(), rtol=1e-6)
    assert rec99.info.condition == {"load_hp": 2, "rpm": 1750.0, "rpm_source": "nominal"}
    rec97 = reader.load(infos["97"])
    assert rec97.channels["de"].rate_hz == 48_000.0 and rec97.info.condition["rpm"] == 1796.0
    assert reader.load(infos["105"]).channels["de"].rate_hz == 12_000.0
    assert sources.CWRU_FILES is not None


# ----------------------------------------------------------------------------- KAIST


@pytest.mark.parametrize(
    "name, expected",
    [
        ("0Nm_Normal.mat", {"label": "normal", "severity": None, "load_nm": 0}),
        ("2Nm_BPFI_03", {"label": "inner_race", "severity": "0.3 mm", "load_nm": 2}),
        ("4Nm_BPFO_30.tdms", {"label": "outer_race", "severity": "3.0 mm", "load_nm": 4}),
        ("0Nm_Misalign_05", {"label": "misalignment", "severity": "0.5 mm", "load_nm": 0}),
        ("2Nm_Unbalalnce_1169mg.mat", {"label": "unbalance", "severity": "1169 mg", "load_nm": 2}),
    ],
)
def test_kaist_condition_names(name, expected):
    assert parse_condition(name) == expected


def test_kaist_name_normalization_and_rejection():
    assert normalize_name("2Nm_Unbalalnce_0583mg.mat") == "2Nm_Unbalance_0583mg"
    with pytest.raises(ValueError):
        parse_condition("5Nm_Broken")


def _write_kaist(
    root,
    stem,
    n=2560,
    vib_time="2021-06-28 11:32:36 ms 799.212",
    ele_time="2021-06-28T11:29:13.000000",
    empty=(),
):
    from nptdms import ChannelObject, GroupObject, TdmsWriter

    (root / "vibration").mkdir(parents=True, exist_ok=True)
    (root / "current,temp").mkdir(parents=True, exist_ok=True)
    signal = {
        "x_values": {"start_value": 0.0, "increment": 1 / 25_600, "number_of_values": n},
        "y_values": {"values": RNG.standard_normal((n, 4))},
        "function_record": {"TL_export_properties_annotation": {"absolute_time": vib_time}},
    }
    savemat(root / "vibration" / f"{stem}.mat", {"Signal": signal})
    channels = []
    for module, index, unit in (
        ("Mod1", 0, "°C"),
        ("Mod1", 1, "°C"),
        ("Mod2", 0, "A"),
        ("Mod2", 2, "A"),
        ("Mod2", 3, "A"),
    ):
        name = f"cDAQ9185-1F486B5{module}/ai{index}"
        if f"{module}/ai{index}" in empty:  # as in the real outer-race files
            channels.append(ChannelObject("Log", name, np.zeros(0), properties={"wf_samples": 1}))
            continue
        channels.append(
            ChannelObject(
                "Log",
                name,
                RNG.standard_normal(n + 1),
                properties={
                    "wf_increment": 3.905e-05,
                    "wf_start_time": np.datetime64(ele_time),
                    "unit_string": unit,
                },
            )
        )
    with TdmsWriter(str(root / "current,temp" / f"{stem}.tdms")) as writer:
        writer.write_segment([GroupObject("Test Information"), *channels])


def test_kaist_reader_combines_both_devices(tmp_path):
    _write_kaist(tmp_path, "0Nm_BPFI_03")
    _write_kaist(tmp_path, "2Nm_Unbalalnce_0583mg")
    reader = get_reader("kaist", root=tmp_path, raw=tmp_path / "missing")
    infos = {i.recording_id: i for i in reader.index()}
    assert set(infos) == {"0Nm_BPFI_03", "2Nm_Unbalance_0583mg"}
    rec = reader.load(infos["0Nm_BPFI_03"])
    assert {k for k, c in rec.channels.items() if c.family == "vibration"} == {
        "vib_ax",
        "vib_ay",
        "vib_bx",
        "vib_by",
    }
    assert rec.channels["vib_ax"].rate_hz == 25_600.0
    assert rec.channels["temp_1"].family == "slow" and rec.channels["temp_1"].unit == "°C"
    assert rec.channels["current_3"].family == "current"
    assert abs(rec.channels["current_1"].rate_hz - 25_608.19) < 0.1
    assert rec.info.meta["device_offset_s"] == pytest.approx(-203.0)  # 3 min 23 s apart
    assert infos["2Nm_Unbalance_0583mg"].unit == "unbalance_583mg"
    vibration_only = reader.load(infos["0Nm_BPFI_03"], electrical=False)
    assert set(vibration_only.channels) == {"vib_ax", "vib_ay", "vib_bx", "vib_by"}


def test_kaist_reader_skips_and_reports_empty_channels(tmp_path):
    _write_kaist(tmp_path, "0Nm_BPFO_03", empty=("Mod2/ai2", "Mod2/ai3"))
    reader = get_reader("kaist", root=tmp_path, raw=tmp_path / "missing")
    rec = reader.load(reader.index()[0])
    assert rec.info.meta["empty_channels"] == ["current_2", "current_3"]
    assert "current_2" not in rec.channels and "current_1" in rec.channels
    from pdmfm.datasets.kaist import CLASS_DEPENDENT_GAPS, CURRENTS_USED

    assert set(CLASS_DEPENDENT_GAPS) == {"current_2", "current_3"}
    assert CURRENTS_USED == ("current_1",)


def test_kaist_reader_reads_straight_from_zips(tmp_path):
    import zipfile

    staging = tmp_path / "staging"
    _write_kaist(staging, "4Nm_Normal")
    raw = tmp_path / "raw"
    raw.mkdir()
    for folder, suffix in (("vibration", ".mat"), ("current,temp", ".tdms")):
        with zipfile.ZipFile(raw / f"{folder}.zip", "w") as archive:
            archive.write(staging / folder / f"4Nm_Normal{suffix}", f"4Nm_Normal{suffix}")
    reader = get_reader("kaist", root=tmp_path / "not-extracted", raw=raw)
    (info,) = reader.index()
    assert len(reader.load(info).channels) == 9


# ----------------------------------------------------------------------------- XJTU-SY


def test_xjtu_sy_reader_orders_snapshots_by_minute(tmp_path):
    bearing = tmp_path / "35Hz12kN" / "Bearing1_1"
    bearing.mkdir(parents=True)
    for minute in (1, 2, 10):  # lexical order would put 10 before 2
        values = np.full((64, 2), float(minute), dtype=np.float32)
        header = "Horizontal_vibration_signals,Vertical_vibration_signals"
        np.savetxt(bearing / f"{minute}.csv", values, delimiter=",", header=header, comments="")
    reader = get_reader("xjtu_sy", root=tmp_path)
    infos = reader.index()
    assert [i.meta["minute"] for i in infos] == [1, 2, 3]
    assert infos[0].meta["lifetime_min"] == 3 and infos[0].meta["failure"] == "outer race"
    assert infos[0].condition["rpm"] == 2100.0
    last = reader.load(infos[-1])
    assert last.channels["vib_v"].signal[0] == 10.0 and last.t_offset == 120.0


# ----------------------------------------------------------------------------- Najafi


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("Noload", ("healthy", None)),
        ("Fan", ("cooling_fan_failure", None)),
        ("Rotor-0", ("stuck_rotor", None)),
        ("A30", ("stator_short_circuit", "1 phase(s), 30 %")),
        ("A&C&B10", ("stator_short_circuit", "3 phase(s), 10 %")),
    ],
)
def test_najafi_folder_labels(folder, expected):
    assert parse_folder(folder) == expected


def test_najafi_reader_skips_the_annotated_subset(tmp_path):
    for folder in ("A10", "Noload", "Subset_40_Thermal_GT"):
        (tmp_path / folder).mkdir()
        Image.new("RGB", (320, 240)).save(tmp_path / folder / "1.bmp")
    infos = get_reader("najafi_thermal", root=tmp_path).index()
    assert sorted(i.condition["folder"] for i in infos) == ["A10", "Noload"]


# ----------------------------------------------------------------------------- Nonastreda


def test_nonastreda_reader_clock_photos_and_targets(tmp_path, monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class FakeRecord:
        record_id: str
        tool: str
        run: int
        block: int
        force: np.ndarray
        label: int

    records = [
        FakeRecord("T1R1B2", "T1", 1, 2, np.zeros((2000, 3), np.float32), 0),
        FakeRecord("T1R1B1", "T1", 1, 1, np.ones((1000, 3), np.float32), 0),
        FakeRecord("T2R1B1", "T2", 1, 1, np.ones((500, 3), np.float32), 2),
    ]
    import types

    fake = types.ModuleType("omnifuser.data")
    fake.load_nonastreda = lambda root, **kwargs: records
    monkeypatch.setitem(__import__("sys").modules, "omnifuser.data", fake)
    (tmp_path / "labels_reg.csv").write_text(
        "id,gaps,flank_wear,overhang\nT1R1B1,1,30.0,2\nT1R1B2,1,40.0,2\nT2R1B1,1,120.0,2\n"
    )
    for folder, suffix in (("tool", ".jpg"), ("chip", ".jpg"), ("work", ".png")):
        (tmp_path / folder).mkdir()
        for rid in ("T1R1B1", "T1R1B2"):
            (tmp_path / folder / f"{rid}{suffix}").write_bytes(b"x")
    reader = get_reader("nonastreda", root=tmp_path)
    infos = reader.index()
    assert [i.recording_id for i in infos] == ["T1R1B1", "T1R1B2", "T2R1B1"]  # run/block order
    from pdmfm.datasets.nonastreda import PASS_GAP_S
    from pdmfm.schema import as_of_join

    first, second, other = infos
    assert (first.meta["t_start"], first.meta["t_end"]) == (0.0, 1.0)
    gap = PASS_GAP_S  # passes are separated by the photo pause
    assert (second.meta["t_start"], second.meta["t_end"]) == (1.0 + gap, 3.0 + gap)
    assert other.meta["t_start"] == 0.0  # each tool has its own clock
    assert other.label == "dulled" and other.meta["flank_wear"] == 120.0
    rec = reader.load(second)
    assert {i.kind for i in rec.images} == {"tool", "chip", "workpiece"}
    assert all(i.captured_at == 3.0 + 1.5 * gap for i in rec.images)  # taken after the pass
    # The first pass's photos fall strictly between the passes: the second pass's very
    # first window gets them, and no window of the first pass does.
    photos = reader.load(first).images
    assert len(as_of_join(photos, second.meta["t_start"])) == 3
    assert as_of_join(photos, first.meta["t_end"]) == ()
    assert rec.targets == {"wear_um": 40.0} and rec.channels["fx"].rate_hz == 1000.0
    assert "run" not in rec.info.condition  # the run-number shortcut stays in meta


# ----------------------------------------------------------------------------- Paderborn


def _paderborn_struct(n_fast=640, n_mech=40, n_temp=1):
    entries = np.zeros((1, 7), dtype=[("Name", "O"), ("Data", "O"), ("Raster", "O")])
    spec = [
        ("force", n_mech, "Mech_4kHz"),
        ("phase_current_1", n_fast, "HostService"),
        ("phase_current_2", n_fast, "HostService"),
        ("speed", n_mech, "Mech_4kHz"),
        ("temp_2_bearing_module", n_temp, "Temp_1Hz"),
        ("torque", n_mech, "Mech_4kHz"),
        ("vibration_1", n_fast, "HostService"),
    ]
    for i, (name, n, raster) in enumerate(spec):
        entries[0, i] = (name, RNG.standard_normal((1, n)), raster)
    return {"Info": "synthetic", "Y": entries}


def test_paderborn_reader_on_the_documented_layout(tmp_path):
    folder = tmp_path / "KA04"
    folder.mkdir()
    stem = "N15_M07_F10_KA04_1"
    savemat(folder / f"{stem}.mat", {stem: _paderborn_struct()})
    reader = get_reader("paderborn", root=tmp_path)
    (info,) = reader.index()
    assert info.unit == "KA04" and info.label == "outer_race" and info.severity == "level 1"
    assert info.condition == {
        "setting": "N15_M07_F10",
        "rpm": 1500.0,
        "torque_nm": 0.7,
        "radial_n": 1000.0,
    }
    rec = reader.load(info)
    assert rec.channels["vibration"].rate_hz == 64_000.0
    assert rec.channels["current_2"].family == "current"
    assert rec.channels["temperature"].rate_hz == 1.0
    assert rec.channels["torque"].rate_hz == 4_000.0


def test_paderborn_file_without_channels_is_reported(tmp_path):
    folder = tmp_path / "K001"
    folder.mkdir()
    stem = "N15_M07_F10_K001_1"
    savemat(folder / f"{stem}.mat", {stem: {"Info": "no Y field"}})
    reader = get_reader("paderborn", root=tmp_path)
    with pytest.raises(ValueError, match="no 'Y' channel list"):
        reader.load(reader.index()[0])


# ----------------------------------------------------------------------------- PHM 2010


def test_phm2010_reader_on_the_documented_layout(tmp_path):
    cutter = tmp_path / "c1"
    (cutter / "c1").mkdir(parents=True)
    (cutter / "c1_wear.csv").write_text("cut,flute_1,flute_2,flute_3\n1,30,32,31\n2,40,45,41\n")
    for cut in (1, 2):
        np.savetxt(
            cutter / "c1" / f"c_1_{cut:03d}.csv", RNG.standard_normal((50, 7)), delimiter=","
        )
    reader = get_reader("phm2010", root=tmp_path)
    infos = reader.index()
    assert [i.meta["cut"] for i in infos] == [1, 2]
    assert infos[1].meta["wear_um"] == 45.0 and infos[1].meta["labelled"]
    rec = reader.load(infos[0])
    assert len(rec.channels) == 7 and rec.channels["ae_rms"].family == "slow"
    assert rec.channels["vx"].rate_hz == 50_000.0 and rec.targets == {"wear_um": 32.0}


def test_phm2010_reader_skips_csv_files_without_a_cut_number(tmp_path):
    cutter = tmp_path / "c4"
    cutter.mkdir()
    (cutter / "c4_wear.csv").write_text("cut,flute_1,flute_2,flute_3\n1,30,32,31\n")
    np.savetxt(cutter / "c_4_001.csv", RNG.standard_normal((5, 7)), delimiter=",")
    (cutter / "notes.csv").write_text("written,by,hand\n")  # a mirror's extra file
    infos = get_reader("phm2010", root=tmp_path).index()
    assert [(i.unit, i.meta["cut"]) for i in infos] == [("c4", 1)]


def test_phm2010_wear_file_without_flute_columns_is_rejected(tmp_path):
    from pdmfm.datasets.phm2010 import read_wear

    path = tmp_path / "c1_wear.csv"
    path.write_text("cut,vb\n1,30\n")
    with pytest.raises(ValueError, match="flute"):
        read_wear(path)


# ----------------------------------------------------------------------------- NASA milling


def test_nasa_milling_reader(tmp_path):
    fields = [
        "case",
        "run",
        "VB",
        "time",
        "DOC",
        "feed",
        "material",
        "smcAC",
        "smcDC",
        "vib_table",
        "vib_spindle",
        "AE_table",
        "AE_spindle",
    ]
    runs = np.zeros((1, 2), dtype=[(f, "O") for f in fields])
    for i, vb in enumerate((0.1, np.nan)):
        runs[0, i] = (
            1,
            i + 1,
            vb,
            2 * i,
            1.5,
            0.5,
            1,
            *(RNG.standard_normal(90) for _ in range(6)),
        )
    savemat(tmp_path / "mill.mat", {"mill": runs})
    reader = get_reader("nasa_milling", root=tmp_path)
    infos = reader.index()
    assert [i.recording_id for i in infos] == ["case01/run01", "case01/run02"]
    assert infos[0].meta["wear_um"] == pytest.approx(100.0) and infos[1].meta["wear_um"] is None
    assert infos[0].condition["material"] == "cast iron"
    rec = reader.load(infos[0])
    assert rec.channels["current_ac"].rate_hz == 250.0 and len(rec.channels) == 6
    assert rec.targets == {"wear_um": pytest.approx(100.0)}
    assert reader.load(infos[1]).targets == {}


def test_unknown_reader():
    with pytest.raises(KeyError):
        get_reader("nope")
