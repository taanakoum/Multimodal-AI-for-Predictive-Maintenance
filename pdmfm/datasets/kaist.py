"""KAIST rotating-machine dataset (load-varying set, Jung et al., Data in Brief 2023).

Formats (checked on the downloaded files, see PROJECT_LOG.md):

* ``vibration/<cond>.mat``: Siemens Test.Lab export, struct ``Signal``; ``y_values.values``
  is (n, 4) in g (housing accelerometers Point1-4), ``x_values.increment`` = 1/25 600 s.
* ``current,temp/<cond>.tdms``: group ``Log`` (the first group, ``Test Information``, is
  empty); ``Mod1/ai0-ai1`` thermocouples in °C, ``Mod2/ai0, ai2, ai3`` currents in A, at
  1 / 39.05 µs ≈ 25 608 Hz.
* Condition names ``<load>Nm_<kind>[_<severity>]``; the 2 Nm unbalance files are spelled
  ``Unbalalnce`` and are normalized here.

The two devices are not sample-synchronous: their clocks differ by about 3.4 min, and five
conditions were recorded at different times. ``meta["device_offset_s"]`` keeps the measured
offset, and the plan treats KAIST modalities as pairs of the same machine state.

Currents 2 and 3 (``Mod2/ai2``, ``Mod2/ai3``) are empty in all nine outer-race files and
present everywhere else. Their absence would identify the class, so the reader skips empty
channels and records them in ``meta["empty_channels"]``, and models use only
:data:`CURRENTS_USED`.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from nptdms import TdmsFile
from scipy.io import loadmat

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import extracted_dir, raw_dir

RPM = 3010.0
# NSK 6205 DDU (Jung et al. 2023).
GEOMETRY = {"n_balls": 9, "ball_d": 7.90, "pitch_d": 38.5, "contact_deg": 0.0}
_NAME = re.compile(
    r"^(?P<load>\d+)Nm_(?P<kind>Normal|BPFI|BPFO|Misalign|Unbalance)(?:_(?P<sev>\d+)(?P<mg>mg)?)?$"
)
_KIND = {
    "Normal": "normal",
    "BPFI": "inner_race",
    "BPFO": "outer_race",
    "Misalign": "misalignment",
    "Unbalance": "unbalance",
}
_VIBRATION_CHANNELS = (  # column order of y_values.values
    ("vib_ax", "x", "housing_a"),
    ("vib_ay", "y", "housing_a"),
    ("vib_bx", "x", "housing_b"),
    ("vib_by", "y", "housing_b"),
)
# Channels whose presence depends on the class (empty in every outer-race file): never model
# inputs, or "channel missing" would reveal the label.
CLASS_DEPENDENT_GAPS = ("current_2", "current_3")
CURRENTS_USED = ("current_1",)
_TDMS_KEY = re.compile(r"(Mod\d+/ai\d+)$")
_TDMS_CHANNELS = {
    "Mod1/ai0": ("temp_1", "slow", "housing"),
    "Mod1/ai1": ("temp_2", "slow", "housing"),
    "Mod2/ai0": ("current_1", "current", "motor_phase"),
    "Mod2/ai2": ("current_2", "current", "motor_phase"),
    "Mod2/ai3": ("current_3", "current", "motor_phase"),
}


def normalize_name(name: str) -> str:
    """'2Nm_Unbalalnce_0583mg.mat' -> '2Nm_Unbalance_0583mg'."""
    return Path(name).stem.replace("Unbalalnce", "Unbalance")


def parse_condition(stem: str) -> dict:
    """Label, severity and load of a KAIST condition name."""
    match = _NAME.match(normalize_name(stem))
    if match is None:
        raise ValueError(f"not a KAIST condition name: {stem!r}")
    kind, sev = match["kind"], match["sev"]
    label = _KIND[kind]
    if kind == "Normal":
        severity = None
    elif match["mg"]:
        severity = f"{int(sev)} mg"
    else:
        severity = f"{int(sev) / 10:.1f} mm"  # 03 -> 0.3 mm, 10 -> 1.0 mm, 30 -> 3.0 mm
    return {"label": label, "severity": severity, "load_nm": int(match["load"])}


def _text(value) -> str:
    """Strings in Test.Lab structs come back as arrays of characters, sometimes per channel."""
    array = np.atleast_1d(np.asarray(value, dtype=object))
    first = array[0]
    if isinstance(first, str):
        return first if array.size == 1 or len(first) > 1 else "".join(map(str, array.ravel()))
    return "".join(str(c) for c in np.atleast_1d(first).ravel())


class KAIST:
    name = "kaist"

    def __init__(self, root: Path | None = None, raw: Path | None = None) -> None:
        self.root = Path(root) if root else extracted_dir("kaist")
        self.raw = Path(raw) if raw else raw_dir("kaist")

    def _read(self, folder: str, filename: str) -> bytes:
        """Bytes of one member, from the extracted folder or straight from the zip."""
        path = self.root / folder / filename
        if path.exists():
            return path.read_bytes()
        with zipfile.ZipFile(self.raw / f"{folder}.zip") as archive:
            return archive.read(filename)

    def _names(self, folder: str, suffix: str) -> dict[str, str]:
        directory = self.root / folder
        if directory.is_dir():
            names = [p.name for p in directory.glob(f"*{suffix}")]
        else:
            with zipfile.ZipFile(self.raw / f"{folder}.zip") as archive:
                names = [n for n in archive.namelist() if n.endswith(suffix)]
        return {normalize_name(n): n for n in names}

    def index(self) -> list[RecordingInfo]:
        vibration = self._names("vibration", ".mat")
        electrical = self._names("current,temp", ".tdms")
        infos = []
        for stem in sorted(vibration):
            parsed = parse_condition(stem)
            unit = (
                parsed["label"]
                if parsed["severity"] is None
                else (f"{parsed['label']}_{parsed['severity'].replace(' ', '')}")
            )
            infos.append(
                RecordingInfo(
                    dataset=self.name,
                    recording_id=stem,
                    unit=unit,
                    label=parsed["label"],
                    severity=parsed["severity"],
                    condition={"load_nm": parsed["load_nm"], "rpm": RPM},
                    meta={
                        "vibration_file": vibration[stem],
                        "electrical_file": electrical.get(stem),
                    },
                )
            )
        return infos

    def load(self, info: RecordingInfo, electrical: bool = True) -> Recording:
        mat = loadmat(
            io.BytesIO(self._read("vibration", info.meta["vibration_file"])),
            squeeze_me=True,
            struct_as_record=False,
            variable_names=["Signal"],
        )["Signal"]
        rate = 1.0 / float(mat.x_values.increment)
        values = np.asarray(mat.y_values.values)
        channels = {
            name: ChannelData(
                name, "vibration", axis, placement, rate, as_float32(values[:, i]), "g"
            )
            for i, (name, axis, placement) in enumerate(_VIBRATION_CHANNELS)
        }
        meta = dict(info.meta)
        vib_start = _text(mat.function_record.TL_export_properties_annotation.absolute_time)
        meta["vibration_start"] = vib_start
        if electrical and info.meta.get("electrical_file"):
            tdms = TdmsFile.read(
                io.BytesIO(self._read("current,temp", info.meta["electrical_file"]))
            )
            empty = []
            for channel in tdms["Log"].channels():
                found = _TDMS_KEY.search(channel.name)  # 'cDAQ9185-1F486B5Mod1/ai0'
                mapped = _TDMS_CHANNELS.get(found.group(1)) if found else None
                if mapped is None:
                    continue
                name, family, placement = mapped
                props = dict(channel.properties)
                if len(channel) == 0 or "wf_increment" not in props:
                    empty.append(name)  # e.g. currents 2 and 3 in every outer-race file
                    continue
                channels[name] = ChannelData(
                    name,
                    family,
                    "none",
                    placement,
                    1.0 / float(props["wf_increment"]),
                    as_float32(channel[:]),
                    "°C" if family == "slow" else "A",
                )
                meta.setdefault("electrical_start", str(props.get("wf_start_time")))
            meta["empty_channels"] = empty
            meta["device_offset_s"] = _offset_seconds(vib_start, meta.get("electrical_start"))
        return Recording(info=RecordingInfo(**{**info.__dict__, "meta": meta}), channels=channels)


def _offset_seconds(vibration_start: str, electrical_start: str | None) -> float | None:
    """Electrical start minus vibration start, in seconds (None if either is unknown)."""
    if not electrical_start:
        return None
    vib = datetime.strptime(vibration_start.split(" ms")[0], "%Y-%m-%d %H:%M:%S")
    ele = datetime.fromisoformat(electrical_start.split(".")[0])
    return (ele - vib).total_seconds()
