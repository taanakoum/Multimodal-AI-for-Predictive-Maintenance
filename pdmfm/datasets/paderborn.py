"""Paderborn (KAt) bearing data, written against the published format.

**Not yet verified on the real files** (the host refuses connections from the development
machine; see docs/USER_ACTIONS.md). The structure follows the KAt documentation and the
widely used layout: each ``<bearing>/N15_M07_F10_<bearing>_<n>.mat`` holds one struct named
like the file, whose ``Y`` field lists channels with ``Name``, ``Data`` and ``Raster``
(``HostService`` = 64 kHz for the motor currents and vibration, ``Mech_4kHz`` for force,
speed and torque, ``Temp_1Hz`` for the bearing-module temperature). When a raster name is
unknown, the rate is inferred from the channel length and the vibration channel's duration.
Run ``python scripts/eda/run_eda.py paderborn`` after the download to confirm the layout.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import extracted_dir
from pdmfm.sources import PADERBORN_SETTINGS

VERIFIED_ON_REAL_DATA = False
RASTER_HZ = {"HostService": 64_000.0, "Mech_4kHz": 4_000.0, "Temp_1Hz": 1.0}
# 6203 deep-groove ball bearing; geometry consistent with the PHME 2016 paper's
# BPFO = 76.25 Hz at 1500 rpm.
GEOMETRY = {"n_balls": 8, "ball_d": 6.75, "pitch_d": 28.5, "contact_deg": 0.0}
# Damage location, extent level (1-3) and origin (PHME 2016, Tables 4, 5 and 7).
DAMAGE = {
    **{f"K00{i}": ("healthy", None, "healthy") for i in range(1, 7)},
    "KA01": ("outer_race", 1, "artificial: EDM"),
    "KA03": ("outer_race", 2, "artificial: engraver"),
    "KA05": ("outer_race", 1, "artificial: engraver"),
    "KA06": ("outer_race", 2, "artificial: engraver"),
    "KA07": ("outer_race", 1, "artificial: drilling"),
    "KA08": ("outer_race", 2, "artificial: drilling"),
    "KA09": ("outer_race", 2, "artificial: drilling"),
    "KI01": ("inner_race", 1, "artificial: EDM"),
    "KI03": ("inner_race", 1, "artificial: engraver"),
    "KI05": ("inner_race", 1, "artificial: engraver"),
    "KI07": ("inner_race", 2, "artificial: engraver"),
    "KI08": ("inner_race", 2, "artificial: engraver"),
    "KA04": ("outer_race", 1, "real: fatigue pitting"),
    "KA15": ("outer_race", 1, "real: indentations"),
    "KA16": ("outer_race", 2, "real: fatigue pitting"),
    "KA22": ("outer_race", 1, "real: fatigue pitting"),
    "KA30": ("outer_race", 1, "real: indentations"),
    "KB23": ("inner_race", 2, "real: fatigue pitting, IR (+OR)"),
    "KB24": ("inner_race", 3, "real: fatigue pitting, IR (+OR)"),
    "KB27": ("combined", 1, "real: indentations, OR + IR"),
    "KI04": ("inner_race", 1, "real: fatigue pitting"),
    "KI14": ("inner_race", 1, "real: fatigue pitting"),
    "KI16": ("inner_race", 3, "real: fatigue pitting"),
    "KI17": ("inner_race", 1, "real: fatigue pitting"),
    "KI18": ("inner_race", 2, "real: fatigue pitting"),
    "KI21": ("inner_race", 1, "real: fatigue pitting"),
}
_FILE = re.compile(r"^(?P<setting>N\d\d_M\d\d_F\d\d)_(?P<bearing>K[ABI]?\d\d|K00\d)_(?P<n>\d+)$")
_CHANNELS = {
    "vibration_1": ("vibration", "vibration", "none", "housing", ""),  # unit: confirm on real data
    "phase_current_1": ("current_1", "current", "none", "motor_phase", "A"),
    "phase_current_2": ("current_2", "current", "none", "motor_phase", "A"),
    "temp_2_bearing_module": ("temperature", "slow", "none", "bearing_module", "°C"),
    "force": ("radial_force", "slow", "radial", "bearing_module", "N"),
    "speed": ("speed", "slow", "none", "none", "rpm"),
    "torque": ("torque", "slow", "none", "none", "Nm"),
}


def _field(entry, name):
    value = getattr(entry, name, None)
    if value is None and isinstance(entry, np.void):
        value = entry[name]
    return value


def _name(value) -> str:
    return value if isinstance(value, str) else "".join(np.atleast_1d(value).astype(str))


class Paderborn:
    name = "paderborn"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else extracted_dir("paderborn")

    def index(self) -> list[RecordingInfo]:
        infos = []
        for path in sorted(self.root.rglob("*.mat")):
            match = _FILE.match(path.stem)
            if match is None:
                continue
            bearing = match["bearing"]
            label, level, origin = DAMAGE.get(bearing, (None, None, "unknown"))
            rpm, torque, force = PADERBORN_SETTINGS[match["setting"]]
            infos.append(
                RecordingInfo(
                    dataset=self.name,
                    recording_id=path.stem,
                    unit=bearing,
                    label=label,
                    severity=None if level is None else f"level {level}",
                    condition={
                        "setting": match["setting"],
                        "rpm": float(rpm),
                        "torque_nm": torque,
                        "radial_n": float(force),
                    },
                    meta={"path": str(path), "measurement": int(match["n"]), "origin": origin},
                )
            )
        return infos

    def load(self, info: RecordingInfo) -> Recording:
        mat = loadmat(info.meta["path"], squeeze_me=True, struct_as_record=False)
        key = next(k for k in mat if not k.startswith("__"))
        entries = np.atleast_1d(_field(mat[key], "Y"))
        raw = {}
        for entry in entries:
            name = _name(_field(entry, "Name"))
            raster = _name(_field(entry, "Raster")) if _field(entry, "Raster") is not None else ""
            raw[name] = (np.asarray(_field(entry, "Data"), dtype=np.float64).ravel(), raster)
        duration = (
            len(raw["vibration_1"][0]) / RASTER_HZ["HostService"] if "vibration_1" in raw else None
        )
        channels = {}
        for source, (name, family, axis, placement, unit) in _CHANNELS.items():
            if source not in raw:
                continue
            values, raster = raw[source]
            rate = RASTER_HZ.get(raster) or (len(values) / duration if duration else np.nan)
            channels[name] = ChannelData(
                name, family, axis, placement, float(rate), as_float32(values), unit
            )
        return Recording(info=info, channels=channels)
