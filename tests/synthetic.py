"""Synthetic readers in the layout of each dataset, for pipeline tests (no data needed).

Signals are built so that a working pipeline can tell the classes apart (a class-specific
tone; a wear-dependent force level; vibration that grows after 60 % of a bearing's life),
and so that the tests can check exact values where the preprocessing is the identity.
"""

from __future__ import annotations

import numpy as np

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo
from pdmfm.schema import ImageItem


class FakeReader:
    def __init__(self, recordings: list[Recording]) -> None:
        self.recordings = {r.info.recording_id: r for r in recordings}

    def index(self) -> list[RecordingInfo]:
        return [r.info for r in self.recordings.values()]

    def load(self, info: RecordingInfo) -> Recording:
        return self.recordings[info.recording_id]


def _tone(freq: float, rate: float, n: int, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / rate)


def cwru_reader(seconds: float = 2.1, seed: int = 0) -> FakeReader:
    """4 classes x 4 loads; each class has its own tone in the drive-end signal."""
    from pdmfm.datasets.cwru import NOMINAL_RPM

    rng = np.random.default_rng(seed)
    rate = 12_000.0
    n = int(seconds * rate)
    classes = {
        ("normal", None): 500.0,
        ("inner_race", "0.007 in"): 900.0,
        ("outer_race", "0.007 in"): 1300.0,
        ("ball", "0.007 in"): 1700.0,
    }
    recordings = []
    for load in range(4):
        rpm = NOMINAL_RPM[load]
        for (label, severity), freq in classes.items():
            unit = "normal" if label == "normal" else f"{label}_0.007in"
            channels = {}
            for name, placement in (("de", "drive_end"), ("fe", "fan_end")):
                signal = _tone(freq, rate, n) + 0.3 * _tone(rpm / 60, rate, n)
                signal = signal + 0.2 * rng.standard_normal(n)
                channels[name] = ChannelData(
                    name, "vibration", "none", placement, rate, signal.astype(np.float32), "g"
                )
            info = RecordingInfo(
                "cwru", f"{label}-{load}", unit, label, severity, {"load_hp": load, "rpm": rpm}
            )
            recordings.append(Recording(info, channels))
    return FakeReader(recordings)


def xjtu_reader(snapshots: int = 30, seed: int = 0) -> FakeReader:
    """15 bearings; vibration grows after 60 % of each life."""
    from pdmfm.datasets.xjtu_sy import CONDITIONS

    rng = np.random.default_rng(seed)
    rate, n = 12_000.0, 13_000
    names = list(CONDITIONS)
    recordings = []
    for c in (1, 2, 3):
        condition = names[c - 1]
        for i in range(1, 6):
            bearing = f"Bearing{c}_{i}"
            for minute in range(1, snapshots + 1):
                life = minute / snapshots
                amplitude = 1.0 + 6.0 * max(0.0, life - 0.6)
                channels = {
                    name: ChannelData(
                        name, "vibration", axis, "housing", rate,
                        (amplitude * rng.standard_normal(n)).astype(np.float32), "g",
                    )
                    for name, axis in (("vib_h", "x"), ("vib_v", "y"))
                }  # fmt: skip
                info = RecordingInfo(
                    "xjtu_sy", f"{bearing}/{minute}", bearing, None, None,
                    {"condition": condition, **CONDITIONS[condition]},
                    {"minute": minute, "lifetime_min": snapshots},
                )  # fmt: skip
                recordings.append(Recording(info, channels))
    return FakeReader(recordings)


NONASTREDA_CLASSES = ("sharp", "sharp", "used", "used", "dulled", "dulled")


def nonastreda_reader(photo_root=None, seconds: float = 16.5, seed: int = 0) -> FakeReader:
    """10 tools x 6 passes; the Fx/Fy level rises with the wear class. Passes follow each
    other on each tool's clock with the reader's photo pause; photos (if ``photo_root``) are
    stamped in the middle of the pause, as pdmfm.datasets.nonastreda does."""
    from pdmfm.datasets.nonastreda import PASS_GAP_S

    rng = np.random.default_rng(seed)
    rate = 1000.0
    n = int(seconds * rate)
    recordings = []
    for t in range(1, 11):
        tool, clock = f"T{t}", 0.0
        for p, label in enumerate(NONASTREDA_CLASSES):
            level = 50.0 + 25.0 * ("sharp", "used", "dulled").index(label) + rng.normal(0, 2)
            channels = {
                name: ChannelData(
                    name, "force", axis, "table", rate,
                    (level + 5 * rng.standard_normal(n)).astype(np.float32), "N",
                )
                for name, axis in (("fx", "x"), ("fy", "y"), ("fz", "z"))
            }  # fmt: skip
            record_id = f"{tool}R1B{p + 1}"
            images = ()
            if photo_root is not None:
                images = tuple(
                    ImageItem(
                        kind,
                        str(photo_root / f"{kind}-{record_id}.png"),
                        clock + seconds + PASS_GAP_S / 2,
                    )
                    for kind in ("tool", "chip")
                )
            info = RecordingInfo(
                "nonastreda", record_id, tool, label, None, {},
                {"run": 1, "block": p + 1, "t_start": clock, "t_end": clock + seconds},
            )  # fmt: skip
            targets = {"wear_um": 40.0 + 30.0 * ("sharp", "used", "dulled").index(label)}
            recordings.append(Recording(info, channels, images, clock, targets))
            clock += seconds + PASS_GAP_S
    return FakeReader(recordings)
