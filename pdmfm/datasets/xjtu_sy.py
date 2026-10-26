"""XJTU-SY run-to-failure bearing data (Wang et al., IEEE Trans. Reliability 2020).

Layout after extraction: ``XJTU-SY_Bearing_Datasets/<35Hz12kN|37.5Hz11kN|40Hz10kN>/
Bearing<c>_<k>/<minute>.csv``. Each CSV is one 1.28 s snapshot (32,768 rows at 25.6 kHz)
with columns ``Horizontal_vibration_signals,Vertical_vibration_signals`` in g. The number
of snapshots equals the bearing's lifetime in minutes (all 15 checked).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import extracted_dir

RATE_HZ = 25_600.0
CONDITIONS = {
    "35Hz12kN": {"rpm": 2100.0, "radial_kn": 12.0},
    "37.5Hz11kN": {"rpm": 2250.0, "radial_kn": 11.0},
    "40Hz10kN": {"rpm": 2400.0, "radial_kn": 10.0},
}
# LDK UER204 (Wang et al. 2020).
GEOMETRY = {"n_balls": 8, "ball_d": 7.92, "pitch_d": 34.55, "contact_deg": 0.0}
# Failure location at end of life as reported by Wang et al. (2020); kept as metadata
# (it is known only after failure, so it is never a model input).
FAILURE = {
    "Bearing1_1": "outer race",
    "Bearing1_2": "outer race",
    "Bearing1_3": "outer race",
    "Bearing1_4": "cage",
    "Bearing1_5": "inner race, outer race",
    "Bearing2_1": "inner race",
    "Bearing2_2": "outer race",
    "Bearing2_3": "cage",
    "Bearing2_4": "outer race",
    "Bearing2_5": "outer race",
    "Bearing3_1": "outer race",
    "Bearing3_2": "inner race, ball, cage, outer race",
    "Bearing3_3": "inner race",
    "Bearing3_4": "inner race",
    "Bearing3_5": "outer race",
}


def read_snapshot(path: Path) -> np.ndarray:
    """(32768, 2) float32 from one CSV (pandas C parser; about 20x faster than numpy)."""
    try:
        import pandas as pd

        return pd.read_csv(path, dtype=np.float32).to_numpy()
    except ImportError:
        return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


class XJTUSY:
    name = "xjtu_sy"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else extracted_dir("xjtu_sy") / "XJTU-SY_Bearing_Datasets"

    def bearings(self) -> dict[str, tuple[str, list[Path]]]:
        """Bearing -> (condition folder, snapshot paths in time order)."""
        found = {}
        for condition in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for bearing in sorted(p for p in condition.iterdir() if p.is_dir()):
                paths = sorted(bearing.glob("*.csv"), key=lambda p: int(p.stem))
                found[bearing.name] = (condition.name, paths)
        return found

    def index(self) -> list[RecordingInfo]:
        infos = []
        for bearing, (condition, paths) in self.bearings().items():
            for minute, path in enumerate(paths, 1):
                infos.append(
                    RecordingInfo(
                        dataset=self.name,
                        recording_id=f"{bearing}/{minute}",
                        unit=bearing,
                        label=None,
                        condition={"condition": condition, **CONDITIONS.get(condition, {})},
                        meta={
                            "path": str(path),
                            "minute": minute,
                            "lifetime_min": len(paths),
                            "failure": FAILURE.get(bearing),
                        },
                    )
                )
        return infos

    def load(self, info: RecordingInfo) -> Recording:
        values = read_snapshot(Path(info.meta["path"]))
        channels = {
            "vib_h": ChannelData(
                "vib_h", "vibration", "x", "housing", RATE_HZ, as_float32(values[:, 0]), "g"
            ),
            "vib_v": ChannelData(
                "vib_v", "vibration", "y", "housing", RATE_HZ, as_float32(values[:, 1]), "g"
            ),
        }
        minute = info.meta["minute"]
        return Recording(info=info, channels=channels, t_offset=60.0 * (minute - 1))
