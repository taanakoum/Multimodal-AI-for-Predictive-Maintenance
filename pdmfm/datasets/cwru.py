"""CWRU bearing data: the 12 kHz drive-end fault set with the 48 kHz normal baseline.

Quirks handled (checked on the downloaded files, see PROJECT_LOG.md):

* ``99.mat`` also contains ``X098_*``, an exact copy of file 98; only the variables that carry
  the file's own number are read.
* ``98.mat`` and ``99.mat`` have no RPM field; the nominal speed of the load is used.
* Each fault size is one physical specimen tested at four loads, so the split ``unit`` is the
  specimen (class + size) and the load is the operating condition.
"""

from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import raw_dir
from pdmfm.sources import CWRU_FILES

NOMINAL_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
# SKF 6205-2RS JEM drive-end bearing (CWRU bearing data center).
GEOMETRY = {"n_balls": 9, "ball_d": 7.94, "pitch_d": 39.04, "contact_deg": 0.0}


class CWRU:
    name = "cwru"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else raw_dir("cwru")

    def index(self) -> list[RecordingInfo]:
        infos = []
        for number, (kind, size, load) in sorted(CWRU_FILES.items()):
            infos.append(
                RecordingInfo(
                    dataset=self.name,
                    recording_id=str(number),
                    unit="normal" if kind == "normal" else f"{kind}_{size:.3f}in",
                    label=kind,
                    severity=None if kind == "normal" else f"{size:.3f} in",
                    condition={"load_hp": load},
                    meta={
                        "file": f"{number}.mat",
                        "rate_hz": 48_000.0 if kind == "normal" else 12_000.0,
                    },
                )
            )
        return infos

    def load(self, info: RecordingInfo) -> Recording:
        number = int(info.recording_id)
        mat = loadmat(self.root / info.meta["file"])
        prefix = f"X{number:03d}_"
        rate = info.meta["rate_hz"]
        channels = {}
        for key, name, placement in (("DE_time", "de", "drive_end"), ("FE_time", "fe", "fan_end")):
            if prefix + key in mat:
                channels[name] = ChannelData(
                    name=name,
                    family="vibration",
                    axis="none",
                    placement=placement,
                    rate_hz=rate,
                    signal=as_float32(mat[prefix + key]),
                    unit="g",
                )
        rpm_key = f"X{number:03d}RPM"
        rpm = (
            float(mat[rpm_key].squeeze())
            if rpm_key in mat
            else NOMINAL_RPM[info.condition["load_hp"]]
        )
        condition = {
            **info.condition,
            "rpm": rpm,
            "rpm_source": "file" if rpm_key in mat else "nominal",
        }
        return Recording(
            info=RecordingInfo(**{**info.__dict__, "condition": condition}), channels=channels
        )
