"""NASA Ames milling data (Agogino and Goebel 2007), checked on the downloaded ``mill.mat``.

``mill`` is a 1x167 struct array (runs) over 16 cases; each case is one insert set under one
setting (depth of cut 0.75 or 1.5 mm, feed 0.25 or 0.5 mm/rev, cast iron or J45 steel;
826 rpm, KC710 inserts). Six channels of 9,000 samples at 250 Hz (36 s): AC and DC spindle
motor current, and table and spindle vibration and acoustic emission. Vibration and AE are
already RMS envelopes (8 ms time constant), not raw waveforms. Flank wear ``VB`` is in mm
and is missing for 21 runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import extracted_dir

RATE_HZ = 250.0
SPINDLE_RPM = 826.0
MATERIALS = {1: "cast iron", 2: "steel J45"}
CHANNELS = (
    ("smcAC", "current_ac", "current", "none", "spindle"),
    ("smcDC", "current_dc", "slow", "none", "spindle"),
    ("vib_table", "vib_table_rms", "slow", "none", "table"),
    ("vib_spindle", "vib_spindle_rms", "slow", "none", "spindle"),
    ("AE_table", "ae_table_rms", "slow", "none", "table"),
    ("AE_spindle", "ae_spindle_rms", "slow", "none", "spindle"),
)


class NASAMilling:
    name = "nasa_milling"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else extracted_dir("nasa_milling")
        self._runs = None

    def _load(self):
        if self._runs is None:
            path = next(self.root.rglob("mill.mat"))
            self._runs = np.atleast_1d(
                loadmat(path, squeeze_me=True, struct_as_record=False)["mill"]
            )
        return self._runs

    def index(self) -> list[RecordingInfo]:
        infos = []
        for position, run in enumerate(self._load()):
            vb = float(run.VB)
            infos.append(
                RecordingInfo(
                    dataset=self.name,
                    recording_id=f"case{int(run.case):02d}/run{int(run.run):02d}",
                    unit=f"case{int(run.case):02d}",
                    label=None,
                    condition={
                        "rpm": SPINDLE_RPM,
                        "depth_of_cut_mm": float(run.DOC),
                        "feed_mm_per_rev": float(run.feed),
                        "material": MATERIALS.get(int(run.material), str(run.material)),
                    },
                    meta={
                        "position": position,
                        "time_min": float(run.time),
                        "wear_um": vb * 1000.0 if np.isfinite(vb) else None,
                    },
                )
            )
        return infos

    def load(self, info: RecordingInfo) -> Recording:
        run = self._load()[info.meta["position"]]
        channels = {
            name: ChannelData(
                name, family, axis, placement, RATE_HZ, as_float32(getattr(run, field)), ""
            )
            for field, name, family, axis, placement in CHANNELS
        }
        targets = {"wear_um": info.meta["wear_um"]} if info.meta["wear_um"] is not None else {}
        return Recording(info=info, channels=channels, targets=targets)
