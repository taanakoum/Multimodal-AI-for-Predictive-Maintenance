"""PHM Society 2010 milling data, written against the challenge's published description.

**Not yet verified on the real files** (the original download host is gone; see
docs/USER_ACTIONS.md). Per the challenge page, each cutter ``cN`` has one wear file (wear
after each cut, per flute, in 10^-3 mm = µm) and about 300 CSV files, one per cut, with
seven unlabelled columns: Fx, Fy, Fz (N), Vx, Vy, Vz (g), AE-RMS (V), sampled at 50 kHz.
File names are matched loosely (``*wear*.csv``; cut files ending in a cut number).
Settings: 10,400 rpm, 1,555 mm/min feed, 0.125 mm radial and 0.2 mm axial depth, 6 mm
ball-nose tungsten-carbide cutter.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import extracted_dir

VERIFIED_ON_REAL_DATA = False
RATE_HZ = 50_000.0
SPINDLE_RPM = 10_400.0
LABELLED = ("c1", "c4", "c6")
COLUMNS = (
    ("fx", "force", "x", "table", "N"),
    ("fy", "force", "y", "table", "N"),
    ("fz", "force", "z", "table", "N"),
    ("vx", "vibration", "x", "table", "g"),
    ("vy", "vibration", "y", "table", "g"),
    ("vz", "vibration", "z", "table", "g"),
    ("ae_rms", "slow", "none", "table", "V"),
)
_CUT = re.compile(r"(\d+)$")


def read_wear(path: Path) -> dict[int, float]:
    """Cut number -> flank wear in µm (maximum over the flutes)."""
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    names = rows.dtype.names
    flutes = [n for n in names or () if n.lower().startswith("flute")]
    if not names or not flutes:
        raise ValueError(f"{path}: expected a header with a cut column and flute_* columns")
    return {int(r[names[0]]): float(max(r[f] for f in flutes)) for r in rows}


class PHM2010:
    name = "phm2010"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else extracted_dir("phm2010")

    def index(self) -> list[RecordingInfo]:
        infos = []
        for cutter_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            cutter = cutter_dir.name
            wear_files = list(cutter_dir.rglob("*wear*.csv"))
            wear = read_wear(wear_files[0]) if wear_files else {}
            cuts = []
            for path in cutter_dir.rglob("*.csv"):
                match = _CUT.search(path.stem)
                if match and "wear" not in path.name.lower():  # other CSVs are not cuts
                    cuts.append((int(match.group(1)), path))
            for cut, path in sorted(cuts):
                infos.append(
                    RecordingInfo(
                        dataset=self.name,
                        recording_id=f"{cutter}/{cut}",
                        unit=cutter,
                        label=None,
                        condition={"rpm": SPINDLE_RPM},
                        meta={
                            "path": str(path),
                            "cut": cut,
                            "labelled": cutter in LABELLED,
                            "wear_um": wear.get(cut),
                        },
                    )
                )
        return infos

    def load(self, info: RecordingInfo) -> Recording:
        values = np.loadtxt(info.meta["path"], delimiter=",", dtype=np.float64, ndmin=2)
        channels = {
            name: ChannelData(
                name, family, axis, placement, RATE_HZ, as_float32(values[:, i]), unit
            )
            for i, (name, family, axis, placement, unit) in enumerate(COLUMNS)
            if i < values.shape[1]
        }
        targets = {"wear_um": info.meta["wear_um"]} if info.meta.get("wear_um") is not None else {}
        return Recording(info=info, channels=channels, targets=targets)
