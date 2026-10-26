"""Najafi et al. induction-motor thermal images (320x240 pseudo-colour BMP, 11 conditions).

Folder names encode the condition (readme, Table 3): ``A``, ``A&C`` and ``A&C&B`` are stator
short circuits in 1, 2 or 3 phases at 10, 30 or 50 % of the winding; ``Fan`` is a cooling-
fan failure; ``Rotor-0`` a stuck rotor; ``Noload`` the healthy motor without load. The
``Subset_40_Thermal_GT`` folder holds 40 annotated images and is excluded. One motor
only, so there is no physical unit to split on.
"""

from __future__ import annotations

import re
from pathlib import Path

from pdmfm.datasets.base import Recording, RecordingInfo
from pdmfm.paths import extracted_dir
from pdmfm.schema import ImageItem

_STATOR = re.compile(r"^(?P<phases>[ABC](?:&[ABC])*)(?P<percent>\d+)$")


def parse_folder(name: str) -> tuple[str, str | None]:
    """(label, severity) of a condition folder."""
    if name == "Noload":
        return "healthy", None
    if name == "Fan":
        return "cooling_fan_failure", None
    if name.startswith("Rotor"):
        return "stuck_rotor", None
    match = _STATOR.match(name)
    if match is None:
        raise ValueError(f"unknown Najafi condition folder {name!r}")
    phases = match["phases"].count("&") + 1
    return "stator_short_circuit", f"{phases} phase(s), {match['percent']} %"


class Najafi:
    name = "najafi_thermal"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else extracted_dir("najafi_thermal")

    def index(self) -> list[RecordingInfo]:
        infos = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if folder.name.startswith("Subset"):
                continue
            label, severity = parse_folder(folder.name)
            for image in sorted(folder.glob("*.bmp")):
                infos.append(
                    RecordingInfo(
                        dataset=self.name,
                        recording_id=f"{folder.name}/{image.stem}",
                        unit="motor",
                        label=label,
                        severity=severity,
                        condition={"folder": folder.name},
                        meta={"path": str(image)},
                    )
                )
        return infos

    def load(self, info: RecordingInfo) -> Recording:
        item = ImageItem(kind="thermal", path=info.meta["path"], captured_at=-1.0)
        return Recording(info=info, channels={}, images=(item,))
