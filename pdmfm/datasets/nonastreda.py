"""Nonastreda milling tool-wear data, read through OmniFuser's loader.

``omnifuser.data.load_nonastreda(pool=1)`` gives the 378 deduplicated, edge-trimmed passes
at 1 kHz (the dataset's own rate). This reader adds what OmniFuser's ``Record`` lacks:
flank wear, gaps and overhang from ``labels_reg.csv``; the chip and workpiece photos; and a
clock per tool. Passes are laid end to end in (run, block) order, and every photo of a pass
is stamped with the pass's end time, because the photos are taken after the pass. The as-of
join (:func:`pdmfm.schema.as_of_join`) then attaches only the previous pass's photos to a
window. Run and block numbers stay in ``meta`` and are never model inputs (the run number
alone predicts the class with 71 % accuracy, OmniFuser finding F2).
"""

from __future__ import annotations

import csv
from pathlib import Path

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import raw_dir
from pdmfm.schema import ImageItem

RATE_HZ = 1000.0
CLASS_NAMES = ("sharp", "used", "dulled")  # flank wear < 71 µm, < 110 µm, >= 110 µm
PHOTO_FOLDERS = {"tool": ("tool", ".jpg"), "chip": ("chip", ".jpg"), "workpiece": ("work", ".png")}
FORCE_AXES = (("fx", "x"), ("fy", "y"), ("fz", "z"))


def _read_regression(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["id"]: {k: float(v) for k, v in row.items() if k != "id"}
            for row in csv.DictReader(handle)
        }


class Nonastreda:
    name = "nonastreda"

    def __init__(self, root: Path | None = None, dedup: bool = True, trim: bool = True) -> None:
        self.root = Path(root) if root else raw_dir("nonastreda")
        self.dedup, self.trim = dedup, trim
        self._records = None

    def _load_records(self):
        if self._records is None:
            from omnifuser.data import load_nonastreda  # optional dependency, imported lazily

            records = load_nonastreda(
                self.root, pool=1, dedup_identical_forces=self.dedup, trim_pass=self.trim
            )
            self._records = {r.record_id: r for r in records}
        return self._records

    def index(self) -> list[RecordingInfo]:
        records = self._load_records()
        regression = _read_regression(self.root / "labels_reg.csv")
        clocks: dict[str, float] = {}
        infos = []
        for record in sorted(records.values(), key=lambda r: (int(r.tool[1:]), r.run, r.block)):
            start = clocks.get(record.tool, 0.0)
            seconds = len(record.force) / RATE_HZ
            clocks[record.tool] = start + seconds
            infos.append(
                RecordingInfo(
                    dataset=self.name,
                    recording_id=record.record_id,
                    unit=record.tool,
                    label=CLASS_NAMES[record.label],
                    condition={},
                    meta={
                        "run": record.run,  # meta only: a 71 % shortcut if used as input
                        "block": record.block,
                        "t_start": start,
                        "t_end": start + seconds,
                        "seconds": seconds,
                        **{
                            k: regression[record.record_id][k]
                            for k in ("flank_wear", "gaps", "overhang")
                        },
                    },
                )
            )
        return infos

    def photos(self, info: RecordingInfo) -> tuple[ImageItem, ...]:
        """This pass's photos, stamped with the pass's end time (taken after the pass)."""
        items = []
        for kind, (folder, suffix) in PHOTO_FOLDERS.items():
            path = self.root / folder / f"{info.recording_id}{suffix}"
            if path.exists():
                items.append(ImageItem(kind=kind, path=str(path), captured_at=info.meta["t_end"]))
        return tuple(items)

    def load(self, info: RecordingInfo) -> Recording:
        record = self._load_records()[info.recording_id]
        channels = {
            name: ChannelData(
                name, "force", axis, "table", RATE_HZ, as_float32(record.force[:, i]), "N"
            )
            for i, (name, axis) in enumerate(FORCE_AXES)
        }
        return Recording(
            info=info,
            channels=channels,
            images=self.photos(info),
            t_offset=info.meta["t_start"],
            targets={"wear_um": info.meta["flank_wear"]},
        )
