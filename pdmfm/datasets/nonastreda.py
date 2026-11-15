"""Native Nonastreda MAT/CSV reader, including post-pass photographs.

The default 378-pass view reproduces historical deduplication and trimming. Lifecycle
conversion separately preserves all 512 passes, with duplicate signals available only at
the latest member. The nominal photo pause is an ordering device, not operating time.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d

from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import raw_dir
from pdmfm.schema import ImageItem

RATE_HZ = 1000.0
# Nominal pause between passes (not recorded in the data). Its length only sets the photo
# ages; what matters is that the photos fall strictly between two passes.
PASS_GAP_S = 60.0
# The publisher's image labels (annotators' visual judgement). 360 of the 378 passes agree
# with flank wear < 71 µm, < 110 µm, >= 110 µm; 18 do not, mostly T6 (PROJECT_LOG.md, week 6).
CLASS_NAMES = ("sharp", "used", "dulled")
PHOTO_FOLDERS = {"tool": ("tool", ".jpg"), "chip": ("chip", ".jpg"), "workpiece": ("work", ".png")}
FORCE_AXES = (("fx", "x"), ("fy", "y"), ("fz", "z"))


@dataclass
class ForceRecord:
    record_id: str
    tool: str
    run: int
    block: int
    force: np.ndarray
    label: int


def trim_bounds(force: np.ndarray, causal: bool = False) -> tuple[int, int]:
    """Retain the active span; lifecycle reference uses a fixed first five seconds.

    Predictions are made at pass end, when both span bounds are available. The legacy
    five-percent reference remains available for reproducing earlier window caches.
    """
    if len(force) < 20:
        return 0, len(force)
    envelope = np.sqrt(uniform_filter1d(np.square(force.astype(float)).sum(1), 20, mode="nearest"))
    count = min(5000, len(force)) if causal else max(20, int(0.05 * len(force)))
    reference = float(np.median(envelope[:count]))
    active = np.flatnonzero(envelope >= reference / 2)
    if reference <= np.finfo(float).eps or not len(active):
        return 0, len(force)
    return int(active[0]), int(active[-1]) + 1


def _read_regression(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
        result = {row["id"]: {k: float(v) for k, v in row.items() if k != "id"} for row in rows}
    if len(result) != len(rows) or not all(
        np.isfinite(v) for values in result.values() for v in values.values()
    ):
        raise ValueError("regression IDs must be unique and measurements finite")
    return result


class Nonastreda:
    name = "nonastreda"

    def __init__(self, root: Path | None = None, dedup: bool = True, trim: bool = True) -> None:
        self.root = Path(root) if root else raw_dir("nonastreda")
        self.dedup, self.trim = dedup, trim
        self._records = None
        self._all = None

    def all_records(self) -> dict[str, ForceRecord]:
        """Native MAT/CSV ingestion, without a sibling checkout or private package."""
        if self._all is not None:
            return self._all
        with (self.root / "labels.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        labels = {r["id"]: r["image_label"].lower() for r in rows}
        regression = _read_regression(self.root / "labels_reg.csv")
        if len(labels) != len(rows) or set(labels) != set(regression):
            raise ValueError("Nonastreda label IDs must be unique and match regression IDs")
        raw = loadmat(
            self.root / "forces_xyz_raw.mat", variable_names=["baseDatastore"], simplify_cells=True
        ).get("baseDatastore")
        if isinstance(raw, np.ndarray) and raw.ndim == 1:
            raw = raw[None, :]
        if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.shape[1] < 4:
            raise ValueError("baseDatastore must contain filename, label and force columns")
        records = {}
        for row in raw:
            rid = Path(str(row[0])).stem
            match = re.fullmatch(r"(T\d+)R(\d+)B(\d+)", rid)
            if match is None or rid in records or rid not in labels:
                raise ValueError(f"invalid or duplicate Nonastreda record ID: {rid}")
            label = str(row[1]).lower()
            force = np.asarray(row[3])
            if label != labels[rid] or label not in CLASS_NAMES:
                raise ValueError(f"label mismatch for {rid}")
            if force.ndim != 2 or force.shape[0] != 3 or not np.isfinite(force).all():
                raise ValueError(f"{rid}: force must be finite (3, samples)")
            records[rid] = ForceRecord(
                rid,
                match[1],
                int(match[2]),
                int(match[3]),
                np.ascontiguousarray(force.T, dtype=np.float32),
                CLASS_NAMES.index(label),
            )
        if set(records) != set(labels):
            raise ValueError("force IDs must match labels.csv")
        self._all = records
        return records

    def duplicate_groups(self) -> list[list[str]]:
        """Exact copies within a tool/run; chronological membership is label independent."""
        import hashlib

        groups = {}
        for r in sorted(self.all_records().values(), key=lambda r: (r.run, r.block)):
            digest = hashlib.sha256(r.force.tobytes()).hexdigest()
            groups.setdefault((r.tool, r.run, r.force.shape, digest), []).append(r.record_id)
        return [g for g in groups.values() if len(g) > 1]

    def _load_records(self):
        if self._records is None:
            records = dict(self.all_records())
            if self.dedup:
                regression = _read_regression(self.root / "labels_reg.csv")
                for group in self.duplicate_groups():
                    keep = max(
                        group, key=lambda k: (regression[k]["flank_wear"], -records[k].block)
                    )
                    for rid in group:
                        if rid != keep:
                            del records[rid]
            self._records = {}
            for rid, r in records.items():
                start, end = trim_bounds(r.force) if self.trim else (0, len(r.force))
                self._records[rid] = ForceRecord(
                    rid, r.tool, r.run, r.block, r.force[start:end], r.label
                )
        return self._records

    def index(self) -> list[RecordingInfo]:
        records = self._load_records()
        regression = _read_regression(self.root / "labels_reg.csv")
        clocks: dict[str, float] = {}
        infos = []
        for record in sorted(records.values(), key=lambda r: (int(r.tool[1:]), r.run, r.block)):
            start = clocks.get(record.tool, 0.0)
            seconds = len(record.force) / RATE_HZ
            clocks[record.tool] = start + seconds + PASS_GAP_S
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
        """This pass's photos, stamped in the pause after the pass (taken after it)."""
        taken = info.meta["t_end"] + PASS_GAP_S / 2
        items = []
        for kind, (folder, suffix) in PHOTO_FOLDERS.items():
            path = self.root / folder / f"{info.recording_id}{suffix}"
            if path.exists():
                items.append(ImageItem(kind=kind, path=str(path), captured_at=taken))
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
