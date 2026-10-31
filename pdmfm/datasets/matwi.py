"""MATWI: multimodal automatic tool wear inspection (De Pauw, Jacobs & Goedemé, KU Leuven,
2023; CC BY-SA 4.0; doi:10.48804/GK6LHH).

17 single-insert carbide milling tools (15 mm; Set 17 has two inserts), run to failure in
C45 (Sets 1-11) and 304 stainless steel (Sets 12-17). Each measurement is a cut of about
55.6 s, recorded at 1/0.000611 s ≈ 1,636.7 Hz with five channels (accelerometer, microphone,
Fx, Fy, Fz), followed about 1.5 min later by a photo of the insert, on which an expert
measured the flank wear in µm. Checked on the downloaded files (PROJECT_LOG.md, week 5).

What this reader does:

* **One recording per cut**, labelled by timestamps, not by the rows of ``labels.csv``: the
  wear on the first photo taken after the cut starts and before the next cut (the wear at
  the end of the cut). The rows are shifted by one cut in places (the publisher's
  synchronisation errors; Set 9's median row gap is 175 s instead of about 85 s).
  Cuts without such a photo are unlabelled. For Set 17 (two inserts, photographed in turn)
  the label is the worse insert: the maximum over the latest photo of each insert.
* **Classes**: the flank-wear thresholds of Nonastreda (sharp < 71 µm ≤ used < 110 µm ≤
  dulled), so both milling sets share one text-anchored label space.
* **Rate**: every channel is resampled to 1 kHz, the rate of Nonastreda's force, so milling
  spectra share one frequency axis (the tooth-passing lines, 42-106 Hz, are far below 500 Hz).
* **Photos**: each photo is cropped to the insert with the publisher's per-set crop box
  (``sets.csv``; 600 x 400 px) and saved once to ``extracted/matwi/crops/``. A cut's photo is
  stamped at its capture time, after the cut, so the as-of join attaches it to the *next*
  cut's windows and never to its own (the leak OmniFuser's finding F1 describes).
* **Clock**: seconds since the tool's first event; meta ``t_start`` orders the cuts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm import preprocess as pp
from pdmfm.datasets.base import ChannelData, Recording, RecordingInfo, as_float32
from pdmfm.paths import extracted_dir
from pdmfm.schema import ImageItem

NOMINAL_RATE_HZ = 1 / 0.000611  # every sample 0.611 ms apart (checked on the files)
MODEL_RATE_HZ = 1000.0  # Nonastreda's force rate
CHANNELS = (
    ("acc", "vibration", "none", "spindle"),
    ("mic", "acoustic", "none", "machine"),
    ("fx", "force", "x", "table"),
    ("fy", "force", "y", "table"),
    ("fz", "force", "z", "table"),
)
WEAR_EDGES = (71.0, 110.0)  # µm; Nonastreda's class thresholds
CLASS_NAMES = ("sharp", "used", "dulled")
MATERIAL = {"CK45": "C45 steel", "RVS 304": "304 stainless steel", "RVS304": "304 stainless steel"}


def wear_class(wear_um: float) -> str | None:
    if not np.isfinite(wear_um):
        return None
    return CLASS_NAMES[int(np.searchsorted(WEAR_EDGES, wear_um, side="right"))]


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None  # Set 1's parameters are "?"; Set 6's feed is "185/148"


def read_sets(path: Path) -> dict[int, dict]:
    """Cutting parameters and the insert crop box (x0, y0, x1, y1) per set."""
    table = pd.read_csv(path)
    sets = {}
    for row in table.to_dict("records"):
        number = int(str(row["Unnamed: 0"]).split()[-1])
        crop = tuple(int(v) for v in str(row["crop"]).split(","))
        rpm = _number(row["n"])
        sets[number] = {
            "material": MATERIAL.get(str(row["material"]).strip(), str(row["material"]).strip()),
            "rpm": rpm,
            "speed_text": f"{rpm:.0f} rpm" if rpm else "an unrecorded spindle speed",
            "cutting_speed_m_min": _number(row["Vc"]),
            "feed_per_tooth_mm": _number(row["fz"]),
            "radial_depth_mm": _number(row["Ae"]),
            "axial_depth_mm": _number(row["Ap"]),
            "inserts": int(_number(row["z"]) or 1),
            "crop": crop,
        }
    return sets


def _seconds(stamps) -> np.ndarray:
    return pd.to_datetime(stamps).astype("int64").to_numpy() / 1e9


class MATWI:
    name = "matwi"

    def __init__(self, root: Path | None = None, crops: Path | None = None) -> None:
        self.root = Path(root) if root else extracted_dir("matwi")
        self.crops = Path(crops) if crops else self.root / "crops"
        self._table: pd.DataFrame | None = None

    # ------------------------------------------------------------------ metadata

    def _file(self, relative) -> Path | None:
        """labels.csv paths start with ``MATWI/``; the zips unpack to ``SetN/...``."""
        if not isinstance(relative, str) or not relative:
            return None
        return self.root / relative.removeprefix("MATWI/")

    def table(self) -> pd.DataFrame:
        """labels.csv with clocks: seconds since each set's first event, for cuts and photos."""
        if self._table is None:
            labels = pd.read_csv(self.root / "labels.csv")
            sensor = _seconds(labels["SensorDateTime"].fillna(labels["ImageDateTime"]))
            image = _seconds(labels["ImageDateTime"].fillna(labels["SensorDateTime"]))
            labels["sensor_s"] = np.where(labels["SensorDateTime"].notna(), sensor, np.nan)
            labels["image_s"] = np.where(labels["ImageDateTime"].notna(), image, np.nan)
            origin = labels.groupby("Set")[["sensor_s", "image_s"]].transform("min").min(axis=1)
            labels["sensor_s"] -= origin
            labels["image_s"] -= origin
            self._table = labels
        return self._table

    def crop_path(self, image_file: str) -> Path:
        relative = Path(image_file.removeprefix("MATWI/"))
        return self.crops / relative.parent.parent.name / relative.name

    def prepare_crops(self, log=print) -> int:
        """Crop every photo to its set's insert box (once); returns how many were written."""
        from PIL import Image

        sets = read_sets(self.root / "sets.csv")
        table = self.table()
        written = 0
        for row in table.loc[table["ImageFile"].notna()].to_dict("records"):
            target = self.crop_path(str(row["ImageFile"]))
            if target.exists():
                continue
            source = self._file(row["ImageFile"])
            if source is None or not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.crop(sets[int(row["Set"])]["crop"]).save(target, quality=95)
            written += 1
            if written % 200 == 0:
                log(f"[matwi] {written} insert crops written")
        return written

    # ------------------------------------------------------------------ reader API

    def index(self) -> list[RecordingInfo]:
        sets = read_sets(self.root / "sets.csv")
        table = self.table()
        cuts = table.loc[table["SensorFile"].notna()].sort_values(["Set", "sensor_s"])
        labels = self.cut_wear(cuts, sets)
        infos = []
        for index, row in zip(cuts.index, cuts.to_dict("records"), strict=True):
            number = int(row["Set"])
            wear, wear_type = labels[index]
            condition = {k: v for k, v in sets[number].items() if k != "crop"}
            infos.append(
                RecordingInfo(
                    dataset=self.name,
                    recording_id=Path(str(row["SensorName"])).stem,
                    unit=f"Set{number:02d}",
                    label=wear_class(wear),
                    condition=condition,
                    meta={
                        "path": str(self._file(row["SensorFile"])),
                        "t_start": float(row["sensor_s"]),
                        "wear_um": wear,
                        "wear_type": wear_type,
                        "set": number,
                    },
                )
            )
        return infos

    def cut_wear(self, cuts: pd.DataFrame, sets: dict[int, dict]) -> dict:
        """(wear in µm, wear type) per cut row, from the photos taken after the cut and before
        the next one; NaN where there is none."""
        table = self.table()
        photos = table.loc[table["ImageFile"].notna() & table["wear"].notna()]
        photos = photos.sort_values("image_s")
        out: dict = {}  # cut row index -> (wear, wear type)
        for _, mine in cuts.groupby("Set"):
            number = int(mine["Set"].iloc[0])
            starts = mine["sensor_s"].to_numpy()
            ends = np.append(starts[1:], np.inf)
            taken = photos.loc[photos["Set"] == number]
            times, wears = taken["image_s"].to_numpy(), taken["wear"].to_numpy()
            kinds = taken["type"].fillna("").astype(str).to_numpy()
            inserts = sets[number]["inserts"]
            for index, start, end in zip(mine.index, starts, ends, strict=True):
                own = np.flatnonzero((times >= start) & (times < end))
                if not len(own):
                    out[index] = (np.nan, "")
                    continue
                last = own[0]  # the first photo after the cut shows its end state
                recent = wears[max(0, last - inserts + 1) : last + 1]  # one per insert
                out[index] = (float(np.max(recent)), str(kinds[last]))
        return out

    def photos(self, unit: str, after: float, before: float) -> tuple[ImageItem, ...]:
        """Insert photos of ``unit`` captured in [after, before), stamped at capture."""
        table = self.table()
        mine = table[
            (table["Set"] == int(unit.removeprefix("Set")))
            & table["ImageFile"].notna()
            & (table["image_s"] >= after)
            & (table["image_s"] < before)
        ]
        return tuple(
            ImageItem("insert", str(self.crop_path(str(f))), float(t))
            for f, t in zip(mine["ImageFile"], mine["image_s"], strict=True)
        )

    def load(self, info: RecordingInfo) -> Recording:
        values = pd.read_csv(info.meta["path"], header=None).iloc[:, :5].to_numpy(np.float64)
        channels = {
            name: ChannelData(
                name,
                family,
                axis,
                placement,
                MODEL_RATE_HZ,
                as_float32(pp.resample(values[:, i], NOMINAL_RATE_HZ, MODEL_RATE_HZ)),
                "N" if family == "force" else "",
            )
            for i, (name, family, axis, placement) in enumerate(CHANNELS)
        }
        start = info.meta["t_start"]
        later = [
            i.meta["t_start"] for i in self._unit_infos(info.unit) if i.meta["t_start"] > start
        ]
        until = min(later) if later else np.inf
        return Recording(
            info=info,
            channels=channels,
            images=self.photos(info.unit, start, until),  # photos after this cut, before the next
            t_offset=start,
            targets={"wear_um": info.meta["wear_um"]},
        )

    def _unit_infos(self, unit: str) -> list[RecordingInfo]:
        if not hasattr(self, "_by_unit"):
            self._by_unit: dict[str, list[RecordingInfo]] = {}
            for i in self.index():
                self._by_unit.setdefault(i.unit, []).append(i)
        return self._by_unit[unit]
