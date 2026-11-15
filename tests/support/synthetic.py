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


def paderborn_reader(seconds: float = 2.1, seed: int = 0) -> FakeReader:
    """The 15 source bearings, a real-damage and a held-out healthy target bearing, and the
    out-of-space KB27, at two settings. Each class has its own vibration tone; the
    temperature (1 Hz, a slow channel) sits 3 K higher per class above a setting offset."""
    from pdmfm.datasets.paderborn import DAMAGE
    from pdmfm.protocols import PADERBORN_SOURCE_FOLDS
    from pdmfm.sources import PADERBORN_SETTINGS

    rng = np.random.default_rng(seed)
    rate = 12_000.0
    n = int(seconds * rate)
    bearings = [b for fold in PADERBORN_SOURCE_FOLDS for b in fold] + ["KA04", "K004", "KB27"]
    tones = {"healthy": None, "outer_race": 1300.0, "inner_race": 1700.0, "combined": 900.0}
    offsets = {"healthy": 0.0, "outer_race": 3.0, "inner_race": 6.0, "combined": 9.0}
    recordings = []
    for setting in ("N15_M07_F10", "N09_M07_F10"):
        rpm, torque, force = PADERBORN_SETTINGS[setting]
        for bearing in bearings:
            label, level, _ = DAMAGE[bearing]
            vibration = _tone(rpm / 60, rate, n) + 0.2 * rng.standard_normal(n)
            if tones[label] is not None:
                vibration = vibration + _tone(tones[label], rate, n, 0.8)
            channels = {
                "vibration": ChannelData(
                    "vibration", "vibration", "none", "housing", rate, vibration.astype(np.float32)
                ),
            }
            for name in ("current_1", "current_2"):
                current = _tone(50.0, rate, n) + 0.1 * rng.standard_normal(n)
                channels[name] = ChannelData(
                    name, "current", "none", "motor_phase", rate, current.astype(np.float32)
                )
            base = 40.0 + (5.0 if setting.startswith("N15") else 0.0) + offsets[label]
            temperature = base + 0.1 * np.arange(int(seconds) + 1) + rng.normal(0, 0.05)
            channels["temperature"] = ChannelData(
                "temperature", "slow", "none", "bearing_module", 1.0,
                temperature.astype(np.float32),
            )  # fmt: skip
            condition = {"setting": setting, "rpm": float(rpm), "torque_nm": torque}
            info = RecordingInfo(
                "paderborn", f"{setting}_{bearing}_1", bearing, label,
                None if level is None else f"level {level}",
                {**condition, "radial_n": float(force)},
            )  # fmt: skip
            recordings.append(Recording(info, channels))
    return FakeReader(recordings)


def matwi_reader(photo_root=None, cuts: int = 4, seed: int = 0) -> FakeReader:
    """17 single-insert tools x ``cuts`` cuts of 8.5 s at 1 kHz (force, vibration, sound),
    each followed by an insert photo; the Fx level and the vibration grow with the wear."""
    rng = np.random.default_rng(seed)
    rate, seconds, gap = 1000.0, 8.5, 30.0
    n = int(seconds * rate)
    recordings = []
    for number in range(1, 18):
        unit, clock = f"Set{number:02d}", 0.0
        for c in range(cuts):
            wear = 40.0 + 30.0 * c + rng.normal(0, 3)
            label = ("sharp", "used", "dulled")[int(np.searchsorted((71.0, 110.0), wear, "right"))]
            channels = {
                name: ChannelData(
                    name, family, axis, placement, rate,
                    (offset + scale * wear / 40 * rng.standard_normal(n)).astype(np.float32),
                    "N" if family == "force" else "",
                )
                for name, family, axis, placement, offset, scale in (
                    ("fx", "force", "x", "table", 100.0 + wear, 5.0),
                    ("fy", "force", "y", "table", 50.0, 5.0),
                    ("fz", "force", "z", "table", 10.0, 2.0),
                    ("acc", "vibration", "none", "spindle", 0.0, 1.0),
                    ("mic", "acoustic", "none", "machine", 0.0, 0.5),
                )
            }  # fmt: skip
            record_id = f"{unit}_cut{c}"
            images = ()
            if photo_root is not None:
                images = (
                    ImageItem("insert", str(photo_root / f"{record_id}.jpg"), clock + seconds + 5),
                )
            info = RecordingInfo(
                "matwi", record_id, unit, label, None,
                {"material": "C45 steel" if number <= 11 else "304 stainless steel",
                 "inserts": 1, "speed_text": "2547 rpm"},
                {"t_start": clock, "wear_um": wear, "set": number},
            )  # fmt: skip
            recordings.append(Recording(info, channels, images, clock, {"wear_um": wear}))
            clock += seconds + gap
    return FakeReader(recordings)


# ----------------------------------------------------------------------------- caches


class FakeImageEncoder:
    """Stand-in for DINOv2: 5 tokens of ``dim`` per photo, a fixed function of the path."""

    name = "dinov2"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def signature(self, kind):
        return {"encoder": "fake-dinov2", "kind": kind, "dim": self.dim}

    def encode_paths(self, paths, kind):
        return np.stack([_vector(f"{kind}:{p}", (5, self.dim)) for p in paths]).astype(np.float32)


class FakeTextEncoder:
    """Stand-in for Qwen3-Embedding: unit vectors, a fixed function of the sentence."""

    name = "qwen3_embedding"

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def signature(self):
        return {"encoder": "fake-qwen3", "dim": self.dim}

    def encode(self, sentences):
        z = np.stack([_vector(s, (self.dim,)) for s in sentences]).astype(np.float32)
        return z / np.linalg.norm(z, axis=1, keepdims=True)


def _vector(text: str, shape: tuple[int, ...]) -> np.ndarray:
    import hashlib

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(shape)


def build_planb_root(root, datasets=("nonastreda", "xjtu_sy", "cwru", "paderborn", "matwi")):
    """Window stores, stats features (arms T, F and, for bearings, E), photo caches and text
    prototypes under ``root`` laid out as a data root (``interim/windows``,
    ``interim/features``). Call with ``PDMFM_DATA_ROOT`` set to ``root``."""
    from pdmfm import features, windows
    from pdmfm.encoders import StatsEncoder

    def quiet(*_):
        pass

    readers = {
        "nonastreda": lambda: nonastreda_reader(photo_root=root / "raw" / "nonastreda"),
        "xjtu_sy": lambda: xjtu_reader(30),
        "cwru": cwru_reader,
        "paderborn": paderborn_reader,
        "matwi": lambda: matwi_reader(photo_root=root / "raw" / "matwi"),
    }
    cache = features.FeatureStore(root / "interim" / "features")
    for name in datasets:
        store = windows.build(name, reader=readers[name](), log=quiet)
        for arm in ("T", "F", "E") if store.spec.bearing else ("T", "F"):
            cache.extract(store, StatsEncoder(), arm, log=quiet)
        if store.spec.image_kinds:
            attached = {
                kind: sorted(p for p in store.manifest[f"image_{kind}"].unique() if p)
                for kind in store.spec.image_kinds
            }
            # Every kind gets a cache, as on the real data; one photo is never attached.
            attached = {k: v + [f"raw/{name}/unattached-{k}.png"] for k, v in attached.items()}
            original = features.image_paths
            features.image_paths = lambda dataset, found=attached: found
            try:
                features.extract_images(name, FakeImageEncoder(), root=cache.root, log=quiet)
            finally:
                features.image_paths = original
    features.extract_texts(FakeTextEncoder(), list(datasets), root=cache.root, log=quiet)
    return root
