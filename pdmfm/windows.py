"""Stage 1 of the pipeline: raw recordings -> window shards and a manifest.

One store per dataset, in ``$PDMFM_DATA_ROOT/interim/windows/<dataset>/``:

* ``signals.f32``: (N, C, L) float32, the model input of every channel
  (:func:`pdmfm.preprocess.model_input`: resampled to 12 kHz when faster, band-limited to
  5 kHz), cut into non-overlapping windows. float32, not float16: float16 rounding is white
  noise up to Nyquist and would undo the band limit.
* ``levels.npy``: (N, C, 4) absolute level descriptors. Relative levels need a healthy
  baseline from each fold's training units, so they are computed at training time.
* ``slow.npy``: (N, S, 2) mean and slope of the slow channels over each window.
* ``manifest.csv``: one row per window: unit, bag, labels, condition, measured shaft speed,
  context sentence and, for Nonastreda, the photos attached by the as-of join (previous
  pass only) with their age.
* ``meta.json``: signature (spec, preprocessing version, source fingerprint), shapes and
  ``"complete": true``. The store is written to a temporary folder and renamed when complete.

    python -m pdmfm.windows build cwru kaist xjtu_sy nonastreda
    python -m pdmfm.windows info
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pdmfm import preprocess as pp
from pdmfm import texts
from pdmfm.paths import data_root, interim_dir, raw_dir
from pdmfm.schema import ImageItem, as_of_join
from pdmfm.utils import stable_hash

# Part of every store's signature. Bump it whenever the stores' content changes: this
# module, or a reader's output (as the Nonastreda photo pause did before version 1 was
# first used). Stores of another version are refused when opened.
WINDOWS_VERSION = 1
# Never parsed as numbers when the manifest is read back.
TEXT_COLUMNS = {
    "dataset",
    "recording_id",
    "unit",
    "bag",
    "label",
    "severity",
    "class_key",
    "label_space",
    "context",
    "shaft_source",
    "fpt_source",
    "setting",
    "condition",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    channels: tuple[str, ...]  # high-rate channels, in this order
    slow: tuple[str, ...] = ()  # slow channels: mean and slope per window
    window: int = pp.WINDOW_SAMPLES  # samples at the model rate
    bearing: bool = True  # the E arm (envelope on a shaft-order axis) is defined
    speed: str = "measured"  # shaft speed: "measured" from vibration, "nominal", or "none"
    first_window_only: bool = False  # XJTU-SY: the first 1.024 s of each 1.28 s snapshot
    image_kinds: tuple[str, ...] = ()  # photos attached by the as-of join

    @property
    def segments(self) -> int:
        return self.window // pp.SEGMENT


SPECS = {
    "cwru": DatasetSpec("cwru", ("de", "fe")),
    "kaist": DatasetSpec(
        "kaist",
        ("vib_ax", "vib_ay", "vib_bx", "vib_by", "current_1"),  # currents 2-3: class-dependent gaps
        slow=("temp_1", "temp_2"),
    ),
    # Short snapshots (1.28 s) cannot resolve the shaft line well; the motor speed is set.
    "xjtu_sy": DatasetSpec("xjtu_sy", ("vib_h", "vib_v"), speed="nominal", first_window_only=True),
    "nonastreda": DatasetSpec(
        "nonastreda",
        ("fx", "fy", "fz"),
        window=8192,  # 8.192 s at the native 1 kHz
        bearing=False,
        speed="none",
        image_kinds=("chip", "tool", "workpiece"),
    ),
    "paderborn": DatasetSpec(
        "paderborn", ("vibration", "current_1", "current_2"), slow=("temperature",)
    ),
    "phm2010": DatasetSpec(
        "phm2010",
        ("fx", "fy", "fz", "vx", "vy", "vz"),
        slow=("ae_rms",),
        bearing=False,
        speed="none",
    ),
}


def source_fingerprint(dataset: str) -> str | None:
    """Hash of the (path, SHA-256) pairs in the raw MANIFEST.json: it changes when a source
    file changes, not when the manifest is merely rewritten."""
    path = raw_dir(dataset) / "MANIFEST.json"
    if not path.is_file():
        return None
    files = json.loads(path.read_text(encoding="utf-8")).get("files", {})
    return stable_hash(sorted((k, v.get("sha256")) for k, v in files.items()))[:16]


def window_signature(spec: DatasetSpec, limit: int | None = None) -> dict:
    return {
        "dataset": spec.name,
        "spec": asdict(spec),
        "windows_version": WINDOWS_VERSION,
        "preprocess_version": pp.PREPROCESS_VERSION,
        "model_rate_hz": pp.MODEL_RATE_HZ,
        "common_band_hz": pp.COMMON_BAND_HZ,
        "source": source_fingerprint(spec.name),
        "limit": limit,
    }


# ----------------------------------------------------------------------------- store


def _read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame.columns:
        if column in TEXT_COLUMNS or column.startswith("image_"):
            continue
        values = frame[column]
        numbers = pd.to_numeric(values, errors="coerce")  # "" becomes NaN
        if numbers.notna().sum() == (values != "").sum():  # every non-empty value is a number
            frame[column] = numbers
    return frame


def stale_reasons(signature: dict) -> list[str]:
    """Why a store's signature no longer matches this code (empty when it does). The source
    fingerprint is not compared: a store copied to Ruche has no raw files beside it."""
    current = window_signature(SPECS[signature["dataset"]], signature.get("limit"))
    reasons = [
        f"{key} {signature.get(key)} != {current[key]}"
        for key in ("windows_version", "preprocess_version", "model_rate_hz", "common_band_hz")
        if signature.get(key) != current[key]
    ]
    if json.loads(json.dumps(signature["spec"])) != json.loads(json.dumps(current["spec"])):
        reasons.append("dataset spec changed")
    return reasons


class WindowStore:
    """A built window store, read lazily (signals are memory-mapped). A store built by other
    code (another windows or preprocessing version, or another spec) is refused."""

    def __init__(self, folder: Path) -> None:
        folder = Path(folder)
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"no window store in {folder}; run python -m pdmfm.windows")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not self.meta.get("complete"):
            raise ValueError(f"window store {folder} is incomplete; rebuild it")
        self.folder = folder
        self.dataset: str = self.meta["dataset"]
        self.channels: tuple[str, ...] = tuple(self.meta["channels"])
        self.families: tuple[str, ...] = tuple(self.meta["families"])
        self.slow_channels: tuple[str, ...] = tuple(self.meta["slow_channels"])
        self.rate: float = float(self.meta["rate_hz"])
        self.window: int = int(self.meta["window"])
        self.signature: dict = self.meta["signature"]
        self.key: str = self.meta["key"]
        stale = stale_reasons(self.signature)
        if stale:
            raise ValueError(
                f"window store {folder} was built by other code ({'; '.join(stale)}); "
                f"rebuild it: python -m pdmfm.windows build {self.dataset}"
            )
        self.limit: int | None = self.signature.get("limit")
        self.manifest = _read_manifest(folder / "manifest.csv")
        n = len(self.manifest)
        self.signals = np.memmap(
            folder / "signals.f32", dtype=np.float32, mode="r",
            shape=(n, len(self.channels), self.window),
        )  # fmt: skip
        self.levels = np.load(folder / "levels.npy")
        self.slow = np.load(folder / "slow.npy")

    def __len__(self) -> int:
        return len(self.manifest)

    @property
    def spec(self) -> DatasetSpec:
        spec = dict(self.signature["spec"])
        for key in ("channels", "slow", "image_kinds"):
            spec[key] = tuple(spec[key])
        return DatasetSpec(**spec)

    @classmethod
    def open(cls, dataset: str, root: Path | None = None) -> WindowStore:
        """The full store of ``dataset`` (smoke-test stores live in ``<dataset>-limit<N>``)."""
        return cls((Path(root) if root else interim_dir("windows")) / dataset)


# ----------------------------------------------------------------------------- build


def _class_key(dataset_texts: texts.DatasetTexts, label: str | None, severity: str | None) -> str:
    """Key in the dataset's label space, or "" (no label, or outside the space)."""
    if label is None:
        return ""
    key = texts.class_key(label, severity)
    if key in dataset_texts.descriptions:
        return key
    return label if label in dataset_texts.descriptions else ""


def _cwru_context(condition: dict) -> dict:
    """CWRU's per-file rpm (1721-1729 rpm at 3 hp) would identify the recording, and so the
    class: the context shows the nominal speed of the load."""
    from pdmfm.datasets.cwru import NOMINAL_RPM

    return {**condition, "rpm": NOMINAL_RPM[condition["load_hp"]]}


# Per-dataset hooks of the builder (kept out of DatasetSpec so store signatures stay stable).
CONTEXT_CONDITION = {"cwru": _cwru_context}  # condition shown in the context sentence
RUL_INDICATOR = {"xjtu_sy": "vib_h"}  # run-to-failure: channel whose RMS and kurtosis set FPT


def _shaft(spec: DatasetSpec, recording) -> tuple[float, str]:
    rpm = recording.info.condition.get("rpm")
    if spec.speed == "none" or rpm is None:
        return float("nan"), "none"
    nominal = float(rpm) / 60.0
    if spec.speed == "nominal":
        return nominal, "nominal"
    channel = recording.channels[spec.channels[0]]
    measured = pp.dominant_frequency(
        channel.signal, channel.rate_hz, 0.95 * nominal, 1.05 * nominal
    )
    return measured, "measured"


def _relative(path: str) -> str:
    """Photo paths relative to the data root, so a store works on another machine."""
    try:
        return Path(os.path.relpath(path, data_root())).as_posix()
    except ValueError:  # another drive on Windows
        return Path(path).as_posix()


def _scalar_meta(meta: dict) -> dict:
    return {
        f"meta_{k}": v
        for k, v in meta.items()
        if isinstance(v, int | float | str | bool) and "path" not in k and "file" not in k
    }


def build(
    dataset: str,
    out: Path | None = None,
    limit: int | None = None,
    reader=None,
    log=print,
) -> WindowStore:
    """Build (or rebuild) one dataset's window store. ``limit`` keeps the first recordings
    only (smoke tests), in a separate ``<dataset>-limit<N>`` folder; ``reader`` replaces the
    registered reader (tests)."""
    from pdmfm.datasets import get_reader

    spec = SPECS[dataset]
    reader = reader or get_reader(dataset)
    dataset_texts = texts.load(dataset)
    infos = reader.index()
    if limit is not None:
        infos = infos[:limit]
    name = dataset if limit is None else f"{dataset}-limit{limit}"  # never replaces the full store
    folder = Path(out) if out else interim_dir("windows") / name
    partial = folder.with_name(f".{folder.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)

    try:
        rows, levels, slow = [], [], []
        families: tuple[str, ...] | None = None
        rate: float | None = None
        photos: dict[str, list[ImageItem]] = defaultdict(list)
        rul_series: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        start_time = time.perf_counter()
        with (partial / "signals.f32").open("wb") as handle:
            for number, info in enumerate(infos, 1):
                recording = reader.load(info)
                missing = [c for c in spec.channels if c not in recording.channels]
                if missing:
                    raise ValueError(f"{dataset} {info.recording_id}: channels missing: {missing}")
                prepared = []
                for name in spec.channels:
                    channel = recording.channels[name]
                    signal, channel_rate = pp.model_input(channel.signal, channel.rate_hz)
                    if rate is None:
                        rate = channel_rate
                    elif channel_rate != rate:
                        raise ValueError(f"{dataset}: channels at different model rates")
                    prepared.append(signal)
                families = families or tuple(recording.channels[c].family for c in spec.channels)
                assert rate is not None
                length = min(len(s) for s in prepared)  # two devices may differ in length
                if spec.first_window_only:
                    starts = np.array([0]) if length >= spec.window else np.array([], dtype=int)
                else:
                    starts = pp.window_starts(length, spec.window, spec.window)
                stack = np.stack([s[:length] for s in prepared])
                shaft_hz, shaft_source = _shaft(spec, recording)
                condition = dict(recording.info.condition)
                shown = CONTEXT_CONDITION.get(dataset, lambda c: c)(condition)
                context = dataset_texts.context(shown)
                if dataset in RUL_INDICATOR and len(starts):  # health indicators, for RUL labels
                    x = recording.channels[RUL_INDICATOR[dataset]].signal.astype(np.float64)
                    rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
                    kurt = float(stats.kurtosis(x, fisher=False))
                    rul_series[info.unit].append((len(rows), rms, kurt))
                for index, start in enumerate(starts):
                    window = stack[:, start : start + spec.window]
                    handle.write(np.ascontiguousarray(window, dtype=np.float32).tobytes())
                    levels.append([pp.level_descriptors(w) for w in window])
                    t0 = start / rate
                    t1 = t0 + spec.window / rate
                    summaries = []
                    for name in spec.slow:
                        sc = recording.channels.get(name)
                        if sc is None:
                            summaries.append([np.nan, np.nan])
                            continue
                        piece = sc.signal[int(t0 * sc.rate_hz) : max(int(t1 * sc.rate_hz), 1)]
                        summaries.append(
                            pp.slow_summary(piece, sc.rate_hz) if len(piece) else [np.nan, np.nan]
                        )
                    slow.append(summaries)
                    t_start = recording.t_offset + t0
                    row = {
                        "dataset": dataset,
                        "recording_id": info.recording_id,
                        "unit": info.unit,
                        "bag": info.recording_id,
                        "window": index,
                        "start": int(start),
                        "t_start": t_start,
                        "t_end": recording.t_offset + t1,
                        "label": info.label or "",
                        "severity": info.severity or "",
                        "class_key": _class_key(dataset_texts, info.label, info.severity),
                        "label_space": dataset_texts.label_space or "",
                        "context": context,
                        "shaft_hz": shaft_hz,
                        "shaft_source": shaft_source,
                        "wear_um": recording.targets.get("wear_um", info.meta.get("wear_um")),
                        **condition,
                        **_scalar_meta(info.meta),
                    }
                    if spec.image_kinds:
                        attached = as_of_join(photos[info.unit], t_start, spec.image_kinds)
                        for item in attached:
                            row[f"image_{item.kind}"] = _relative(item.path)
                            row[f"age_{item.kind}"] = t_start - item.captured_at
                    rows.append(row)
                photos[info.unit].extend(recording.images)
                if number % 100 == 0 or number == len(infos):
                    elapsed = time.perf_counter() - start_time
                    log(
                        f"[windows] {dataset}: {number}/{len(infos)} recordings, "
                        f"{len(rows)} windows, {elapsed:.0f} s"
                    )

        if not rows:
            shutil.rmtree(partial, ignore_errors=True)
            raise ValueError(f"{dataset}: no windows (recordings shorter than one window?)")
        manifest = pd.DataFrame(rows)
        # Every image kind gets its columns, empty where nothing is attached.
        for kind in spec.image_kinds:
            if f"image_{kind}" not in manifest:
                manifest[f"image_{kind}"] = ""
                manifest[f"age_{kind}"] = np.nan
            manifest[f"image_{kind}"] = manifest[f"image_{kind}"].fillna("")
        if dataset in RUL_INDICATOR:
            manifest["rul"] = np.nan
            manifest["fpt"] = np.nan
            manifest["fpt_source"] = ""
            for series in rul_series.values():
                index = [r for r, _, _ in series]
                fpt, source = pp.label_fpt(
                    np.array([s[1] for s in series]), np.array([s[2] for s in series])
                )
                manifest.loc[index, "rul"] = pp.normalized_rul(len(series), fpt)
                manifest.loc[index, "fpt"] = np.nan if fpt is None else fpt
                manifest.loc[index, "fpt_source"] = source
        manifest.insert(0, "row", np.arange(len(manifest)))
        manifest.to_csv(partial / "manifest.csv", index=False)
        np.save(partial / "levels.npy", np.asarray(levels, dtype=np.float32))
        np.save(
            partial / "slow.npy",
            np.asarray(slow, dtype=np.float32).reshape(len(rows), len(spec.slow), 2),
        )
        signature = window_signature(spec, limit)
        meta = {
            "dataset": dataset,
            "channels": list(spec.channels),
            "families": list(families or ()),
            "slow_channels": list(spec.slow),
            "rate_hz": rate,
            "window": spec.window,
            "n_windows": len(rows),
            "signature": signature,
            "key": stable_hash(signature)[:12],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.perf_counter() - start_time, 1),
            "complete": True,
        }
        (partial / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
        )
    except BaseException:  # a failed build leaves nothing behind
        shutil.rmtree(partial, ignore_errors=True)
        raise
    old = folder.with_name(f".{folder.name}.old")
    shutil.rmtree(old, ignore_errors=True)
    if folder.exists():
        try:
            folder.rename(old)
        except PermissionError as error:  # Windows: a process still maps the old signals
            shutil.rmtree(partial, ignore_errors=True)
            raise PermissionError(
                f"cannot replace {folder}: another process (or this one) still reads it; "
                "close it and build again"
            ) from error
    partial.rename(folder)
    shutil.rmtree(old, ignore_errors=True)
    log(f"[windows] {dataset}: {len(rows)} windows x {len(spec.channels)} channels -> {folder}")
    return WindowStore(folder)


# ----------------------------------------------------------------------------- command line


def _cmd_build(args: argparse.Namespace) -> None:
    from pdmfm.resources import claim_resources

    claim_resources("windows", min_idle_cpus=1.0, min_free_ram_gb=2.0, limit=4, force=args.force)
    for dataset in args.datasets:
        build(dataset, limit=args.limit)


def _cmd_info(args: argparse.Namespace) -> None:
    root = interim_dir("windows")
    names = args.datasets or sorted(
        p.name for p in root.glob("[!.]*") if (p / "meta.json").is_file()
    )
    for name in names:
        try:
            store = WindowStore.open(name)
        except (FileNotFoundError, ValueError) as error:
            print(f"{name:12s} {error}")
            continue
        size = (store.folder / "signals.f32").stat().st_size / 1e9
        print(
            f"{name:12s} {len(store):6d} windows x {len(store.channels)} channels x {store.window}"
            f" at {store.rate:g} Hz  {size:.2f} GB  key {store.key}  built {store.meta['created']}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.windows", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="build window stores (checks CPU and RAM first)")
    p.add_argument("datasets", nargs="+", choices=sorted(SPECS))
    p.add_argument("--limit", type=int, help="first N recordings only (smoke test)")
    p.add_argument("--force", action="store_true", help="run even if the machine is busy")
    p.set_defaults(func=_cmd_build)
    p = sub.add_parser("info", help="list the built stores")
    p.add_argument("datasets", nargs="*")
    p.set_defaults(func=_cmd_info)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
