"""Per-modality token sets from the caches (plan, "Model architecture": token set).

For one dataset, :func:`assemble` gathers everything a supervised head needs, row-aligned
with the window store's manifest:

* **Sensor modalities**, one per channel family (force, vibration, current): every
  channel's cached tokens of the chosen arms (T: 6 or 4 tokens, F and E: 1 each), with a
  channel index and a token role per token.
* **Level tokens** per channel: log RMS, log peak, kurtosis and crest factor (no Fz token on
  Nonastreda: Fz drifts by recording block, EDA F10). ``levels`` chooses the descriptors
  minus the unit's healthy baseline (``relative``, the plan's step 5 and the default), the
  absolute descriptors, or ``both``. ``force_mean`` adds the window mean of force channels
  (the static cutting force). Both options are ablations: on the Nonastreda
  inner-validation tools, absolute levels made no difference and the force mean lowered the
  force expert (PROJECT_LOG.md, week 5), although Fx and Fy means follow flank wear across
  passes (Spearman ±0.67).
  Baselines: Nonastreda, the tool's first pass; XJTU-SY, the bearing's first 10
  snapshots; CWRU, the normal recording at the same load (Gate 1 and pretraining only);
  Paderborn, the healthy *training* bearings at the same setting, so it depends on the fold
  (``train_units``). Each uses only the unit's healthy start or training units, so it is
  available at prediction time. KAIST's baseline is its support set's normal windows
  (stage 8), not given here.
* **Slow channels** (Paderborn temperature): one token per channel from its per-window mean
  and slope minus the same baseline, embedded by a small MLP (plan, "Slow token"); always
  relative, because absolute temperature identifies the recording session (EDA F3).
* **Photos** (Nonastreda): the DINOv2 tokens of the previous pass's tool, chip and workpiece
  photos attached by the as-of join; a pass without photos is a missing modality (mask
  false), never zeros. Their age is counted in passes (the seconds between passes are not
  recorded). :class:`PhotoViews` also holds, for training augmentation and the staleness
  test, the photos of the 2nd and 3rd previous passes and augmented DINOv2 copies of each
  photo (``python -m pdmfm.features images --views 4``).
* **Targets:** class index in the dataset's label space (Qwen prototypes from the text cache)
  and the regression target (wear in µm, normalized RUL), NaN where absent.

Context text is not a modality here: it never names the state (by construction), so an
opinion from it would be vacuous; it serves the leakage check and later fusion stages.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm import arms
from pdmfm import preprocess as pp
from pdmfm import texts as text_specs
from pdmfm.features import FeatureStore
from pdmfm.paths import interim_dir
from pdmfm.windows import WindowStore

ROLES = ("T", "F", "E", "S", "level", "tool", "chip", "workpiece", "thermal", "slow")
ROLE = {name: i for i, name in enumerate(ROLES)}
PHOTO_KINDS = ("tool", "chip", "workpiece")
MAX_PHOTO_LAG = 3  # photos of up to the 3rd previous pass
NO_LEVEL = {"nonastreda": ("fz",)}  # Fz drifts by block (EDA F10)
DC_FAMILIES = ("force",)  # channels whose window mean ``force_mean`` adds
LEVEL_MODES = ("relative", "both", "absolute")
FOLD_BASELINE = ("paderborn",)  # level baseline from the fold's training units
REGRESSION_TARGET = {"nonastreda": "wear_um", "xjtu_sy": "rul", "phm2010": "wear_um"}


@dataclass
class Modality:
    name: str
    tokens: np.ndarray  # (N, T, D) float32
    roles: np.ndarray  # (T,) role index
    channels: np.ndarray  # (T,) channel index within the dataset
    present: np.ndarray  # (N,) bool: the modality exists for the row
    token_present: np.ndarray | None = None  # (N, T) bool, None: all tokens present
    levels: np.ndarray | None = None  # (N, L, F) level descriptors (see ``level_names``)
    level_channels: np.ndarray | None = None  # (L,) channel index of each level token
    age: np.ndarray | None = None  # (N, T) passes since capture (photos)
    descriptor: bool = False  # tokens are small descriptor vectors (slow channels)
    views: PhotoViews | None = None  # alternative photo token sets (photos only)

    @property
    def dim(self) -> int:
        return int(self.tokens.shape[-1])


@dataclass
class PhotoViews:
    """Photo tokens of other passes and augmented copies, looked up per row on demand.

    ``banks[j]`` is (V, n_photos, T_j, D) for photo kind j: view 0 is the photo itself, views
    1.. are augmented DINOv2 copies. ``index`` is (MAX_PHOTO_LAG, N, kinds): the bank row of
    the photo taken after the (lag)-th previous pass, or -1."""

    banks: list[np.ndarray]
    index: np.ndarray

    @property
    def n_views(self) -> int:
        return int(self.banks[0].shape[0])

    def available_lags(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows), MAX_PHOTO_LAG) bool: some photo of that lag exists."""
        return np.transpose((self.index[:, rows] >= 0).any(axis=2))

    def tokens(self, rows: np.ndarray, lags: np.ndarray, views: np.ndarray):
        """(tokens (B, T, D), token_present (B, T), age (B, T)) for each row's lag (1-based)
        and view."""
        blocks, present, ages = [], [], []
        for j, bank in enumerate(self.banks):
            photo = self.index[lags - 1, rows, j]
            have = photo >= 0
            block = np.zeros((len(rows),) + bank.shape[2:], dtype=np.float32)
            block[have] = bank[views[have], photo[have]]
            blocks.append(block)
            present.append(np.repeat(have[:, None], bank.shape[2], axis=1))
            ages.append(np.repeat(np.where(have, lags, 0)[:, None], bank.shape[2], axis=1))
        return (
            np.concatenate(blocks, axis=1),
            np.concatenate(present, axis=1),
            np.concatenate(ages, axis=1).astype(np.float32),
        )


@dataclass
class TaskInputs:
    dataset: str
    manifest: pd.DataFrame
    modalities: dict[str, Modality]
    classes: list[str] = field(default_factory=list)
    prototypes: np.ndarray | None = None  # (K, 1024) L2-normalized
    y_class: np.ndarray | None = None  # (N,) int, -1 unlabelled
    y_reg: np.ndarray | None = None  # (N,) float, NaN unlabelled
    regression_target: str | None = None
    description: dict = field(default_factory=dict)


def baseline_groups(
    dataset: str, manifest: pd.DataFrame, train_units=None
) -> tuple[np.ndarray, np.ndarray]:
    """(group key per row, reference mask): a row's levels are taken relative to the median
    of the reference rows of its group. ``train_units`` is required where the reference is
    made of training units (``FOLD_BASELINE``)."""
    if dataset == "nonastreda":  # the tool's first pass
        start = manifest.groupby("unit")["meta_t_start"].transform("min").to_numpy()
        first = manifest["meta_t_start"].to_numpy() == start
        return manifest["unit"].astype(str).to_numpy(), np.asarray(first)
    if dataset == "xjtu_sy":  # the bearing's first 10 snapshots
        early = manifest["meta_minute"].to_numpy() <= 10
        return manifest["unit"].astype(str).to_numpy(), np.asarray(early)
    if dataset == "cwru":  # the normal recording at the same load (Gate 1 and pretraining)
        normal = manifest["class_key"] == "normal"
        return manifest["load_hp"].astype(str).to_numpy(), normal.to_numpy()
    if dataset == "paderborn":  # healthy training bearings at the same operating setting
        if train_units is None:
            raise ValueError("paderborn's level baseline uses the fold's training bearings")
        healthy = manifest["class_key"] == "healthy"
        reference = healthy & manifest["unit"].astype(str).isin(list(train_units))
        return manifest["setting"].astype(str).to_numpy(), reference.to_numpy()
    # KAIST: the plan's baseline is the normal windows of the support set (stage 8); the
    # normal recording at the test load would be a test window.
    raise NotImplementedError(f"no level baseline rule for {dataset} yet")


def _relative(values: np.ndarray, keys: np.ndarray, reference: np.ndarray, dataset: str):
    out = np.empty_like(values, dtype=np.float32)
    for key in np.unique(keys):
        members = keys == key
        rows = np.flatnonzero(members & reference)
        if not len(rows):
            raise ValueError(f"{dataset}: no reference windows for {key}")
        with warnings.catch_warnings():  # a channel missing in every reference row stays NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            out[members] = values[members] - np.nanmedian(values[rows], axis=0)
    return out


_MEANS: dict[str, np.ndarray] = {}


def window_means(store: WindowStore) -> np.ndarray:
    """(N, C) mean of every window and channel (the stored model input keeps DC), computed
    once per store and process."""
    if store.key not in _MEANS:
        _MEANS[store.key] = np.concatenate(
            [
                np.asarray(store.signals[i : i + 512], dtype=np.float64).mean(axis=2)
                for i in range(0, len(store), 512)
            ]
        ).astype(np.float32)
    return _MEANS[store.key]


def absolute_levels(store: WindowStore, with_mean: bool = False) -> np.ndarray:
    """(N, C, 4) stored level descriptors; (N, C, 5) with the window mean."""
    if not with_mean:
        return store.levels
    return np.concatenate([store.levels, window_means(store)[:, :, None]], axis=2)


def relative_levels(store: WindowStore, train_units=None, with_mean: bool = False) -> np.ndarray:
    """Level descriptors minus the median of the row's reference windows."""
    keys, reference = baseline_groups(store.dataset, store.manifest, train_units)
    return _relative(absolute_levels(store, with_mean), keys, reference, store.dataset)


def level_names(family: str, mode: str, force_mean: bool = False) -> list[str]:
    """Names of the level features of one channel family, in order."""
    with_mean = force_mean and family in DC_FAMILIES
    base = list(pp.LEVEL_NAMES) + (["mean"] if with_mean else [])
    if mode == "both":
        return [f"absolute_{n}" for n in base] + [f"relative_{n}" for n in base]
    return [f"{mode}_{n}" for n in base]


def relative_slow(store: WindowStore, train_units=None) -> np.ndarray:
    """(N, S, 2) slow-channel mean and slope minus the median of the reference windows;
    NaN where the channel is missing."""
    keys, reference = baseline_groups(store.dataset, store.manifest, train_units)
    return _relative(store.slow, keys, reference, store.dataset)


def _sensor_modalities(
    store: WindowStore,
    encoder: str,
    arm_list,
    feature_store: FeatureStore,
    train_units=None,
    levels: str = "relative",
    force_mean: bool = False,
):
    if levels not in LEVEL_MODES:
        raise ValueError(f"levels must be one of {LEVEL_MODES}, not {levels!r}")
    families = sorted(set(store.families), key=store.families.index)
    with_mean = force_mean and any(f in DC_FAMILIES for f in families)
    forms = []
    if levels in ("both", "absolute"):
        forms.append(absolute_levels(store, with_mean))
    if levels in ("both", "relative"):
        forms.append(relative_levels(store, train_units, with_mean))
    skip = set(NO_LEVEL.get(store.dataset, ()))
    modalities = {}
    done = np.ones(len(store), dtype=bool)
    for family in families:
        blocks, roles, channels = [], [], []
        for arm in arm_list:
            if arm in arms.unavailable(store):
                continue
            arm_ch = arms.arm_channels(store, arm)
            members = [j for j, c in enumerate(arm_ch) if store.families[c] == family]
            if not members:
                continue
            cached = feature_store.find(store, encoder, arm)
            done &= cached.done
            block = np.asarray(cached.features[:, members], dtype=np.float32)  # (N, c, K, D)
            n, c, k, d = block.shape
            blocks.append(block.reshape(n, c * k, d))
            roles += [ROLE[arm]] * (c * k)
            channels += [arm_ch[j] for j in members for _ in range(k)]
        if not blocks:
            continue
        level_ch = [
            i for i, f in enumerate(store.families) if f == family and store.channels[i] not in skip
        ]
        n_features = 5 if with_mean and family in DC_FAMILIES else 4
        family_levels = (
            np.concatenate([form[:, level_ch, :n_features] for form in forms], axis=2)
            if level_ch
            else None
        )
        modalities[family] = Modality(
            name=family,
            tokens=np.concatenate(blocks, axis=1),
            roles=np.asarray(roles),
            channels=np.asarray(channels),
            present=np.ones(len(store), dtype=bool),
            levels=family_levels,
            level_channels=np.asarray(level_ch) if level_ch else None,
        )
    return modalities, done


def _slow_modality(store: WindowStore, train_units=None) -> Modality:
    """One token per slow channel; channel indices follow the high-rate channels."""
    values = relative_slow(store, train_units)  # (N, S, 2)
    token_present = np.asarray(~np.any(np.isnan(values), axis=2))
    n_slow = values.shape[1]
    return Modality(
        name="slow",
        tokens=np.nan_to_num(values).astype(np.float32),
        roles=np.full(n_slow, ROLE["slow"]),
        channels=len(store.channels) + np.arange(n_slow),
        present=np.asarray(np.any(token_present, axis=1)),
        token_present=token_present,
        descriptor=True,
    )


def _image_cache(root: Path, dataset: str, kind: str) -> tuple[list[str], np.ndarray]:
    folders = sorted(
        (root / "images").glob(f"{dataset}-{kind}-dinov2-*/meta.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not folders:
        raise FileNotFoundError(
            f"no DINOv2 {kind} photo cache for {dataset} (python -m pdmfm.features images)"
        )
    folder = folders[-1].parent
    paths = pd.read_csv(folder / "paths.csv")["path"].astype(str).tolist()
    return paths, np.load(folder / "features.npy", mmap_mode="r")


def _view_cache(root: Path, dataset: str, kind: str, paths: list[str]) -> np.ndarray | None:
    """Augmented copies (V, n, T, D) of the photo cache, if extracted for the same photos."""
    folders = sorted(
        (root / "images").glob(f"{dataset}-{kind}-dinov2views-*/meta.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not folders:
        return None
    folder = folders[-1].parent
    if pd.read_csv(folder / "paths.csv")["path"].astype(str).tolist() != paths:
        raise ValueError(f"{folder.name} was extracted for other photos; extract the views again")
    return np.load(folder / "features.npy", mmap_mode="r")


def lagged_photo_index(manifest: pd.DataFrame, kind: str, index: dict[str, int]) -> np.ndarray:
    """(MAX_PHOTO_LAG, N) cache row of the photo taken after the k-th previous pass, or -1.

    A pass's windows carry the previous pass's photo (the as-of join), so the photo k passes
    back is the one carried by the pass k - 1 positions earlier on the same tool."""
    out = np.full((MAX_PHOTO_LAG, len(manifest)), -1, dtype=np.int64)
    attached = manifest[f"image_{kind}"].astype(str).to_numpy()
    by_bag = manifest.groupby("bag", sort=False)
    first = by_bag.head(1)
    for _, passes in first.sort_values(["unit", "meta_t_start"]).groupby("unit", sort=False):
        bags = passes["bag"].tolist()
        photos = [index.get(str(attached[i]), -1) for i in passes.index]
        for position, bag in enumerate(bags):
            rows = by_bag.indices[bag]
            for lag in range(1, MAX_PHOTO_LAG + 1):
                source = position - (lag - 1)
                if source >= 0:
                    out[lag - 1, rows] = photos[source]
    return out


def _photo_modality(store: WindowStore, root: Path) -> Modality:
    manifest = store.manifest
    banks, lag_index, roles = [], [], []
    for kind in PHOTO_KINDS:
        paths, features = _image_cache(root, store.dataset, kind)
        index = {p: i for i, p in enumerate(paths)}
        wanted = manifest[f"image_{kind}"].astype(str).to_numpy()
        unknown = np.array([p != "" and p not in index for p in wanted])
        if unknown.any():
            raise ValueError(
                f"{int(unknown.sum())} attached {kind} photos are not in the photo cache"
            )
        views = _view_cache(root, store.dataset, kind, paths)
        bank = np.asarray(features, dtype=np.float32)[None]
        if views is not None:
            bank = np.concatenate([bank, np.asarray(views, dtype=np.float32)], axis=0)
        banks.append(bank)
        lag_index.append(lagged_photo_index(manifest, kind, index))
        roles += [ROLE[kind]] * features.shape[1]
    photo_views = PhotoViews(banks=banks, index=np.stack(lag_index, axis=2))
    rows = np.arange(len(manifest))
    ones = np.ones(len(manifest), dtype=np.int64)
    tokens, token_present, age = photo_views.tokens(rows, ones, np.zeros_like(ones))
    return Modality(
        name="photo",
        tokens=tokens,
        roles=np.asarray(roles),
        channels=np.zeros(len(roles), dtype=int),
        present=np.asarray(np.any(token_present, axis=1)),
        token_present=token_present,
        age=age,
        views=photo_views,
    )


def _prototypes(dataset: str, root: Path) -> tuple[list[str], np.ndarray]:
    files = sorted(
        (root / "texts").glob(f"{dataset}-qwen3_embedding-*.npz"), key=lambda p: p.stat().st_mtime
    )
    if not files:
        raise FileNotFoundError(
            f"no text prototypes for {dataset} (python -m pdmfm.features texts)"
        )
    data = np.load(files[-1])
    classes = [str(c) for c in data["classes"]]
    expected = list(text_specs.load(dataset).descriptions)
    if classes != expected:
        raise ValueError(f"{dataset}: text cache classes {classes} differ from the YAML {expected}")
    return classes, np.asarray(data["prototypes"], dtype=np.float32)


def assemble(
    dataset: str,
    encoder: str,
    arm_list: tuple[str, ...],
    photos: bool = True,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    train_units=None,
    levels: str = "relative",
    force_mean: bool = False,
) -> TaskInputs:
    """Every modality, label and target of ``dataset`` for one sensor encoder and its arms.
    Rows whose sensor features are not cached are marked absent in every modality.
    ``train_units`` (the fold's training units) is required for ``FOLD_BASELINE`` datasets;
    ``levels`` is ``relative`` (default), ``both`` or ``absolute``; ``force_mean`` adds the
    window mean of force channels to their level tokens."""
    store = WindowStore.open(dataset, window_root)
    root = Path(feature_root) if feature_root else interim_dir("features")
    if dataset not in FOLD_BASELINE:
        train_units = None  # the baseline does not depend on the fold
    modalities, done = _sensor_modalities(
        store, encoder, arm_list, FeatureStore(root), train_units, levels, force_mean
    )
    if not modalities:
        raise ValueError(f"{dataset}: no {encoder} features for arms {arm_list}")
    if store.slow_channels:
        modalities["slow"] = _slow_modality(store, train_units)
    if photos and store.spec.image_kinds:
        modalities["photo"] = _photo_modality(store, root)
    for modality in modalities.values():
        modality.present &= done
    manifest = store.manifest
    inputs = TaskInputs(dataset=dataset, manifest=manifest, modalities=modalities)
    label_space = text_specs.load(dataset).label_space
    if label_space:
        classes, prototypes = _prototypes(dataset, root)
        index = {c: i for i, c in enumerate(classes)}
        inputs.classes, inputs.prototypes = classes, prototypes
        inputs.y_class = np.array([index.get(c, -1) for c in manifest["class_key"].astype(str)])
    target = REGRESSION_TARGET.get(dataset)
    if target and target in manifest:
        inputs.regression_target = target
        inputs.y_reg = manifest[target].to_numpy(dtype=float)
    inputs.description = {
        "dataset": dataset,
        "encoder": encoder,
        "arms": list(arm_list),
        "window_store": store.key,
        "levels": levels,
        "modalities": {
            m.name: {
                "tokens": int(m.tokens.shape[1]),
                "dim": m.dim,
                "levels": 0 if m.levels is None else int(m.levels.shape[1]),
                "level_features": (
                    [] if m.levels is None else level_names(m.name, levels, force_mean)
                ),
            }
            for m in modalities.values()
        },
        "classes": inputs.classes,
        "regression_target": inputs.regression_target,
        "baseline_units": None if train_units is None else sorted(map(str, train_units)),
    }
    json.dumps(inputs.description)  # must stay JSON-serializable: it goes into every result
    return inputs
