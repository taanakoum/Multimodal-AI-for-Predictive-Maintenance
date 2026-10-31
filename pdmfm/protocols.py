"""Splits by physical unit (plan, "Cross-validation and evaluation protocol").

Every split is a set of boolean row masks over one dataset's window manifest
(:class:`pdmfm.windows.WindowStore`). Rows outside train, validation and test (for example
target bearings in a source fold) belong to no part.

* Outer folds (5, shared by the source datasets): Nonastreda tools T(2k-1), T(2k) with
  T(2k+1) for inner validation; XJTU-SY bearings Bearing{1,2,3}_k with those of fold k+1 for
  validation; Paderborn's 15 source bearings in 5 stratified triples.
* Gate 1 uses the inner-validation parts only; CWRU's validation is its 3 hp recordings.
* :func:`inner_folds` (Gate 2): within outer fold k, validation rotates over the other four
  folds' test groups, so every non-test unit is validated once per outer fold.
* Held-out targets: KAIST (support from 0 and 2 Nm, test at 4 Nm), Paderborn real damage
  (one bearing per class as support, redrawn per draw), PHM 2010 (one labelled cutter as test).
* :func:`pretraining_mask`: what self-supervised pretraining of fold k may see; no target
  dataset, and none of fold k's test or inner-validation units.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdmfm.sources import PADERBORN_HEALTHY, PADERBORN_REAL

N_FOLDS = 5
NONASTREDA_TOOLS = tuple(f"T{i}" for i in range(1, 11))
XJTU_BEARINGS = tuple(f"Bearing{c}_{i}" for c in (1, 2, 3) for i in range(1, 6))
PHM2010_LABELLED = ("c1", "c4", "c6")
TARGETS = ("kaist", "phm2010")  # Paderborn is split by bearing: source and target parts

# Paderborn source bearings (K001-K003 + 12 artificial) in 5 test triples. Folds 1-3 hold one
# healthy, one outer-race and one inner-race bearing; folds 4-5 two outer and one inner. The
# damage methods (EDM, engraver, drilling) are spread over the folds.
PADERBORN_SOURCE_FOLDS = (
    ("K001", "KA01", "KI01"),
    ("K002", "KA03", "KI03"),
    ("K003", "KA05", "KI05"),
    ("KA06", "KA08", "KI07"),
    ("KA07", "KA09", "KI08"),
)
PADERBORN_TARGET_HEALTHY = PADERBORN_HEALTHY[3:]  # K004-K006
# MATWI's 17 tools in 5 groups, dealt round-robin over C45 (Sets 1-11) then 304 stainless
# (Sets 12-17), so every group mixes both materials.
MATWI_GROUPS = tuple(
    tuple(f"Set{n:02d}" for n in range(1, 18) if (n - 1) % 5 == g) for g in range(5)
)
PADERBORN_OUT_OF_SPACE = ("KB27",)  # combined outer + inner damage: not in the 3-class space


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        if not self.train.shape == self.val.shape == self.test.shape:
            raise ValueError("masks must have the same length")
        overlap = (self.train & self.val) | (self.train & self.test) | (self.val & self.test)
        if overlap.any():
            raise ValueError(f"{self.name}: {int(overlap.sum())} rows in two parts")

    def units(self, manifest: pd.DataFrame, part: str) -> set[str]:
        return set(manifest.loc[getattr(self, part), "unit"].astype(str))

    def assert_unit_disjoint(self, manifest: pd.DataFrame) -> None:
        parts = {p: self.units(manifest, p) for p in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = parts[a] & parts[b]
            if shared:
                raise AssertionError(f"{self.name}: units in {a} and {b}: {sorted(shared)}")


def _units(manifest: pd.DataFrame) -> np.ndarray:
    return manifest["unit"].astype(str).to_numpy()


def unit_split(
    manifest: pd.DataFrame,
    name: str,
    test: Iterable[str],
    val: Iterable[str],
    exclude: Iterable[str] = (),
    train: Iterable[str] | None = None,
) -> Split:
    """Masks by unit: ``train`` is every other unit unless given explicitly."""
    unit = _units(manifest)
    test_mask = np.isin(unit, list(test))
    val_mask = np.isin(unit, list(val))
    if train is None:
        train_mask = ~(test_mask | val_mask | np.isin(unit, list(exclude)))
    else:
        train_mask = np.isin(unit, list(train)) & ~(test_mask | val_mask)
    return Split(name, train_mask, val_mask, test_mask)


def _check_fold(k: int, n: int = N_FOLDS) -> None:
    if not 1 <= k <= n:
        raise ValueError(f"fold must be 1..{n}, got {k}")


# ----------------------------------------------------------------------------- source folds


def nonastreda_fold(manifest: pd.DataFrame, k: int) -> Split:
    """Outer fold k: test T(2k-1) and T(2k); inner validation T(2k+1), wrapping round."""
    _check_fold(k)
    tools = NONASTREDA_TOOLS
    test = (tools[2 * k - 2], tools[2 * k - 1])
    val = (tools[(2 * k) % len(tools)],)
    return unit_split(manifest, f"nonastreda-fold{k}", test, val)


def nonastreda_loto(manifest: pd.DataFrame, i: int) -> Split:
    """Leave-one-tool-out fold i (1..10): test Ti, validation the next tool (stage 7)."""
    _check_fold(i, len(NONASTREDA_TOOLS))
    tools = NONASTREDA_TOOLS
    return unit_split(manifest, f"nonastreda-loto{i}", (tools[i - 1],), (tools[i % len(tools)],))


def xjtu_fold(manifest: pd.DataFrame, k: int) -> Split:
    """Outer fold k: one bearing per operating condition for test, those of fold k+1 for
    inner validation."""
    _check_fold(k)
    nxt = k % N_FOLDS + 1
    test = tuple(f"Bearing{c}_{k}" for c in (1, 2, 3))
    val = tuple(f"Bearing{c}_{nxt}" for c in (1, 2, 3))
    return unit_split(manifest, f"xjtu_sy-fold{k}", test, val)


def matwi_fold(manifest: pd.DataFrame, k: int) -> Split:
    """Outer fold k: MATWI group k for test (3-4 tools), group k+1 for inner validation."""
    _check_fold(k)
    test, val = MATWI_GROUPS[k - 1], MATWI_GROUPS[k % N_FOLDS]
    return unit_split(manifest, f"matwi-fold{k}", test, val)


def paderborn_fold(manifest: pd.DataFrame, k: int) -> Split:
    """Outer fold k over the 15 source bearings; target bearings belong to no part."""
    _check_fold(k)
    source = {b for fold in PADERBORN_SOURCE_FOLDS for b in fold}
    test = PADERBORN_SOURCE_FOLDS[k - 1]
    val = PADERBORN_SOURCE_FOLDS[k % N_FOLDS]
    return unit_split(manifest, f"paderborn-fold{k}", test, val, train=source)


def cwru_gate1(manifest: pd.DataFrame) -> Split:
    """CWRU (pretraining and Gate 1 only): train on 0-2 hp, validate on 3 hp. CWRU's units
    repeat across loads, so this measures load shift, never unit shift."""
    load = manifest["load_hp"].to_numpy()
    none = np.zeros(len(manifest), dtype=bool)
    return Split("cwru-gate1", np.isin(load, (0, 1, 2)), load == 3, none)


SOURCE_FOLDS = {
    "nonastreda": nonastreda_fold,
    "matwi": matwi_fold,
    "xjtu_sy": xjtu_fold,
    "paderborn": paderborn_fold,
}


def inner_folds(dataset: str, manifest: pd.DataFrame, k: int) -> list[Split]:
    """The inner folds of outer fold k: validation on outer fold j's test group (j != k),
    training on the rest of fold k's training and validation units; fold k's test units
    belong to no part. Every non-test unit is validated exactly once."""
    if dataset not in SOURCE_FOLDS:
        raise ValueError(f"no outer folds for {dataset!r}")
    outer = SOURCE_FOLDS[dataset](manifest, k)
    pool = outer.train | outer.val
    none = np.zeros(len(manifest), dtype=bool)
    splits = []
    for j in range(1, N_FOLDS + 1):
        if j == k:
            continue
        group = SOURCE_FOLDS[dataset](manifest, j).test & pool
        splits.append(Split(f"{outer.name}-inner{j}", pool & ~group, group, none))
    return splits


def pretraining_mask(dataset: str, manifest: pd.DataFrame, fold: int | None) -> np.ndarray:
    """Rows that self-supervised pretraining for outer fold ``fold`` may use.

    Target datasets: none. Paderborn: only the fold's training bearings among the 15 source
    bearings. CWRU: everything except its 3 hp validation recordings. ``fold=None`` allows
    every source unit (for smoke tests only: Gate 1 must use per-fold models).
    """
    n = len(manifest)
    if dataset in TARGETS:
        return np.zeros(n, dtype=bool)
    if dataset == "cwru":
        return cwru_gate1(manifest).train
    if dataset == "paderborn" and fold is None:
        source = {b for f in PADERBORN_SOURCE_FOLDS for b in f}
        return np.isin(_units(manifest), list(source))
    if fold is None:
        return np.ones(n, dtype=bool)
    if dataset not in SOURCE_FOLDS:
        raise ValueError(f"no outer folds for {dataset!r}")
    return SOURCE_FOLDS[dataset](manifest, fold).train


# ----------------------------------------------------------------------------- targets


def kaist_target(manifest: pd.DataFrame) -> Split:
    """KAIST: the support pool is 0 and 2 Nm, the test set is every 4 Nm window. Draw the
    support set itself with :func:`few_shot`. The same specimens appear at every load, so
    this is a condition shift, not a unit shift."""
    load = manifest["load_nm"].to_numpy()
    none = np.zeros(len(manifest), dtype=bool)
    return Split("kaist-target", np.isin(load, (0, 2)), none, load == 4)


def paderborn_target(manifest: pd.DataFrame, k: int, draw: int) -> Split:
    """Paderborn real damage: one bearing per class (healthy from K004-K006) gives k support
    windows each; every other target bearing is the test set. Bearings are redrawn per draw."""
    rng = np.random.default_rng(draw)
    unit = _units(manifest)
    labels = manifest["class_key"].astype(str).to_numpy()
    pools = {
        "healthy": [b for b in PADERBORN_TARGET_HEALTHY if b in set(unit)],
        "outer_race": sorted(
            {u for u, c in zip(unit, labels, strict=True) if c == "outer_race"}
            & set(PADERBORN_REAL)
        ),
        "inner_race": sorted(
            {u for u, c in zip(unit, labels, strict=True) if c == "inner_race"}
            & set(PADERBORN_REAL)
        ),
    }
    target = set(PADERBORN_TARGET_HEALTHY) | (set(PADERBORN_REAL) - set(PADERBORN_OUT_OF_SPACE))
    chosen = {c: str(rng.choice(p)) for c, p in pools.items() if p}
    support = np.zeros(len(manifest), dtype=bool)
    for bearing in chosen.values():
        rows = np.flatnonzero(unit == bearing)
        support[rng.choice(rows, size=min(k, len(rows)), replace=False)] = True
    test = np.isin(unit, sorted(target - set(chosen.values())))
    none = np.zeros(len(manifest), dtype=bool)
    return Split(f"paderborn-target-k{k}-draw{draw}", support, none, test)


def phm2010_fold(manifest: pd.DataFrame, test_cutter: str) -> Split:
    """PHM 2010: one labelled cutter is the test set; the other two form the support pool.
    The unlabelled cutters c2, c3 and c5 belong to no part."""
    if test_cutter not in PHM2010_LABELLED:
        raise ValueError(f"test cutter must be one of {PHM2010_LABELLED}")
    others = [c for c in PHM2010_LABELLED if c != test_cutter]
    return unit_split(manifest, f"phm2010-{test_cutter}", (test_cutter,), (), train=others)


def few_shot(
    manifest: pd.DataFrame, pool: np.ndarray, k: int, seed: int, by: str = "class_key"
) -> np.ndarray:
    """k rows per value of ``by`` drawn from ``pool`` (all of them if fewer exist)."""
    rng = np.random.default_rng(seed)
    values = manifest[by].astype(str).to_numpy()
    chosen = np.zeros(len(manifest), dtype=bool)
    for value in sorted(set(values[pool])):
        rows = np.flatnonzero(pool & (values == value))
        chosen[rng.choice(rows, size=min(k, len(rows)), replace=False)] = True
    return chosen


def subsample(manifest: pd.DataFrame, fraction: float, seed: int = 0) -> np.ndarray:
    """A unit- and class-stratified fraction of the rows (at least one per group). The same
    seed gives the same rows to every encoder, so subsampled caches stay comparable."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    keys = manifest["unit"].astype(str) + "|" + manifest["class_key"].astype(str)
    chosen = np.zeros(len(manifest), dtype=bool)
    for _, rows in pd.Series(np.arange(len(manifest))).groupby(keys.to_numpy()):
        rows = rows.to_numpy()
        size = max(1, int(np.ceil(fraction * len(rows))))
        chosen[rng.choice(rows, size=size, replace=False)] = True
    return chosen
