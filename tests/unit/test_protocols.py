"""Unit-level splits (pdmfm.protocols)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdmfm import protocols
from pdmfm.datasets.paderborn import DAMAGE
from pdmfm.sources import PADERBORN_ARTIFICIAL, PADERBORN_HEALTHY, PADERBORN_REAL


def manifest(units, windows=4, **columns):
    rows = [{"unit": u, "window": w} for u in units for w in range(windows)]
    frame = pd.DataFrame(rows)
    for name, values in columns.items():
        frame[name] = [values[u] for u in frame["unit"]]
    return frame


@pytest.fixture
def nonastreda():
    tools = protocols.NONASTREDA_TOOLS
    return manifest(tools, class_key={t: "sharp" for t in tools})


@pytest.fixture
def paderborn():
    bearings = list(DAMAGE)
    classes = {b: DAMAGE[b][0] for b in bearings}
    return manifest(bearings, windows=6, class_key=classes)


@pytest.mark.parametrize("k", range(1, 6))
def test_nonastreda_folds(nonastreda, k):
    split = protocols.nonastreda_fold(nonastreda, k)
    split.assert_unit_disjoint(nonastreda)
    assert split.units(nonastreda, "test") == {f"T{2 * k - 1}", f"T{2 * k}"}
    assert split.units(nonastreda, "val") == {f"T{(2 * k) % 10 + 1}"}
    assert len(split.units(nonastreda, "train")) == 7


def test_every_tool_is_tested_exactly_once(nonastreda):
    tested = [
        u
        for k in range(1, 6)
        for u in protocols.nonastreda_fold(nonastreda, k).units(nonastreda, "test")
    ]
    assert sorted(tested) == sorted(protocols.NONASTREDA_TOOLS)


def test_leave_one_tool_out_wraps_round(nonastreda):
    split = protocols.nonastreda_loto(nonastreda, 10)
    assert split.units(nonastreda, "test") == {"T10"} and split.units(nonastreda, "val") == {"T1"}
    assert len(split.units(nonastreda, "train")) == 8


@pytest.mark.parametrize("k", range(1, 6))
def test_xjtu_folds_hold_one_bearing_per_condition(k):
    frame = manifest(protocols.XJTU_BEARINGS)
    split = protocols.xjtu_fold(frame, k)
    split.assert_unit_disjoint(frame)
    test = split.units(frame, "test")
    assert {u.split("_")[0] for u in test} == {"Bearing1", "Bearing2", "Bearing3"}
    assert {u.split("_")[1] for u in split.units(frame, "val")} == {str(k % 5 + 1)}


def test_paderborn_folds_cover_the_15_source_bearings_once(paderborn):
    source = set(PADERBORN_HEALTHY[:3]) | set(PADERBORN_ARTIFICIAL)
    tested = []
    for k in range(1, 6):
        split = protocols.paderborn_fold(paderborn, k)
        split.assert_unit_disjoint(paderborn)
        test = split.units(paderborn, "test")
        classes = {DAMAGE[b][0] for b in test}
        assert {"outer_race", "inner_race"} <= classes  # stratified: both damage locations
        everything = split.units(paderborn, "train") | test | split.units(paderborn, "val")
        assert everything == source  # target bearings never enter a source fold
        tested += list(test)
    assert sorted(tested) == sorted(source)


def test_cwru_gate1_splits_by_load():
    frame = pd.DataFrame({"unit": ["a", "a", "a", "a"], "load_hp": [0, 1, 2, 3]})
    split = protocols.cwru_gate1(frame)
    assert split.train.tolist() == [True, True, True, False]
    assert split.val.tolist() == [False, False, False, True] and not split.test.any()


@pytest.mark.parametrize("k", range(1, 6))
def test_pretraining_never_sees_test_or_validation_units(paderborn, nonastreda, k):
    for dataset, frame in (("paderborn", paderborn), ("nonastreda", nonastreda)):
        split = protocols.SOURCE_FOLDS[dataset](frame, k)
        allowed = protocols.pretraining_mask(dataset, frame, k)
        assert not (allowed & (split.test | split.val)).any()
        assert set(frame.loc[allowed, "unit"]) == split.units(frame, "train")
    assert not protocols.pretraining_mask("kaist", manifest(["a"]), k).any()
    assert not protocols.pretraining_mask("phm2010", manifest(["c1"]), k).any()


def test_pretraining_without_fold_excludes_targets_and_cwru_validation(paderborn):
    allowed = protocols.pretraining_mask("paderborn", paderborn, None)
    assert not set(paderborn.loc[allowed, "unit"]) & (
        set(PADERBORN_REAL) | set(PADERBORN_HEALTHY[3:])
    )
    cwru = pd.DataFrame({"unit": ["a"] * 4, "load_hp": [0, 1, 2, 3]})
    assert protocols.pretraining_mask("cwru", cwru, 2).tolist() == [True, True, True, False]


def test_kaist_target_support_pool_and_test_set():
    frame = pd.DataFrame(
        {"unit": list("aaabbb"), "load_nm": [0, 2, 4, 0, 2, 4], "class_key": list("xxxyyy")}
    )
    split = protocols.kaist_target(frame)
    assert split.train.tolist() == [True, True, False, True, True, False]
    assert split.test.tolist() == [False, False, True, False, False, True]


def test_few_shot_draws_k_per_class_from_the_pool_only():
    frame = pd.DataFrame({"class_key": ["a"] * 10 + ["b"] * 10 + ["c"] * 2})
    pool = np.ones(len(frame), dtype=bool)
    pool[:5] = False
    chosen = protocols.few_shot(frame, pool, k=3, seed=1)
    counts = frame.loc[chosen, "class_key"].value_counts().to_dict()
    assert counts == {"a": 3, "b": 3, "c": 2} and not chosen[:5].any()
    assert (chosen == protocols.few_shot(frame, pool, k=3, seed=1)).all()  # deterministic
    assert (chosen != protocols.few_shot(frame, pool, k=3, seed=2)).any()  # and per draw


def test_paderborn_target_draws_one_bearing_per_class(paderborn):
    split = protocols.paderborn_target(paderborn, k=5, draw=0)
    support = set(paderborn.loc[split.train, "unit"])
    assert len(support) == 3 and int(split.train.sum()) == 15
    classes = {DAMAGE[b][0] for b in support}
    assert classes == {"healthy", "outer_race", "inner_race"}
    assert support & set(PADERBORN_HEALTHY) <= set(PADERBORN_HEALTHY[3:])
    test = set(paderborn.loc[split.test, "unit"])
    assert not test & support and "KB27" not in test
    assert not test & (set(PADERBORN_HEALTHY[:3]) | set(PADERBORN_ARTIFICIAL))
    draws = [protocols.paderborn_target(paderborn, k=5, draw=d) for d in range(5)]
    supports = {frozenset(paderborn.loc[d.train, "unit"]) for d in draws}
    assert len(supports) > 1  # bearings are redrawn per draw
    again = protocols.paderborn_target(paderborn, k=5, draw=0)
    assert (again.train == split.train).all()  # and each draw is reproducible


def test_phm2010_fold():
    frame = manifest(["c1", "c2", "c3", "c4", "c5", "c6"])
    split = protocols.phm2010_fold(frame, "c4")
    assert split.units(frame, "test") == {"c4"} and split.units(frame, "train") == {"c1", "c6"}
    with pytest.raises(ValueError):
        protocols.phm2010_fold(frame, "c2")


def test_subsample_is_stratified_and_reproducible():
    frame = manifest(
        [f"u{i}" for i in range(5)], windows=40, class_key={f"u{i}": "x" for i in range(5)}
    )
    chosen = protocols.subsample(frame, 0.1, seed=0)
    assert frame.loc[chosen].groupby("unit").size().tolist() == [4] * 5
    assert (chosen == protocols.subsample(frame, 0.1, seed=0)).all()
    with pytest.raises(ValueError):
        protocols.subsample(frame, 0.0)


def test_split_rejects_overlapping_masks():
    mask = np.array([True, False])
    with pytest.raises(ValueError, match="two parts"):
        protocols.Split("bad", mask, mask, ~mask)


def test_fold_numbers_are_checked(nonastreda):
    with pytest.raises(ValueError):
        protocols.nonastreda_fold(nonastreda, 6)
    with pytest.raises(ValueError):
        protocols.pretraining_mask("najafi_thermal", manifest(["m"]), 1)


@pytest.fixture
def matwi():
    tools = [t for group in protocols.MATWI_GROUPS for t in group]
    return manifest(tools, class_key={t: "sharp" for t in tools})


@pytest.mark.parametrize("k", range(1, 6))
def test_matwi_folds_mix_both_materials(matwi, k):
    split = protocols.matwi_fold(matwi, k)
    split.assert_unit_disjoint(matwi)
    test = split.units(matwi, "test")
    assert 3 <= len(test) <= 4 and len(split.units(matwi, "val")) >= 3
    numbers = {int(t.removeprefix("Set")) for t in test}
    assert any(n <= 11 for n in numbers) and any(n >= 12 for n in numbers)  # C45 and 304


def test_every_matwi_tool_is_tested_once(matwi):
    tested = [t for k in range(1, 6) for t in protocols.matwi_fold(matwi, k).units(matwi, "test")]
    assert sorted(tested) == sorted(f"Set{n:02d}" for n in range(1, 18))
