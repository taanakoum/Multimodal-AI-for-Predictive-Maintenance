"""Hyperparameter search by nested cross-validation (pdmfm.tune)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import build_planb_root

from pdmfm import protocols, tune
from pdmfm.train import TrainConfig
from pdmfm.windows import WindowStore


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tune")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root, datasets=("xjtu_sy",))
        yield root


def search(root, grid, run_id="t", log=quiet, folds=(1, 2)):
    return tune.search(
        "xjtu_sy", "stats", ("F",), run_id, grid, TrainConfig(epochs=2, batch_size=64),
        list(folds), out_root=root / "runs", window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features", log=log,
    )  # fmt: skip


def test_parse_grid():
    assert tune.parse_grid(["lr=3e-4,1e-3", "d=64"]) == {"lr": [3e-4, 1e-3], "d": [64]}
    with pytest.raises(ValueError):
        tune.parse_grid(["batch=1"])


def test_every_configuration_is_scored_on_every_inner_split_and_the_search_resumes(root):
    report = search(root, {"lr": [1e-3, 3e-3], "d": [16, 32]})
    results = pd.read_csv(root / "runs" / "t" / "results.csv")
    assert len(results) == 4 * 2 * 4  # configurations x outer folds x inner splits
    assert set(results["fold"]) == {1, 2} and results["split"].nunique() == 8
    assert set(report["chosen"]) == {"lr", "d"}
    best = json.loads((root / "runs" / "t" / "best.json").read_text())
    assert best["flags"].startswith("--") and len(best["ranking"]) == 4
    assert set(best["per_fold"]) == {"1", "2"}
    for entry in best["per_fold"].values():
        assert set(entry["overrides"]) == {"lr", "d"} and entry["splits"] == 4
    assert 0.0 <= best["per_fold_agreement"] <= 1.0
    messages = []
    search(root, {"lr": [1e-3, 3e-3], "d": [16, 32]}, log=messages.append)
    assert not any("validation" in m and "chosen" not in m for m in messages)  # nothing refitted


def test_the_search_never_reads_an_outer_test_unit(root, monkeypatch):
    seen = []
    original = tune.fit

    def recording(inputs, train_rows, val_rows, cfg, seed, log=print):
        seen.append((train_rows, val_rows))
        return original(inputs, train_rows, val_rows, cfg, seed, log)

    monkeypatch.setattr(tune, "fit", recording)
    search(root, {"d": [16]}, run_id="spy", folds=(3,))
    manifest = WindowStore.open("xjtu_sy", root / "interim" / "windows").manifest
    outer_test = protocols.xjtu_fold(manifest, 3).test
    assert len(seen) == 4
    validated = set()
    for train_rows, val_rows in seen:
        assert not outer_test[train_rows].any() and not outer_test[val_rows].any()
        validated |= set(manifest["unit"].to_numpy()[val_rows])
    # Every non-test bearing of fold 3 is validated once across its inner splits.
    non_test = set(manifest.loc[~outer_test, "unit"])
    assert validated == non_test


def test_each_fold_chooses_on_its_own_splits(root, monkeypatch):
    """Width 16 wins every inner split of fold 1, width 32 every one of fold 2: each fold
    keeps its own choice; the pooled choice is a near-tie, so the cheaper width."""
    original = tune.fit

    def rigged(inputs, train_rows, val_rows, cfg, seed, log=print):
        fitted = original(inputs, train_rows, val_rows, cfg, seed, log)
        manifest = inputs.manifest
        in_fold_2 = protocols.xjtu_fold(manifest, 2).train | protocols.xjtu_fold(manifest, 2).val
        fold_2 = (
            bool(in_fold_2[train_rows].all())
            and not protocols.xjtu_fold(manifest, 2).test[train_rows].any()
        )
        wins = (cfg.d == 32) == fold_2
        for h in fitted.history:
            h["val_score"] = 0.6 if wins else 0.5
        return fitted

    monkeypatch.setattr(tune, "fit", rigged)
    report = search(root, {"d": [16, 32]}, run_id="own")
    assert report["per_fold"]["1"]["overrides"] == {"d": 16}
    assert report["per_fold"]["2"]["overrides"] == {"d": 32}
    assert report["per_fold_agreement"] == pytest.approx(0.5)


def test_near_ties_go_to_the_cheaper_configuration(root, monkeypatch):
    """Two configurations with the same validation scores: the smaller width wins."""
    original = tune.fit

    def same_score(inputs, train_rows, val_rows, cfg, seed, log=print):
        fitted = original(inputs, train_rows, val_rows, cfg, seed, log)
        for h in fitted.history:
            h["val_score"] = 0.5
        return fitted

    monkeypatch.setattr(tune, "fit", same_score)
    report = search(root, {"d": [32, 16]}, run_id="ties")
    assert report["chosen"] == {"d": 16}
    assert all(v["overrides"] == {"d": 16} for v in report["per_fold"].values())


def test_results_of_the_single_split_search_are_refused(root):
    old = root / "runs" / "old"
    old.mkdir(parents=True)
    pd.DataFrame([{"config": "{}", "fold": 1, "val_score": 0.5}]).to_csv(
        old / "results.csv", index=False
    )
    with pytest.raises(ValueError, match="single-split"):
        search(root, {"d": [16]}, run_id="old")


def test_choose_ranks_by_mean_and_breaks_near_ties_by_cost():
    rows = []
    for split, (a, b) in enumerate([(0.60, 0.40), (0.62, 0.41), (0.58, 0.39), (0.61, 0.40)]):
        rows.append({"config": json.dumps({"d": 64}), "split": f"s{split}", "val_score": a})
        rows.append({"config": json.dumps({"d": 16}), "split": f"s{split}", "val_score": b})
    chosen, ranking = tune.choose(pd.DataFrame(rows))
    assert json.loads(chosen) == {"d": 64}  # a clear lead is not a near-tie
    assert np.isclose(ranking["mean"].max(), 0.6025)
