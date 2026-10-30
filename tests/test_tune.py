"""Hyperparameter search on inner-validation units (pdmfm.tune)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from synthetic import build_planb_root

from pdmfm import tune
from pdmfm.train import TrainConfig


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tune")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root, datasets=("xjtu_sy",))
        yield root


def search(root, grid, run_id="t"):
    return tune.search(
        "xjtu_sy", "stats", ("F",), run_id, grid, TrainConfig(epochs=2, batch_size=64), [1, 2],
        out_root=root / "runs", window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features", log=quiet,
    )  # fmt: skip


def test_parse_grid():
    assert tune.parse_grid(["lr=3e-4,1e-3", "d=64"]) == {"lr": [3e-4, 1e-3], "d": [64]}
    with pytest.raises(ValueError):
        tune.parse_grid(["batch=1"])


def test_search_scores_every_configuration_on_every_fold_and_resumes(root):
    report = search(root, {"lr": [1e-3, 3e-3], "d": [16, 32]})
    results = pd.read_csv(root / "runs" / "t" / "results.csv")
    assert len(results) == 4 * 2 and set(results["fold"]) == {1, 2}
    assert set(report["chosen"]) == {"lr", "d"}
    best = json.loads((root / "runs" / "t" / "best.json").read_text())
    assert best["flags"].startswith("--") and len(best["ranking"]) == 4
    messages = []
    tune.search(
        "xjtu_sy", "stats", ("F",), "t", {"lr": [1e-3, 3e-3], "d": [16, 32]},
        TrainConfig(epochs=2, batch_size=64), [1, 2], out_root=root / "runs",
        window_root=root / "interim" / "windows", feature_root=root / "interim" / "features",
        log=messages.append,
    )  # fmt: skip
    assert not any(m.startswith("[tune] fold") for m in messages)  # nothing refitted


def test_near_ties_go_to_the_cheaper_configuration(root, monkeypatch):
    """Two configurations with the same validation scores: the smaller width wins."""
    from pdmfm import train

    original = train.fit

    def same_score(inputs, train_rows, val_rows, cfg, seed, log=print):
        fitted = original(inputs, train_rows, val_rows, cfg, seed, log)
        for h in fitted.history:
            h["val_score"] = 0.5
        return fitted

    monkeypatch.setattr(tune, "fit", same_score)
    report = search(root, {"d": [32, 16]}, run_id="ties")
    assert report["chosen"] == {"d": 16}
