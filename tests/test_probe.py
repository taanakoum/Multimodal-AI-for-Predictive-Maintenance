"""Gate 1 (pdmfm.probe) on synthetic window stores with the handcrafted encoder."""

from __future__ import annotations

import pandas as pd
import pytest
from synthetic import cwru_reader, nonastreda_reader, xjtu_reader

from pdmfm import probe, protocols, windows
from pdmfm.encoders import StatsEncoder
from pdmfm.features import FeatureStore


def quiet(*_):
    pass


def named(name: str):
    """A stats encoder under another name (a stand-in for another checkpoint)."""
    return type(f"Enc_{name}", (StatsEncoder,), {"name": name})()


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate1")
    stores = FeatureStore(root / "features")
    readers = {"cwru": cwru_reader(), "nonastreda": nonastreda_reader(), "xjtu_sy": xjtu_reader(30)}
    for name, reader in readers.items():
        store = windows.build(name, out=root / "windows" / name, reader=reader, log=quiet)
        for arm in ("F", "T", "E") if store.spec.bearing else ("F", "T"):
            stores.extract(store, StatsEncoder(), arm, log=quiet)
        for fold in range(1, 6):  # per-fold stand-ins for "patchtst-fold{fold}"
            stores.extract(store, named(f"foldenc-{fold}"), "F", log=quiet)
    return root


def run(root, tasks, candidates, references=(), run_id="test", jobs=1):
    return probe.run(
        tasks, candidates, list(references), run_id, out_root=root / "runs",
        window_root=root / "windows", feature_root=root / "features", log=quiet, jobs=jobs,
    )  # fmt: skip


def test_probe_scores_every_available_task(root):
    report = run(root, list(probe.TASKS), ["stats:F", "stats:T", "stats:T+F"])
    assert set(report["skipped"]) == {"paderborn"}  # no Paderborn store here
    assert report["winner"] in {"stats:F", "stats:T", "stats:T+F"}
    out = root / "runs" / "test"
    results = pd.read_csv(out / "results.csv")
    assert len(results) == (1 + 5 + 5) * 3  # cwru 1 split, nonastreda and xjtu_sy 5 folds
    score = results.groupby(["task", "candidate"])[["balanced_accuracy", "rmse"]].mean()
    assert score.loc[("cwru", "stats:F"), "balanced_accuracy"] >= 0.9  # class tones: spectrum
    assert score.loc[("nonastreda", "stats:T"), "balanced_accuracy"] >= 0.8  # force level: time
    assert score.loc[("xjtu_sy", "stats:T"), "rmse"] < 0.3  # growing vibration: RUL
    assert "Winner" in (out / "summary.md").read_text()


def test_validation_units_are_scored_never_test_units(root):
    run(root, ["nonastreda"], ["stats:T"], run_id="units")
    results = pd.read_csv(root / "runs" / "units" / "results.csv")
    store = windows.WindowStore(root / "windows" / "nonastreda")
    for k, row in enumerate(results.itertuples(), 1):
        split = protocols.nonastreda_fold(store.manifest, k)
        assert row.n_val == int(split.val.sum()) and row.n_train == int(split.train.sum())


def test_candidates_share_the_same_rows(root):
    store = windows.WindowStore(root / "windows" / "cwru")
    half = protocols.subsample(store.manifest, 0.5, seed=0)
    FeatureStore(root / "features").extract(store, named("half"), "F", rows=half, log=quiet)
    run(root, ["cwru"], ["stats:F", "half:F"], run_id="common")
    results = pd.read_csv(root / "runs" / "common" / "results.csv")
    assert results["n_val"].nunique() == 1 and results["n_train"].nunique() == 1
    assert results["n_val"].iloc[0] == int((half & protocols.cwru_gate1(store.manifest).val).sum())


def test_fold_placeholder_selects_each_folds_model(root):
    run(root, ["nonastreda"], ["foldenc-{fold}:F", "stats:F"], run_id="folds")
    results = pd.read_csv(root / "runs" / "folds" / "results.csv")
    by_candidate = results.pivot(index="split", columns="candidate", values="balanced_accuracy")
    pd.testing.assert_series_equal(
        by_candidate["foldenc-{fold}:F"], by_candidate["stats:F"], check_names=False
    )


def test_missing_features_skip_the_task_instead_of_failing(root):
    report = run(root, ["cwru"], ["mantis:F"], run_id="missing")
    assert report["winner"] is None and "cwru/cwru-gate1" in report["skipped"]


def test_ties_go_to_the_cheaper_encoder():
    results = pd.DataFrame(
        [
            {"task": "cwru", "candidate": c, "is_candidate": True, "balanced_accuracy": 0.9}
            for c in ("moment:T", "mantis:T", "stats:F")
        ]
    )
    results.loc[2, "is_candidate"] = False  # a reference is never ranked
    summary = probe.summarize(results)
    assert summary["candidate"].tolist()[:2] == ["mantis:T", "moment:T"]
    assert summary.loc[summary["candidate"] == "stats:F", "mean_rank"].isna().all()


def test_candidate_syntax():
    assert probe.parse_candidate("moment:T+E") == ("moment", ("T", "E"))
    with pytest.raises(ValueError):
        probe.parse_candidate("moment")


def test_arms_a_dataset_lacks_are_dropped_or_ranked_last(root):
    report = run(root, ["cwru", "nonastreda"], ["stats:E", "stats:T+E", "stats:T"], run_id="arms")
    results = pd.read_csv(root / "runs" / "arms" / "results.csv")
    nona = results[results["task"] == "nonastreda"].pivot(
        index="split", columns="candidate", values="balanced_accuracy"
    )
    assert nona["stats:E"].isna().all()  # Nonastreda has no E arm
    pd.testing.assert_series_equal(
        nona["stats:T+E"], nona["stats:T"], check_names=False
    )  # T+E is T there
    arms_used = results.loc[results["task"] == "nonastreda", "arms"].fillna("")  # "" reads as NaN
    assert set(arms_used) == {"", "T"}
    summary = pd.read_csv(root / "runs" / "arms" / "summary.csv").set_index("candidate")
    assert summary.loc["stats:E", "rank_nonastreda"] == 3.0  # worst rank, not a free pass
    assert report["winner"] in {"stats:T+E", "stats:T", "stats:E"}


def test_parallel_fits_give_the_same_results(root):
    run(root, ["nonastreda"], ["stats:F", "stats:T"], run_id="serial", jobs=1)
    run(root, ["nonastreda"], ["stats:F", "stats:T"], run_id="parallel", jobs=2)
    serial = pd.read_csv(root / "runs" / "serial" / "results.csv")
    parallel = pd.read_csv(root / "runs" / "parallel" / "results.csv")
    pd.testing.assert_frame_equal(serial, parallel)


def test_a_missing_reference_is_dropped_not_the_task(root):
    report = run(root, ["cwru"], ["stats:F"], references=["mantis:F"], run_id="ref")
    assert report["winner"] == "stats:F" and report["complete"]
    assert "reference dropped" in report["skipped"]["cwru/cwru-gate1/mantis:F"]
    results = pd.read_csv(root / "runs" / "ref" / "results.csv")
    assert results["candidate"].tolist() == ["stats:F"]


def test_a_missing_candidate_marks_gate1_incomplete(root):
    report = run(root, ["cwru", "nonastreda"], ["stats:F", "mantis:F"], run_id="gap")
    assert not report["complete"] and report["winner"] is None
    assert "Incomplete" in (root / "runs" / "gap" / "summary.md").read_text()
