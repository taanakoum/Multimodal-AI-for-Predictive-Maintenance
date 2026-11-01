"""Supervised baselines (pdmfm.baselines) on synthetic stores: features, the selection on
validation units only, Plan B's output layout, and the paired report across runs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import build_planb_root

from pdmfm import baselines, planb, report
from pdmfm.model.inputs import assemble
from pdmfm.train import TrainConfig
from pdmfm.windows import WindowStore


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("baselines")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        mp.setattr(planb, "BOOTSTRAP_REPLICATES", 50)
        build_planb_root(root, datasets=("nonastreda", "xjtu_sy", "paderborn"))
        yield root


def paths(root):
    return {
        "window_root": root / "interim" / "windows",
        "feature_root": root / "interim" / "features",
    }


def run(root, method, dataset, run_id, **kwargs):
    return baselines.run(
        method, dataset, run_id, kwargs.pop("folds", [1]), kwargs.pop("seeds", [0]),
        out_root=root / "runs", log=quiet, **paths(root), **kwargs,
    )  # fmt: skip


# ----------------------------------------------------------------------------- features


def test_line_prominence():
    spectrum = np.zeros(512)
    order = 3.58
    spectrum[int(order / (25 / 512))] = 2.0
    assert baselines.line_prominence(spectrum, order) == pytest.approx(2.0)
    assert baselines.line_prominence(np.ones(512), order) == 0.0
    assert np.isnan(baselines.line_prominence(spectrum, 30.0))  # beyond the axis


def test_feature_blocks(root):
    store = WindowStore.open("xjtu_sy", root / "interim" / "windows")
    lines = baselines.defect_line_features(store)
    assert lines is not None and lines.shape == (len(store), 2 * 4 * 3)
    assert baselines.defect_line_features(store) is lines  # computed once per store
    nona = WindowStore.open("nonastreda", root / "interim" / "windows")
    assert baselines.defect_line_features(nona) is None
    inputs = assemble("nonastreda", "stats", ("F", "E"), photos=False, **paths(root))
    x = baselines.handcrafted_features(inputs, nona)
    # 3 force channels x 36 stats (F only) + Fx, Fy levels x 4 (no Fz level token).
    assert x.shape == (len(nona), 3 * 36 + 2 * 4)
    pooled = baselines.pooled_features(inputs)
    assert pooled.shape == (len(nona), 3 * 36)


# ----------------------------------------------------------------------------- runs


def test_gbm_writes_plan_b_layout_for_both_tasks(root):
    out = run(root, "gbm", "nonastreda", "gbm-nona")
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    assert result["method"] == "gbm" and set(result["selected"]) == {"class", "reg"}
    assert result["test"]["combined"]["bag"]["macro_f1"] >= 0
    assert result["test"]["regression"]["combined"]["bag"]["rmse"] >= 0
    bags = pd.read_csv(out / "fold1-seed0-bags.csv")
    for column in ("pred_class", "pred_class_ordinal", "pred_reg", "persistence", "schedule"):
        assert column in bags
    test_units = planb.FOLDS["nonastreda"](
        WindowStore.open("nonastreda", paths(root)["window_root"]).manifest, 1
    )
    split_units = test_units.units(
        WindowStore.open("nonastreda", paths(root)["window_root"]).manifest, "test"
    )
    assert set(bags["unit"].astype(str)) == split_units  # only test units are scored
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# Gradient boosting on handcrafted features: gbm-nona")


def test_runs_resume_and_refuse_other_settings(root):
    out = run(root, "gbm", "nonastreda", "gbm-nona")  # done earlier: skipped
    assert (out / "fold1-seed0.json").is_file()
    with pytest.raises(ValueError, match="other settings"):
        run(root, "probe", "nonastreda", "gbm-nona", encoder="stats", arm_list=("F",))


def test_probe_on_rul(root):
    out = run(root, "probe", "xjtu_sy", "probe-xjtu", encoder="stats", arm_list=("F", "E"))
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    assert list(result["selected"]) == ["reg"]
    assert "post_fpt" in result["test"]["regression"]["combined"]["bag"]
    assert "pred_reg" in pd.read_csv(out / "fold1-seed0-bags.csv")
    # The synthetic bearings grow louder after 60 % of life: the probe beats the mean.
    learned = result["test"]["regression"]["combined"]["bag"]["rmse"]
    assert learned < 0.8 * result["baselines"]["train_mean"]["rmse"]


def test_gbm_on_paderborn_uses_each_folds_baseline(root):
    out = run(root, "gbm", "paderborn", "gbm-pad", folds=[1, 2])
    for fold in (1, 2):
        result = json.loads((out / f"fold{fold}-seed0.json").read_text(encoding="utf-8"))
        assert result["split"] == f"paderborn-fold{fold}" and "class" in result["selected"]


def test_selection_never_reads_test_units(root, monkeypatch):
    seen = {}
    original = baselines.fit_tabular

    def spy(method, inputs, x, rows, seed):
        seen.update(rows)
        return original(method, inputs, x, rows, seed)

    monkeypatch.setattr(baselines, "fit_tabular", spy)
    run(root, "gbm", "nonastreda", "gbm-spy")
    manifest = WindowStore.open("nonastreda", paths(root)["window_root"]).manifest
    unit = manifest["unit"].astype(str).to_numpy()
    assert not set(unit[seen["train"]]) & set(unit[seen["test"]])
    assert not set(unit[seen["val"]]) & set(unit[seen["test"]])


def test_patchtst_from_scratch_runs_and_selects_an_epoch(root):
    pytest.importorskip("transformers")  # optional "encoders" extra; CI skips this test
    out = run(root, "patchtst", "nonastreda", "pt-nona",
              patchtst_cfg={"epochs": 2, "batch_size": 16, "train_segments": 2})  # fmt: skip
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    assert result["best_epoch"] in (1, 2) and len(result["history"]) == 2
    assert set(result["selected"]) == {"class", "reg"}
    assert np.isfinite(result["test"]["regression"]["combined"]["bag"]["rmse"])


def test_unknown_methods_and_datasets_are_refused(root):
    with pytest.raises(ValueError, match="method"):
        run(root, "svm", "nonastreda", "x")
    with pytest.raises(ValueError, match="source datasets"):
        run(root, "gbm", "cwru", "x")


def test_the_report_pairs_a_baseline_with_plan_b(root):
    planb.run(
        "nonastreda", "stats", ("T", "F"), "planb-nona", [1], [0],
        TrainConfig(epochs=2, batch_size=32, d=32), out_root=root / "runs", log=quiet,
        **paths(root),
    )  # fmt: skip
    tables = report.run(
        "nona", {"planb": root / "runs" / "planb-nona", "gbm": root / "runs" / "gbm-nona"},
        n=50, out_root=root / "report", log=quiet,
    )  # fmt: skip
    classes = tables["class"].set_index("method")
    assert {"planb", "gbm", "majority", "persistence", "schedule"} <= set(classes.index)
    assert classes.loc["gbm", "units"] == 2  # fold 1 holds two test tools
    assert "reg" in tables
