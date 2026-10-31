"""Gate 2 (pdmfm.gate2): inner folds, candidates, the decision rule and the search runner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import build_planb_root

from pdmfm import gate2, planb, protocols
from pdmfm.train import TrainConfig
from pdmfm.windows import WindowStore

TINY = TrainConfig(epochs=2, batch_size=32, d=16)


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate2")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root, datasets=("nonastreda", "xjtu_sy", "paderborn", "matwi"))
        yield root


def manifest_of(root, dataset):
    return WindowStore.open(dataset, root / "interim" / "windows").manifest


# ----------------------------------------------------------------------------- inner folds


@pytest.mark.parametrize("dataset", ["nonastreda", "xjtu_sy", "paderborn", "matwi"])
@pytest.mark.parametrize("k", range(1, 6))
def test_inner_folds_rotate_validation_over_the_other_test_groups(root, dataset, k):
    manifest = manifest_of(root, dataset)
    outer = protocols.SOURCE_FOLDS[dataset](manifest, k)
    pool = outer.units(manifest, "train") | outer.units(manifest, "val")
    inner = protocols.inner_folds(dataset, manifest, k)
    assert len(inner) == 4
    validated = []
    for j, split in zip([j for j in range(1, 6) if j != k], inner, strict=True):
        split.assert_unit_disjoint(manifest)
        assert not split.test.any()
        val = split.units(manifest, "val")
        assert val == protocols.SOURCE_FOLDS[dataset](manifest, j).units(manifest, "test")
        assert split.units(manifest, "train") == pool - val
        assert not (val | split.units(manifest, "train")) & outer.units(manifest, "test")
        validated += sorted(val)
    assert sorted(validated) == sorted(pool)  # every non-test unit validated exactly once


def test_inner_folds_need_outer_folds(root):
    with pytest.raises(ValueError, match="no outer folds"):
        protocols.inner_folds("cwru", manifest_of(root, "nonastreda"), 1)


# ----------------------------------------------------------------------------- candidates


def test_candidates_share_the_search_recipe_and_plan_b_keeps_its_own():
    base = TrainConfig(epochs=7, d=64, modality_dropout=0.2)
    configs = gate2.candidates(base)
    assert list(configs) == ["planb", "late", "concat", "perceiver", "mbt", "c2f"]
    assert configs["planb"] == base  # fusion late, lr 1e-3, constant, its own options
    for name in ("late", "concat", "perceiver", "mbt", "c2f"):
        cfg = configs[name]
        assert (cfg.fusion, cfg.lr, cfg.schedule, cfg.input_dropout) == (
            name, 3e-4, "cosine", 0.3,
        )  # fmt: skip
        assert (cfg.epochs, cfg.d, cfg.modality_dropout) == (7, 64, 0.0)
    with pytest.raises(ValueError, match="unknown candidate"):
        gate2.candidates(base, ("planb", "gated"))


# ----------------------------------------------------------------------------- decision rule


def table(scores: dict[str, list[float]], **columns) -> pd.DataFrame:
    """Results of candidates over the same inner folds (one row per candidate and fold)."""
    rows = []
    for name, values in scores.items():
        for i, value in enumerate(values):
            row = {"candidate": name, "outer": 1 + i // 4, "split": f"s{i}", "score": value,
                   "robustness": 0.5, "ece": 0.1, "n_params": 1000, "seconds": 1.0}  # fmt: skip
            for column, per_candidate in columns.items():
                row[column] = per_candidate.get(name, row[column])
            rows.append(row)
    return pd.DataFrame(rows)


CONFIGS = gate2.candidates(TrainConfig())
FOLDS = 20


def steady(level: float, noise: float = 0.0, seed: int = 0) -> list[float]:
    return list(level + noise * np.random.default_rng(seed).standard_normal(FOLDS))


def test_a_clear_fusion_gain_replaces_late_fusion():
    report = gate2.decide(
        table({"planb": steady(0.60, 0.02), "late": steady(0.58, 0.02, 1),
               "c2f": steady(0.65, 0.02, 2)}),
        CONFIGS,
    )  # fmt: skip
    assert report["reference"] == "planb" and report["model"] == "c2f"
    assert "replaces late fusion" in report["verdict"]
    assert report["against_reference"]["gain"] == pytest.approx(0.05, abs=0.02)
    assert report["flags"] == "--fusion c2f --lr 0.0003 --schedule cosine --input-dropout 0.3"


def test_a_small_gain_keeps_plan_b():
    report = gate2.decide(
        table({"planb": steady(0.60), "late": steady(0.55), "concat": steady(0.605)}), CONFIGS
    )
    assert report["model"] == "planb" and "Plan B stays the model" in report["verdict"]
    assert report["flags"] == "--fusion late"


def test_a_gain_from_few_folds_keeps_plan_b():
    """+0.03 on average, but from 6 large wins out of 20 folds: below the 60 % win rule."""
    fusion = [0.70] * 6 + [0.59] * 14
    report = gate2.decide(table({"planb": steady(0.60), "mbt": fusion}), CONFIGS)
    assert report["against_reference"]["gain"] >= gate2.MIN_GAIN
    assert report["against_reference"]["win_rate"] == pytest.approx(0.3)
    assert report["model"] == "planb"


def test_the_reference_is_the_better_late_variant():
    report = gate2.decide(
        table({"planb": steady(0.55), "late": steady(0.62), "perceiver": steady(0.64)}), CONFIGS
    )
    assert report["reference"] == "late"
    assert report["model"] == "perceiver"  # +0.02 on every fold


def test_near_ties_go_to_robustness_then_calibration_then_size():
    noisy = {"planb": steady(0.50, 0.01), "concat": steady(0.60, 0.05, 1),
             "c2f": steady(0.60, 0.05, 2)}  # fmt: skip
    robust = gate2.decide(table(noisy, robustness={"concat": 0.4, "c2f": 0.55}), CONFIGS)
    assert robust["model"] == "c2f"
    calibrated = gate2.decide(table(noisy, ece={"concat": 0.05, "c2f": 0.2}), CONFIGS)
    assert calibrated["model"] == "concat"
    smaller = gate2.decide(table(noisy, n_params={"concat": 900, "c2f": 5000}), CONFIGS)
    assert smaller["model"] == "concat"
    larger = gate2.decide(table(noisy, n_params={"concat": 5000, "c2f": 900}), CONFIGS)
    assert larger["model"] == "c2f"


def test_a_clear_score_lead_is_not_overruled_by_robustness():
    scores = {"planb": steady(0.50), "concat": steady(0.70), "mbt": steady(0.60)}
    report = gate2.decide(table(scores, robustness={"concat": 0.3, "mbt": 0.6}), CONFIGS)
    assert report["model"] == "concat"


def test_regression_only_datasets_rank_by_interval_coverage():
    t = table({"planb": steady(0.4), "concat": steady(0.5)}).drop(columns="ece")
    t["coverage_error"] = 0.05
    report = gate2.decide(t, CONFIGS)
    assert report["calibration_metric"] == "coverage_error"
    assert report["model"] == "concat"


def test_robustness_scores_only_rows_that_keep_some_input(root):
    """Without the force, Nonastreda rows that had no photo have no input left: they are not
    scored, as in Plan B's missing-modality results."""
    from pdmfm.model.inputs import assemble
    from pdmfm.model.late import LateEvidential
    from pdmfm.planb import without_each
    from pdmfm.train import predict, selection_score

    inputs = assemble(
        "nonastreda", "stats", ("T", "F"), window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features",
    )  # fmt: skip
    model = LateEvidential(inputs, d=16)
    rows = np.flatnonzero(inputs.y_class >= 0)
    dropped = without_each(model, inputs, rows, TINY)
    photo = inputs.modalities["photo"].present[rows]
    assert 0 < photo.sum() < len(rows)
    without_force = predict(model, inputs, rows[photo], TINY, drop=("force",))
    expected = np.mean(
        [
            selection_score(inputs, rows[photo], without_force, 0.0, 1.0),
            selection_score(inputs, rows, dropped["photo"], 0.0, 1.0),
        ]
    )
    assert gate2.robustness(inputs, rows, dropped, 0.0, 1.0) == pytest.approx(expected)


# ----------------------------------------------------------------------------- the runner


def test_search_scores_inner_folds_only_and_resumes(root, monkeypatch):
    manifest = manifest_of(root, "nonastreda")
    outer_test = protocols.nonastreda_fold(manifest, 1).test
    seen = []
    original = gate2.fit

    def recording_fit(inputs, train_rows, val_rows, cfg, seed, log=print):
        seen.append((train_rows, val_rows, cfg.fusion))
        return original(inputs, train_rows, val_rows, cfg, seed, log=log)

    monkeypatch.setattr(gate2, "fit", recording_fit)
    kwargs = dict(
        names=("planb", "concat"), folds=[1], out_root=root / "runs",
        window_root=root / "interim" / "windows", feature_root=root / "interim" / "features",
        log=quiet,
    )  # fmt: skip
    report = gate2.search("nonastreda", "stats", ("T", "F"), "search", TINY, **kwargs)
    assert len(seen) == 8  # 4 inner folds x 2 candidates
    for train_rows, val_rows, _ in seen:
        assert not outer_test[train_rows].any() and not outer_test[val_rows].any()
        assert not set(train_rows) & set(val_rows)
    out = root / "runs" / "search"
    results = pd.read_csv(out / "results.csv")
    assert len(results) == 8 and set(results["candidate"]) == {"planb", "concat"}
    assert {"score", "robustness", "ece", "coverage_error", "n_params"} <= set(results)
    assert results["score"].between(-5, 1).all() and results["robustness"].notna().all()
    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert decision["model"] in ("planb", "concat") and decision["model"] == report["model"]
    assert "## " not in (out / "summary.md").read_text(encoding="utf-8")[:2]
    assert "| concat |" in (out / "summary.md").read_text(encoding="utf-8")
    gate2.search("nonastreda", "stats", ("T", "F"), "search", TINY, **kwargs)
    assert len(seen) == 8  # resumed: nothing refitted


def test_paderborn_inputs_use_each_inner_folds_training_bearings(root, monkeypatch):
    calls = []
    original = gate2.prepare_inputs

    def recording(dataset, encoder, arm_list, cfg, train_units, options, log):
        calls.append(tuple(train_units))
        return original(dataset, encoder, arm_list, cfg, train_units, options, log)

    monkeypatch.setattr(gate2, "prepare_inputs", recording)
    gate2.search(
        "paderborn", "stats", ("T", "F"), "paderborn", TINY, names=("planb",), folds=[2],
        out_root=root / "runs", window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features", log=quiet,
    )  # fmt: skip
    manifest = manifest_of(root, "paderborn")
    expected = [
        tuple(sorted(split.units(manifest, "train")))
        for split in protocols.inner_folds("paderborn", manifest, 2)
    ]
    assert calls == expected


def test_command_line(root, tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(root))
    monkeypatch.setattr(planb, "BOOTSTRAP_REPLICATES", 100)
    gate2.main(
        ["--dataset", "xjtu_sy", "--encoder", "stats", "--arms", "F", "--run-id", "cli",
         "--candidates", "planb", "mbt", "--folds", "3", "--epochs", "1", "--d", "16",
         "--out", str(tmp_path), "--force"]
    )  # fmt: skip
    decision = json.loads((tmp_path / "cli" / "decision.json").read_text(encoding="utf-8"))
    assert decision["calibration_metric"] == "coverage_error"  # RUL: regression only
    assert {r["candidate"] for r in decision["ranking"]} == {"planb", "mbt"}
    assert all(r["folds"] == 4 for r in decision["ranking"])
