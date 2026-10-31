"""Plan B (pdmfm.model.late, pdmfm.train, pdmfm.planb) on synthetic stores and caches."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from synthetic import build_planb_root

from pdmfm import planb, protocols
from pdmfm.model.inputs import Modality, TaskInputs, assemble
from pdmfm.model.late import LateEvidential
from pdmfm.train import TrainConfig, _losses, bag_mean, batch, fit, predict, selection_score

SMALL = TrainConfig(epochs=2, batch_size=32, d=32)


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("planb")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        mp.setattr(planb, "BOOTSTRAP_REPLICATES", 100)  # intervals are tested in test_stats
        build_planb_root(root)
        yield root


def load(root, dataset, arms, photos=True, train_units=None):
    return assemble(
        dataset, "stats", arms, photos=photos, window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features", train_units=train_units,
    )  # fmt: skip


@pytest.fixture(scope="module")
def nona(root):
    return load(root, "nonastreda", ("T", "F"))


def run(
    root, dataset, arms, run_id, cfg=SMALL, folds=(1,), seeds=(0,), photos=True, encoder="stats"
):
    return planb.run(
        dataset, encoder, arms, run_id, list(folds), list(seeds), cfg, photos=photos,
        out_root=root / "runs", window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features", log=quiet,
    )  # fmt: skip


def outputs(model, inputs, rows, drop=()):
    model.eval()
    with torch.no_grad():
        return model(batch(inputs, rows, torch.device("cpu"), drop))


# ----------------------------------------------------------------------------- the model


def test_forward_shapes(nona):
    torch.manual_seed(0)
    model = LateEvidential(nona, d=32)
    rows = np.arange(10)
    out = outputs(model, nona, rows)
    k = len(nona.classes)
    assert out["evidence"].shape == (10, 2, k) and out["combined"].shape == (10, k)
    assert out["conflict"].shape == (10, 1)
    assert out["nig"].shape == (10, 2, 4) and out["fused_nig"].shape == (10, 4)
    assert torch.all(out["evidence"] >= 0) and torch.isfinite(out["fused_nig"]).all()
    assert out["attention"]["force"].shape == (10, 15 + 2)  # tokens + level tokens


def test_an_absent_photo_leaves_the_force_opinion_unchanged(nona):
    torch.manual_seed(0)
    model = LateEvidential(nona, d=32)
    absent = np.flatnonzero(~nona.modalities["photo"].present)
    assert len(absent) > 0
    out = outputs(model, nona, absent)
    torch.testing.assert_close(out["evidence"][:, 1], torch.zeros_like(out["evidence"][:, 1]))
    torch.testing.assert_close(out["combined"], out["evidence"][:, 0])
    torch.testing.assert_close(out["fused_nig"], out["nig"][:, 0])
    torch.testing.assert_close(out["conflict"], torch.zeros_like(out["conflict"]))


def test_dropping_a_modality_at_test_time(nona):
    torch.manual_seed(0)
    model = LateEvidential(nona, d=32)
    rows = np.flatnonzero(nona.modalities["photo"].present)[:20]
    both = outputs(model, nona, rows)
    without = outputs(model, nona, rows, drop=("photo",))
    assert not without["present"][:, 1].any()
    torch.testing.assert_close(without["combined"], both["evidence"][:, 0])
    torch.testing.assert_close(without["fused_nig"], both["nig"][:, 0])
    assert not torch.allclose(both["combined"], both["evidence"][:, 0])  # photo counted


def test_masked_tokens_have_no_effect(nona):
    """Tokens of photos that are not attached (and the workpiece kind, never attached here)
    may hold anything: the evidence and the fused NIG do not change."""
    torch.manual_seed(0)
    model = LateEvidential(nona, d=32)
    rows = np.arange(len(nona.manifest))
    before = outputs(model, nona, rows)
    photo = nona.modalities["photo"]
    noisy = photo.tokens.copy()
    noisy[~photo.token_present] = np.random.default_rng(0).normal(
        0, 100, size=noisy[~photo.token_present].shape
    )
    changed = TaskInputs(**{**nona.__dict__, "modalities": {**nona.modalities}})
    changed.modalities["photo"] = Modality(**{**photo.__dict__, "tokens": noisy})
    after = outputs(model, changed, rows)
    for key in ("evidence", "combined", "fused_nig"):
        torch.testing.assert_close(after[key], before[key])


def test_parameter_budget_at_full_size():
    """At d = 256 with real widths (Mantis 256, DINOv2 384, Qwen 1024) the model stays
    within the plan's 3 M parameters per fusion candidate."""
    n = 4
    modalities = {
        name: Modality(
            name, np.zeros((n, tokens, dim), np.float32), np.zeros(tokens, int),
            np.arange(tokens) % 3, np.ones(n, bool),
            levels=np.zeros((n, 2, 4), np.float32) if name != "photo" else None,
            level_channels=np.array([0, 1]) if name != "photo" else None,
        )
        for name, tokens, dim in (("force", 15, 256), ("vibration", 16, 256), ("photo", 15, 384))
    }  # fmt: skip
    inputs = TaskInputs(
        "x", pd.DataFrame(index=range(n)), modalities, ["a", "b", "c"],
        np.eye(3, 1024, dtype=np.float32), np.zeros(n, int), np.zeros(n),
    )  # fmt: skip
    model = LateEvidential(inputs, d=256)
    assert sum(p.numel() for p in model.parameters()) <= 3_000_000


def test_level_channels_beyond_the_token_channels_get_embeddings():
    n = 3
    modality = Modality(
        "vibration", np.zeros((n, 1, 4), np.float32), np.zeros(1, int), np.zeros(1, int),
        np.ones(n, bool), levels=np.zeros((n, 2, 4), np.float32), level_channels=np.array([0, 5]),
    )  # fmt: skip
    inputs = TaskInputs("x", pd.DataFrame(index=range(n)), {"vibration": modality})
    inputs.y_reg = np.zeros(n)
    out = outputs(LateEvidential(inputs, d=8), inputs, np.arange(n))
    assert out["fused_nig"].shape == (n, 4)


# ----------------------------------------------------------------------------- training


def test_bag_mean_keeps_first_appearance_order():
    values = np.array([[1.0], [2.0], [10.0], [3.0], [20.0]])
    keys, means = bag_mean(values, np.array(["b", "b", "a", "b", "a"]))
    assert keys.tolist() == ["b", "a"]
    np.testing.assert_allclose(means, [[2.0], [15.0]])


def test_fit_is_reproducible(nona):
    split = protocols.nonastreda_fold(nona.manifest, 1)
    train, val = np.flatnonzero(split.train), np.flatnonzero(split.val)
    a = fit(nona, train, val, SMALL, seed=3, log=quiet)
    b = fit(nona, train, val, SMALL, seed=3, log=quiet)
    assert [h["loss"] for h in a.history] == pytest.approx([h["loss"] for h in b.history])
    assert a.y_mean == pytest.approx(float(np.mean(nona.y_reg[train])))


def test_selection_score_combines_f1_and_rmse(nona):
    rows = np.arange(len(nona.manifest))
    k = len(nona.classes)
    perfect = {
        "combined": np.eye(k)[nona.y_class] * 100,
        "fused_nig": np.stack(
            [
                nona.y_reg,
                np.ones_like(nona.y_reg),
                np.full_like(nona.y_reg, 2),
                np.ones_like(nona.y_reg),
            ],
            1,
        ),
    }
    assert selection_score(nona, rows, perfect, 0.0, 1.0) == pytest.approx(1.0)
    # Standardized means (y / 10 + 1) with std 10 are 10 too high: RMSE / std = 1.
    shifted = perfect["fused_nig"].copy()
    shifted[:, 0] = nona.y_reg / 10 + 1
    score = selection_score(nona, rows, dict(perfect, fused_nig=shifted), 0.0, 10.0)
    assert score == pytest.approx((1 + 0) / 2)


def test_plan_b_learns_the_synthetic_wear_classes(root):
    """The synthetic force level rises with the wear class. Over three seeds, Plan B must
    beat the majority class on the held-out tools by a wide margin, and every seed's fused
    wear error must be well below the training mean's."""
    cfg = TrainConfig(epochs=60, batch_size=16, d=32, lr=3e-3)
    out = run(root, "nonastreda", ("T", "F"), "learns", cfg=cfg, photos=False, seeds=(0, 1, 2))
    results = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("fold1-seed*.json")]
    assert len(results) == 3
    accuracy = [r["test"]["combined"]["bag"]["balanced_accuracy"] for r in results]
    majority = results[0]["baselines"]["majority"]["balanced_accuracy"]
    assert np.mean(accuracy) >= 0.85 and min(accuracy) > majority
    for r in results:
        rmse = r["test"]["regression"]["combined"]["bag"]["rmse"]
        assert rmse < r["baselines"]["train_mean"]["rmse"] / 2


# ----------------------------------------------------------------------------- runs


def test_run_writes_results_resumes_and_refuses_other_settings(root):
    messages = []
    out = run(root, "nonastreda", ("T", "F"), "resume", seeds=(0, 1))
    assert sorted(p.name for p in out.glob("fold*")) == [
        "fold1-seed0-bags.csv", "fold1-seed0.json", "fold1-seed0.npz",
        "fold1-seed1-bags.csv", "fold1-seed1.json", "fold1-seed1.npz",
    ]  # fmt: skip
    stamp = (out / "fold1-seed0.json").stat().st_mtime_ns
    planb.run(
        "nonastreda", "stats", ("T", "F"), "resume", [1], [0, 1], SMALL, out_root=root / "runs",
        window_root=root / "interim" / "windows", feature_root=root / "interim" / "features",
        log=messages.append,
    )  # fmt: skip
    assert sum("skipped" in m for m in messages) == 2
    assert (out / "fold1-seed0.json").stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="other settings"):
        run(root, "nonastreda", ("T", "F"), "resume", cfg=TrainConfig(epochs=3, d=32))


def test_result_contents(root):
    out = run(root, "nonastreda", ("T", "F"), "contents")
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    test = result["test"]
    for key in ("combined", "combined_calibrated", "only_force", "only_photo"):
        assert {"window", "bag"} <= set(test[key])
    for key in ("without_force", "without_photo"):
        assert "bag" in test[key]
    assert {"combined", "only_force", "without_photo"} <= set(test["regression"])
    assert {"uncertainty", "conflict", "trust_force", "trust_photo"} <= set(test["trust"])
    assert 0.05 <= result["temperature"] <= 20 and 0.05 <= result["interval_scale"] <= 20
    # The calibrated intervals are the raw ones times the fitted factor, nothing else.
    raw, calibrated = test["regression"]["combined"], test["regression"]["combined_calibrated"]
    for level in ("window", "bag"):
        assert calibrated[level]["width_90"] == pytest.approx(
            raw[level]["width_90"] * result["interval_scale"], rel=1e-5
        )
        assert calibrated[level]["rmse"] == pytest.approx(raw[level]["rmse"])
    assert result["split"] == "nonastreda-fold1" and len(result["history"]) == SMALL.epochs
    manifest = load(root, "nonastreda", ("T", "F")).manifest
    saved = np.load(out / "fold1-seed0.npz")
    assert set(manifest.loc[saved["rows"], "unit"]) == {"T1", "T2"}  # test tools of fold 1
    assert saved["evidence"].shape[:2] == (len(saved["rows"]), 2)
    # Only rows with a photo are scored for the photo alone.
    assert test["only_photo"]["window"]["n"] < test["only_force"]["window"]["n"]


def test_without_photo_equals_the_force_opinion(root):
    out = run(root, "nonastreda", ("T", "F"), "without")
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    flat_without, flat_force = {}, {}
    planb._flatten("", result["test"]["without_photo"], flat_without)
    planb._flatten("", result["test"]["only_force"], flat_force)
    assert flat_without == pytest.approx(flat_force)  # a vacuous photo opinion: equal


def test_late_fusion_without_a_modality_combines_the_other_experts(nona):
    """The missing-modality test marks the modality absent at prediction time; for late
    fusion that is exactly Dempster's rule and MoNIG over the remaining experts."""
    from pdmfm import losses

    torch.manual_seed(0)
    model = LateEvidential(nona, d=32)
    rows = np.arange(len(nona.manifest))
    pred = predict(model, nona, rows, SMALL)
    dropped = planb.without_each(model, nona, rows, SMALL)
    assert set(dropped) == {"force", "photo"}
    combined = losses.ds_combine_all(torch.from_numpy(pred["evidence"][:, [0]]))[0].numpy()
    np.testing.assert_allclose(dropped["photo"]["combined"], combined, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dropped["photo"]["fused_nig"], pred["nig"][:, 0], rtol=1e-5)


def test_persistence_baselines_on_the_known_sequence(nona):
    """Each synthetic tool runs sharp, sharp, used, used, dulled, dulled (wear 40 to 100 µm):
    the previous pass's label is right on passes 2, 4, 6; pass 1 falls back to the training
    majority (a three-way tie, so the first class, sharp: right)."""
    split = protocols.nonastreda_fold(nona.manifest, 1)
    train, test = np.flatnonzero(split.train), np.flatnonzero(split.test)
    result = planb.baselines(nona, train, test)
    assert result["persistence"]["accuracy"] == pytest.approx(4 / 6)
    assert result["majority"]["accuracy"] == pytest.approx(1 / 3)
    mean = float(np.mean(nona.y_reg[train]))
    errors = np.array([40 - mean, 0, 30, 0, 30, 0])
    assert result["persistence_regression"]["mae"] == pytest.approx(np.abs(errors).mean())
    assert result["persistence_regression"]["rmse"] == pytest.approx(np.sqrt((errors**2).mean()))
    assert result["train_mean"]["mae"] == pytest.approx(
        np.mean(np.abs([40, 40, 70, 70, 100, 100] - np.float64(mean)))
    )


def test_schedule_baseline_knows_only_the_position_in_each_tools_life(nona):
    """Every synthetic tool follows the same schedule (sharp, sharp, used, used, dulled,
    dulled; 40 to 100 um), so the training tools' class and mean wear at each pass number
    predict the test tools exactly: the schedule shortcut a model must beat."""
    split = protocols.nonastreda_fold(nona.manifest, 1)
    train, test = np.flatnonzero(split.train), np.flatnonzero(split.test)
    table = planb.baseline_table(nona, train, test)
    np.testing.assert_array_equal(table["schedule"], [0, 0, 1, 1, 2, 2] * 2)
    np.testing.assert_allclose(table["schedule_reg"], table["y_reg"], atol=1e-4)
    result = planb.baselines(nona, train, test)
    assert result["schedule"]["accuracy"] == 1.0 and result["schedule"]["mace"] == 0.0
    assert result["schedule_regression"]["rmse"] == pytest.approx(0, abs=1e-4)


def test_schedule_baseline_falls_back_beyond_the_training_lives():
    position = np.array([0, 1, 2, 0, 1])
    values = np.array([1, 1, 2, 0, 1])
    out = planb._schedule(position, values, np.array([0, 1, 2, 3]), -1, lambda v: int(v.max()))
    np.testing.assert_array_equal(out, [1, 1, 2, -1])


def test_ordered_classes_get_ordinal_metrics_and_decisions(root):
    out = run(root, "nonastreda", ("T", "F"), "ordinal")
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    bag = result["test"]["combined"]["bag"]
    assert {"mace", "ordinal", "recall"} <= set(bag)
    assert {"accuracy", "macro_f1", "mace", "recall"} <= set(bag["ordinal"])
    bags = pd.read_csv(out / "fold1-seed0-bags.csv")
    assert {"pred_class_ordinal", "schedule", "schedule_reg"} <= set(bags)
    pooled = set(pd.read_csv(out / "pooled.csv")["quantity"])
    assert {
        "macro-F1, model, ordinal median", "macro-F1, schedule", "macro-F1, model - schedule",
        "accuracy where the class changes, model", "RMSE, schedule",
    } <= pooled  # fmt: skip
    assert "macro-F1, model - model, ordinal median" not in pooled
    xjtu = run(root, "xjtu_sy", ("F",), "unordered")
    assert "ordinal" not in json.dumps(json.loads((xjtu / "fold1-seed0.json").read_text()))


def test_xjtu_rul_run(root):
    out = run(root, "xjtu_sy", ("F", "E"), "xjtu")
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    combined = result["test"]["regression"]["combined"]["bag"]
    assert {"rmse", "mae", "coverage_90", "width_90", "phm2012"} <= set(combined)
    assert "combined" not in result["test"] and result["temperature"] is None
    assert result["interval_scale"] > 0 and "combined_calibrated" in result["test"]["regression"]
    # No RUL persistence; the schedule baseline is the training bearings' mean RUL at the
    # same snapshot number, exact here because every synthetic bearing lives 30 minutes.
    assert set(result["baselines"]) == {"train_mean", "schedule_regression"}
    assert result["baselines"]["schedule_regression"]["rmse"] == pytest.approx(0, abs=1e-6)
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Regression" in summary and "baseline: train_mean" in summary
    assert "interval scale" in summary and "| combined_calibrated |" in summary


def test_summary_over_folds_and_seeds(root):
    out = run(root, "nonastreda", ("T", "F"), "summary", folds=(1, 2), seeds=(0, 1))
    summary = pd.read_csv(out / "summary.csv", index_col=0)
    assert summary.loc["test.combined.bag.macro_f1", "runs"] == 4
    text = (out / "summary.md").read_text(encoding="utf-8")
    for line in ("| combined |", "| without_photo |", "| baseline: persistence |", "trust_force"):
        assert line in text


def test_datasets_without_outer_folds_are_refused(root):
    with pytest.raises(ValueError, match="outer folds"):
        run(root, "cwru", ("F",), "cwru")


def test_command_line(root, tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(root))
    planb.main(
        ["--dataset", "xjtu_sy", "--encoder", "stats", "--arms", "F", "--run-id", "cli",
         "--folds", "2", "--seeds", "0", "--epochs", "1", "--batch-size", "64",
         "--out", str(tmp_path), "--force"]
    )  # fmt: skip
    result = json.loads((tmp_path / "cli" / "fold2-seed0.json").read_text(encoding="utf-8"))
    assert result["settings"]["inputs"]["arms"] == ["F"]
    assert result["settings"]["train"]["epochs"] == 1


def test_losses_skip_unlabelled_rows(nona):
    torch.manual_seed(0)
    model = LateEvidential(nona, d=16)
    rows = np.arange(16)
    unlabelled = TaskInputs(**{**nona.__dict__})
    unlabelled.y_class = np.full_like(nona.y_class, -1)
    unlabelled.y_reg = np.full_like(nona.y_reg, np.nan)
    out = model(batch(unlabelled, rows, torch.device("cpu")))
    assert float(_losses(out, unlabelled, rows, 1.0, SMALL, 0.0, 1.0, "cpu")) == 0.0
    out = model(batch(nona, rows, torch.device("cpu")))
    loss = _losses(out, nona, rows, 1.0, SMALL, 50.0, 20.0, "cpu")
    loss.backward()
    assert float(loss.detach()) > 0 and all(
        p.grad is not None for p in model.encoders["force"].adapter.parameters()
    )


def test_paderborn_inputs_are_assembled_per_fold(root):
    messages = []
    out = planb.run(
        "paderborn", "stats", ("F", "E"), "paderborn", [1, 2], [0], SMALL, out_root=root / "runs",
        window_root=root / "interim" / "windows", feature_root=root / "interim" / "features",
        log=messages.append,
    )  # fmt: skip
    assert sum("modalities" in m for m in messages) == 2  # once per fold
    results = {
        k: json.loads((out / f"fold{k}-seed0.json").read_text(encoding="utf-8")) for k in (1, 2)
    }
    units = {k: r["settings"]["inputs"]["baseline_units"] for k, r in results.items()}
    assert units[1] != units[2]
    for k, r in results.items():
        manifest = load(root, "paderborn", ("F",), train_units=units[k]).manifest
        split = protocols.paderborn_fold(manifest, k)
        assert set(units[k]) == split.units(manifest, "train")
        assert r["settings_key"] != results[3 - k]["settings_key"]
        assert {"only_slow", "without_slow", "only_current"} <= set(r["test"])
        assert "persistence" not in r["baselines"]  # recordings of a bearing are not a sequence


def test_fold_placeholder_uses_each_folds_encoder(root):
    from pdmfm import features
    from pdmfm.encoders import StatsEncoder
    from pdmfm.windows import WindowStore

    s = WindowStore.open("xjtu_sy", root / "interim" / "windows")
    cache = features.FeatureStore(root / "interim" / "features")
    for k in (1, 2):
        encoder = type(f"Fold{k}", (StatsEncoder,), {"name": f"foldenc-{k}"})()
        cache.extract(s, encoder, "F", log=quiet)
    out = run(root, "xjtu_sy", ("F",), "placeholder", folds=(1, 2), encoder="foldenc-{fold}")
    for k in (1, 2):
        result = json.loads((out / f"fold{k}-seed0.json").read_text(encoding="utf-8"))
        assert result["settings"]["inputs"]["encoder"] == f"foldenc-{k}"


def test_modalities_can_be_left_out(root):
    out = planb.run(
        "paderborn", "stats", ("F",), "without-slow", [1], [0], SMALL, out_root=root / "runs",
        window_root=root / "interim" / "windows", feature_root=root / "interim" / "features",
        log=quiet, without=("slow",),
    )  # fmt: skip
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    assert result["settings"]["without"] == ["slow"]
    assert "slow" not in result["settings"]["inputs"]["modalities"]
    assert "only_slow" not in result["test"] and "only_current" in result["test"]
    with pytest.raises(ValueError, match="no modality"):  # Nonastreda has no slow channel
        planb.run(
            "nonastreda", "stats", ("F",), "unknown", [1], [0], SMALL, out_root=root / "runs",
            window_root=root / "interim" / "windows", feature_root=root / "interim" / "features",
            log=quiet, without=("slow",),
        )  # fmt: skip


def test_levels_are_standardized_with_training_rows_only(nona):
    split = protocols.nonastreda_fold(nona.manifest, 1)
    train = np.flatnonzero(split.train)
    model = LateEvidential(nona, d=16, train_rows=train)
    levels = nona.modalities["force"].levels
    np.testing.assert_allclose(
        model.levels_force_mean.numpy(), levels[train].mean(axis=0), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        model.levels_force_std.numpy(), levels[train].std(axis=0), rtol=1e-4, atol=1e-5
    )
    # Test rows never enter the statistics.
    changed = TaskInputs(**{**nona.__dict__, "modalities": {**nona.modalities}})
    shifted = levels.copy()
    shifted[np.flatnonzero(split.test)] += 1000.0
    changed.modalities["force"] = Modality(
        **{**nona.modalities["force"].__dict__, "levels": shifted}
    )
    again = LateEvidential(changed, d=16, train_rows=train)
    torch.testing.assert_close(again.levels_force_mean, model.levels_force_mean)
    identity = LateEvidential(nona, d=16)
    assert torch.all(identity.levels_force_mean == 0) and torch.all(identity.levels_force_std == 1)


def test_slow_tokens_are_standardized_over_present_tokens(root):
    manifest = load(root, "paderborn", ("F",), train_units=["K003"]).manifest
    split = protocols.paderborn_fold(manifest, 1)
    units = sorted(split.units(manifest, "train"))
    inputs = load(root, "paderborn", ("F",), train_units=units)
    slow = inputs.modalities["slow"]
    slow.token_present[:4] = False  # a channel missing in some training rows
    train = np.flatnonzero(split.train)
    model = LateEvidential(inputs, d=16, train_rows=train)
    present = train[slow.token_present[train, 0]]
    np.testing.assert_allclose(
        model.tokens_slow_mean.numpy()[0], slow.tokens[present, 0].mean(axis=0), rtol=1e-4,
        atol=1e-5,
    )  # fmt: skip


# ----------------------------------------------------------------------------- options


def test_photo_augmentation_samples_older_photos_in_training_only(nona):
    rows = np.flatnonzero(nona.modalities["photo"].present)[:64]
    rng = np.random.default_rng(0)
    cfg = TrainConfig(photo_lags=3)
    plain = batch(nona, rows, torch.device("cpu"))
    mixed = batch(nona, rows, torch.device("cpu"), augment=(cfg, rng))
    ages = mixed["photo"]["age"][:, 0]
    assert set(ages.unique().tolist()) <= {0.0, 1.0, 2.0, 3.0} and (ages > 1).any()
    torch.testing.assert_close(plain["photo"]["age"][:, 0], torch.ones(len(rows)))
    # Rows keep their photo whenever some lag has one; tokens of lag 1 are the attached ones.
    assert bool(mixed["photo"]["present"].all())
    same = ages == 1
    torch.testing.assert_close(mixed["photo"]["tokens"][same], plain["photo"]["tokens"][same])


def test_modality_dropout_changes_only_the_combined_losses(nona):
    torch.manual_seed(0)
    model = LateEvidential(nona, d=16)
    rows = np.flatnonzero(nona.modalities["photo"].present)[:32]
    out = model(batch(nona, rows, torch.device("cpu")))
    plain = _losses(out, nona, rows, 1.0, SMALL, 50.0, 20.0, "cpu")
    torch.manual_seed(1)
    dropped = _losses(
        out, nona, rows, 1.0, TrainConfig(epochs=2, d=32, modality_dropout=0.9), 50.0, 20.0, "cpu"
    )
    assert float(dropped.detach()) != pytest.approx(float(plain.detach()))
    # p = 1 still keeps one present modality per row: the losses stay finite.
    torch.manual_seed(2)
    all_dropped = _losses(
        out, nona, rows, 1.0, TrainConfig(epochs=2, d=32, modality_dropout=1.0), 50.0, 20.0, "cpu"
    )
    assert torch.isfinite(all_dropped)


def test_linear_head_option(nona):
    model = LateEvidential(nona, d=16, head="linear")
    out = outputs(model, nona, np.arange(8))
    assert out["evidence"].shape == (8, 2, len(nona.classes))
    with pytest.raises(ValueError, match="head must be"):
        LateEvidential(nona, d=16, head="cosine")


def test_history_tracks_the_fit_on_training_rows(nona):
    split = protocols.nonastreda_fold(nona.manifest, 1)
    fitted = fit(
        nona, np.flatnonzero(split.train), np.flatnonzero(split.val), SMALL, seed=0, log=quiet
    )
    assert all({"train_score", "val_score", "loss"} <= set(h) for h in fitted.history)


def test_results_hold_fit_staleness_and_bag_tables(root):
    out = run(root, "nonastreda", ("T", "F"), "diagnostics", seeds=(0, 1))
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    assert {"bag", "window"} <= set(result["train"]["combined"])
    assert {"photo_lag2", "photo_lag3"} <= set(result["test"])
    assert {"photo_lag2", "photo_lag3"} <= set(result["test"]["regression"])
    bags = pd.read_csv(out / "fold1-seed0-bags.csv")
    manifest = load(root, "nonastreda", ("T", "F")).manifest
    assert set(bags["unit"]) == {"T1", "T2"} and len(bags) == 12  # two tools x 6 passes
    assert {"y_class", "pred_class", "persistence", "majority", "y_reg", "pred_reg"} <= set(bags)
    passes = manifest.groupby("bag").head(1).set_index("bag")
    np.testing.assert_array_equal(bags["y_reg"], passes.loc[bags["bag"], "wear_um"])
    pooled = pd.read_csv(out / "pooled.csv").set_index("quantity")
    assert {"macro-F1, model", "macro-F1, model - persistence", "RMSE, model"} <= set(pooled.index)
    assert (pooled["low"] <= pooled["high"]).all()
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Pooled out-of-fold" in text and "Fit (bag level)" in text
    assert "| photo_lag2 |" in text


def test_rul_is_also_scored_after_the_first_prediction_time(root):
    out = run(root, "xjtu_sy", ("F",), "post-fpt")
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    post = result["test"]["regression"]["combined"]["window"]["post_fpt"]
    assert {"rmse", "coverage_90", "phm2012"} <= set(post)
    assert post["n"] < result["test"]["regression"]["combined"]["window"]["n"]
    pooled = pd.read_csv(out / "pooled.csv")
    assert "RMSE after FPT, model" in set(pooled["quantity"])


def test_command_line_options(root, tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(root))
    planb.main(
        ["--dataset", "nonastreda", "--encoder", "stats", "--arms", "F", "--run-id", "options",
         "--folds", "1", "--seeds", "0", "--epochs", "1", "--d", "16", "--lr", "3e-3",
         "--modality-dropout", "0.3", "--photo-lags", "3", "--head", "linear",
         "--out", str(tmp_path), "--force"]
    )  # fmt: skip
    train = json.loads((tmp_path / "options" / "fold1-seed0.json").read_text())["settings"]["train"]
    assert (train["d"], train["lr"], train["modality_dropout"]) == (16, 3e-3, 0.3)
    assert (train["photo_lags"], train["head"]) == (3, "linear")


def test_projected_head_starts_at_the_text_prototypes_and_can_separate_them(nona):
    from pdmfm.model.layers import PrototypeEvidentialHead

    prototypes = torch.from_numpy(nona.prototypes)
    head = PrototypeEvidentialHead(16, prototypes, text_rank=4)
    torch.testing.assert_close(head.class_vectors(), head.prototypes)  # W_t = I at the start
    with torch.no_grad():
        head.text_up.normal_()
    moved = head.class_vectors()
    torch.testing.assert_close(moved.norm(dim=-1), torch.ones(len(prototypes)))
    assert not torch.allclose(moved, head.prototypes)
    model = LateEvidential(nona, d=16, head="projected")
    params = sum(p.numel() for p in model.class_heads["force"].parameters())
    plain = sum(p.numel() for p in LateEvidential(nona, d=16).class_heads["force"].parameters())
    assert params - plain == 2 * 16 * nona.prototypes.shape[1]  # rank-16 A and B only
