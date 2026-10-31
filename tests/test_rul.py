"""The snapshot-history RUL model (pdmfm.rul): causal histories, learning from trends that a
single snapshot cannot show, and runs that Plan B's summaries and bootstrap can read."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from synthetic import build_planb_root

from pdmfm import planb, rul
from pdmfm.train import TrainConfig


def quiet(*_):
    pass


def make_snapshots(lives, features, minutes=None) -> rul.Snapshots:
    units = np.concatenate([[f"B{i}"] * n for i, n in enumerate(lives)])
    position = np.concatenate([np.arange(n) for n in lives])
    minute = position + 1.0 if minutes is None else minutes
    table = pd.DataFrame(
        {"bag": [f"{u}/{p}" for u, p in zip(units, position, strict=True)], "unit": units,
         "minute": minute, "y": np.zeros(len(units))}
    )  # fmt: skip
    return rul.Snapshots(table, np.asarray(features, np.float32), position)


# ----------------------------------------------------------------------------- histories


def test_histories_read_only_the_units_own_past():
    snaps = make_snapshots([4, 3], np.arange(7)[:, None])
    index, length = rul.histories(snaps, np.arange(7), history=3)
    np.testing.assert_array_equal(length, [1, 2, 3, 3, 1, 2, 3])
    np.testing.assert_array_equal(
        index,
        [[0, -1, -1], [0, 1, -1], [0, 1, 2], [1, 2, 3], [4, -1, -1], [4, 5, -1], [4, 5, 6]],
    )
    single, _ = rul.histories(snaps, np.arange(7), history=1)
    np.testing.assert_array_equal(single[:, 0], np.arange(7))


def test_predictions_never_depend_on_later_snapshots_or_other_units():
    rng = np.random.default_rng(0)
    lives = [12, 9]
    features = rng.standard_normal((21, 5))
    snaps = make_snapshots(lives, features)
    cfg = rul.RULConfig(history=5, epochs=1, d=16)
    torch.manual_seed(0)
    fitted = rul.fit(snaps, np.arange(21), np.arange(21), cfg, seed=0, log=quiet)
    before = rul.predict(fitted, np.arange(21))
    changed = features.copy()
    changed[8:12] += 50.0  # bearing 0's last four snapshots
    changed[12:] -= 50.0  # bearing 1, every snapshot
    fitted.sequences.x = fitted.sequences.x + (changed - features)  # shifted inputs
    after = rul.predict(fitted, np.arange(21))
    np.testing.assert_allclose(after[:8], before[:8], rtol=1e-5, atol=1e-6)
    assert not np.allclose(after[8:12], before[8:12])


def test_time_inputs_are_gaps_and_elapsed_minutes():
    snaps = make_snapshots([3], np.zeros((3, 2)), minutes=np.array([10.0, 11.0, 15.0]))
    sequences = rul.Sequences(snaps, np.arange(3), rul.RULConfig(history=3, elapsed=True))
    x, length = sequences.batch(np.array([2]), "cpu")
    assert x.shape == (1, 3, 2 + 2) and int(length[0]) == 3
    gaps, elapsed = x[0, :, 2].numpy(), x[0, :, 3].numpy()
    np.testing.assert_allclose(gaps * rul.TIME_SCALE, np.log1p([0, 1, 4]), rtol=1e-6)
    np.testing.assert_allclose(elapsed * rul.TIME_SCALE, np.log1p([0, 1, 5]), rtol=1e-6)
    without = rul.Sequences(snaps, np.arange(3), rul.RULConfig(history=3))  # the default
    assert without.batch(np.array([2]), "cpu")[0].shape == (1, 3, 3)


# ----------------------------------------------------------------------------- learning


def trend_snapshots(seed: int = 0, units: int = 24):
    """Each bearing has its own random level (so one snapshot says little) and starts to
    degrade halfway through a random life of 40-80 snapshots; the feature then climbs
    linearly. Normalized RUL is 1, then falls to 0. Only the trend, which a history shows,
    gives the RUL away; elapsed time does not (lives differ)."""
    rng = np.random.default_rng(seed)
    lives = rng.integers(40, 81, units)
    features, targets = [], []
    for life in lives:
        fpt = life // 2
        t = np.arange(life)
        progress = np.clip((t - fpt) / (life - 1 - fpt), 0, 1)
        level = rng.uniform(-3, 3)
        x = level + 3.0 * progress + 0.05 * rng.standard_normal(life)
        features.append(np.stack([x, rng.standard_normal(life)], 1))
        targets.append(1.0 - progress)
    snaps = make_snapshots(lives, np.concatenate(features))
    snaps.table["y"] = np.concatenate(targets)
    return snaps


def test_a_history_learns_what_a_single_snapshot_cannot():
    snaps = trend_snapshots()
    unit = snaps.table["unit"].to_numpy()
    names = sorted(set(unit), key=lambda u: int(u[1:]))
    train = np.flatnonzero(np.isin(unit, names[:16]))
    val = np.flatnonzero(np.isin(unit, names[16:20]))
    test = np.flatnonzero(np.isin(unit, names[20:]))
    y = snaps.table["y"].to_numpy()
    errors = {}
    for history in (1, 20):  # absolute features: the per-bearing level stays in
        cfg = rul.RULConfig(history=history, relative=False, epochs=40, d=32, lr=3e-3)
        fitted = rul.fit(snaps, train, val, cfg, seed=0, log=quiet)
        mean = rul.predict(fitted, test)[:, 0] * fitted.y_std + fitted.y_mean
        errors[history] = float(np.sqrt(np.mean((mean - y[test]) ** 2)))
    spread = float(np.std(y[test]))
    assert errors[20] < 0.6 * errors[1], errors  # seeds 0-2: 0.10-0.13 against 0.30-0.34
    assert errors[20] < 0.5 * spread


def test_inputs_relative_to_the_healthy_start_are_causal():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((30, 3))
    snaps = make_snapshots([18, 12], features)
    relative = rul.relative_to_start(snaps)
    np.testing.assert_allclose(relative[[0, 18]], 0, atol=1e-6)  # each unit's first snapshot
    np.testing.assert_allclose(relative[3], features[3] - features[:4].mean(0), rtol=1e-5)
    np.testing.assert_allclose(relative[15], features[15] - features[:10].mean(0), rtol=1e-5)
    later = features.copy()
    later[12:18] += 100.0  # unit 0 after its 12th snapshot
    changed = rul.relative_to_start(make_snapshots([18, 12], later))
    np.testing.assert_allclose(changed[:12], relative[:12], atol=1e-6)
    np.testing.assert_allclose(changed[18:], relative[18:], atol=1e-6)


def test_relative_inputs_remove_each_bearings_level():
    """On the trend snapshots, the healthy-start reference takes away the per-bearing level
    that made one snapshot ambiguous: a single relative snapshot is enough there."""
    snaps = trend_snapshots()
    unit = snaps.table["unit"].to_numpy()
    names = sorted(set(unit), key=lambda u: int(u[1:]))
    train = np.flatnonzero(np.isin(unit, names[:16]))
    val = np.flatnonzero(np.isin(unit, names[16:20]))
    test = np.flatnonzero(np.isin(unit, names[20:]))
    y = snaps.table["y"].to_numpy()
    cfg = rul.RULConfig(history=1, epochs=40, d=32, lr=3e-3)
    fitted = rul.fit(snaps, train, val, cfg, seed=0, log=quiet)
    mean = rul.predict(fitted, test)[:, 0] * fitted.y_std + fitted.y_mean
    assert np.sqrt(np.mean((mean - y[test]) ** 2)) < 0.5 * np.std(y[test])


# ----------------------------------------------------------------------------- runs


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rul")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        mp.setattr(planb, "BOOTSTRAP_REPLICATES", 100)
        mp.setattr(rul, "BOOTSTRAP_REPLICATES", 100)
        build_planb_root(root, datasets=("xjtu_sy",))
        yield root


def paths(root):
    return {
        "out_root": root / "runs",
        "window_root": root / "interim" / "windows",
        "feature_root": root / "interim" / "features",
        "log": quiet,
    }


def test_snapshots_of_the_synthetic_bearings(root):
    from pdmfm.model.inputs import assemble

    inputs = assemble(
        "xjtu_sy", "stats", ("F", "E"), photos=False, window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features",
    )  # fmt: skip
    snaps = rul.snapshots(inputs)
    assert len(snaps.table) == 15 * 30 == len(snaps.features)
    first = snaps.table.groupby("unit").head(1)
    assert (first["minute"] == 1).all() and (snaps.position[first.index] == 0).all()
    assert snaps.table.groupby("unit")["minute"].apply(lambda m: m.is_monotonic_increasing).all()
    # 2 channels x 2 arms of stats features, then the level descriptors.
    vibration = inputs.modalities["vibration"]
    n_levels = 0 if vibration.levels is None else int(np.prod(vibration.levels.shape[1:]))
    assert snaps.features.shape[1] == 4 * vibration.tokens.shape[2] + n_levels
    # Normalized RUL per snapshot is the windows' common label.
    manifest = inputs.manifest.groupby("bag").head(1).set_index("bag")
    np.testing.assert_allclose(snaps.table["y"], manifest.loc[snaps.table["bag"], "rul"])


def test_run_writes_plan_b_compatible_results_and_resumes(root):
    cfg = rul.RULConfig(history=5, epochs=3, d=16)
    out = rul.run("stats", ("F", "E"), "rul", [1], [0, 1], cfg, **paths(root))
    result = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))
    combined = result["test"]["regression"]["combined"]["bag"]
    assert {"rmse", "mae", "nll", "coverage_90", "width_90", "phm2012", "post_fpt"} <= set(combined)
    calibrated = result["test"]["regression"]["combined_calibrated"]["bag"]
    assert calibrated["width_90"] == pytest.approx(combined["width_90"] * result["interval_scale"])
    bags = pd.read_csv(out / "fold1-seed0-bags.csv")
    assert set(bags["unit"]) == {"Bearing1_1", "Bearing2_1", "Bearing3_1"} and len(bags) == 90
    assert {"y_reg", "pred_reg", "train_mean", "schedule_reg"} <= set(bags)
    # Every synthetic bearing lives 30 minutes: the schedule baseline is exact here.
    np.testing.assert_allclose(bags["schedule_reg"], bags["y_reg"], atol=1e-6)
    assert result["baselines"]["schedule_regression"]["rmse"] == pytest.approx(0, abs=1e-6)
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Regression (bag level)" in summary and "combined, after FPT" in summary
    pooled = pd.read_csv(out / "pooled.csv")
    assert {"RMSE, model", "RMSE after FPT, model", "RMSE, schedule"} <= set(pooled["quantity"])
    messages = []
    rul.run("stats", ("F", "E"), "rul", [1], [0, 1], cfg, **{**paths(root), "log": messages.append})
    assert sum("skipped" in m for m in messages) == 2
    with pytest.raises(ValueError, match="other settings"):
        rul.run("stats", ("F", "E"), "rul", [1], [0], rul.RULConfig(epochs=4, d=16), **paths(root))


def test_comparison_with_plan_b_on_the_same_snapshots(root):
    out = rul.run("stats", ("F", "E"), "cmp", [1], [0], rul.RULConfig(epochs=2, d=16),
                  **paths(root))  # fmt: skip
    reference = planb.run(
        "xjtu_sy", "stats", ("F", "E"), "planb-ref", [1], [0], TrainConfig(epochs=2, d=16),
        photos=False, **paths(root),
    )  # fmt: skip
    table = rul.compare(out, reference).set_index("quantity")
    assert set(table.index) == {"RMSE, this - reference", "RMSE after FPT, this - reference"}
    assert (table["low"] <= table["estimate"]).all() and (table["estimate"] <= table["high"]).all()
    assert (table["bags"] == 90).all()
    mine = pd.read_csv(out / "fold1-seed0-bags.csv")
    theirs = pd.read_csv(reference / "fold1-seed0-bags.csv")
    gap = np.sqrt(np.mean((mine["pred_reg"] - mine["y_reg"]) ** 2)) - np.sqrt(
        np.mean((theirs.set_index("bag").loc[mine["bag"], "pred_reg"].to_numpy()
                 - mine["y_reg"].to_numpy()) ** 2)
    )  # fmt: skip
    assert table.loc["RMSE, this - reference", "estimate"] == pytest.approx(gap, rel=1e-6)
    with pytest.raises(ValueError, match="no fold and seed in common"):
        rul.compare(out, root / "runs" / "nothing")


def test_command_line(root, tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(root))
    rul.main(
        ["--encoder", "stats", "--arms", "F", "--run-id", "cli", "--folds", "2", "--seeds", "0",
         "--history", "3", "--elapsed", "--absolute", "--epochs", "1", "--d", "16",
         "--out", str(tmp_path),
         "--force"]
    )  # fmt: skip
    train = json.loads((tmp_path / "cli" / "fold2-seed0.json").read_text())["settings"]["train"]
    assert (train["history"], train["elapsed"], train["relative"]) == (3, True, False)
    with pytest.raises(SystemExit):
        rul.main(["--encoder", "stats", "--run-id", "x", "--history", "0", "--force"])
