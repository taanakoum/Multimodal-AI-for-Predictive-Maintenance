"""Gate-2 fusion candidates (pdmfm.model.fusion) and their training options in pdmfm.train,
on the synthetic stores and caches of tests/synthetic.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from synthetic import build_planb_root

from pdmfm import losses, planb, protocols
from pdmfm.model.fusion import FUSIONS, STAGES, FusionModel, build_model
from pdmfm.model.inputs import Modality, TaskInputs, assemble
from pdmfm.model.late import LateEvidential
from pdmfm.train import (
    TrainConfig,
    _drop_modalities,
    _losses,
    batch,
    fit,
    make_scheduler,
    predict,
)

CANDIDATES = [f for f in FUSIONS if f != "late"]
SMALL = TrainConfig(epochs=2, batch_size=32, d=32)


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fusion")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        mp.setattr(planb, "BOOTSTRAP_REPLICATES", 100)
        build_planb_root(root, datasets=("nonastreda", "matwi", "paderborn"))
        yield root


def load(root, dataset, arms=("T", "F"), photos=True, train_units=None):
    return assemble(
        dataset, "stats", arms, photos=photos, window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features", train_units=train_units,
    )  # fmt: skip


@pytest.fixture(scope="module")
def nona(root):
    return load(root, "nonastreda")


@pytest.fixture(scope="module")
def matwi(root):
    return load(root, "matwi")


def outputs(model, inputs, rows, drop=()):
    model.eval()
    with torch.no_grad():
        return model(batch(inputs, rows, torch.device("cpu"), drop))


def with_tokens(inputs: TaskInputs, name: str, tokens: np.ndarray) -> TaskInputs:
    changed = TaskInputs(**{**inputs.__dict__, "modalities": {**inputs.modalities}})
    changed.modalities[name] = Modality(**{**inputs.modalities[name].__dict__, "tokens": tokens})
    return changed


# ----------------------------------------------------------------------------- the candidates


@pytest.mark.parametrize("fusion", CANDIDATES)
def test_every_candidate_gives_finite_outputs_of_the_right_shape(matwi, fusion):
    """MATWI has four modalities (force, vibration, acoustic, photo) and both tasks."""
    torch.manual_seed(0)
    model = build_model(matwi, fusion, 32, 0.1, None, "prototype")
    assert isinstance(model, FusionModel) and model.fusion_name == fusion
    rows = np.arange(min(40, len(matwi.manifest)))
    out = outputs(model, matwi, rows)
    m, k = len(matwi.modalities), len(matwi.classes)
    assert out["evidence"].shape == (len(rows), m, k)
    assert out["fused_evidence"].shape == (len(rows), k) and out["combined"].shape == (len(rows), k)
    assert out["fused_own_nig"].shape == (len(rows), 4) and out["fused_nig"].shape == (len(rows), 4)
    for key in ("evidence", "fused_evidence", "combined", "fused_own_nig", "fused_nig"):
        assert torch.isfinite(out[key]).all(), key
    assert (out["fused_evidence"] >= 0).all()


def test_late_is_plan_b_and_unknown_names_are_refused(nona):
    model = build_model(nona, "late", 16, 0.1, None, "prototype")
    assert type(model) is LateEvidential
    with pytest.raises(ValueError, match="fusion must be one of"):
        build_model(nona, "gated", 16, 0.1, None, "prototype")


@pytest.mark.parametrize("fusion", CANDIDATES)
def test_tokens_of_an_absent_modality_never_reach_the_fusion(nona, fusion):
    """Rows whose photo is not attached, and every row with the photo dropped: the photo
    tokens may hold anything without changing any output."""
    torch.manual_seed(0)
    model = build_model(nona, fusion, 32, 0.1, None, "prototype")
    photo = nona.modalities["photo"]
    noisy = with_tokens(
        nona, "photo", np.random.default_rng(1).normal(0, 100, photo.tokens.shape).astype("f4")
    )
    absent = np.flatnonzero(~photo.present)
    every = np.arange(len(nona.manifest))
    for rows, drop in ((absent, ()), (every, ("photo",))):
        before, after = outputs(model, nona, rows, drop), outputs(model, noisy, rows, drop)
        for key in ("fused_evidence", "combined", "fused_own_nig", "fused_nig"):
            torch.testing.assert_close(after[key], before[key], msg=f"{fusion} {key} {drop}")


@pytest.mark.parametrize("fusion", CANDIDATES)
def test_the_fusion_stage_uses_a_present_photo(nona, fusion):
    """Not vacuous: with the photo present, its tokens change the fused opinion."""
    torch.manual_seed(0)
    model = build_model(nona, fusion, 32, 0.1, None, "prototype")
    rows = np.flatnonzero(nona.modalities["photo"].present)[:20]
    photo = nona.modalities["photo"]
    noisy = with_tokens(
        nona, "photo", np.random.default_rng(1).normal(0, 3, photo.tokens.shape).astype("f4")
    )
    before, after = outputs(model, nona, rows), outputs(model, noisy, rows)
    assert not torch.allclose(before["fused_evidence"], after["fused_evidence"])


@pytest.mark.parametrize("fusion", CANDIDATES)
def test_one_modality_alone_is_enough(matwi, fusion):
    torch.manual_seed(0)
    model = build_model(matwi, fusion, 32, 0.1, None, "prototype")
    rows = np.arange(20)
    others = tuple(n for n in matwi.modalities if n != "force")
    out = outputs(model, matwi, rows, drop=others)
    assert torch.isfinite(out["fused_evidence"]).all() and torch.isfinite(out["fused_nig"]).all()
    assert out["present"][:, 1:].sum() == 0


def test_the_fused_opinion_joins_the_combination(nona):
    torch.manual_seed(0)
    model = build_model(nona, "concat", 32, 0.1, None, "prototype")
    rows = np.flatnonzero(nona.modalities["photo"].present)[:20]
    out = outputs(model, nona, rows)
    stacked = torch.cat([out["fused_evidence"].unsqueeze(1), out["evidence"]], 1)
    torch.testing.assert_close(out["combined"], losses.ds_combine_all(stacked)[0])
    experts_only = losses.ds_combine_all(out["evidence"])[0]
    assert not torch.allclose(out["combined"], experts_only)
    flags = torch.cat([out["present"].any(1, keepdim=True), out["present"]], 1)
    nig = torch.cat([out["fused_own_nig"].unsqueeze(1), out["nig"]], 1)
    torch.testing.assert_close(out["fused_nig"], losses.nig_fuse_all(nig, flags))


def test_fusion_stages_fit_the_parameter_budget():
    """At d = 256 with the real token counts (Mantis T+F 26 + levels, DINOv2 3 kinds x 5,
    Qwen 1) every fusion stage stays within the plan's 3 M parameters."""
    n = 2
    modalities = {
        name: Modality(
            name, np.zeros((n, tokens, dim), np.float32), np.zeros(tokens, int),
            np.arange(tokens) % 3, np.ones(n, bool),
        )
        for name, tokens, dim in (
            ("force", 26, 256), ("vibration", 26, 256), ("photo", 15, 384), ("slow", 3, 256)
        )
    }  # fmt: skip
    inputs = TaskInputs(
        "x", pd.DataFrame(index=range(n)), modalities, ["a", "b", "c"],
        np.eye(3, 1024, dtype=np.float32), np.zeros(n, int), np.zeros(n),
    )  # fmt: skip
    for name in STAGES:
        model = FusionModel(inputs, name, d=256)
        stage = sum(p.numel() for p in model.stage.parameters())
        assert stage <= 3_000_000, (name, stage)
        out = model(batch(inputs, np.arange(n), torch.device("cpu")))
        assert torch.isfinite(out["combined"]).all()


def test_c2f_puts_sensors_in_the_hub(root):
    from pdmfm.windows import WindowStore

    manifest = WindowStore.open("paderborn", root / "interim" / "windows").manifest
    train = sorted(protocols.paderborn_fold(manifest, 1).units(manifest, "train"))
    inputs = load(root, "paderborn", train_units=train)
    model = FusionModel(inputs, "c2f", d=16)
    names = list(inputs.modalities)
    assert [names[i] for i in model.stage.hub] == [n for n in names if n not in ("photo", "slow")]
    assert "slow" in [names[i] for i in model.stage.others]


# ----------------------------------------------------------------------------- training


def test_input_dropout_never_removes_every_modality():
    rng = np.random.default_rng(0)
    present = rng.random((4000, 4)) < 0.7
    parts = {f"m{j}": {"present": torch.from_numpy(present[:, j].copy())} for j in range(4)}
    _drop_modalities(parts, 0.3, rng)
    kept = np.stack([parts[f"m{j}"]["present"].numpy() for j in range(4)], 1)
    assert not (kept & ~present).any()  # absent stays absent
    assert (kept.any(1) == present.any(1)).all()  # a row keeps at least one modality
    rate = 1 - kept[present].mean()
    assert 0.25 < rate < 0.32  # about p, slightly less because one is always kept
    parts = {f"m{j}": {"present": torch.from_numpy(present[:, j].copy())} for j in range(4)}
    _drop_modalities(parts, 1.0, rng)
    kept = np.stack([parts[f"m{j}"]["present"].numpy() for j in range(4)], 1)
    assert (kept.sum(1) == present.any(1)).all()  # exactly one left where any was present


def test_input_dropout_applies_to_training_batches_only(nona):
    rows = np.flatnonzero(nona.modalities["photo"].present)[:200]
    cfg = TrainConfig(input_dropout=0.5)
    trained = batch(nona, rows, torch.device("cpu"), augment=(cfg, np.random.default_rng(0)))
    plain = batch(nona, rows, torch.device("cpu"))
    assert bool(plain["photo"]["present"].all()) and bool(plain["force"]["present"].all())
    dropped = ~trained["photo"]["present"] | ~trained["force"]["present"]
    assert 0.3 < float(dropped.float().mean()) < 0.9
    assert bool((trained["photo"]["present"] | trained["force"]["present"]).all())


def test_cosine_schedule_anneals_to_zero_over_every_batch():
    weight = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([weight], lr=1e-3)
    cfg = TrainConfig(epochs=4, batch_size=10, schedule="cosine")
    scheduler = make_scheduler(optimizer, cfg, n_rows=25)  # 3 batches per epoch, 12 steps
    assert scheduler is not None
    rates = []
    for _ in range(12):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[5] == pytest.approx(5e-4, rel=1e-6) and rates[-1] == pytest.approx(0, abs=1e-12)
    assert make_scheduler(optimizer, TrainConfig(), 25) is None
    with pytest.raises(ValueError, match="schedule"):
        make_scheduler(optimizer, TrainConfig(schedule="step"), 25)


def test_loss_level_dropout_is_refused_for_fusion_candidates(nona):
    rows = np.arange(len(nona.manifest))
    cfg = TrainConfig(epochs=1, d=16, fusion="concat", modality_dropout=0.3)
    with pytest.raises(ValueError, match="input_dropout"):
        fit(nona, rows, rows, cfg, seed=0, log=quiet)


def test_fused_heads_add_their_own_losses(nona):
    """Both tasks: the total is Plan B's terms plus TMC on the fused opinion and the NIG
    loss on the fused head's own NIG (the regression term uses the NIG, not the evidence)."""
    torch.manual_seed(0)
    model = build_model(nona, "perceiver", 16, 0.1, None, "prototype")
    rows = np.flatnonzero(nona.y_class >= 0)[:64]
    out = model(batch(nona, rows, torch.device("cpu")))
    y_mean, y_std = 50.0, 20.0
    full = _losses(out, nona, rows, 1.0, SMALL, y_mean, y_std, "cpu")
    base = {k: v for k, v in out.items() if k not in ("fused_evidence", "fused_own_nig")}
    plain = _losses(base, nona, rows, 1.0, SMALL, y_mean, y_std, "cpu")
    y = torch.from_numpy(nona.y_class[rows])
    target = torch.from_numpy(((nona.y_reg[rows] - y_mean) / y_std).astype(np.float32))
    extra = (
        losses.tmc_loss(out["fused_evidence"], y, 1.0).mean()
        + losses.nig_loss(out["fused_own_nig"], target, SMALL.nig_regularizer).mean()
    )
    torch.testing.assert_close(full, plain + extra)


@pytest.mark.parametrize("fusion", CANDIDATES)
def test_each_candidate_learns_the_synthetic_wear(root, fusion):
    """MATWI's synthetic force level and vibration grow with wear (68 cuts, one window
    each; the second cut's 70 +- 3 um straddles the 71 um class edge, so some labels are
    noisy). With the search recipe (input dropout 0.3, cosine schedule) every candidate must
    beat the majority class clearly on the held-out tools and halve the training mean's wear
    error. The fused head's own estimate must learn too (below 0.6 of that error; it lags
    the experts on 50 training cuts, and MoNIG weights it by its evidence: seeds 0-2 gave
    0.19-0.48 of the training mean's error)."""
    inputs = load(root, "matwi")
    split = protocols.matwi_fold(inputs.manifest, 1)
    cfg = TrainConfig(
        epochs=40, batch_size=16, d=32, lr=3e-3, fusion=fusion, input_dropout=0.3,
        schedule="cosine",
    )  # fmt: skip
    train, val, test = (np.flatnonzero(m) for m in (split.train, split.val, split.test))
    fitted = fit(inputs, train, val, cfg, seed=0, log=quiet)
    pred = predict(fitted.model, inputs, test, cfg)
    result = planb.evaluate(inputs, test, pred, fitted.y_mean, fitted.y_std, None)
    reference = planb.baselines(inputs, train, test)
    assert reference["majority"]["balanced_accuracy"] == pytest.approx(1 / 3)
    assert result["combined"]["bag"]["balanced_accuracy"] >= 0.6
    limit = reference["train_mean"]["rmse"] / 2
    assert result["regression"]["combined"]["bag"]["rmse"] < limit
    assert result["regression"]["only_fused"]["bag"]["rmse"] < 0.6 * reference["train_mean"]["rmse"]


# ----------------------------------------------------------------------------- Plan B runs


def test_plan_b_runs_a_fusion_candidate(root, tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(root))
    planb.main(
        ["--dataset", "nonastreda", "--encoder", "stats", "--arms", "T", "F", "--run-id", "c2f",
         "--folds", "1", "--seeds", "0", "--epochs", "2", "--d", "16", "--fusion", "c2f",
         "--input-dropout", "0.3", "--schedule", "cosine", "--lr", "3e-4",
         "--out", str(tmp_path), "--force"]
    )  # fmt: skip
    result = json.loads((tmp_path / "c2f" / "fold1-seed0.json").read_text(encoding="utf-8"))
    train = result["settings"]["train"]
    assert (train["fusion"], train["input_dropout"], train["schedule"]) == ("c2f", 0.3, "cosine")
    test = result["test"]
    assert {"only_fused", "only_force", "only_photo", "without_force", "without_photo"} <= set(test)
    assert "only_fused" in test["regression"] and "trust_fused" in test["trust"]
    # With a fusion stage, removing the photo is not the force expert alone: the fused
    # opinion (from force only) still joins the combination.
    assert test["without_photo"] != test["only_force"]
    assert "without_photo" in result["validation"]


def test_without_each_skips_rows_left_with_nothing(nona):
    """Without the force, rows that had no photo have no input; they are not scored."""
    torch.manual_seed(0)
    model = LateEvidential(nona, d=16)
    rows = np.arange(len(nona.manifest))
    pred = predict(model, nona, rows, SMALL)
    dropped = planb.without_each(model, nona, rows, SMALL)
    result = planb.evaluate(nona, rows, pred, 0.0, 1.0, None, dropped=dropped)
    with_photo = int((nona.modalities["photo"].present & (nona.y_class >= 0)).sum())
    assert result["without_force"]["window"]["n"] == with_photo
    without, photo = {}, {}  # Dempster's rule with a vacuous opinion: equal
    planb._flatten("", result["without_force"], without)
    planb._flatten("", result["only_photo"], photo)
    assert without == pytest.approx(photo)
