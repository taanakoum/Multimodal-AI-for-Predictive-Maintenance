"""Stage-6 pretraining (pdmfm.pretrain) on the synthetic source datasets.

What must hold: pretraining never reads a fold's test or inner-validation units nor a target;
the backbone is shared across datasets (one expert per family); hidden tokens and dropped
modalities cannot leak into what predicts them; the contrastive views are what they claim
and never use a window of the same unit as a negative; datasets are drawn in proportion to
the square root of their size with units balanced; runs resume; fine-tuning loads a backbone
only for its own fold.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from synthetic import build_planb_root

from pdmfm import planb, pretrain, protocols
from pdmfm.model.inputs import assemble
from pdmfm.model.late import LateEvidential
from pdmfm.train import TrainConfig, batch, build_model
from pdmfm.windows import WindowStore

TINY = pretrain.PretrainConfig(epochs=2, batch_size=16, d=32, steps_per_epoch=8)


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("pretrain")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root)
        yield root


def paths(root):
    return {
        "window_root": root / "interim" / "windows",
        "feature_root": root / "interim" / "features",
    }


@pytest.fixture(scope="module")
def sources(root):
    return pretrain.load_sources(pretrain.SOURCES, "stats", ("T", "F", "E"), 1, log=quiet,
                                 **paths(root))  # fmt: skip


@pytest.fixture(scope="module")
def pre(sources):
    torch.manual_seed(0)
    return pretrain.Pretrainer(sources, TINY)


# ----------------------------------------------------------------------------- data


def test_pretraining_never_reads_a_folds_test_or_validation_units(root, sources):
    assert [s.name for s in sources] == list(pretrain.SOURCES)
    for source in sources:
        manifest = source.inputs.manifest
        unit = manifest["unit"].astype(str).to_numpy()
        allowed = protocols.pretraining_mask(source.name, manifest, 1)
        assert allowed[source.train_rows].all()
        if source.name == "cwru":
            assert set(manifest.loc[source.train_rows, "load_hp"]) == {0, 1, 2}
            assert set(manifest.loc[source.val_rows, "load_hp"]) == {3}
            continue
        split = protocols.SOURCE_FOLDS[source.name](manifest, 1)
        held_out = set(unit[split.test]) | set(unit[split.val])
        assert not held_out & set(unit[source.train_rows])
        assert set(unit[source.val_rows]) == set(unit[split.val])
    with pytest.raises(ValueError, match="held-out target"):
        pretrain.load_sources(("kaist",), "stats", ("F",), 1, log=quiet, **paths(root))


def test_unlabelled_windows_are_pretrained_on_but_not_validated(root, monkeypatch):
    """MATWI has cuts without a wear label (no photo before the next cut): self-supervised
    losses use them, the checkpoint choice cannot."""
    original = pretrain.assemble

    def unlabelled(dataset, *args, **kwargs):
        inputs = original(dataset, *args, **kwargs)
        unit = inputs.manifest["unit"].astype(str).to_numpy()
        gone = np.isin(unit, ["Set03", "Set02"])  # fold 1: a training tool, a validation tool
        assert inputs.y_class is not None and inputs.y_reg is not None
        inputs.y_class = np.where(gone, -1, inputs.y_class)
        inputs.y_reg = np.where(gone, np.nan, inputs.y_reg)
        return inputs

    monkeypatch.setattr(pretrain, "assemble", unlabelled)
    (matwi,) = pretrain.load_sources(("matwi",), "stats", ("T", "F"), 1, log=quiet, **paths(root))
    unit = matwi.inputs.manifest["unit"].astype(str).to_numpy()
    assert "Set03" in set(unit[matwi.train_rows])  # unlabelled, still pretrained on
    assert "Set02" not in set(unit[matwi.val_rows])  # never scored for the checkpoint
    assert len(matwi.val_rows) > 0


def test_datasets_are_drawn_by_square_root_size_with_units_balanced(sources):
    weights = pretrain.dataset_weights(sources)
    sizes = np.array([len(s.train_rows) for s in sources], dtype=float)
    np.testing.assert_allclose(weights, np.sqrt(sizes) / np.sqrt(sizes).sum())
    rng = np.random.default_rng(0)
    xjtu = next(s for s in sources if s.name == "xjtu_sy")
    rows = np.concatenate([pretrain.sample_rows(xjtu, 64, rng) for _ in range(50)])
    assert set(rows) <= set(xjtu.train_rows)
    counts = pd.Series(xjtu.units[rows]).value_counts()
    assert counts.min() / counts.max() > 0.7  # every training bearing about equally often


# ----------------------------------------------------------------------------- the backbone


def test_one_expert_per_family_is_shared_by_every_dataset(pre, sources):
    by_name = {s.name: s.model for s in sources}
    vibration = pre.experts["vibration"]
    for name in ("cwru", "xjtu_sy", "matwi", "paderborn"):
        assert by_name[name].encoders["vibration"] is vibration
    assert by_name["nonastreda"].encoders["force"] is by_name["matwi"].encoders["force"]
    nona_head = by_name["nonastreda"].class_heads["force"]
    matwi_head = by_name["matwi"].class_heads["force"]
    assert nona_head.project is matwi_head.project  # the text projection W_q is shared
    assert nona_head.bias is not matwi_head.bias  # each label space keeps its own bias
    assert pre.families == ["vibration", "force", "current", "acoustic", "slow", "photo"]


def test_a_step_on_one_dataset_updates_what_another_uses(pre, sources):
    cwru = next(s for s in sources if s.name == "cwru")
    before = pre.experts["vibration"].adapter[1].weight.detach().clone()
    optimizer = torch.optim.SGD(pre.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(0)
    rows = cwru.train_rows[:8]
    losses = pretrain.step_losses(pre, cwru, rows, TINY, 1.0, generator, TINY.train_config())
    sum(losses.values()).backward()
    optimizer.step()
    optimizer.zero_grad()
    after = next(s for s in sources if s.name == "xjtu_sy").model.encoders["vibration"]
    assert not torch.equal(after.adapter[1].weight, before)


# ----------------------------------------------------------------------------- masking


def _parts(source, rows):
    return batch(source.inputs, rows, torch.device("cpu"))


def test_masking_drops_modalities_but_never_all_and_hides_present_tokens(sources):
    nona = next(s for s in sources if s.name == "nonastreda")
    rows = nona.train_rows[:64]
    parts = _parts(nona, rows)
    original = {n: p["present"].clone() for n, p in parts.items()}
    hidden = pretrain.mask_parts(parts, 0.4, 0.3, torch.Generator().manual_seed(0))
    kept = torch.stack([hidden[n][2] for n in parts], 1)
    had = torch.stack([original[n] for n in parts], 1)
    assert (kept.any(1) == had.any(1)).all()  # a row keeps at least one modality
    assert (kept <= had).all() and (kept < had).any()  # some modalities dropped
    force_hidden, _, force_kept = hidden["force"]
    hide = parts["force"]["hide"]
    assert not hide[~force_kept].any()  # a dropped modality's tokens are not "hidden" but gone
    assert force_hidden[~force_kept & original["force"]].all()  # ...and must be predicted
    share = hide[force_kept].float().mean().item()
    assert 0.25 < share < 0.55
    photo = parts["photo"]
    assert not photo["hide"][~photo["token_present"]].any()  # absent photo tokens never hidden


def test_hidden_content_and_dropped_modalities_do_not_reach_the_context(pre, sources):
    """Whatever the hidden tokens and the dropped modalities contain, the summaries the
    decoder reads (and the model's outputs) are the same: nothing can leak."""
    nona = next(s for s in sources if s.name == "nonastreda")
    model = nona.model
    pre.eval()
    rows = nona.train_rows[:32]
    parts = _parts(nona, rows)
    hidden = pretrain.mask_parts(parts, 0.4, 0.5, torch.Generator().manual_seed(1))
    with torch.no_grad():
        first = model(parts)
        changed = pretrain._copy(parts)
        for name, part in changed.items():
            noise = torch.randn_like(part["tokens"]) * 100
            gone = (part["hide"] | ~hidden[name][2].unsqueeze(1)).unsqueeze(-1)
            part["tokens"] = torch.where(gone, noise, part["tokens"])
        second = model(changed)
    kept = torch.stack([hidden[n][2] for n in model.names], 1)
    torch.testing.assert_close(first["z"][kept], second["z"][kept])
    torch.testing.assert_close(first["combined"], second["combined"])
    pre.train()


def test_the_mask_loss_is_finite_and_covers_tokens_levels_and_slow_channels(pre, sources):
    for name in ("nonastreda", "paderborn"):
        source = next(s for s in sources if s.name == name)
        parts = _parts(source, source.train_rows[:32])
        hidden = pretrain.mask_parts(parts, 0.4, 0.3, torch.Generator().manual_seed(2))
        out = source.model(parts)
        loss = pretrain.mask_loss(pre, source.model, parts, out, hidden)
        assert torch.isfinite(loss) and loss > 0


# ----------------------------------------------------------------------------- contrastive views


def test_views_hold_what_they_claim(pre, sources):
    cwru = next(s for s in sources if s.name == "cwru")
    parts = _parts(cwru, cwru.train_rows[:8])
    spec_a, spec_b = pretrain.VIEWS["cwru"]
    a = pretrain.view_parts(parts, spec_a, cwru, cwru.model)["vibration"]
    b = pretrain.view_parts(parts, spec_b, cwru, cwru.model)["vibration"]
    channels = cwru.model.channels_vibration
    de, fe = cwru.channel_names.index("de"), cwru.channel_names.index("fe")
    assert set(channels[a["token_present"][0]].tolist()) == {de}
    assert set(channels[b["token_present"][0]].tolist()) == {fe}
    level_channels = cwru.model.level_channels_vibration
    assert set(level_channels[a["level_present"][0]].tolist()) == {de}
    nona = next(s for s in sources if s.name == "nonastreda")
    parts = _parts(nona, nona.train_rows[:8])
    force_view = pretrain.view_parts(parts, pretrain.VIEWS["nonastreda"][0], nona, nona.model)
    assert not force_view["photo"]["present"].any() and force_view["force"]["present"].all()


def test_info_nce_never_uses_the_same_unit_as_a_negative(pre):
    torch.manual_seed(0)
    a = torch.randn(4, 32)
    units = np.array(["u1", "u1", "u2", "u3"])
    aligned = pretrain.info_nce(pre, a, a, units)
    shuffled = pretrain.info_nce(pre, a, a[[1, 0, 3, 2]], units)
    assert aligned < shuffled
    # Rows 0 and 1 share a unit: their mutual similarity is not a negative of either.
    twin = a.clone()
    twin[1] = twin[0]
    same_unit = pretrain.info_nce(pre, twin, twin, units)
    other_unit = pretrain.info_nce(pre, twin, twin, np.array(["u1", "u4", "u2", "u3"]))
    assert same_unit < other_unit


# ----------------------------------------------------------------------------- runs


def run(root, run_id, cfg=TINY, folds=(1,), log=quiet):
    return pretrain.run("stats", ("T", "F", "E"), run_id, list(folds), [0], cfg,
                        out_root=root / "runs", log=log, **paths(root))  # fmt: skip


def test_a_run_writes_its_files_and_resumes(root):
    out = run(root, "pre")
    folder = out / "fold1-seed0"
    for name in ("best.pt", "last.pt", "history.csv", "summary.json", "progress.json"):
        assert (folder / name).is_file()
    history = pd.read_csv(folder / "history.csv")
    assert len(history) == 2 and history[["loss_sup", "loss_mask", "loss_nce"]].notna().all().all()
    assert {"val_score", "z_std", "tau", "val_nonastreda", "val_cwru"} <= set(history)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["fold"] == 1 and summary["datasets"] == list(pretrain.SOURCES)
    assert abs(sum(summary["dataset_weights"].values()) - 1) < 1e-9
    assert "| 1 | 0 |" in (out / "summary.md").read_text(encoding="utf-8")
    messages = []
    run(root, "pre", log=messages.append)
    assert any("done earlier" in m for m in messages)
    with pytest.raises(ValueError, match="other settings"):
        run(root, "pre", cfg=pretrain.PretrainConfig(epochs=3, batch_size=16, d=32,
                                                     steps_per_epoch=8))  # fmt: skip


def test_an_interrupted_run_resumes_after_its_last_epoch(root):
    """A job stopped at its time limit after epoch 1 continues with epoch 2."""
    cfg = pretrain.PretrainConfig(epochs=2, batch_size=16, d=32, steps_per_epoch=4)

    class Stop(Exception):
        pass

    def stop_after_first(message):
        if "epoch 1/2" in message:
            raise Stop

    with pytest.raises(Stop):
        run(root, "cut", cfg=cfg, log=stop_after_first)
    folder = root / "runs" / "cut" / "fold1-seed0"
    assert torch.load(folder / "last.pt", weights_only=False)["epoch"] == 1
    assert not (folder / "summary.json").is_file()
    messages = []
    run(root, "cut", cfg=cfg, log=messages.append)
    assert any("resuming after epoch 1" in m for m in messages)
    assert len(pd.read_csv(folder / "history.csv")) == 2


def test_loss_ablations_switch_terms_off(root):
    cfg = pretrain.PretrainConfig(epochs=1, batch_size=16, d=32, steps_per_epoch=6, w_nce=0.0,
                                  w_mask=0.0)  # fmt: skip
    history = pd.read_csv(run(root, "sup-only", cfg=cfg) / "fold1-seed0" / "history.csv")
    assert history["loss_mask"].isna().all() and history["loss_nce"].isna().all()


def test_a_fusion_stage_is_pretrained_across_datasets(root, sources):
    cfg = pretrain.PretrainConfig(epochs=1, batch_size=16, d=32, steps_per_epoch=6, fusion="c2f")
    pre = pretrain.Pretrainer(sources, cfg)
    assert all(s.model.stage is pre.stage for s in sources)
    assert all(s.model.stage_names == pre.families for s in sources)
    out = run(root, "c2f", cfg=cfg)
    assert (out / "fold1-seed0" / "best.pt").is_file()


# ----------------------------------------------------------------------------- fine-tuning


def test_fine_tuning_starts_from_the_backbone_of_its_own_fold(root):
    out = run(root, "pre")
    checkpoint = out / "fold1-seed0" / "best.pt"
    backbone = torch.load(checkpoint, weights_only=False)["backbone"]
    inputs = assemble("nonastreda", "stats", ("T", "F"), **paths(root))
    model = build_model(inputs, "late", 32, 0.1, np.arange(len(inputs.manifest)), "prototype")
    report = pretrain.initialize(model, checkpoint, "nonastreda", fold=1, encoder="stats",
                                 log=quiet)  # fmt: skip
    assert report["loaded"] > 0
    expected = backbone["models"]["nonastreda"]["encoders.force.adapter.1.weight"]
    torch.testing.assert_close(model.encoders["force"].adapter[1].weight.detach(), expected)
    with pytest.raises(ValueError, match="must use its own"):
        pretrain.initialize(model, checkpoint, "nonastreda", fold=2, log=quiet)
    with pytest.raises(ValueError, match="pretrained on stats"):
        pretrain.initialize(model, checkpoint, "nonastreda", fold=1, encoder="mantis", log=quiet)
    fused = build_model(inputs, "concat", 32, 0.1, np.arange(len(inputs.manifest)), "prototype")
    with pytest.raises(ValueError, match="pretrained a late stage"):
        pretrain.initialize(fused, checkpoint, "nonastreda", fold=1, log=quiet)


def test_a_dataset_outside_pretraining_takes_the_shared_experts(root):
    """A held-out target (here KAIST-like: a dataset the backbone never saw) gets the shared
    vibration expert and text projection; any fold's backbone may serve it."""
    out = run(root, "pre")
    checkpoint = out / "fold1-seed0" / "best.pt"
    backbone = torch.load(checkpoint, weights_only=False)["backbone"]
    backbone["models"].pop("cwru")
    reduced = root / "runs" / "pre" / "without-cwru.pt"
    meta = torch.load(checkpoint, weights_only=False)["meta"]
    torch.save({"meta": {**meta, "datasets": [d for d in meta["datasets"] if d != "cwru"]},
                "backbone": backbone}, reduced)  # fmt: skip
    inputs = assemble("cwru", "stats", ("T", "F", "E"), **paths(root))
    model = build_model(inputs, "late", 32, 0.1, np.arange(len(inputs.manifest)), "prototype")
    assert isinstance(model, LateEvidential)
    pretrain.initialize(model, reduced, "cwru", fold=4, log=quiet)
    torch.testing.assert_close(
        model.encoders["vibration"].adapter[1].weight.detach(),
        backbone["experts"]["vibration"]["adapter.1.weight"],
    )


def test_plan_b_fine_tunes_each_fold_from_its_own_backbone(root):
    run(root, "pre-two", folds=(1, 2))
    template = str(root / "runs" / "pre-two" / "fold{fold}-seed{seed}" / "best.pt")
    out = planb.run(
        "nonastreda", "stats", ("T", "F"), "fine", [1, 2], [0],
        TrainConfig(epochs=2, batch_size=32, d=32, init=template), out_root=root / "runs",
        log=quiet, **paths(root),
    )  # fmt: skip
    for fold in (1, 2):
        result = json.loads((out / f"fold{fold}-seed0.json").read_text(encoding="utf-8"))
        assert result["settings"]["train"]["init"] == template
    manifest = WindowStore.open("nonastreda", paths(root)["window_root"]).manifest
    assert protocols.nonastreda_fold(manifest, 1).test.any()


def test_the_backbone_fixes_width_and_head_while_tuning_keeps_the_recipe(root):
    """Ruche tunes d over {64, 256} from scratch, per dataset and fold; one backbone (d 256)
    serves every dataset. Fine-tuning takes the backbone's width and head, keeps the tuned
    learning rate and dropout, and records what ran."""
    run(root, "pre-two", folds=(1, 2))
    template = str(root / "runs" / "pre-two" / "fold{fold}-seed{seed}" / "best.pt")
    messages = []
    out = planb.run(
        "nonastreda", "stats", ("T", "F"), "fine-tuned-width", [1], [0],
        TrainConfig(epochs=1, batch_size=32, d=32, head="projected", init=template),
        out_root=root / "runs", log=messages.append,
        per_fold={1: {"d": 16, "lr": 3e-4, "dropout": 0.3}}, **paths(root),
    )  # fmt: skip
    train = json.loads((out / "fold1-seed0.json").read_text(encoding="utf-8"))["settings"]["train"]
    assert (train["d"], train["head"]) == (32, "prototype")
    assert (train["lr"], train["dropout"]) == (3e-4, 0.3)
    assert any("from the backbone instead of {'d': 16, 'head': 'projected'}" in m for m in messages)
    assert any("pretrained tensors loaded" in m for m in messages)
    unchanged = TrainConfig(d=32, init=template)
    assert pretrain.architecture(unchanged, 1, 0, log=quiet) == unchanged
