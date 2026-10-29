"""Per-modality token sets from the caches (pdmfm.model.inputs) on synthetic stores."""

from __future__ import annotations

import shutil
import time

import numpy as np
import pandas as pd
import pytest
from synthetic import FakeTextEncoder, build_planb_root

from pdmfm import features, protocols, texts
from pdmfm.encoders import StatsEncoder
from pdmfm.model import inputs as I
from pdmfm.windows import WindowStore


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("planb-inputs")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root)
        yield root


def assemble(root, dataset, arms, **kwargs):
    return I.assemble(
        dataset, "stats", arms, window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features", **kwargs,
    )  # fmt: skip


def store(root, dataset):
    return WindowStore.open(dataset, root / "interim" / "windows")


def test_nonastreda_force_and_photo_modalities(root):
    inputs = assemble(root, "nonastreda", ("T", "F"))
    assert list(inputs.modalities) == ["force", "photo"]
    force = inputs.modalities["force"]
    # T: 3 channels x 4 pooled tokens; F: 3 channels x 1 token.
    assert force.tokens.shape == (len(inputs.manifest), 15, StatsEncoder.dim)
    assert force.roles.tolist() == [I.ROLE["T"]] * 12 + [I.ROLE["F"]] * 3
    assert force.channels.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [0, 1, 2]
    assert force.present.all() and force.token_present is None


def test_cached_tokens_are_copied_unchanged(root):
    inputs = assemble(root, "nonastreda", ("T", "F"))
    cache = features.FeatureStore(root / "interim" / "features")
    s = store(root, "nonastreda")
    t_tokens = np.asarray(cache.find(s, "stats", "T").features, dtype=np.float32)
    f_tokens = np.asarray(cache.find(s, "stats", "F").features, dtype=np.float32)
    force = inputs.modalities["force"].tokens
    np.testing.assert_array_equal(force[:, :4], t_tokens[:, 0])  # fx, T arm
    np.testing.assert_array_equal(force[:, 12:], f_tokens[:, :, 0])


def test_no_fz_level_token_on_nonastreda(root):
    force = assemble(root, "nonastreda", ("T", "F")).modalities["force"]
    assert force.level_channels.tolist() == [0, 1]  # fx, fy; Fz drifts by block
    assert force.levels.shape == (len(force.present), 2, 4)


def test_levels_are_relative_to_each_tools_first_pass(root):
    s = store(root, "nonastreda")
    levels = I.relative_levels(s)
    keys, reference = I.baseline_groups("nonastreda", s.manifest)
    first_pass = s.manifest.groupby("unit")["meta_t_start"].transform("min")
    np.testing.assert_array_equal(reference, s.manifest["meta_t_start"] == first_pass)
    for unit in np.unique(keys):
        rows = (keys == unit) & reference
        np.testing.assert_allclose(np.median(levels[rows], axis=0), 0.0, atol=1e-5)
        expected = s.levels[keys == unit] - np.median(s.levels[rows], axis=0)
        np.testing.assert_allclose(levels[keys == unit], expected, rtol=1e-6)
    # The synthetic force level rises with wear, so later passes sit above the baseline.
    mean_level = levels[:, 0, :].mean(-1)
    assert mean_level[s.manifest["class_key"] == "dulled"].mean() > mean_level[reference].mean()


def test_xjtu_and_cwru_level_baselines(root):
    xjtu = store(root, "xjtu_sy")
    keys, reference = I.baseline_groups("xjtu_sy", xjtu.manifest)
    assert reference.sum() == 15 * 10 and set(keys) == set(xjtu.manifest["unit"])
    cwru = store(root, "cwru")
    keys, reference = I.baseline_groups("cwru", cwru.manifest)
    assert set(keys) == {"0", "1", "2", "3"}  # one group per load
    assert set(cwru.manifest.loc[reference, "class_key"]) == {"normal"}
    with pytest.raises(NotImplementedError):
        I.baseline_groups("phm2010", cwru.manifest)


def test_a_group_without_reference_windows_is_an_error(root):
    s = store(root, "cwru")
    s.manifest = s.manifest.assign(class_key=s.manifest["class_key"].replace("normal", "ball"))
    with pytest.raises(ValueError, match="no reference windows"):
        I.relative_levels(s)


def test_photos_of_the_previous_pass_with_their_age(root):
    inputs = assemble(root, "nonastreda", ("T", "F"))
    photo, manifest = inputs.modalities["photo"], inputs.manifest
    # 3 kinds x 5 tokens; the synthetic reader has tool and chip photos, no workpiece.
    assert photo.tokens.shape[1:] == (15, 8)
    assert (
        photo.roles.tolist()
        == [I.ROLE["tool"]] * 5 + [I.ROLE["chip"]] * 5 + [I.ROLE["workpiece"]] * 5
    )
    has_tool = (manifest["image_tool"] != "").to_numpy()
    first_pass = (
        manifest.groupby("unit")["meta_t_start"].transform("min") == manifest["meta_t_start"]
    )
    np.testing.assert_array_equal(has_tool, ~first_pass.to_numpy())  # nothing before pass 1
    np.testing.assert_array_equal(photo.present, has_tool)
    np.testing.assert_array_equal(photo.token_present[:, :5], np.repeat(has_tool[:, None], 5, 1))
    assert not photo.token_present[:, 10:].any()  # workpiece never attached
    np.testing.assert_array_equal(photo.tokens[~has_tool], 0.0)
    np.testing.assert_allclose(photo.age[has_tool, 0], manifest.loc[has_tool, "age_tool"])
    assert (photo.age >= 0).all()


def test_photo_tokens_come_from_the_cache_row_of_the_attached_path(root):
    inputs = assemble(root, "nonastreda", ("T", "F"))
    folder = next((root / "interim" / "features" / "images").glob("nonastreda-tool-*"))
    paths = pd.read_csv(folder / "paths.csv")["path"].tolist()
    cached = np.load(folder / "features.npy").astype(np.float32)
    row = int(np.flatnonzero(inputs.modalities["photo"].present)[3])
    wanted = inputs.manifest.loc[row, "image_tool"]
    np.testing.assert_array_equal(
        inputs.modalities["photo"].tokens[row, :5], cached[paths.index(wanted)]
    )


def test_photos_can_be_left_out(root):
    inputs = assemble(root, "nonastreda", ("T", "F"), photos=False)
    assert list(inputs.modalities) == ["force"]


def test_an_attached_photo_missing_from_the_cache_is_an_error(root, tmp_path):
    s = store(root, "nonastreda")
    s.manifest = s.manifest.copy()
    s.manifest.loc[s.manifest["image_chip"] != "", "image_chip"] = "raw/elsewhere.png"
    with pytest.raises(ValueError, match="not in the photo cache"):
        I._photo_modality(s, root / "interim" / "features")
    with pytest.raises(FileNotFoundError, match="photo cache"):
        I._photo_modality(s, tmp_path)


def test_targets_and_text_prototypes(root):
    inputs = assemble(root, "nonastreda", ("T", "F"))
    assert inputs.classes == list(texts.load("nonastreda").descriptions)
    np.testing.assert_allclose(np.linalg.norm(inputs.prototypes, axis=1), 1.0, rtol=1e-5)
    names = np.array(inputs.classes)[inputs.y_class]
    np.testing.assert_array_equal(names, inputs.manifest["class_key"])
    assert inputs.regression_target == "wear_um"
    np.testing.assert_allclose(inputs.y_reg, inputs.manifest["wear_um"])


def test_xjtu_has_rul_and_no_classes(root):
    inputs = assemble(root, "xjtu_sy", ("F", "E"))
    assert list(inputs.modalities) == ["vibration"]
    assert inputs.y_class is None and inputs.prototypes is None
    assert inputs.regression_target == "rul" and np.isfinite(inputs.y_reg).all()
    vibration = inputs.modalities["vibration"]
    assert vibration.roles.tolist() == [I.ROLE["F"]] * 2 + [I.ROLE["E"]] * 2


def test_arms_a_dataset_lacks_are_skipped(root):
    inputs = assemble(root, "nonastreda", ("T", "F", "E"))  # no envelope arm on milling
    assert inputs.modalities["force"].tokens.shape[1] == 15


def test_text_cache_with_other_classes_is_refused(root, tmp_path):
    folder = tmp_path / "texts"
    folder.mkdir()
    prototypes = FakeTextEncoder().encode(["a", "b", "c"])
    np.savez(
        folder / "nonastreda-qwen3_embedding-x.npz",
        classes=np.array(["dulled", "used", "sharp"]),
        prototypes=prototypes,
    )
    with pytest.raises(ValueError, match="differ from the YAML"):
        I._prototypes("nonastreda", tmp_path)
    with pytest.raises(FileNotFoundError, match="text prototypes"):
        I._prototypes("cwru", tmp_path)


def test_windows_without_cached_features_are_absent(root, tmp_path):
    """A partly extracted cache (e.g. a subsample) marks its missing rows absent everywhere."""
    s = store(root, "nonastreda")
    cache = features.FeatureStore(tmp_path / "features")
    half = np.arange(len(s)) % 2 == 0
    for arm in ("T", "F"):
        cache.extract(s, StatsEncoder(), arm, rows=half, log=lambda *_: None)
    for kind in ("images", "texts"):
        shutil.copytree(root / "interim" / "features" / kind, cache.root / kind)
    inputs = I.assemble(
        "nonastreda", "stats", ("T", "F"),
        window_root=root / "interim" / "windows", feature_root=cache.root,
    )  # fmt: skip
    np.testing.assert_array_equal(inputs.modalities["force"].present, half)
    assert not inputs.modalities["photo"].present[~half].any()


def test_the_description_names_the_inputs(root):
    inputs = assemble(root, "nonastreda", ("T", "F"))
    d = inputs.description
    assert d["window_store"] == store(root, "nonastreda").key
    assert d["modalities"]["force"] == {"tokens": 15, "dim": StatsEncoder.dim, "levels": 2}
    assert d["arms"] == ["T", "F"] and d["classes"] == inputs.classes


def test_newest_image_cache_is_used(root, tmp_path):
    shutil.copytree(root / "interim" / "features" / "images", tmp_path / "images")
    old = next((tmp_path / "images").glob("nonastreda-tool-*"))
    newer = tmp_path / "images" / "nonastreda-tool-dinov2-newer"
    newer.mkdir()
    (newer / "paths.csv").write_text("path\nraw/x.png\n")
    np.save(newer / "features.npy", np.ones((1, 5, 8), dtype=np.float16))
    time.sleep(0.05)
    (newer / "meta.json").write_text("{}")
    paths, cached = I._image_cache(tmp_path, "nonastreda", "tool")
    assert paths == ["raw/x.png"] and cached.shape == (1, 5, 8)
    assert old.exists()


# ----------------------------------------------------------------------------- Paderborn


def paderborn_train_units(root, fold=1):
    manifest = store(root, "paderborn").manifest
    return sorted(protocols.paderborn_fold(manifest, fold).units(manifest, "train"))


def test_paderborn_baseline_is_the_healthy_training_bearings_per_setting(root):
    s = store(root, "paderborn")
    with pytest.raises(ValueError, match="training bearings"):
        I.baseline_groups("paderborn", s.manifest)
    train = paderborn_train_units(root)
    keys, reference = I.baseline_groups("paderborn", s.manifest, train)
    np.testing.assert_array_equal(keys, s.manifest["setting"])
    referenced = set(s.manifest.loc[reference, "unit"])
    assert referenced == {u for u in train if u.startswith("K00")} == {"K003"}
    assert "K001" not in referenced  # fold 1's healthy test bearing


def test_paderborn_levels_and_slow_tokens_relative_to_the_reference(root):
    s = store(root, "paderborn")
    train = paderborn_train_units(root)
    keys, reference = I.baseline_groups("paderborn", s.manifest, train)
    levels, slow = I.relative_levels(s, train), I.relative_slow(s, train)
    for setting in np.unique(keys):
        rows = (keys == setting) & reference
        np.testing.assert_allclose(np.median(levels[rows], axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.median(slow[rows], axis=0), 0.0, atol=1e-5)
    # The synthetic temperature sits 3 K per class above healthy, whatever the setting.
    mean_slow = pd.Series(slow[:, 0, 0]).groupby(s.manifest["class_key"].to_numpy()).mean()
    assert mean_slow["healthy"] == pytest.approx(0.0, abs=0.2)
    assert mean_slow["outer_race"] == pytest.approx(3.0, abs=0.2)
    assert mean_slow["inner_race"] == pytest.approx(6.0, abs=0.2)


def test_paderborn_modalities_include_the_slow_channel(root):
    train = paderborn_train_units(root)
    inputs = assemble(root, "paderborn", ("F", "E"), train_units=train)
    assert list(inputs.modalities) == ["vibration", "current", "slow"]
    slow = inputs.modalities["slow"]
    assert slow.descriptor and slow.tokens.shape == (len(inputs.manifest), 1, 2)
    assert slow.roles.tolist() == [I.ROLE["slow"]]
    assert slow.channels.tolist() == [3]  # after the three high-rate channels
    assert inputs.description["baseline_units"] == train
    assert inputs.classes == ["healthy", "outer_race", "inner_race"]
    assert (inputs.y_class[inputs.manifest["unit"] == "KB27"] == -1).all()  # out of space


def test_other_datasets_ignore_train_units(root):
    inputs = assemble(root, "xjtu_sy", ("F",), train_units=["Bearing1_1"])
    assert inputs.description["baseline_units"] is None


def test_a_missing_slow_channel_is_masked(root):
    s = store(root, "paderborn")
    s.slow = s.slow.copy()
    s.slow[:5] = np.nan
    slow = I._slow_modality(s, paderborn_train_units(root))
    assert not slow.token_present[:5].any() and not slow.present[:5].any()
    assert slow.present[5:].all()
    np.testing.assert_array_equal(slow.tokens[:5], 0.0)
