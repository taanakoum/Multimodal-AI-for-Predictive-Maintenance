"""Channel descriptions and their embedding (pdmfm.sensors, pdmfm.model.layers.ChannelEmbedding).

The purpose (plan, "Sensor tokens"; PROJECT_LOG week 7, E2): a token's channel vector depends
only on what the channel physically is, so the same kind of sensor means the same thing in
every dataset, and no two channels of one dataset look alike. The tests check the catalogue
against the readers, the cross-dataset identity of physically identical channels, that the
model's output depends on the descriptions and not on the channels' order, and the token
positions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from synthetic import (
    build_planb_root,
    cwru_reader,
    matwi_reader,
    nonastreda_reader,
    paderborn_reader,
    xjtu_reader,
)

from pdmfm import sensors
from pdmfm.model.inputs import Modality, TaskInputs, assemble
from pdmfm.model.late import LateEvidential
from pdmfm.model.layers import ChannelEmbedding
from pdmfm.train import batch
from pdmfm.windows import SPECS


def describe(dataset: str, image: bool = False) -> list[sensors.ChannelInfo]:
    spec = SPECS[dataset]
    rate = 1000.0 if spec.window == 8192 else 12_000.0
    return sensors.describe(dataset, spec.channels, spec.slow, rate, spec.window / rate, image)


# ----------------------------------------------------------------------------- the catalogue


@pytest.mark.parametrize("dataset", sorted(SPECS))
def test_every_store_channel_is_described_once(dataset):
    infos = describe(dataset, image=True)
    spec = SPECS[dataset]
    assert len(infos) == len(spec.channels) + len(spec.slow) + 1
    keys = [(i.family, i.axis, i.location, i.site) for i in infos]
    assert len(set(keys)) == len(keys)  # no two channels of a dataset look alike
    assert infos[-1].family == "image"
    assert all(i.family != "other" for i in infos)


@pytest.mark.parametrize(
    "dataset, reader",
    [
        ("cwru", cwru_reader),
        ("xjtu_sy", lambda: xjtu_reader(3)),
        ("nonastreda", nonastreda_reader),
        ("paderborn", paderborn_reader),
        ("matwi", lambda: matwi_reader(cuts=1)),
    ],
)
def test_the_catalogue_matches_the_synthetic_readers(dataset, reader):
    fake = reader()
    recording = fake.load(fake.index()[0])
    for name, (family, axis, placement) in sensors.CHANNELS[dataset].items():
        if name not in recording.channels:
            continue
        channel = recording.channels[name]
        assert (channel.family, channel.axis, channel.placement) == (family, axis, placement)


REAL = ["cwru", "kaist", "xjtu_sy", "nonastreda", "matwi", "paderborn", "phm2010"]


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(
            d, marks=pytest.mark.data(d, extracted=d in ("xjtu_sy", "paderborn", "phm2010"))
        )
        for d in REAL
    ],
)
def test_the_catalogue_matches_the_real_readers(dataset):
    from pdmfm.datasets import get_reader

    reader = get_reader(dataset)
    recording = reader.load(reader.index()[0])
    used = set(SPECS[dataset].channels) | set(SPECS[dataset].slow)
    for name in used:
        channel = recording.channels[name]
        family, axis, placement = sensors.CHANNELS[dataset][name]
        assert (channel.family, channel.axis, channel.placement) == (family, axis, placement), name


def test_sites_rank_channels_that_share_family_axis_and_location():
    cwru = {i.site for i in describe("cwru")}
    assert cwru == {0, 1}  # drive end and fan end: both bearing-housing accelerometers
    kaist = describe("kaist")
    names = SPECS["kaist"].channels + SPECS["kaist"].slow
    by_name = dict(zip(names, kaist, strict=True))
    assert by_name["vib_ax"].site == 0 and by_name["vib_bx"].site == 1
    assert by_name["temp_1"].site == 0 and by_name["temp_2"].site == 1
    assert by_name["temp_1"].rate_hz == sensors.SLOW_RATE_HZ


def test_unknown_channels_and_places_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="not in pdmfm.sensors.CHANNELS"):
        sensors.describe("cwru", ("de", "ba"), (), 12_000.0, 1.024)
    with pytest.raises(ValueError, match="no channel catalogue"):
        sensors.describe("cmapss", ("s1",), (), 1.0, 1.0)
    monkeypatch.setitem(sensors.CHANNELS, "odd", {"a": ("vibration", "x", "gearbox")})
    with pytest.raises(ValueError, match="no location class"):
        sensors.describe("odd", ("a",), (), 12_000.0, 1.024)
    with pytest.raises(ValueError, match="at most"):
        sensors.generic(sensors.MAX_SITES + 1)


def test_the_scale_carries_each_tokens_physical_duration():
    force, vibration = describe("nonastreda")[0], describe("xjtu_sy")[0]
    assert 512 / force.rate_hz == pytest.approx(0.512)  # a T segment at 1 kHz
    assert 512 / vibration.rate_hz == pytest.approx(0.0427, abs=1e-4)  # at 12 kHz
    assert min(5000.0, vibration.rate_hz / 2) / 512 == pytest.approx(9.77, abs=0.01)  # F bin
    assert min(5000.0, force.rate_hz / 2) / 512 == pytest.approx(0.977, abs=0.001)
    assert force.scale() == pytest.approx((0.0, np.log10(8.192)))
    assert vibration.scale() == pytest.approx((np.log10(12.0), np.log10(1.024)))


# ----------------------------------------------------------------------------- the embedding


def embed(module: ChannelEmbedding, infos) -> torch.Tensor:
    codes, scale = sensors.table(infos)
    return module(torch.from_numpy(codes), torch.from_numpy(scale))


def test_physically_identical_channels_get_identical_vectors_across_datasets():
    torch.manual_seed(0)
    module = ChannelEmbedding(16)
    torch.nn.init.normal_(module.scale.weight)  # as after training: the scale matters
    xjtu = dict(zip(SPECS["xjtu_sy"].channels, embed(module, describe("xjtu_sy")), strict=True))
    kaist_names = SPECS["kaist"].channels + SPECS["kaist"].slow
    kaist = dict(zip(kaist_names, embed(module, describe("kaist")), strict=True))
    torch.testing.assert_close(xjtu["vib_h"], kaist["vib_ax"])  # x accelerometer on a housing
    torch.testing.assert_close(xjtu["vib_v"], kaist["vib_ay"])
    nona = dict(zip(SPECS["nonastreda"].channels, embed(module, describe("nonastreda")),
                    strict=True))  # fmt: skip
    matwi = dict(zip(SPECS["matwi"].channels, embed(module, describe("matwi")), strict=True))
    for axis in ("fx", "fy", "fz"):  # the same table dynamometer axes, both at 1 kHz
        torch.testing.assert_close(nona[axis], matwi[axis])
    cwru = dict(zip(SPECS["cwru"].channels, embed(module, describe("cwru")), strict=True))
    assert not torch.allclose(cwru["de"], nona["fx"])  # an accelerometer is not a force axis
    assert not torch.allclose(cwru["de"], cwru["fe"])  # two sites of one dataset differ


def test_the_sampling_rate_separates_otherwise_identical_channels_once_learned():
    module = ChannelEmbedding(16)
    slow = sensors.ChannelInfo("vibration", "none", "spindle", 0, 1000.0, 8.192)
    fast = sensors.ChannelInfo("vibration", "none", "spindle", 0, 12_000.0, 1.024)
    first = embed(module, [slow, fast])
    torch.testing.assert_close(first[0], first[1])  # the scale map starts at zero
    first.sum().backward()
    assert module.scale.weight.grad is not None and module.scale.weight.grad.abs().sum() > 0
    torch.nn.init.normal_(module.scale.weight)
    later = embed(module, [slow, fast])
    assert not torch.allclose(later[0], later[1])


# ----------------------------------------------------------------------------- the model


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("sensors")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root, datasets=("nonastreda",))
        yield root


@pytest.fixture(scope="module")
def nona(root):
    return assemble(
        "nonastreda", "stats", ("T", "F"), window_root=root / "interim" / "windows",
        feature_root=root / "interim" / "features",
    )  # fmt: skip


def outputs(model, inputs, rows):
    model.eval()
    with torch.no_grad():
        return model(batch(inputs, rows, torch.device("cpu")))


def renumbered(inputs: TaskInputs, order: list[int]) -> TaskInputs:
    """The same data with the channel table in ``order``: every token keeps its own
    description under a new index (as another dataset might number the same sensors)."""
    new_index = {old: new for new, old in enumerate(order)}
    modalities = {}
    for name, m in inputs.modalities.items():
        remap = np.vectorize(new_index.get)
        level_channels = None if m.level_channels is None else remap(m.level_channels)
        modalities[name] = Modality(**{**m.__dict__, "channels": remap(m.channels),
                                       "level_channels": level_channels})  # fmt: skip
    assert inputs.channels is not None
    channels = [inputs.channels[old] for old in order]
    return TaskInputs(**{**inputs.__dict__, "modalities": modalities, "channels": channels})


def parameters_only(model):
    names = {n for n, _ in model.named_parameters()}
    return {k: v for k, v in model.state_dict().items() if k in names}


def test_the_output_depends_on_descriptions_not_on_channel_order(nona):
    torch.manual_seed(0)
    model = LateEvidential(nona, d=16)
    rows = np.arange(40)
    before = outputs(model, nona, rows)
    assert nona.channels is not None
    order = list(range(len(nona.channels)))[::-1]  # Fx last, the image entry first
    again = LateEvidential(renumbered(nona, order), d=16)
    missing, _ = again.load_state_dict(parameters_only(model), strict=False)
    assert not [k for k in missing if k in parameters_only(again)]
    after = outputs(again, renumbered(nona, order), rows)
    for key in ("evidence", "combined", "fused_nig"):
        torch.testing.assert_close(after[key], before[key])


def test_swapping_descriptions_changes_the_output(nona):
    torch.manual_seed(0)
    model = LateEvidential(nona, d=16)
    rows = np.arange(40)
    before = outputs(model, nona, rows)
    assert nona.channels is not None
    swapped = list(nona.channels)
    swapped[0], swapped[1] = swapped[1], swapped[0]  # Fx tokens now described as Fy
    changed = TaskInputs(**{**nona.__dict__, "channels": swapped})
    again = LateEvidential(changed, d=16)
    again.load_state_dict(parameters_only(model), strict=False)
    after = outputs(again, changed, rows)
    assert not torch.allclose(after["evidence"][:, 0], before["evidence"][:, 0])


def test_photo_tokens_and_token_positions(nona):
    assert nona.channels is not None
    photo, force = nona.modalities["photo"], nona.modalities["force"]
    image = len(nona.channels) - 1
    assert set(photo.channels.tolist()) == {image} and nona.channels[image].family == "image"
    assert photo.positions is not None and photo.positions.max() < 16
    # 3 channels x (4 T tokens at slots 0-3, then 1 F token at slot 0).
    assert force.positions is not None
    assert force.positions.tolist() == [0, 1, 2, 3] * 3 + [0, 0, 0]
    assert nona.description["channel_embedding"] == "semantic-v1"
    assert [c["family"] for c in nona.description["channels"]] == ["force"] * 3 + ["image"]
