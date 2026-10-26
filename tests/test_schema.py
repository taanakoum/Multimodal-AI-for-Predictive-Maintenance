import numpy as np
import pytest

from pdmfm.schema import ChannelWindow, ImageItem, Sample, as_of_join, item_ages


def _channel(**overrides):
    values = dict(
        name="fx",
        family="force",
        axis="x",
        placement="table",
        rate_hz=1000.0,
        signal=np.zeros(8, dtype=np.float32),
        unit="N",
    )
    values.update(overrides)
    return ChannelWindow(**values)


def test_as_of_join_takes_the_newest_item_strictly_before_the_window():
    items = [
        ImageItem("tool", "pass1.jpg", captured_at=100.0),
        ImageItem("tool", "pass2.jpg", captured_at=200.0),
        ImageItem("tool", "pass3.jpg", captured_at=300.0),  # this pass's own photo: excluded
        ImageItem("chip", "pass1_chip.jpg", captured_at=100.0),
    ]
    joined = as_of_join(items, t_start=300.0)
    assert [(i.kind, i.path) for i in joined] == [("chip", "pass1_chip.jpg"), ("tool", "pass2.jpg")]
    assert item_ages(joined, 300.0) == {"chip": 200.0, "tool": 100.0}


def test_as_of_join_leaves_missing_kinds_absent():
    items = [ImageItem("tool", "a.jpg", captured_at=10.0)]
    assert as_of_join(items, t_start=5.0) == ()  # first pass of a tool: no photo yet
    assert as_of_join(items, t_start=20.0, kinds=("chip",)) == ()


def test_sample_rejects_items_captured_after_its_start():
    late = ImageItem("tool", "own_pass.jpg", captured_at=50.0)
    with pytest.raises(ValueError, match="as-of join"):
        Sample(
            "nonastreda",
            "T1",
            "T1R1B1",
            t_start=50.0,
            t_end=58.0,
            channels=(_channel(),),
            images=(late,),
        )


def test_sample_reports_its_modalities():
    sample = Sample(
        "nonastreda",
        "T1",
        "T1R1B2",
        t_start=60.0,
        t_end=68.0,
        channels=(_channel(),),
        images=(ImageItem("tool", "p.jpg", 59.0),),
        context_text="milling C45 steel",
    )
    assert sample.modalities() == {"force", "tool", "text"}


@pytest.mark.parametrize(
    "field, value", [("family", "sound"), ("axis", "w"), ("placement", "roof")]
)
def test_channel_vocabulary_is_closed(field, value):
    with pytest.raises(ValueError):
        _channel(**{field: value})


def test_channel_must_be_one_dimensional():
    with pytest.raises(ValueError):
        _channel(signal=np.zeros((2, 4)))


def test_window_must_have_positive_length():
    with pytest.raises(ValueError):
        Sample("x", "u", "b", t_start=1.0, t_end=1.0, channels=())
