"""The as-of join (pdmfm.schema): photos never come from the window's own pass."""

import pytest

from pdmfm.schema import ImageItem, as_of_join


def test_as_of_join_takes_the_newest_item_strictly_before_the_window():
    items = [
        ImageItem("tool", "pass1.jpg", captured_at=100.0),
        ImageItem("tool", "pass2.jpg", captured_at=200.0),
        ImageItem("tool", "pass3.jpg", captured_at=300.0),  # this pass's own photo: excluded
        ImageItem("chip", "pass1_chip.jpg", captured_at=100.0),
    ]
    joined = as_of_join(items, t_start=300.0)
    assert [(i.kind, i.path) for i in joined] == [("chip", "pass1_chip.jpg"), ("tool", "pass2.jpg")]


def test_as_of_join_leaves_missing_kinds_absent():
    items = [ImageItem("tool", "a.jpg", captured_at=10.0)]
    assert as_of_join(items, t_start=5.0) == ()  # first pass of a tool: no photo yet
    assert as_of_join(items, t_start=20.0, kinds=("chip",)) == ()


def test_an_item_captured_at_the_window_start_is_not_attached():
    items = [ImageItem("tool", "own_pass.jpg", captured_at=50.0)]
    assert as_of_join(items, t_start=50.0) == ()


def test_image_kinds_are_closed():
    with pytest.raises(ValueError, match="unknown image kind"):
        ImageItem("sound", "x.wav", 0.0)
