"""One sample format for every dataset (plan section "Unified sample format").

A :class:`Sample` is one analysis window of one unit (tool, bearing, cutter). It holds the
window's high-rate channels, the slow items attached to it (photos, context text), labels
and the keys that decide splits. Slow items are attached by :func:`as_of_join`, which only
ever takes items captured *before* the window starts, so a photo taken after a pass (which
shows that pass's wear, OmniFuser finding F1) can never leak into the pass's own window.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

FAMILIES = ("vibration", "force", "current", "acoustic", "ae", "slow")
AXES = ("x", "y", "z", "radial", "none")
PLACEMENTS = (
    "drive_end",
    "fan_end",
    "housing",
    "housing_a",
    "housing_b",
    "spindle",
    "table",
    "motor_phase",
    "bearing_module",
    "none",
)
IMAGE_KINDS = ("tool", "chip", "workpiece", "thermal")


@dataclass(frozen=True)
class ChannelWindow:
    """One channel of one window, already at the model rate."""

    name: str
    family: str
    axis: str
    placement: str
    rate_hz: float
    signal: np.ndarray  # (L,)
    unit: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if self.axis not in AXES:
            raise ValueError(f"unknown axis {self.axis!r}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {self.placement!r}")
        if self.signal.ndim != 1:
            raise ValueError("a channel window is one-dimensional")


@dataclass(frozen=True)
class ImageItem:
    kind: str
    path: str
    captured_at: float  # seconds on the unit's clock

    def __post_init__(self) -> None:
        if self.kind not in IMAGE_KINDS:
            raise ValueError(f"unknown image kind {self.kind!r}")


@dataclass(frozen=True)
class Sample:
    dataset: str
    unit: str  # physical unit used for splits: tool, bearing, cutter, specimen
    bag: str  # pass, cut, recording: windows of one bag share a label
    t_start: float
    t_end: float
    channels: tuple[ChannelWindow, ...]
    images: tuple[ImageItem, ...] = ()
    context_text: str = ""
    condition: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)  # class, severity, wear_um, rul
    label_space: str = ""

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError("a window must end after it starts")
        late = [item for item in self.images if item.captured_at >= self.t_start]
        if late:
            raise ValueError(
                f"{len(late)} image(s) captured at or after the window start: "
                "slow items must come from the as-of join"
            )

    def modalities(self) -> set[str]:
        present = {channel.family for channel in self.channels}
        present |= {item.kind for item in self.images}
        if self.context_text:
            present.add("text")
        return present


def as_of_join(
    items: Iterable[ImageItem], t_start: float, kinds: Sequence[str] | None = None
) -> tuple[ImageItem, ...]:
    """For each image kind, the newest item captured strictly before ``t_start``.

    Returns at most one item per kind, ordered by kind; kinds with no earlier item are
    simply absent (a missing modality), never filled with zeros.
    """
    newest: dict[str, ImageItem] = {}
    for item in items:
        if kinds is not None and item.kind not in kinds:
            continue
        if item.captured_at >= t_start:
            continue
        current = newest.get(item.kind)
        if current is None or item.captured_at > current.captured_at:
            newest[item.kind] = item
    return tuple(newest[kind] for kind in sorted(newest))


def item_ages(items: Iterable[ImageItem], t_start: float) -> dict[str, float]:
    """Age in seconds of each attached item at the window start (always > 0)."""
    return {item.kind: t_start - item.captured_at for item in items}
