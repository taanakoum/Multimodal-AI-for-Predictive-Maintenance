"""Photo items and the as-of join (plan section "Unified sample format").

Low-frequency items (photos) are attached to the windows of the window store
(:mod:`pdmfm.windows`) by :func:`as_of_join`, which only ever takes items captured *before*
the window starts, so a photo taken after a pass (which shows that pass's wear, OmniFuser
finding F1) can never leak into the pass's own windows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

FAMILIES = ("vibration", "force", "current", "acoustic", "ae", "slow")
IMAGE_KINDS = ("tool", "chip", "workpiece", "thermal", "insert")


@dataclass(frozen=True)
class ImageItem:
    kind: str
    path: str
    captured_at: float  # seconds on the unit's clock

    def __post_init__(self) -> None:
        if self.kind not in IMAGE_KINDS:
            raise ValueError(f"unknown image kind {self.kind!r}")


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
