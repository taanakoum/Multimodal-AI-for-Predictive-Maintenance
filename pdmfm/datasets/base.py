"""Shared types for dataset readers.

A reader turns one publisher's files into :class:`Recording` objects at the native sampling
rate, with units, label, severity, operating condition and the physical ``unit`` that splits
are made on. Readers are lazy: :meth:`Reader.index` lists recordings cheaply and
:meth:`Reader.load` reads one, so a 300 s KAIST file is in memory only while it is used.
Resampling and windowing happen later (:mod:`pdmfm.preprocess`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from pdmfm.schema import ImageItem


@dataclass(frozen=True)
class ChannelData:
    name: str
    family: str  # see pdmfm.schema.FAMILIES
    axis: str
    placement: str
    rate_hz: float
    signal: np.ndarray  # (n,) float32 at the native rate
    unit: str = ""

    @property
    def seconds(self) -> float:
        return len(self.signal) / self.rate_hz


@dataclass(frozen=True)
class RecordingInfo:
    dataset: str
    recording_id: str
    unit: str  # physical unit for splits (tool, bearing, cutter, fault specimen)
    label: str | None  # class name in the dataset's own label space
    severity: str | None = None
    condition: dict = field(default_factory=dict)  # rpm, load, ...
    meta: dict = field(default_factory=dict)  # file names and facts not given to the model


@dataclass(frozen=True)
class Recording:
    info: RecordingInfo
    channels: dict[str, ChannelData]
    images: tuple[ImageItem, ...] = ()
    t_offset: float = 0.0  # start time on the unit's clock, in seconds
    targets: dict = field(default_factory=dict)  # wear_um, rul, ... when known

    def channel(self, name: str) -> ChannelData:
        return self.channels[name]


class Reader(Protocol):
    name: str

    def index(self) -> list[RecordingInfo]: ...

    def load(self, info: RecordingInfo) -> Recording: ...


def iter_recordings(
    reader: Reader, infos: list[RecordingInfo] | None = None
) -> Iterator[Recording]:
    """Load recordings one at a time (bounded memory)."""
    for info in infos if infos is not None else reader.index():
        yield reader.load(info)


def as_float32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32).ravel())
