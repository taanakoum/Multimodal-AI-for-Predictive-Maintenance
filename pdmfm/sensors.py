"""What every channel physically is (plan, "Sensor tokens": embeddings for sensor type, axis
and sampling rate), in one vocabulary shared by all datasets.

A token's channel embedding is built from its channel's description, never from the channel's
position in its dataset, so the same kind of sensor gets the same vector in every dataset:
CWRU's drive-end accelerometer and XJTU-SY's horizontal one share family, axis class and
location; Nonastreda's Fx shares MATWI's. That is what lets one backbone read several
datasets (pretraining, stage 6); with one model per dataset it changes nothing but the
parametrization.

A description (:class:`ChannelInfo`) has:

* **family**: the sensor type (vibration, force, current, acoustic, ae, slow), ``image`` for
  photo tokens;
* **axis**: x, y, z, radial or none;
* **location**: where the sensor sits, in classes shared across datasets (bearing housing,
  motor, table, spindle, machine); each reader's own placement name maps to one;
* **site**: the channel's rank among the dataset's channels with the same family, axis and
  location (CWRU's fan-end accelerometer is site 1 of the bearing-housing vibration
  channels), so no two channels of a dataset share a description;
* **scale**: the model sampling rate and the window length. They fix every token's physical
  scale: a T segment lasts 512 / rate seconds (43 ms at 12 kHz, 512 ms at 1 kHz), an F bin
  covers min(5 kHz, rate / 2) / 512 Hz (9.8 Hz or 0.98 Hz), level descriptors summarize one
  window. Slow channels are window summaries (mean and slope), given a nominal 1 Hz rate.

The descriptions follow the readers (``pdmfm/datasets``); ``tests/test_sensors.py`` checks
them against the synthetic readers and, where the data is present, the real ones.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from pdmfm.schema import FAMILIES as SENSOR_FAMILIES

FAMILIES = (*SENSOR_FAMILIES, "image", "other")
AXES = ("none", "x", "y", "z", "radial", "other")
LOCATIONS = ("none", "bearing_housing", "motor", "table", "spindle", "machine", "other")
MAX_SITES = 8
SLOW_RATE_HZ = 1.0  # nominal: slow tokens summarize a window (mean and slope)
# Each reader's placement name -> a location class shared across datasets.
PLACEMENT_LOCATION = {
    "drive_end": "bearing_housing",
    "fan_end": "bearing_housing",
    "housing": "bearing_housing",
    "housing_a": "bearing_housing",
    "housing_b": "bearing_housing",
    "bearing_module": "bearing_housing",
    "motor_phase": "motor",
    "table": "table",
    "spindle": "spindle",
    "machine": "machine",
    "none": "none",
}
# dataset -> channel -> (family, axis, placement), as the readers create them.
CHANNELS: dict[str, dict[str, tuple[str, str, str]]] = {
    "cwru": {
        "de": ("vibration", "none", "drive_end"),
        "fe": ("vibration", "none", "fan_end"),
    },
    "kaist": {
        "vib_ax": ("vibration", "x", "housing_a"),
        "vib_ay": ("vibration", "y", "housing_a"),
        "vib_bx": ("vibration", "x", "housing_b"),
        "vib_by": ("vibration", "y", "housing_b"),
        "current_1": ("current", "none", "motor_phase"),
        "temp_1": ("slow", "none", "housing"),
        "temp_2": ("slow", "none", "housing"),
    },
    "xjtu_sy": {
        "vib_h": ("vibration", "x", "housing"),
        "vib_v": ("vibration", "y", "housing"),
    },
    "nonastreda": {
        "fx": ("force", "x", "table"),
        "fy": ("force", "y", "table"),
        "fz": ("force", "z", "table"),
    },
    "matwi": {
        "fx": ("force", "x", "table"),
        "fy": ("force", "y", "table"),
        "fz": ("force", "z", "table"),
        "acc": ("vibration", "none", "spindle"),
        "mic": ("acoustic", "none", "machine"),
    },
    "paderborn": {
        "vibration": ("vibration", "none", "housing"),
        "current_1": ("current", "none", "motor_phase"),
        "current_2": ("current", "none", "motor_phase"),
        "temperature": ("slow", "none", "bearing_module"),
    },
    "phm2010": {
        "fx": ("force", "x", "table"),
        "fy": ("force", "y", "table"),
        "fz": ("force", "z", "table"),
        "vx": ("vibration", "x", "table"),
        "vy": ("vibration", "y", "table"),
        "vz": ("vibration", "z", "table"),
        "ae_rms": ("slow", "none", "table"),
    },
}


@dataclass(frozen=True)
class ChannelInfo:
    family: str
    axis: str
    location: str
    site: int
    rate_hz: float
    window_s: float

    def codes(self) -> tuple[int, int, int, int]:
        """Vocabulary indices of family, axis, location and site."""
        return (
            FAMILIES.index(self.family),
            AXES.index(self.axis),
            LOCATIONS.index(self.location),
            self.site,
        )

    def scale(self) -> tuple[float, float]:
        """(log10 of the rate in kHz, log10 of the window in seconds)."""
        return (math.log10(self.rate_hz / 1000.0), math.log10(self.window_s))


def _check(info: ChannelInfo) -> ChannelInfo:
    for value, vocabulary, what in (
        (info.family, FAMILIES, "family"),
        (info.axis, AXES, "axis"),
        (info.location, LOCATIONS, "location"),
    ):
        if value not in vocabulary:
            raise ValueError(f"unknown {what} {value!r}; add it to pdmfm.sensors")
    if not 0 <= info.site < MAX_SITES:
        raise ValueError(f"site {info.site} outside 0..{MAX_SITES - 1}")
    if info.rate_hz <= 0 or info.window_s <= 0:
        raise ValueError("rate and window length must be positive")
    return info


def describe(
    dataset: str,
    channels: tuple[str, ...],
    slow: tuple[str, ...],
    rate_hz: float,
    window_s: float,
    image: bool = False,
) -> list[ChannelInfo]:
    """Descriptions of the high-rate ``channels`` (at ``rate_hz``), then the ``slow`` ones, then
    one ``image`` entry for photo tokens, in the token channel indices of
    :mod:`pdmfm.model.inputs`. Raises if a channel is not catalogued or two share a description."""
    table = CHANNELS.get(dataset)
    if table is None:
        raise ValueError(f"no channel catalogue for {dataset}; add it to pdmfm.sensors.CHANNELS")
    infos, seen = [], {}
    for name in (*channels, *slow):
        if name not in table:
            raise ValueError(f"{dataset}: channel {name!r} is not in pdmfm.sensors.CHANNELS")
        family, axis, placement = table[name]
        location = PLACEMENT_LOCATION.get(placement)
        if location is None:
            raise ValueError(f"{dataset}: placement {placement!r} has no location class")
        group = (family, axis, location)
        seen[group] = seen.get(group, -1) + 1
        rate = SLOW_RATE_HZ if name in slow else rate_hz
        infos.append(_check(ChannelInfo(family, axis, location, seen[group], rate, window_s)))
    if image:
        infos.append(_check(ChannelInfo("image", "none", "none", 0, SLOW_RATE_HZ, window_s)))
    keys = [(i.family, i.axis, i.location, i.site, i.rate_hz) for i in infos]
    if len(set(keys)) != len(keys):  # cannot happen with ranked sites; kept as a guard
        raise ValueError(f"{dataset}: two channels share a description")
    return infos


def generic(n: int, rate_hz: float = 12_000.0, window_s: float = 1.024) -> list[ChannelInfo]:
    """``n`` distinct placeholder descriptions (family ``other``), for inputs built by hand
    in tests."""
    if n > MAX_SITES:
        raise ValueError(f"at most {MAX_SITES} generic channels")
    return [ChannelInfo("other", "none", "none", i, rate_hz, window_s) for i in range(n)]


def table(infos: list[ChannelInfo]) -> tuple[np.ndarray, np.ndarray]:
    """(codes (C, 4) int64, scale (C, 2) float32), the input of
    :class:`pdmfm.model.layers.ChannelEmbedding`."""
    codes = np.array([i.codes() for i in infos], dtype=np.int64).reshape(len(infos), 4)
    scale = np.array([i.scale() for i in infos], dtype=np.float32).reshape(len(infos), 2)
    return codes, scale


def as_dicts(infos: list[ChannelInfo]) -> list[dict]:
    return [asdict(i) for i in infos]
