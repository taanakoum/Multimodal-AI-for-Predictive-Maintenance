"""Encoder input arms, computed from a window store (plan, "Sensor tokens").

* **T** (time): the window cut into 512-sample segments, one encoder input each (24 per
  1.024 s window; 16 per 8.192 s force window). The encoder outputs are averaged in groups of
  4, giving 6 (or 4) tokens per channel.
* **F** (spectrum): the log spectrum up to the common 5 kHz band (or Nyquist) in 512 bands.
* **E** (envelope): the envelope spectrum on a shaft-order axis, 0-25 orders in 512 bins,
  using each window's measured shaft speed. Bearing datasets, vibration channels only.
* **S** (waveform): the raw window, for encoders that build their own spectrogram (AST).
"""

from __future__ import annotations

import numpy as np

from pdmfm import preprocess as pp
from pdmfm.windows import WindowStore

ARMS = ("T", "F", "E", "S")
GROUP = 4  # T-arm segment embeddings averaged per token


def unavailable(store: WindowStore) -> set[str]:
    """Arms a store does not have: E without bearings and vibration; S below 12 kHz."""
    missing = set()
    if not store.spec.bearing or "vibration" not in store.families:
        missing.add("E")
    if store.rate != pp.MODEL_RATE_HZ:
        missing.add("S")
    return missing


def arm_channels(store: WindowStore, arm: str) -> tuple[int, ...]:
    """Indices of the store's channels that feed ``arm``."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm in unavailable(store):
        raise ValueError(f"{store.dataset}: no {arm} arm (E needs bearings, S the 12 kHz rate)")
    if arm == "E":
        return tuple(i for i, f in enumerate(store.families) if f == "vibration")
    return tuple(range(len(store.channels)))


def n_inputs(store: WindowStore, arm: str) -> int:
    """Encoder inputs per channel and window."""
    return store.window // pp.SEGMENT if arm == "T" else 1


def n_tokens(store: WindowStore, arm: str) -> int:
    """Cached tokens per channel and window, after pooling."""
    return n_inputs(store, arm) // GROUP if arm == "T" else 1


def arm_input(store: WindowStore, rows: np.ndarray, arm: str) -> np.ndarray:
    """(len(rows), channels, inputs, length) float32 encoder inputs for the given rows."""
    rows = np.asarray(rows)
    channels = list(arm_channels(store, arm))
    windows = np.asarray(store.signals[rows][:, channels], dtype=np.float32)  # (n, C, L)
    shaft = store.manifest["shaft_hz"].to_numpy(dtype=float)[rows]
    try:
        return window_input(windows, store.rate, arm, shaft)
    except ValueError as error:
        raise ValueError(f"{store.dataset}: {error}") from error


def window_input(
    windows: np.ndarray, rate: float, arm: str, shaft: np.ndarray | None = None
) -> np.ndarray:
    """(n, C, inputs, 512) float32 encoder inputs of raw windows (n, C, L) at ``rate``, the
    channels already chosen for ``arm``; E needs each window's shaft speed in Hz. Used for
    stored windows (:func:`arm_input`) and for modified ones (band occlusion)."""
    windows = np.asarray(windows, dtype=np.float32)
    n, c, length = windows.shape
    if arm == "T":
        segments = length // pp.SEGMENT
        return windows[:, :, : segments * pp.SEGMENT].reshape(n, c, segments, pp.SEGMENT)
    if arm == "S":
        return windows[:, :, None, :]
    out = np.empty((n, c, 1, pp.SEGMENT), dtype=np.float32)
    for i in range(n):
        for j in range(c):
            if arm == "F":
                out[i, j, 0] = pp.log_spectrum(windows[i, j], rate, pp.SEGMENT)[0]
            else:
                if shaft is None or not np.isfinite(shaft[i]):
                    raise ValueError(f"window {i}: no shaft speed for E")
                out[i, j, 0] = pp.envelope_order_spectrum(windows[i, j], rate, shaft[i])[0]
    return out


def pool_tokens(embeddings: np.ndarray, arm: str) -> np.ndarray:
    """(n, C, inputs, D) -> (n, C, tokens, D): T-arm inputs averaged in groups of 4."""
    if arm != "T":
        return embeddings
    n, c, k, d = embeddings.shape
    return embeddings.reshape(n, c, k // GROUP, GROUP, d).mean(axis=3)
