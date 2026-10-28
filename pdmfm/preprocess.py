"""Signal preprocessing shared by every dataset (plan section "Datasets and preprocessing").

* :func:`resample`: anti-aliased polyphase resampling with an exact rational ratio
  (64 kHz -> 12 kHz is 3/16, 25.6 kHz -> 12 kHz is 15/32, 50 kHz -> 12 kHz is 6/25).
* :func:`band_limit` / :func:`model_input`: one 5 kHz bandwidth for every source, so the
  recorders' own anti-alias roll-offs cannot identify a dataset or a class.
* :func:`frames` / :func:`window_starts`: fixed windows (12,288 samples = 1.024 s at 12 kHz).
* Encoder inputs per window: :func:`time_segments` (T arm), :func:`log_spectrum` (F arm),
  :func:`envelope_order_spectrum` (E arm, bearings only).
* :func:`level_descriptors` and :func:`slow_summary`: what per-input normalized encoders
  cannot see (absolute level, DC trend); :func:`relative_to_baseline` expresses them against
  the unit's healthy reference.
* :func:`dominant_frequency`: tachless shaft-speed or line-frequency estimate.
* :func:`defect_frequencies`, :func:`first_prediction_time`, :func:`label_fpt`,
  :func:`normalized_rul`.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sps
from scipy import stats

# Part of every window-store and feature-cache signature: bump it whenever a function here
# changes its output, so stale caches are rebuilt instead of mixed with new ones.
PREPROCESS_VERSION = 2
MODEL_RATE_HZ = 12_000.0
# Every source is low-passed here: recorders roll off differently below their Nyquist
# frequency, so content above ~5.25 kHz identifies the source (and, on CWRU, the class).
COMMON_BAND_HZ = 5_000.0
WINDOW_SAMPLES = 12_288  # 1.024 s at 12 kHz = 24 segments of 512
SEGMENT = 512
EPS = 1e-12


# ----------------------------------------------------------------------------- resampling


def rational_ratio(src_hz: float, dst_hz: float, max_denominator: int = 1000) -> tuple[int, int]:
    """(up, down) with up/down ≈ dst/src; exact for the rates used in the plan."""
    if src_hz <= 0 or dst_hz <= 0:
        raise ValueError("rates must be positive")
    ratio = Fraction(dst_hz / src_hz).limit_denominator(max_denominator)
    if abs(float(ratio) - dst_hz / src_hz) > 1e-4 * dst_hz / src_hz:
        raise ValueError(f"no accurate ratio for {src_hz} -> {dst_hz} Hz")
    return ratio.numerator, ratio.denominator


def resample(x: np.ndarray, src_hz: float, dst_hz: float, axis: int = -1) -> np.ndarray:
    """Anti-aliased resampling (Kaiser-windowed polyphase FIR, scipy ``resample_poly``)."""
    if abs(src_hz - dst_hz) <= 1e-9 * dst_hz:
        return np.asarray(x, dtype=np.float32)
    up, down = rational_ratio(src_hz, dst_hz)
    y = sps.resample_poly(
        np.asarray(x, dtype=np.float64), up, down, axis=axis, window=("kaiser", 5.0)
    )
    return y.astype(np.float32)


def band_limit(
    x: np.ndarray,
    rate_hz: float,
    cutoff_hz: float = COMMON_BAND_HZ,
    transition_hz: float = 250.0,
    attenuation_db: float = 80.0,
) -> np.ndarray:
    """Linear-phase low-pass that gives every source the same bandwidth.

    CWRU's 12 kHz fault files lose about two decades between 5.25 and 6 kHz; its 48 kHz
    normal files, resampled to 12 kHz, do not. Kaiser-window FIR, flat to ``cutoff_hz`` and
    at least ``attenuation_db`` down from ``cutoff_hz + transition_hz``, with no delay.
    Apply it to a whole recording before windowing (only the outer ~numtaps/2 samples see
    the edges). A signal whose Nyquist frequency is at or below the cutoff is returned as is.
    """
    x = np.asarray(x, dtype=np.float64)
    nyquist = rate_hz / 2.0
    if cutoff_hz >= nyquist:
        return x.astype(np.float32)
    stop = min(cutoff_hz + transition_hz, nyquist)
    numtaps, beta = sps.kaiserord(attenuation_db, (stop - cutoff_hz) / nyquist)
    numtaps |= 1  # odd length: symmetric taps centred on a sample, so "same" has no delay
    taps = sps.firwin(
        numtaps,
        0.5 * (cutoff_hz + stop),
        # scipy accepts (name, parameter) windows; its type stubs list only str
        window=("kaiser", beta),  # pyright: ignore[reportArgumentType]
        fs=rate_hz,
    )
    return sps.oaconvolve(x, taps, mode="same").astype(np.float32)


def model_input(x: np.ndarray, rate_hz: float) -> tuple[np.ndarray, float]:
    """(signal, rate) as every sensor arm sees it: resampled down to the model rate when
    faster, then band-limited to :data:`COMMON_BAND_HZ`. Slower signals keep their rate."""
    if rate_hz > MODEL_RATE_HZ:
        x, rate_hz = resample(x, rate_hz, MODEL_RATE_HZ), MODEL_RATE_HZ
    return band_limit(x, rate_hz), rate_hz


# ----------------------------------------------------------------------------- windows


def window_starts(n_samples: int, length: int, hop: int) -> np.ndarray:
    """Start indices of full windows of ``length`` every ``hop`` samples."""
    if length <= 0 or hop <= 0:
        raise ValueError("length and hop must be positive")
    if n_samples < length:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_samples - length + 1, hop, dtype=np.int64)


def frames(x: np.ndarray, length: int, hop: int) -> np.ndarray:
    """(n_windows, length) copy of the full windows of a 1-D signal."""
    starts = window_starts(len(x), length, hop)
    if len(starts) == 0:
        return np.zeros((0, length), dtype=x.dtype)
    view = np.lib.stride_tricks.sliding_window_view(x, length)[::hop]
    return np.ascontiguousarray(view[: len(starts)])


def time_segments(window: np.ndarray, seg_len: int = SEGMENT) -> np.ndarray:
    """T arm: (n_segments, seg_len) consecutive segments; a remainder is dropped."""
    n = len(window) // seg_len
    return np.asarray(window[: n * seg_len]).reshape(n, seg_len)


# ----------------------------------------------------------------------------- spectra


def _bin_average(axis_values: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of ``values`` per bin of ``edges``; empty bins are filled by interpolation."""
    n_bins = len(edges) - 1
    index = np.searchsorted(edges, axis_values, side="right") - 1
    keep = (index >= 0) & (index < n_bins)
    sums = np.bincount(index[keep], weights=values[keep], minlength=n_bins)
    counts = np.bincount(index[keep], minlength=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = np.empty(n_bins)
    full = counts > 0
    out[full] = sums[full] / counts[full]
    if not full.all():
        out[~full] = np.interp(centres[~full], axis_values, values)
    return out


def log_spectrum(
    window: np.ndarray, rate_hz: float, n_bins: int = 512, f_max: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """F arm: log10 amplitude spectrum (Hann taper) averaged into ``n_bins`` bands.

    Returns (log amplitudes, band edges in Hz). The axis ends at ``f_max``, by default the
    common band or the Nyquist frequency, whichever is lower. The mean is removed first;
    the absolute level lives in :func:`level_descriptors`.
    """
    x = np.asarray(window, dtype=np.float64)
    x = x - x.mean()
    taper = np.hanning(len(x))
    amplitude = 2.0 * np.abs(np.fft.rfft(x * taper)) / taper.sum()
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate_hz)
    edges = np.linspace(0.0, f_max or min(COMMON_BAND_HZ, rate_hz / 2.0), n_bins + 1)
    return np.log10(_bin_average(freqs, amplitude, edges) + EPS), edges


def envelope_order_spectrum(
    window: np.ndarray,
    rate_hz: float,
    shaft_hz: float,
    band_hz: tuple[float, float] = (1000.0, COMMON_BAND_HZ),
    max_order: float = 25.0,
    n_bins: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """E arm: envelope spectrum on a shaft-order axis (fault lines become speed-invariant).

    Band-pass (4th-order Butterworth, zero phase) -> Hilbert envelope -> amplitude spectrum
    -> resampled onto orders 0..``max_order``. The band is fixed and label-free; its upper
    edge is clipped to 45 % of the rate. Returns (log10 amplitudes, order-bin edges).
    """
    if shaft_hz <= 0:
        raise ValueError("shaft_hz must be positive")
    x = np.asarray(window, dtype=np.float64)
    high = min(band_hz[1], 0.45 * rate_hz)
    low = min(band_hz[0], 0.5 * high)
    sos = sps.butter(4, [low, high], btype="bandpass", fs=rate_hz, output="sos")
    envelope = np.abs(sps.hilbert(sps.sosfiltfilt(sos, x - x.mean())))
    envelope -= envelope.mean()
    taper = np.hanning(len(envelope))
    amplitude = 2.0 * np.abs(np.fft.rfft(envelope * taper)) / taper.sum()
    orders = np.fft.rfftfreq(len(envelope), 1.0 / rate_hz) / shaft_hz
    edges = np.linspace(0.0, max_order, n_bins + 1)
    return np.log10(_bin_average(orders, amplitude, edges) + EPS), edges


def dominant_frequency(
    x: np.ndarray, rate_hz: float, low_hz: float, high_hz: float, seconds: float = 8.0
) -> float:
    """Frequency (Hz) of the strongest line in [``low_hz``, ``high_hz``].

    Tachless speed estimate: the shaft's 1x line in vibration, the supply line in a motor
    current, the tooth-passing line in a cutting force. Welch PSD with Hann segments of
    ``seconds`` (resolution 1/``seconds`` Hz), refined by a parabola through the log PSD.
    """
    x = np.asarray(x, dtype=np.float64)
    if not 0 < low_hz < high_hz <= rate_hz / 2:
        raise ValueError("need 0 < low_hz < high_hz <= rate_hz / 2")
    nperseg = min(len(x), int(round(seconds * rate_hz)))
    freqs, psd = sps.welch(x - x.mean(), fs=rate_hz, nperseg=nperseg)
    band = np.flatnonzero((freqs >= low_hz) & (freqs <= high_hz))
    if len(band) == 0:
        raise ValueError("the band holds no frequency bin; use longer segments")
    i = int(band[np.argmax(psd[band])])
    if 0 < i < len(psd) - 1:
        a, b, c = np.log(psd[i - 1 : i + 2] + EPS)
        denominator = a - 2 * b + c
        offset = 0.5 * (a - c) / denominator if denominator < 0 else 0.0
        return float(freqs[i] + offset * (freqs[1] - freqs[0]))
    return float(freqs[i])


# ----------------------------------------------------------------------------- descriptors

LEVEL_NAMES = ("log10_rms", "log10_peak", "kurtosis", "crest")


def level_descriptors(window: np.ndarray) -> np.ndarray:
    """(4,) log10 RMS and log10 peak of the AC part, Pearson kurtosis (normal = 3), crest."""
    x = np.asarray(window, dtype=np.float64)
    ac = x - x.mean()
    rms = float(np.sqrt(np.mean(ac**2)))
    peak = float(np.max(np.abs(ac)))
    kurt = float(stats.kurtosis(ac, fisher=False)) if rms > 0 else 0.0
    return np.array(
        [np.log10(rms + EPS), np.log10(peak + EPS), kurt, peak / (rms + EPS)], dtype=np.float32
    )


def slow_summary(window: np.ndarray, rate_hz: float) -> np.ndarray:
    """(2,) mean and least-squares slope (units per second) of a slow channel or a DC level."""
    x = np.asarray(window, dtype=np.float64)
    t = np.arange(len(x)) / rate_hz
    slope = np.polyfit(t, x, 1)[0] if len(x) > 1 else 0.0
    return np.array([x.mean(), slope], dtype=np.float32)


def healthy_baseline(descriptors: np.ndarray) -> np.ndarray:
    """Median of descriptor rows from healthy (reference) windows."""
    descriptors = np.atleast_2d(descriptors)
    if len(descriptors) == 0:
        raise ValueError("a baseline needs at least one healthy window")
    return np.median(descriptors, axis=0).astype(np.float32)


def relative_to_baseline(descriptors: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Descriptors minus the healthy baseline (log quantities become log ratios)."""
    return (np.asarray(descriptors) - baseline).astype(np.float32)


# ----------------------------------------------------------------------------- physics and RUL


def defect_frequencies(
    n_balls: int, ball_d: float, pitch_d: float, contact_deg: float, shaft_hz: float
) -> dict[str, float]:
    """Bearing characteristic frequencies in Hz (BPFO, BPFI, BSF, FTF)."""
    ratio = ball_d / pitch_d * np.cos(np.radians(contact_deg))
    return {
        "BPFO": n_balls / 2 * shaft_hz * (1 - ratio),
        "BPFI": n_balls / 2 * shaft_hz * (1 + ratio),
        "BSF": pitch_d / (2 * ball_d) * shaft_hz * (1 - ratio**2),
        "FTF": shaft_hz / 2 * (1 - ratio),
    }


def first_prediction_time(
    rms: np.ndarray, baseline_count: int = 10, sigmas: float = 3.0, consecutive: int = 3
) -> int | None:
    """Index where degradation starts: the first of ``consecutive`` snapshots whose RMS
    exceeds mean + ``sigmas``·std of the first ``baseline_count`` snapshots; None if never.
    Only earlier snapshots enter the threshold, so the rule is causal.
    """
    rms = np.asarray(rms, dtype=np.float64)
    if len(rms) <= baseline_count:
        return None
    reference = rms[:baseline_count]
    threshold = reference.mean() + sigmas * reference.std(ddof=1)
    above = rms > threshold
    for start in range(baseline_count, len(rms) - consecutive + 1):
        if above[start : start + consecutive].all():
            return start
    return None


def label_fpt(
    rms: np.ndarray, kurtosis: np.ndarray | None = None, fraction: float = 0.1
) -> tuple[int | None, str]:
    """FPT for offline RUL labels (docs/EDA_REPORT.md, section 7.2): the healthy reference is
    the first ``fraction`` of life (at least 10 snapshots), 3σ, 10 consecutive exceedances.
    Abrupt failures that RMS misses (XJTU-SY Bearing1_4) fall back to the same rule on
    kurtosis. Returns (index or None, "rms" | "kurtosis" | "none"). Uses the whole
    lifetime, so it labels data; it is never a model input."""
    count = max(10, int(fraction * len(rms)))
    fpt = first_prediction_time(rms, count, 3.0, 10)
    if fpt is not None:
        return fpt, "rms"
    if kurtosis is not None:
        fpt = first_prediction_time(kurtosis, count, 3.0, 10)
        if fpt is not None:
            return fpt, "kurtosis"
    return None, "none"


def normalized_rul(n_snapshots: int, fpt: int | None) -> np.ndarray:
    """1 before the first prediction time, then a linear fall to 0 at the last snapshot."""
    rul = np.ones(n_snapshots, dtype=np.float32)
    if fpt is None or fpt >= n_snapshots - 1:
        return rul
    span = n_snapshots - 1 - fpt
    rul[fpt:] = 1.0 - np.arange(span + 1, dtype=np.float32) / span
    return rul
