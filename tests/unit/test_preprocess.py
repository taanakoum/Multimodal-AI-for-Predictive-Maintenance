import numpy as np
import pytest
from scipy import signal as sps

from pdmfm import preprocess as pp


def tone(freq, rate, seconds, amplitude=1.0):
    t = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def peak_hz(x, rate):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spectrum)]


@pytest.mark.parametrize(
    "src, dst, expected",
    [
        (64_000, 12_000, (3, 16)),
        (25_600, 12_000, (15, 32)),
        (50_000, 12_000, (6, 25)),
        (48_000, 12_000, (1, 4)),
        (12_000, 12_000, (1, 1)),
    ],
)
def test_rational_ratios_of_the_plan_are_exact(src, dst, expected):
    assert pp.rational_ratio(src, dst) == expected


def test_odd_rates_get_a_close_ratio():
    up, down = pp.rational_ratio(25_608.2, 12_000)  # KAIST current/temperature device
    assert abs(up / down - 12_000 / 25_608.2) < 1e-4 * 12_000 / 25_608.2


def test_resampling_keeps_in_band_tones_and_removes_aliases():
    rate = 25_600
    x = tone(1000, rate, 1.0) + tone(9000, rate, 1.0)  # 9 kHz is above the new Nyquist (6 kHz)
    y = pp.resample(x, rate, 12_000)
    assert len(y) == 12_000 and y.dtype == np.float32
    assert abs(peak_hz(y, 12_000) - 1000) < 2
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / 12_000)
    alias = 12_000 - 9000  # where a 9 kHz tone would fold to without filtering
    in_band = spectrum[np.argmin(abs(freqs - 1000))]
    folded = spectrum[np.argmin(abs(freqs - alias))]
    assert folded < 1e-3 * in_band


def test_resampling_at_the_same_rate_is_identity():
    x = np.arange(10, dtype=np.float64)
    np.testing.assert_array_equal(pp.resample(x, 1000, 1000), x.astype(np.float32))


def test_windows_and_segments():
    x = np.arange(30_000, dtype=np.float32)
    starts = pp.window_starts(len(x), pp.WINDOW_SAMPLES, pp.WINDOW_SAMPLES)
    assert list(starts) == [0, 12_288]
    w = pp.frames(x, pp.WINDOW_SAMPLES, pp.WINDOW_SAMPLES)
    assert w.shape == (2, 12_288) and w[1, 0] == 12_288
    overlap = pp.frames(x, pp.WINDOW_SAMPLES, pp.WINDOW_SAMPLES // 2)
    assert overlap.shape == (3, 12_288)
    segments = pp.time_segments(w[0])
    assert segments.shape == (24, 512) and segments[1, 0] == 512
    assert pp.frames(x[:100], 512, 512).shape == (0, 512)


def test_log_spectrum_peaks_at_the_tone():
    rate = 12_000
    x = tone(1500, rate, 1.024, amplitude=2.0)
    spectrum, edges = pp.log_spectrum(x, rate)
    # The axis ends at the common band (5 kHz), not at Nyquist: nothing above it is a feature.
    assert spectrum.shape == (512,) and len(edges) == 513 and edges[-1] == pp.COMMON_BAND_HZ
    centre = 0.5 * (edges[:-1] + edges[1:])
    assert abs(centre[np.argmax(spectrum)] - 1500) < 10  # one 9.8 Hz band
    # Amplitude is preserved: the band mean over ~12 lines of a single tone is about
    # amplitude / lines, but the maximum line equals the amplitude.
    assert 10 ** spectrum.max() > 0.05


def test_log_spectrum_axis_end():
    x = np.random.default_rng(2).standard_normal(4096)
    assert pp.log_spectrum(x, 12_000, f_max=6000)[1][-1] == 6000
    assert pp.log_spectrum(x, 1000)[1][-1] == 500  # Nyquist below the common band


def test_envelope_order_spectrum_finds_the_modulation_order():
    rate, shaft = 12_000, 50.0
    bpfo = 3.58 * shaft  # 179 Hz, as for KAIST's NSK 6205 at 3010 rpm
    t = np.arange(pp.WINDOW_SAMPLES) / rate
    carrier = np.sin(2 * np.pi * 3000 * t)  # resonance inside the 1-5 kHz band
    impacts = 1 + 0.8 * np.sign(np.sin(2 * np.pi * bpfo * t))
    rng = np.random.default_rng(0)
    x = carrier * impacts + 0.1 * rng.standard_normal(len(t))
    spectrum, edges = pp.envelope_order_spectrum(x, rate, shaft)
    centre = 0.5 * (edges[:-1] + edges[1:])
    low = centre > 0.5  # ignore the DC neighbourhood
    assert abs(centre[low][np.argmax(spectrum[low])] - bpfo / shaft) < 0.1


def test_envelope_spectrum_handles_low_speeds():
    rate = 12_000
    x = np.random.default_rng(1).standard_normal(pp.WINDOW_SAMPLES)
    spectrum, _ = pp.envelope_order_spectrum(x, rate, shaft_hz=15.0)  # Paderborn N09
    assert np.isfinite(spectrum).all()


def amplitude_at(x, rate, freq):
    """Hann-weighted amplitude of one frequency (leakage from distant tones is negligible)."""
    taper = np.hanning(len(x))
    t = np.arange(len(x)) / rate
    return 2 * abs(np.sum(taper * x * np.exp(-2j * np.pi * freq * t))) / taper.sum()


def test_band_limit_keeps_the_band_and_removes_the_top():
    rate = 12_000
    freqs = (1000, 4900, 5300, 5800)
    x = sum(tone(f, rate, 10.0) for f in freqs)
    y = pp.band_limit(x, rate)
    assert y.dtype == np.float32 and len(y) == len(x)
    middle = slice(rate, -rate)  # away from the edges
    gains = {f: amplitude_at(y[middle], rate, f) / amplitude_at(x[middle], rate, f) for f in freqs}
    assert abs(gains[1000] - 1) < 1e-3 and abs(gains[4900] - 1) < 1e-3  # flat to the cutoff
    assert gains[5300] < 1e-4 and gains[5800] < 1e-4  # >= 80 dB down past 5.25 kHz


def test_band_limit_has_no_delay():
    rate = 12_000
    x = tone(700, rate, 2.0) + tone(2300, rate, 2.0, amplitude=0.5)
    y = pp.band_limit(x, rate)
    middle = slice(rate // 2, -rate // 2)
    assert np.max(np.abs(y[middle] - x[middle])) < 1e-3


def test_band_limit_leaves_slow_signals_alone():
    x = np.random.default_rng(3).standard_normal(5000)
    np.testing.assert_array_equal(pp.band_limit(x, 1000.0), x.astype(np.float32))


def test_band_limit_suppresses_a_recorder_roll_off_difference():
    """CWRU: 48 kHz normal files keep content up to 6 kHz after resampling; the 12 kHz fault
    files' recorder removed it. After the common band limit neither carries power there."""
    rng = np.random.default_rng(4)
    flat = pp.resample(rng.standard_normal(48_000 * 4), 48_000, 12_000)
    sos = sps.butter(12, 5300, fs=12_000, output="sos")
    rolled_off = sps.sosfiltfilt(sos, rng.standard_normal(12_000 * 4))

    def share_above(x, hz=5250):
        power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2  # taper: no truncation leakage
        return power[np.fft.rfftfreq(len(x), 1 / 12_000) > hz].sum() / power.sum()

    assert share_above(flat) > 10 * share_above(rolled_off)  # the shortcut
    for x in (flat, rolled_off):
        assert share_above(pp.band_limit(x, 12_000)[2000:-2000]) < 1e-7


def test_model_input_resamples_then_band_limits():
    rng = np.random.default_rng(5)
    fast = rng.standard_normal(25_600 * 2)
    y, rate = pp.model_input(fast, 25_600)
    assert rate == pp.MODEL_RATE_HZ and len(y) == 2 * 12_000
    x12 = tone(5800, 12_000, 2.0)
    y12, rate12 = pp.model_input(x12, 12_000)
    assert rate12 == 12_000 and np.abs(y12[2000:-2000]).max() < 1e-3
    slow = rng.standard_normal(3000)
    y1k, rate1k = pp.model_input(slow, 1000.0)
    assert rate1k == 1000.0
    np.testing.assert_array_equal(y1k, slow.astype(np.float32))


def test_dominant_frequency_resolves_below_one_bin():
    rate = 25_600
    rng = np.random.default_rng(6)
    x = tone(50.17, rate, 10.0) + 0.5 * rng.standard_normal(rate * 10)  # KAIST shaft, 0 Nm
    estimate = pp.dominant_frequency(x, rate, 40, 60)  # 8 s segments: 0.125 Hz bins
    assert abs(estimate - 50.17) < 0.02


def test_dominant_frequency_stays_in_its_band():
    rate = 12_000
    # CWRU-like: the 1x shaft line (29.95 Hz) is far weaker than the 60 Hz mains line.
    x = tone(29.95, rate, 10.0, amplitude=0.2) + tone(60.0, rate, 10.0)
    assert abs(pp.dominant_frequency(x, rate, 0.95 * 29.95, 1.05 * 29.95) - 29.95) < 0.02
    assert abs(pp.dominant_frequency(x, rate, 20, 100) - 60.0) < 0.02


@pytest.mark.parametrize(
    "low, high, seconds", [(60, 40, 8.0), (10, 7000, 8.0), (50.01, 50.02, 1.0), (0, 10, 8.0)]
)
def test_dominant_frequency_rejects_bad_bands(low, high, seconds):
    x = np.random.default_rng(7).standard_normal(12_000 * 2)
    with pytest.raises(ValueError):
        pp.dominant_frequency(x, 12_000, low, high, seconds)


def test_level_descriptors_of_a_sine():
    x = tone(100, 12_000, 1.0, amplitude=2.0) + 5.0  # DC offset is ignored
    log_rms, log_peak, kurt, crest = pp.level_descriptors(x)
    assert abs(10**log_rms - 2 / np.sqrt(2)) < 1e-3
    assert abs(10**log_peak - 2) < 1e-3
    assert abs(kurt - 1.5) < 1e-2  # Pearson kurtosis of a sine
    assert abs(crest - np.sqrt(2)) < 1e-3


def test_level_descriptors_of_a_constant_are_finite():
    assert np.isfinite(pp.level_descriptors(np.ones(100))).all()


def test_slow_summary_recovers_mean_and_slope():
    rate = 100.0
    t = np.arange(500) / rate
    mean, slope = pp.slow_summary(20.0 + 0.5 * t, rate)
    assert abs(slope - 0.5) < 1e-6 and abs(mean - (20 + 0.5 * t.mean())) < 1e-4


def test_defect_frequencies_match_published_values():
    # Paderborn 6203 at 1500 rpm: the PHME 2016 paper shows BPFO = 76.25 Hz.
    paderborn = pp.defect_frequencies(8, 6.75, 28.5, 0.0, 25.0)
    assert abs(paderborn["BPFO"] - 76.25) / 76.25 < 0.005
    # KAIST NSK 6205 at 3010 rpm.
    kaist = pp.defect_frequencies(9, 7.90, 38.5, 0.0, 3010 / 60)
    assert abs(kaist["BPFO"] - 179.4) < 0.5 and abs(kaist["BPFI"] - 272.1) < 0.5
    # BPFO + BPFI = n x shaft frequency for any geometry.
    assert abs(kaist["BPFO"] + kaist["BPFI"] - 9 * 3010 / 60) < 1e-9


def test_first_prediction_time_is_causal_and_needs_consecutive_exceedances():
    rng = np.random.default_rng(0)
    rms = 1.0 + 0.01 * rng.standard_normal(100)
    rms[40] = 2.0  # a single spike is not degradation
    rms[70:] = 1.5  # sustained rise
    assert pp.first_prediction_time(rms) == 70
    assert pp.first_prediction_time(np.ones(5)) is None  # too short
    flat = 1.0 + 0.01 * rng.standard_normal(50)
    assert pp.first_prediction_time(flat) is None


def test_normalized_rul():
    rul = pp.normalized_rul(11, fpt=6)
    assert rul[:6].tolist() == [1.0] * 6
    np.testing.assert_allclose(rul[6:], [1.0, 0.75, 0.5, 0.25, 0.0])
    assert pp.normalized_rul(4, None).tolist() == [1.0] * 4
