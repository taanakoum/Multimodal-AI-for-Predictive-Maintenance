"""Exploratory data analysis of every Project 14 dataset that is on disk.

    python scripts/eda/run_eda.py                    # all available datasets
    python scripts/eda/run_eda.py kaist cwru --quick # fewer recordings, for a fast check

Checks CPU and RAM first (pdmfm.resources). For each dataset it writes
``<out>/<dataset>/summary.json`` and figures, and copies the summary and key figures to
``docs/eda/`` for the EDA report (docs/EDA_REPORT.md). What it measures, and why:

* inventory: recordings, units, classes, durations, windows at the model rate -> split and
  sampling decisions;
* level descriptors per class and condition -> whether absolute level separates classes and
  how much it moves with load (the case for baseline-relative level tokens);
* envelope order spectra against bearing defect orders -> whether the E arm and the physics
  alignment score have something to find;
* handcrafted-feature baselines under the plan's splits -> a floor for Gate 1;
* shortcut checks (temperature, missing channels, run order, near-duplicate images) -> leaks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from pdmfm import preprocess as pp  # noqa: E402
from pdmfm.datasets import get_reader  # noqa: E402
from pdmfm.paths import REPO_ROOT, data_root, extracted_dir, raw_dir  # noqa: E402
from pdmfm.resources import claim_resources  # noqa: E402

DOCS = REPO_ROOT / "docs" / "eda"
COPY_TO_DOCS = True  # switched off by --quick: report assets come from full runs only
COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.size": 9,
        "figure.constrained_layout.use": True,  # titles and tick labels never overlap
    }
)


# ----------------------------------------------------------------------------- helpers


def save(fig, out: Path, name: str, report: bool = False) -> str:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    if report and COPY_TO_DOCS:
        (DOCS / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, DOCS / "figures" / f"{out.name}_{name}.png")
    return path.name


def band_features(window: np.ndarray, rate: float, n_bands: int = 16) -> np.ndarray:
    """Level descriptors + mean log amplitude in ``n_bands`` equal bands (handcrafted)."""
    spectrum, _ = pp.log_spectrum(window, rate)
    bands = spectrum.reshape(n_bands, -1).mean(axis=1)
    return np.concatenate([pp.level_descriptors(window), bands])


def windows_of(signal: np.ndarray, rate: float, length=pp.WINDOW_SAMPLES, hop=None, limit=None):
    """Windows of the model input (resampled, band-limited), as the encoders will see them."""
    prepared, _ = pp.model_input(signal, rate)
    w = pp.frames(prepared, length, hop or length)
    return w[:limit] if limit else w


def top_band_ratio(window: np.ndarray, rate: float) -> float:
    """log10 amplitude in 5.7-6 kHz minus that in 4-5 kHz (a recorder's roll-off)."""
    spectrum, edges = pp.log_spectrum(window, rate, f_max=rate / 2)
    centres = 0.5 * (edges[:-1] + edges[1:])
    top = spectrum[(centres > 5700) & (centres < 6000)].mean()
    return float(top - spectrum[(centres > 4000) & (centres < 5000)].mean())


def power_share_above(window: np.ndarray, rate: float, hz: float = 5250.0) -> float:
    """Fraction of the window's AC power above ``hz`` (Hann taper: no truncation leakage)."""
    power = np.abs(np.fft.rfft((window - np.mean(window)) * np.hanning(len(window)))) ** 2
    freqs = np.fft.rfftfreq(len(window), 1.0 / rate)
    return float(power[freqs > hz].sum() / (power.sum() + 1e-30))


def classify(train_x, train_y, test_x, test_y) -> dict:
    """Standardized logistic regression; balanced accuracy and macro-F1."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    predicted = _fit_predict(train_x, train_y, test_x)
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, predicted)), 3),
        "macro_f1": round(float(f1_score(test_y, predicted, average="macro")), 3),
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
    }


def classify_grouped(x, y, groups) -> dict:
    """Leave-one-group-out version of :func:`classify`, scored on the pooled predictions."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    predicted = np.empty_like(y)
    for group in np.unique(groups):
        test = groups == group
        predicted[test] = _fit_predict(x[~test], y[~test], x[test])
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 3),
        "macro_f1": round(float(f1_score(y, predicted, average="macro")), 3),
        "folds": int(len(np.unique(groups))),
        "n": int(len(y)),
    }


def _fit_predict(train_x, train_y, test_x):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    return model.fit(train_x, train_y).predict(test_x)


def line_prominence(spectrum: np.ndarray, edges: np.ndarray, orders: list[float]) -> float:
    """Mean log amplitude within ±2 % of the given orders minus the spectrum's median."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = np.zeros_like(centres, dtype=bool)
    for order in orders:
        mask |= np.abs(centres - order) <= max(0.02 * order, edges[1] - edges[0])
    return float(spectrum[mask].mean() - np.median(spectrum))


def spectral_peaks(
    x: np.ndarray, rate: float, low: float, high: float, n: int = 10, seconds: float = 8.0
) -> list[float]:
    """The ``n`` strongest local maxima of the Welch PSD in [low, high] Hz, by frequency."""
    from scipy import signal as sps

    x = np.asarray(x, dtype=np.float64)
    freqs, psd = sps.welch(x - x.mean(), fs=rate, nperseg=min(len(x), int(rate * seconds)))
    band = (freqs >= low) & (freqs <= high)
    freqs, log_psd = freqs[band], np.log10(psd[band] + 1e-30)
    index, _ = sps.find_peaks(log_psd, distance=max(1, int(1.0 / (freqs[1] - freqs[0]))))
    strongest = index[np.argsort(log_psd[index])[-n:]]
    return sorted(round(float(freqs[i]), 2) for i in strongest)


def bearing_orders(geometry: dict, shaft_hz: float) -> dict[str, float]:
    freqs = pp.defect_frequencies(
        geometry["n_balls"],
        geometry["ball_d"],
        geometry["pitch_d"],
        geometry["contact_deg"],
        shaft_hz,
    )
    return {k: v / shaft_hz for k, v in freqs.items()}


# ----------------------------------------------------------------------------- CWRU


def eda_cwru(out: Path, quick: bool) -> dict:
    from pdmfm.datasets.cwru import GEOMETRY

    reader = get_reader("cwru")
    rows, feats, labels, loads, env = [], [], [], [], defaultdict(list)
    de_fe_corr, top_band = [], []
    for info in reader.index():
        rec = reader.load(info)
        de = rec.channels["de"]
        w = windows_of(de.signal, de.rate_hz, limit=4 if quick else None)
        shaft = rec.info.condition["rpm"] / 60
        estimate = pp.dominant_frequency(de.signal, de.rate_hz, 0.95 * shaft, 1.05 * shaft, 4.0)
        rows.append(
            {
                "file": info.recording_id,
                "label": info.label,
                "severity": info.severity,
                "load_hp": info.condition["load_hp"],
                "rpm": rec.info.condition["rpm"],
                "shaft_hz_from_vibration": round(estimate, 2),
                "rpm_vs_vibration_pct": round(100 * (shaft - estimate) / estimate, 2),
                "seconds": round(de.seconds, 2),
                "windows_1s": int(len(w)),
            }
        )
        if info.label == "normal":
            rows[-1]["raw_peaks_hz"] = spectral_peaks(de.signal, de.rate_hz, 20, 500, 8, 2.0)
        # Recorder roll-off near Nyquist, before and after the common band limit.
        unlimited = pp.frames(
            pp.resample(de.signal, de.rate_hz, pp.MODEL_RATE_HZ),
            pp.WINDOW_SAMPLES,
            pp.WINDOW_SAMPLES,
        )
        for before, after in zip(unlimited[:4], w[:4], strict=False):
            top_band.append(
                (
                    info.condition["load_hp"],
                    info.label,
                    top_band_ratio(before, pp.MODEL_RATE_HZ),
                    power_share_above(before, pp.MODEL_RATE_HZ),
                    power_share_above(after, pp.MODEL_RATE_HZ),
                )
            )
        for window in w:
            feats.append(band_features(window, pp.MODEL_RATE_HZ))
            labels.append(f"{info.label}|{info.severity}")
            loads.append(info.condition["load_hp"])
            spectrum, edges = pp.envelope_order_spectrum(window, pp.MODEL_RATE_HZ, shaft)
            env[info.label].append(spectrum)
        if "fe" in rec.channels:
            fe_w = windows_of(rec.channels["fe"].signal, rec.channels["fe"].rate_hz, limit=len(w))
            rms_de = [pp.level_descriptors(x)[0] for x in w]
            rms_fe = [pp.level_descriptors(x)[0] for x in fe_w[: len(w)]]
            if len(rms_de) > 2:
                de_fe_corr.append(float(np.corrcoef(rms_de, rms_fe)[0, 1]))
    feats, labels, loads = np.array(feats), np.array(labels), np.array(loads)
    train, test = loads < 3, loads == 3
    baseline = classify(feats[train], labels[train], feats[test], labels[test])
    # The recorders' roll-off alone tells normal from fault (resampled, not band-limited);
    # the band limit then leaves only a ~1e-8 power share above 5.25 kHz.
    tb_load = np.array([t[0] for t in top_band])
    tb_normal = np.array([t[1] == "normal" for t in top_band])
    ratio = np.array([[t[2]] for t in top_band])
    artifact = {
        "normal_median": round(float(np.median(ratio[tb_normal])), 2),
        "normal_range": [
            round(float(ratio[tb_normal].min()), 2),
            round(float(ratio[tb_normal].max()), 2),
        ],
        "fault_median": round(float(np.median(ratio[~tb_normal])), 2),
        "fault_range": [
            round(float(ratio[~tb_normal].min()), 2),
            round(float(ratio[~tb_normal].max()), 2),
        ],
        "normal_vs_fault_cross_load_from_this_one_feature": classify(
            ratio[tb_load < 3], tb_normal[tb_load < 3], ratio[tb_load == 3], tb_normal[tb_load == 3]
        ),
        "power_share_above_5p25khz_median": {
            stage: {
                "normal": float(f"{np.median([t[col] for t in top_band if t[1] == 'normal']):.3g}"),
                "fault": float(f"{np.median([t[col] for t in top_band if t[1] != 'normal']):.3g}"),
            }
            for col, stage in ((3, "resampled_only"), (4, "band_limited"))
        },
    }

    orders = bearing_orders(GEOMETRY, 1797 / 60)
    prominence = {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, kind, key in zip(
        axes, ("outer_race", "inner_race", "ball"), ("BPFO", "BPFI", "BSF"), strict=True
    ):
        mean_env = np.mean(env[kind], axis=0)
        normal_env = np.mean(env["normal"], axis=0)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centres, normal_env, color="0.6", lw=0.8, label="normal")
        ax.plot(centres, mean_env, color=COLOURS[0], lw=0.9, label=kind)
        for k in (1, 2, 3):
            ax.axvline(k * orders[key], color=COLOURS[3], lw=0.7, ls="--")
        ax.set_title(f"{kind}: {key} = {orders[key]:.2f} x shaft")
        ax.set_xlabel("shaft order")
        prominence[kind] = {
            "fault": round(
                line_prominence(mean_env, edges, [k * orders[key] for k in (1, 2, 3)]), 3
            ),
            "normal": round(
                line_prominence(normal_env, edges, [k * orders[key] for k in (1, 2, 3)]), 3
            ),
        }
    axes[0].set_ylabel("log10 envelope amplitude")
    axes[0].legend(frameon=False)
    fig.suptitle("CWRU drive end: defect orders stand out in the envelope spectrum of each fault")
    figures = [save(fig, out, "envelope_orders", report=True)]

    levels = feats[:, :4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    classes = sorted(set(labels))
    for ax, j, name in ((axes[0], 0, "log10 RMS (g)"), (axes[1], 2, "kurtosis")):
        data = [levels[labels == c, j] for c in classes]
        ax.boxplot(data, tick_labels=[c.replace("|", "\n") for c in classes], showfliers=False)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.suptitle("CWRU: level and impulsiveness differ by class and fault size")
    figures.append(save(fig, out, "levels_by_class", report=True))
    return {
        "recordings": len(rows),
        "windows_1s_total": int(sum(r["windows_1s"] for r in rows)),
        "classes": dict(Counter(r["label"] for r in rows)),
        "durations_s": sorted({r["seconds"] for r in rows}),
        "rpm_by_load": {
            str(k): sorted({r["rpm"] for r in rows if r["load_hp"] == k}) for k in range(4)
        },
        "rpm_vs_vibration_pct_max_abs": max(abs(r["rpm_vs_vibration_pct"]) for r in rows),
        "top_band_ratio_log10_5p7_6khz_minus_4_5khz": artifact,
        "normal_raw_peaks_hz": {r["file"]: r["raw_peaks_hz"] for r in rows if "raw_peaks_hz" in r},
        "de_fe_rms_correlation_median": round(float(np.median(de_fe_corr)), 3)
        if de_fe_corr
        else None,
        "cross_load_baseline_handcrafted": baseline,
        "envelope_line_prominence": prominence,
        "defect_orders": {k: round(v, 3) for k, v in orders.items()},
        "inventory": rows,
        "figures": figures,
    }


# ----------------------------------------------------------------------------- KAIST


def eda_kaist(out: Path, quick: bool) -> dict:
    from pdmfm.datasets.kaist import GEOMETRY, RPM

    reader = get_reader("kaist")
    infos = reader.index()
    if quick:
        infos = [i for i in infos if i.condition["load_nm"] in (0, 4)][:12]
    rows = []
    by_set = {"vibration": [], "current_1": [], "temperature": [], "all": []}
    y, fault_type, load = [], [], []
    env = defaultdict(list)
    current_spec = defaultdict(list)
    temp_vs_time = []
    shaft = RPM / 60
    for info in infos:
        rec = reader.load(info)
        meta = rec.info.meta
        vib = [rec.channels[k] for k in ("vib_ax", "vib_ay", "vib_bx", "vib_by")]
        vw = [windows_of(c.signal, c.rate_hz, limit=20 if quick else None) for c in vib]
        n = min(len(w) for w in vw)
        cur = rec.channels.get("current_1")
        cw = windows_of(cur.signal, cur.rate_hz, limit=n) if cur else np.zeros((0, 1))
        temps = [rec.channels[k] for k in ("temp_1", "temp_2") if k in rec.channels]
        temp_means = [float(np.mean(t.signal)) for t in temps]
        rows.append(
            {
                "condition": info.recording_id,
                "label": info.label,
                "severity": info.severity,
                "load_nm": info.condition["load_nm"],
                "seconds": round(vib[0].seconds, 1),
                "windows_1s": int(n),
                "empty_channels": meta.get("empty_channels", []),
                "device_offset_min": round(meta["device_offset_s"] / 60, 1)
                if meta.get("device_offset_s") is not None
                else None,
                "temperature_mean_c": [round(t, 2) for t in temp_means],
                "electrical_start": meta.get("electrical_start"),
                "recording_day": meta["vibration_start"][:10],
                "shaft_hz_from_vibration": round(
                    pp.dominant_frequency(vib[0].signal, vib[0].rate_hz, 40, 60), 3
                ),
                "current_fundamental_hz": round(
                    pp.dominant_frequency(cur.signal, cur.rate_hz, 20, 100), 3
                )
                if cur
                else None,
            }
        )
        temp_vs_time.append(
            (meta.get("electrical_start"), np.mean(temp_means) if temp_means else np.nan)
        )
        m = min(n, len(cw))
        for i in range(m):
            v_feat = np.concatenate([band_features(w[i], pp.MODEL_RATE_HZ) for w in vw])
            c_feat = band_features(cw[i], pp.MODEL_RATE_HZ)
            t_feat = np.array(temp_means, dtype=np.float32)
            by_set["vibration"].append(v_feat)
            by_set["current_1"].append(c_feat)
            by_set["temperature"].append(t_feat)
            by_set["all"].append(np.concatenate([v_feat, c_feat, t_feat]))
            y.append(f"{info.label}|{info.severity}")
            fault_type.append(info.label)
            load.append(info.condition["load_nm"])
            if i < 10:
                spectrum, edges = pp.envelope_order_spectrum(
                    vw[0][i], pp.MODEL_RATE_HZ, rows[-1]["shaft_hz_from_vibration"]
                )
                env[info.label].append(spectrum)
                cs, _ = pp.log_spectrum(cw[i], pp.MODEL_RATE_HZ, n_bins=5000)  # 1 Hz bands
                current_spec[info.label].append(cs[:400])
        del rec
    y, fault_type, load = np.array(y), np.array(fault_type), np.array(load)
    train, test = np.isin(load, (0, 2)), load == 4
    baselines, baselines_kind = {}, {}
    if train.any() and test.any():
        for name, feats in by_set.items():
            x = np.array(feats)
            baselines[name] = classify(x[train], y[train], x[test], y[test])
            baselines_kind[name] = classify(x[train], fault_type[train], x[test], fault_type[test])

    orders = bearing_orders(GEOMETRY, shaft)
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    prominence = {}
    for ax, kind, key in zip(axes, ("outer_race", "inner_race"), ("BPFO", "BPFI"), strict=True):
        if not env[kind]:
            continue
        mean_env, normal_env = np.mean(env[kind], axis=0), np.mean(env["normal"], axis=0)
        ax.plot(centres, normal_env, color="0.6", lw=0.8, label="normal")
        ax.plot(centres, mean_env, color=COLOURS[0], lw=0.9, label=kind)
        for k in (1, 2, 3):
            ax.axvline(k * orders[key], color=COLOURS[3], lw=0.7, ls="--")
        ax.set_title(f"{kind}: {key} = {orders[key]:.2f} x shaft (measured per run)")
        ax.set_xlabel("shaft order")
        lines = [k * orders[key] for k in (1, 2, 3)]
        prominence[kind] = {
            "fault": round(line_prominence(mean_env, edges, lines), 3),
            "normal": round(line_prominence(normal_env, edges, lines), 3),
        }
    axes[0].set_ylabel("log10 envelope amplitude")
    axes[0].legend(frameon=False)
    fig.suptitle("KAIST housing A, x: envelope spectra against the NSK 6205 defect orders")
    figures.append(save(fig, out, "envelope_orders", report=True))

    fig, ax = plt.subplots(figsize=(8, 3))
    freqs = np.arange(400) + 0.5
    for j, kind in enumerate(sorted(current_spec)):
        ax.plot(
            freqs, np.mean(current_spec[kind], axis=0), lw=0.8, color=COLOURS[j % 10], label=kind
        )
    supply = float(
        np.median([r["current_fundamental_hz"] for r in rows if r["current_fundamental_hz"]])
    )
    shaft_measured = float(np.median([r["shaft_hz_from_vibration"] for r in rows]))
    ax.axvline(supply, color="0.3", lw=0.6, ls=":")
    ax.axvline(shaft_measured, color=COLOURS[3], lw=0.6, ls="--")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("log10 amplitude (A)")
    ax.legend(frameon=False, ncol=5, fontsize=7)
    ax.set_title(
        f"KAIST current 1: fundamental {supply:.1f} Hz (dotted), below the shaft's "
        f"{shaft_measured:.1f} Hz (dashed)"
    )
    figures.append(save(fig, out, "current_spectra", report=True))

    times = [(np.datetime64(t.split(".")[0]) if t else None, v) for t, v in temp_vs_time]
    valid = [(t, v) for t, v in times if t is not None and np.isfinite(v)]
    rho = None
    if len(valid) > 3:
        order = np.argsort([t for t, _ in valid])
        rho = float(
            stats.spearmanr(np.arange(len(valid)), np.array([v for _, v in valid])[order])[0]
        )
        fig, ax = plt.subplots(figsize=(8, 3))
        ts = np.array([t for t, _ in valid])
        vs = np.array([v for _, v in valid])
        labels_for = [r["label"] for r in rows if r.get("electrical_start")]
        for j, kind in enumerate(sorted(set(labels_for))):
            sel = np.array([lab == kind for lab in labels_for])
            ax.scatter(ts[sel], vs[sel], s=14, color=COLOURS[j % 10], label=kind)
        ax.set_ylabel("mean housing temperature (°C)")
        ax.legend(frameon=False, fontsize=7, ncol=5)
        ax.set_title("KAIST: fault groups were recorded on different days; temperature tracks them")
        fig.autofmt_xdate()
        figures.append(save(fig, out, "temperature_vs_time", report=True))
    return {
        "conditions": len(rows),
        "windows_1s_total": int(sum(r["windows_1s"] for r in rows)),
        "windows_1s_by_load": {
            str(k): int(sum(r["windows_1s"] for r in rows if r["load_nm"] == k)) for k in (0, 2, 4)
        },
        "durations_s": dict(Counter(r["seconds"] for r in rows)),
        "class_dependent_empty_channels": sorted(
            {r["condition"] for r in rows if r["empty_channels"]}
        ),
        "cross_load_classes": int(len(set(y))),
        "cross_load_baselines_handcrafted": baselines,
        "cross_load_baselines_handcrafted_fault_type": baselines_kind,
        "classes_by_recording_day": {
            day: sorted({r["label"] for r in rows if r["recording_day"] == day})
            for day in sorted({r["recording_day"] for r in rows})
        },
        "shaft_hz_from_vibration_by_load": {
            str(k): float(
                np.median([r["shaft_hz_from_vibration"] for r in rows if r["load_nm"] == k])
            )
            for k in sorted({r["load_nm"] for r in rows})
        },
        "current_fundamental_hz_by_load": {
            str(k): float(
                np.median([r["current_fundamental_hz"] for r in rows if r["load_nm"] == k])
            )
            for k in sorted({r["load_nm"] for r in rows})
        },
        "temperature_vs_recording_order_spearman": round(rho, 3) if rho is not None else None,
        "envelope_line_prominence": prominence,
        "defect_orders": {k: round(v, 3) for k, v in orders.items()},
        "inventory": rows,
        "figures": figures,
    }


# ----------------------------------------------------------------------------- XJTU-SY

# (healthy snapshots for the threshold, sigmas, consecutive exceedances); the first is the
# default of pp.first_prediction_time. A baseline below 1 is a fraction of the bearing's
# life (at least 10 snapshots): usable for offline labels only, since it needs the lifetime.
FPT_VARIANTS = {
    "b10_3sd_c3": (10, 3.0, 3),
    "b10_3sd_c10": (10, 3.0, 10),
    "b30_3sd_c10": (30, 3.0, 10),
    "b30_4sd_c30": (30, 4.0, 30),
    "b10pct_3sd_c10": (0.1, 3.0, 10),
}
ROBUST_FPT = "b10pct_3sd_c10"  # recommended for offline RUL labels (docs/EDA_REPORT.md)


def eda_xjtu_sy(out: Path, quick: bool) -> dict:
    from pdmfm.datasets.xjtu_sy import CONDITIONS, FAILURE, GEOMETRY, RATE_HZ, read_snapshot

    reader = get_reader("xjtu_sy")
    bearings = reader.bearings()
    if quick:
        bearings = dict(list(bearings.items())[:3])
    series, summary = {}, {}
    env_end = {}
    for bearing, (condition, paths) in bearings.items():
        step = 5 if quick else 1
        rms_h, rms_v, kurt_h = [], [], []
        for path in paths[::step]:
            values = read_snapshot(path)
            rms_h.append(float(np.sqrt(np.mean((values[:, 0] - values[:, 0].mean()) ** 2))))
            rms_v.append(float(np.sqrt(np.mean((values[:, 1] - values[:, 1].mean()) ** 2))))
            kurt_h.append(float(stats.kurtosis(values[:, 0], fisher=False)))
        series[bearing] = (np.array(rms_h), np.array(kurt_h))
        fpt = pp.first_prediction_time(np.array(rms_h))
        variants = {}
        for name, (count, sigmas, consecutive) in FPT_VARIANTS.items():
            if count < 1:
                count = max(10, int(count * len(rms_h)))
            index = pp.first_prediction_time(np.array(rms_h), count, sigmas, consecutive)
            variants[name] = None if index is None else round(index * step / len(paths), 3)
        shaft = CONDITIONS[condition]["rpm"] / 60
        last = read_snapshot(paths[-1])[:, 0]
        first = read_snapshot(paths[0])[:, 0]
        w_last = pp.model_input(last, RATE_HZ)[0][: pp.WINDOW_SAMPLES]
        w_first = pp.model_input(first, RATE_HZ)[0][: pp.WINDOW_SAMPLES]
        e_last, edges = pp.envelope_order_spectrum(w_last, pp.MODEL_RATE_HZ, shaft)
        e_first, _ = pp.envelope_order_spectrum(w_first, pp.MODEL_RATE_HZ, shaft)
        env_end[bearing] = (e_first, e_last)
        orders = bearing_orders(GEOMETRY, shaft)
        summary[bearing] = {
            "condition": condition,
            "lifetime_min": len(paths),
            "failure": FAILURE[bearing],
            "fpt_index": None if fpt is None else int(fpt * step),
            "healthy_fraction": None if fpt is None else round(fpt * step / len(paths), 3),
            "healthy_fraction_by_fpt_rule": variants,
            "rms_first_g": round(rms_h[0], 3),
            "rms_last_g": round(rms_h[-1], 3),
            "rms_ratio_last_first": round(rms_h[-1] / rms_h[0], 1),
            "kurtosis_last": round(kurt_h[-1], 2),
            "bpfo_prominence_end_minus_start": round(
                line_prominence(e_last, edges, [orders["BPFO"], 2 * orders["BPFO"]])
                - line_prominence(e_first, edges, [orders["BPFO"], 2 * orders["BPFO"]]),
                3,
            ),
            "bpfi_prominence_end_minus_start": round(
                line_prominence(e_last, edges, [orders["BPFI"], 2 * orders["BPFI"]])
                - line_prominence(e_first, edges, [orders["BPFI"], 2 * orders["BPFI"]]),
                3,
            ),
        }
    figures = []
    fig, axes = plt.subplots(3, 5, figsize=(13, 6), sharey=False)
    for ax, (bearing, (rms, _)) in zip(axes.ravel(), series.items(), strict=False):
        life = np.linspace(0, 1, len(rms))
        ax.plot(life, rms, lw=0.8, color=COLOURS[0])
        fpt = summary[bearing]["fpt_index"]
        if fpt is not None:
            ax.axvline(fpt / summary[bearing]["lifetime_min"], color=COLOURS[3], lw=0.8, ls="--")
        robust = summary[bearing]["healthy_fraction_by_fpt_rule"][ROBUST_FPT]
        if robust is not None:
            ax.axvline(robust, color=COLOURS[2], lw=0.8, ls=":")
        ax.set_title(f"{bearing} ({summary[bearing]['lifetime_min']} min)", fontsize=8)
        ax.set_yscale("log")
    for ax in axes.ravel()[len(series) :]:
        ax.axis("off")
    fig.supxlabel("fraction of life")
    fig.supylabel("horizontal RMS (g)")
    fig.suptitle(
        "XJTU-SY: RMS stays flat for most of life, then rises; "
        f"FPT: red dashed = default rule, green dotted = {ROBUST_FPT}"
    )
    figures.append(save(fig, out, "rms_trajectories", report=True))
    healthy = [v["healthy_fraction"] for v in summary.values() if v["healthy_fraction"] is not None]
    return {
        "bearings": len(summary),
        "snapshots": int(sum(v["lifetime_min"] for v in summary.values())),
        "lifetime_min_by_condition": {
            c: [v["lifetime_min"] for v in summary.values() if v["condition"] == c]
            for c in CONDITIONS
        },
        "fpt_found": f"{len(healthy)} of {len(summary)}",
        "healthy_fraction_median": round(float(np.median(healthy)), 3) if healthy else None,
        "fpt_rules": {
            k: dict(zip(("baseline", "sigmas", "consecutive"), v, strict=True))
            for k, v in FPT_VARIANTS.items()
        },
        "fpt_alarms_before_5pct_of_life": {
            name: sorted(
                b
                for b, v in summary.items()
                if (h := v["healthy_fraction_by_fpt_rule"][name]) is not None and h < 0.05
            )
            for name in FPT_VARIANTS
        },
        "per_bearing": summary,
        "figures": figures,
    }


# ----------------------------------------------------------------------------- Nonastreda


def eda_nonastreda(out: Path, quick: bool) -> dict:
    from PIL import Image

    reader = get_reader("nonastreda")
    infos = reader.index()
    if quick:
        infos = [i for i in infos if i.unit in ("T1", "T5")]
    per_pass = []
    spectra = defaultdict(list)
    first_pass_peaks = {}
    for info in infos:
        rec = reader.load(info)
        row = {
            "id": info.recording_id,
            "tool": info.unit,
            "label": info.label,
            "wear_um": info.meta["flank_wear"],
            "seconds": round(info.meta["seconds"], 1),
            "run": info.meta["run"],
            "block": info.meta["block"],
        }
        for name in ("fx", "fy", "fz"):
            x = rec.channels[name].signal
            levels = pp.level_descriptors(x)
            row[f"{name}_mean"] = float(np.mean(x))
            row[f"{name}_rms_ac"] = float(10 ** levels[0])
            row[f"{name}_kurtosis"] = float(levels[2])
        for name in ("fx", "fy", "fz"):
            x = rec.channels[name].signal.astype(np.float64)
            # Phase-locked pattern over 15 samples (the 66.67 Hz line) versus a 16-sample control.
            for period in (15, 16):
                folded = x[: len(x) // period * period].reshape(-1, period).mean(axis=0)
                row[f"{name}_sync{period}_p2p"] = float(np.ptp(folded))
            steps = np.abs(np.diff(x))
            row[f"{name}_adc_step"] = float(steps[steps > 0].min())
        fx = rec.channels["fx"].signal
        row["fx_line_hz"] = round(pp.dominant_frequency(fx, 1000.0, 55, 80, 16.0), 3)
        if info.unit not in first_pass_peaks:
            first_pass_peaks[info.unit] = spectral_peaks(fx, 1000.0, 3, 495, 12, 16.0)
        w = pp.frames(fx, 8192, 8192)
        row["windows_8s"] = int(len(w))
        if len(w):
            s, edges = pp.log_spectrum(w[0], 1000.0, n_bins=500)
            spectra[info.label].append(s)
        tool_photo = next((i for i in rec.images if i.kind == "tool"), None)
        if tool_photo:
            image = np.asarray(
                Image.open(tool_photo.path).convert("L").resize((672, 224)), np.float32
            )
            row["tool_photo_mean"] = float(image.mean())
            row["tool_photo_std"] = float(image.std())
        per_pass.append(row)
    wear = np.array([r["wear_um"] for r in per_pass])
    correlations = {}
    for key in (
        "fx_mean",
        "fy_mean",
        "fz_mean",
        "fx_rms_ac",
        "fy_rms_ac",
        "fz_rms_ac",
        "tool_photo_mean",
        "tool_photo_std",
    ):
        values = np.array([r.get(key, np.nan) for r in per_pass], dtype=float)
        ok = np.isfinite(values)
        correlations[key] = round(float(stats.spearmanr(values[ok], wear[ok])[0]), 3)
    # Baseline-relative levels: divide by each tool's first pass.
    relative = {}
    for key in ("fx_rms_ac", "fy_rms_ac"):
        rel, wr = [], []
        for tool in sorted({r["tool"] for r in per_pass}):
            rows = [r for r in per_pass if r["tool"] == tool]
            first = rows[0][key]
            rel += [r[key] / first for r in rows]
            wr += [r["wear_um"] for r in rows]
        relative[key] = round(float(stats.spearmanr(rel, wr)[0]), 3)
    # Persistence of the label from one pass to the next (what a lag-1 photo can give).
    same, total = 0, 0
    for tool in sorted({r["tool"] for r in per_pass}):
        labels = [r["label"] for r in per_pass if r["tool"] == tool]
        same += sum(a == b for a, b in zip(labels, labels[1:], strict=False))
        total += max(0, len(labels) - 1)
    # Leave-one-tool-out floor from per-pass Fx/Fy levels (Fz is left out: it drifts by block).
    keys = ("fx_mean", "fy_mean", "fx_rms_ac", "fy_rms_ac", "fx_kurtosis", "fy_kurtosis")
    loto = None
    if len({r["tool"] for r in per_pass}) > 1:
        loto = classify_grouped(
            [[r[k] for k in keys] for r in per_pass],
            [r["label"] for r in per_pass],
            [r["tool"] for r in per_pass],
        )
    # Fz drift: mean Fz within a run against block order.
    fz_by_block = defaultdict(list)
    for r in per_pass:
        fz_by_block[r["block"]].append(r["fz_mean"])
    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for j, tool in enumerate(sorted({r["tool"] for r in per_pass}, key=lambda t: int(t[1:]))):
        rows = [r for r in per_pass if r["tool"] == tool]
        axes[0].plot([r["wear_um"] for r in rows], lw=0.8, color=COLOURS[j % 10], label=tool)
        axes[1].scatter(
            [r["wear_um"] for r in rows], [r["fy_rms_ac"] for r in rows], s=6, color=COLOURS[j % 10]
        )
    axes[0].axhline(71, color="0.5", lw=0.6, ls="--")
    axes[0].axhline(110, color="0.5", lw=0.6, ls="--")
    axes[0].set_xlabel("pass (deduplicated, run/block order)")
    axes[0].set_ylabel("flank wear (µm)")
    axes[0].legend(frameon=False, ncol=2, fontsize=6)
    axes[1].set_xlabel("flank wear (µm)")
    axes[1].set_ylabel("Fy AC RMS (N)")
    axes[1].set_title(f"Spearman {correlations['fy_rms_ac']:+.2f}")
    centres = np.arange(500) + 0.5
    for j, label in enumerate(("sharp", "used", "dulled")):
        if spectra[label]:
            axes[2].plot(
                centres, np.mean(spectra[label], axis=0), lw=0.8, color=COLOURS[j], label=label
            )
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("log10 |Fx| (N)")
    axes[2].legend(frameon=False)
    fig.suptitle(
        "Nonastreda: wear grows per tool; force level rises with wear; Fx spectra by class"
    )
    figures.append(save(fig, out, "wear_force", report=True))
    peak_freqs = {}
    for label, s in spectra.items():
        mean = np.mean(s, axis=0)
        peak_freqs[label] = [float(centres[i]) for i in np.argsort(mean[5:])[-3:][::-1] + 5]
    return {
        "passes": len(per_pass),
        "tools": len({r["tool"] for r in per_pass}),
        "classes": dict(Counter(r["label"] for r in per_pass)),
        "pass_seconds": {
            "min": min(r["seconds"] for r in per_pass),
            "median": float(np.median([r["seconds"] for r in per_pass])),
            "max": max(r["seconds"] for r in per_pass),
        },
        "windows_8s_total": int(sum(r["windows_8s"] for r in per_pass)),
        "spearman_with_wear": correlations,
        "spearman_with_wear_relative_to_first_pass": relative,
        "next_pass_same_label_rate": round(same / total, 3) if total else None,
        "leave_one_tool_out_baseline_fx_fy_levels": loto,
        "fz_mean_by_block": {
            str(k): round(float(np.mean(v)), 2) for k, v in sorted(fz_by_block.items())
        },
        "fx_spectrum_top_peaks_hz": peak_freqs,
        "fx_line_hz_across_passes": {
            "min": min(r["fx_line_hz"] for r in per_pass),
            "median": float(np.median([r["fx_line_hz"] for r in per_pass])),
            "max": max(r["fx_line_hz"] for r in per_pass),
        },
        "fx_peaks_hz_first_pass_by_tool": first_pass_peaks,
        "synchronous_average_p2p_n_median": {
            name: {
                f"period_{p}": round(
                    float(np.median([r[f"{name}_sync{p}_p2p"] for r in per_pass])), 3
                )
                for p in (15, 16)
            }
            for name in ("fx", "fy", "fz")
        },
        "adc_step_n": {
            name: round(float(np.median([r[f"{name}_adc_step"] for r in per_pass])), 4)
            for name in ("fx", "fy", "fz")
        },
        "figures": figures,
    }


# ----------------------------------------------------------------------------- Najafi


def eda_najafi(out: Path, quick: bool) -> dict:
    from PIL import Image

    reader = get_reader("najafi_thermal")
    infos = reader.index()
    images, labels, folders = [], [], []
    for info in infos:
        image = np.asarray(Image.open(info.meta["path"]).convert("RGB"), np.float32) / 255.0
        images.append(image)
        labels.append(info.label)
        folders.append(info.condition["folder"])
    stack = np.stack(images)
    folders = np.array(folders)
    # Near duplicates: distance of each image to its nearest neighbour in the same folder
    # versus in another folder.
    flat = stack.reshape(len(stack), -1)
    small = flat[:, ::97]
    dist = np.sqrt(((small[:, None, :] - small[None, :, :]) ** 2).mean(axis=2))
    np.fill_diagonal(dist, np.inf)
    same_folder = folders[:, None] == folders[None, :]
    same = np.where(same_folder, dist, np.inf).min(axis=1)
    other = np.where(~same_folder, dist, np.inf).min(axis=1)
    # Are near duplicates neighbouring files (a frame sequence)? Position = rank in the folder.
    position = np.array([int((folders[:i] == folders[i]).sum()) for i in range(len(folders))])
    nearest = np.where(same_folder, dist, np.inf).argmin(axis=1)
    adjacent = float((np.abs(position - position[nearest]) <= 2).mean())
    # Split inflation: random 70/30 split versus the last 30 % of every folder held out.
    labels_arr = np.array(labels)
    rng = np.random.default_rng(0)
    random_test = rng.random(len(labels_arr)) < 0.3
    sizes = Counter(folders.tolist())
    block_test = np.array([p >= 0.7 * sizes[f] for p, f in zip(position, folders, strict=True)])
    splits = {
        name: classify(small[~t], labels_arr[~t], small[t], labels_arr[t])
        for name, t in (("random_70_30", random_test), ("last_30pct_of_each_folder", block_test))
    }
    figures = []
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.4))
    for ax, folder in zip(axes, ("Noload", "A50", "Fan", "Rotor-0"), strict=True):
        idx = int(np.where(folders == folder)[0][0])
        ax.imshow(images[idx])
        ax.set_title(folder)
        ax.axis("off")
    fig.suptitle("Najafi thermal images (320x240 pseudo-colour): healthy, stator, fan, stuck rotor")
    figures.append(save(fig, out, "examples", report=True))
    return {
        "images": len(infos),
        "classes": dict(Counter(labels)),
        "folders": dict(Counter(folders.tolist())),
        "nearest_neighbour_distance_same_folder_median": round(float(np.median(same)), 4),
        "nearest_neighbour_distance_other_folder_median": round(float(np.median(other)), 4),
        "share_of_images_whose_nearest_neighbour_is_in_the_same_folder": round(
            float((same < other).mean()), 3
        ),
        "share_whose_same_folder_neighbour_is_within_2_files": round(adjacent, 3),
        "pixel_baseline_by_split": splits,
        "figures": figures,
    }


# ----------------------------------------------------------------------------- NASA milling


def eda_nasa_milling(out: Path, quick: bool) -> dict:
    reader = get_reader("nasa_milling")
    infos = reader.index()
    rows = []
    for info in infos:
        rec = reader.load(info)
        row = {"case": info.unit, "wear_um": info.meta["wear_um"], **info.condition}
        for name, channel in rec.channels.items():
            row[f"{name}_mean"] = float(np.mean(channel.signal))
        rows.append(row)
    labelled = [r for r in rows if r["wear_um"] is not None]
    wear = np.array([r["wear_um"] for r in labelled])
    corr = {}
    for key in [k for k in labelled[0] if k.endswith("_mean")]:
        corr[key] = round(float(stats.spearmanr([r[key] for r in labelled], wear)[0]), 3)
    fig, ax = plt.subplots(figsize=(7, 3))
    for j, case in enumerate(sorted({r["case"] for r in rows})):
        values = [r["wear_um"] for r in rows if r["case"] == case and r["wear_um"] is not None]
        ax.plot(values, lw=0.8, marker=".", ms=3, color=COLOURS[j % 10])
    ax.set_xlabel("measured run")
    ax.set_ylabel("flank wear VB (µm)")
    ax.set_title("NASA milling: 16 cases, few runs each, wear to about 0.5 mm and beyond")
    figures = [save(fig, out, "wear_by_case", report=True)]
    return {
        "runs": len(rows),
        "cases": len({r["case"] for r in rows}),
        "runs_with_wear": len(labelled),
        "wear_um_range": [float(wear.min()), float(wear.max())],
        "spearman_with_wear": corr,
        "figures": figures,
    }


# ----------------------------------------------------------------------------- MBHM


def eda_mbhm(out: Path, quick: bool) -> dict:
    con = sqlite3.connect(raw_dir("mbhm") / "metadata.sqlite")
    rows = con.execute(
        """select c.dataset, f.label, count(*) from file_info f
           join condition c on c.condition_id = f.condition_id group by c.dataset, f.label"""
    ).fetchall()
    notes = dict(con.execute("select label, note from label_note").fetchall())
    table = defaultdict(dict)
    for dataset, label, count in rows:
        table[dataset][notes[label]] = count
    usable = {k: v for k, v in table.items() if k not in ("PU", "XJTU")}
    return {
        "windows_by_dataset_and_label": table,
        "usable_windows_without_PU_XJTU": int(sum(sum(v.values()) for v in usable.values())),
        "figures": [],
    }


# ----------------------------------------------------------------------------- Paderborn, PHM 2010


def eda_inventory_only(name: str):
    def run(out: Path, quick: bool) -> dict:
        reader = get_reader(name)
        infos = reader.index()
        first = reader.load(infos[0]) if infos else None
        return {
            "recordings": len(infos),
            "units": len({i.unit for i in infos}),
            "labels": dict(Counter(i.label for i in infos)),
            "first_recording_channels": {
                k: {"rate_hz": c.rate_hz, "samples": len(c.signal), "family": c.family}
                for k, c in (first.channels.items() if first else [])
            },
            "note": "inventory only; extend once the files are verified (VERIFIED_ON_REAL_DATA)",
            "figures": [],
        }

    return run


ANALYSES = {
    "cwru": (eda_cwru, "cwru", False),
    "kaist": (eda_kaist, "kaist", True),
    "xjtu_sy": (eda_xjtu_sy, "xjtu_sy", True),
    "nonastreda": (eda_nonastreda, "nonastreda", False),
    "najafi_thermal": (eda_najafi, "najafi_thermal", True),
    "nasa_milling": (eda_nasa_milling, "nasa_milling", True),
    "mbhm": (eda_mbhm, "mbhm", False),
    "paderborn": (eda_inventory_only("paderborn"), "paderborn", True),
    "phm2010": (eda_inventory_only("phm2010"), "phm2010", True),
}


def available(name: str) -> bool:
    _, folder_name, extracted = ANALYSES[name]
    folder = extracted_dir(folder_name) if extracted else raw_dir(folder_name)
    return folder.is_dir() and any(folder.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="any of: " + ", ".join(sorted(ANALYSES)))
    parser.add_argument("--quick", action="store_true", help="subset of recordings")
    parser.add_argument("--out", type=Path, default=None, help="default: <data root>/../eda")
    parser.add_argument("--force", action="store_true", help="run even if the machine is busy")
    args = parser.parse_args()
    unknown = sorted(set(args.datasets) - set(ANALYSES))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")
    global COPY_TO_DOCS
    COPY_TO_DOCS = not args.quick
    claim_resources("EDA", min_idle_cpus=2.0, min_free_ram_gb=2.5, limit=4, force=args.force)
    out_root = args.out or data_root().parent / "eda"
    names = args.datasets or [n for n in ANALYSES if available(n)]
    (DOCS / "summary").mkdir(parents=True, exist_ok=True)
    for name in names:
        if not available(name):
            print(f"skip {name}: not downloaded")
            continue
        start = time.perf_counter()
        summary = ANALYSES[name][0](out_root / name, args.quick)
        summary["runtime_s"] = round(time.perf_counter() - start, 1)
        summary["quick"] = args.quick
        text = json.dumps(summary, indent=2, default=str) + "\n"  # end-of-file hook
        (out_root / name).mkdir(parents=True, exist_ok=True)
        (out_root / name / "summary.json").write_text(text, encoding="utf-8")
        if not args.quick:
            (DOCS / "summary" / f"{name}.json").write_text(text, encoding="utf-8")
        print(f"{name}: done in {summary['runtime_s']} s -> {out_root / name}")


if __name__ == "__main__":
    main()
