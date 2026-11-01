"""Stage 11 (part): frequency-band occlusion, the physics alignment score and two
faithfulness checks (plan, "Trust: uncertainty, calibration and explanations").

A band's attribution is the drop in the predicted class's probability when the band's
narrow-band content is removed and the window is encoded again. "Removed" is *line removal*:
inside the band every spectral magnitude is replaced by the band's median magnitude, phases
kept, so lines vanish and the broadband floor stays. Zeroing the band would instead hand the
encoder a spectral hole it never saw, and the drop would measure that.

Two axes:

* ``hz``: the raw window's spectrum, 64 bands over 0-5 kHz (the common band); every arm is
  recomputed from the edited window. Defect lines k·f in Hz.
* ``order`` (bearings, models with the E arm): the envelope order spectrum, 64 bands over
  0-25 shaft orders; the other arms are unchanged. A bearing defect modulates a structural
  resonance at the defect rate, so its lines live in the envelope: this is the axis the
  physics check is about. The raw spectrum shows them only weakly.

Physics alignment score (PAS) of one window: the share of its positive attribution that
falls in bands within ±2 % of a defect line (k = 1..5; inner race with ±1x shaft sidebands)
divided by those bands' share of the axis. PAS > 1 means the model relies on the defect
lines more than chance. It is scored on correctly classified faulty windows and compared
with the same model with its weights randomized (Adebayo et al. 2018), whose PAS should stay
near 1; for a probe the trained weights are its classifier, the encoder being frozen.

Deletion test: the bands are removed cumulatively in the order of their attribution and, as
the reference, in random orders. A faithful attribution makes the predicted class's
probability fall faster: a smaller area under the curve.

The model explained here is the Gate-1 probe on CWRU (fit on 0-2 hp, explained on the 3 hp
windows): CWRU's specimens carry seeded faults of known type, so the expected lines are
known. Plan B and the final model plug in through the same interface (``inputs``,
``features``, ``predict_features``) once their models are kept (stage 11 proper).

    python -m pdmfm.explain --dataset cwru --encoder mantis --arms F E --run-id pas-cwru-mantis
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pdmfm import arms
from pdmfm import preprocess as pp
from pdmfm.paths import REPO_ROOT
from pdmfm.utils import write_json_atomic
from pdmfm.windows import WindowStore

N_BANDS = 64
TOLERANCE = 0.02  # ±2 % around each defect line
HARMONICS = 5
MAX_ORDER = 25.0  # the E arm's order axis (pp.envelope_order_spectrum)
DELETION_STEPS = 16
RANDOM_ORDERS = 3
# Fault labels of each dataset (manifest "label", XJTU-SY's failure location) -> line family.
FAULT_KIND = {
    "outer_race": "outer_race",
    "outer race": "outer_race",
    "inner_race": "inner_race",
    "inner race": "inner_race",
    "ball": "ball",
    "cage": "cage",
}


def geometry(dataset: str) -> dict:
    """Bearing geometry of a dataset's tested bearing (from its reader)."""
    if dataset == "cwru":
        from pdmfm.datasets.cwru import GEOMETRY
    elif dataset == "kaist":
        from pdmfm.datasets.kaist import GEOMETRY
    elif dataset == "paderborn":
        from pdmfm.datasets.paderborn import GEOMETRY
    elif dataset == "xjtu_sy":
        from pdmfm.datasets.xjtu_sy import GEOMETRY
    else:
        raise ValueError(f"{dataset} has no bearing geometry (the physics score needs one)")
    return dict(GEOMETRY)


# ----------------------------------------------------------------------------- physics


def defect_lines(
    geometry: dict, fault: str, shaft_hz: float = 1.0, harmonics: int = HARMONICS
) -> np.ndarray:
    """Frequencies of a fault's lines: k x the defect frequency (k = 1..``harmonics``), and
    for the inner race also k x BPFI ± 1x shaft (the inner ring turns through the load
    zone). With ``shaft_hz = 1`` the lines are in shaft orders."""
    kind = FAULT_KIND.get(fault)
    if kind is None:
        raise ValueError(f"no defect lines for fault {fault!r}")
    rates = pp.defect_frequencies(
        geometry["n_balls"], geometry["ball_d"], geometry["pitch_d"],
        geometry["contact_deg"], shaft_hz,
    )  # fmt: skip
    base = {"outer_race": "BPFO", "inner_race": "BPFI", "ball": "BSF", "cage": "FTF"}[kind]
    k = np.arange(1, harmonics + 1)
    lines = [k * rates[base]]
    if kind == "inner_race":
        lines += [k * rates[base] - shaft_hz, k * rates[base] + shaft_hz]
    return np.sort(np.concatenate(lines))


def band_edges(axis: str, rate: float = pp.MODEL_RATE_HZ, n_bands: int = N_BANDS) -> np.ndarray:
    """(n_bands + 1,) band edges: Hz up to the common band (or Nyquist), or shaft orders."""
    if axis == "hz":
        return np.linspace(0.0, min(pp.COMMON_BAND_HZ, rate / 2.0), n_bands + 1)
    if axis == "order":
        return np.linspace(0.0, MAX_ORDER, n_bands + 1)
    raise ValueError(f"axis must be hz or order, not {axis!r}")


def physics_mask(edges: np.ndarray, lines: np.ndarray, tolerance: float = TOLERANCE) -> np.ndarray:
    """(n_bands,) bool: bands that overlap [f(1 - tolerance), f(1 + tolerance)] of a line."""
    low, high = edges[:-1], edges[1:]
    mask = np.zeros(len(low), dtype=bool)
    for f in np.asarray(lines, dtype=float):
        if f <= 0:
            continue
        mask |= (high > f * (1 - tolerance)) & (low < f * (1 + tolerance))
    return mask


def pas(attribution: np.ndarray, mask: np.ndarray) -> float:
    """Physics alignment score of one window: positive attribution share in ``mask`` over
    the mask's share of the bands; NaN without positive attribution or physical bands."""
    positive = np.clip(np.asarray(attribution, dtype=float), 0.0, None)
    total = positive.sum()
    if total <= 0 or not mask.any():
        return float("nan")
    return float((positive[mask].sum() / total) / mask.mean())


# ----------------------------------------------------------------------------- occlusion


def remove_lines(x: np.ndarray, rate: float, bands: list[tuple[float, float]]) -> np.ndarray:
    """Copy of ``x`` (..., L) whose spectral magnitudes inside each band [low, high) are set
    to the band's median magnitude (per series), phases kept; DC is never touched."""
    x = np.asarray(x, dtype=np.float64)
    spectrum = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / rate)
    for low, high in bands:
        inside = (freqs >= low) & (freqs < high) & (freqs > 0)
        if inside.sum() < 2:
            continue
        part = spectrum[..., inside]
        floor = np.median(np.abs(part), axis=-1, keepdims=True)
        spectrum[..., inside] = floor * np.exp(1j * np.angle(part))
    return np.fft.irfft(spectrum, n=x.shape[-1], axis=-1).astype(np.float32)


def flatten_bins(spectrum: np.ndarray, bands: list[tuple[int, int]]) -> np.ndarray:
    """Copy of a binned log spectrum (..., B) whose bins [start, stop) of each band are set
    to their median: lines removed, floor kept (the E arm's counterpart of
    :func:`remove_lines`)."""
    out = np.array(spectrum, dtype=np.float32, copy=True)
    for start, stop in bands:
        out[..., start:stop] = np.median(out[..., start:stop], axis=-1, keepdims=True)
    return out


# ----------------------------------------------------------------------------- the model


@dataclass
class ProbeModel:
    """A probe on frozen features, evaluated online: raw windows -> arm inputs -> encoder ->
    tokens averaged per channel (as :meth:`pdmfm.features.FeatureSet.pooled`) -> classifier.
    ``channels`` maps each arm to the store's channel indices that feed it."""

    rate: float
    arm_list: tuple[str, ...]
    channels: dict[str, tuple[int, ...]]
    encoder: object
    classifier: object
    classes: list[str] = field(default_factory=list)

    def inputs(self, windows: np.ndarray, shaft: np.ndarray) -> dict[str, np.ndarray]:
        return {
            arm: arms.window_input(windows[:, list(self.channels[arm])], self.rate, arm, shaft)
            for arm in self.arm_list
        }

    def arm_features(self, arm: str, x: np.ndarray) -> np.ndarray:
        """(n, C * D) of one arm's inputs (n, C, k, 512)."""
        n, c, k, length = x.shape
        z = np.asarray(self.encoder.encode(x.reshape(n * c * k, length)), dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]
        return z.reshape(n, c, k, -1).mean(axis=2).reshape(n, -1)

    def features(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([self.arm_features(a, inputs[a]) for a in self.arm_list], axis=1)

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(x))  # pyright: ignore[reportAttributeAccessIssue]

    def predict(self, windows: np.ndarray, shaft: np.ndarray) -> np.ndarray:
        return self.predict_features(self.features(self.inputs(windows, shaft)))


def randomized(model: ProbeModel, seed: int = 0) -> ProbeModel:
    """The model with its trained weights randomized: the classifier's coefficients permuted
    (same scale, no learned structure). The frozen encoder is not part of what was trained."""
    rng = np.random.default_rng(seed)
    classifier = copy.deepcopy(model.classifier)
    final: Any = classifier.steps[-1][1] if hasattr(classifier, "steps") else classifier  # pyright: ignore[reportAttributeAccessIssue]
    coef = np.asarray(final.coef_)
    final.coef_ = rng.permutation(coef.ravel()).reshape(coef.shape)
    return ProbeModel(model.rate, model.arm_list, model.channels, model.encoder, classifier,
                      model.classes)  # fmt: skip


def variant_features(
    model: ProbeModel, window: np.ndarray, shaft: float, axis: str, band_sets: list[list[int]]
) -> np.ndarray:
    """(len(band_sets), F) features of ``window`` (C, L) with each set of bands removed. The
    encoder runs once per variant; any classifier on the same encoder can reuse them."""
    m = len(band_sets)
    if axis == "hz":
        edges = band_edges("hz", model.rate)
        edited = np.stack(
            [remove_lines(window, model.rate, [(edges[b], edges[b + 1]) for b in s])
             for s in band_sets]
        )  # fmt: skip
        return model.features(model.inputs(edited, np.full(m, shaft)))
    if "E" not in model.arm_list:
        raise ValueError("the order axis needs a model with the E arm")
    base_inputs = model.inputs(window[None], np.array([shaft]))
    per_band = pp.SEGMENT // N_BANDS
    blocks = []
    for arm in model.arm_list:
        x = base_inputs[arm]
        if arm == "E":
            edited = np.stack(
                [flatten_bins(x[0], [(b * per_band, (b + 1) * per_band) for b in s])
                 for s in band_sets]
            )  # fmt: skip
            blocks.append(model.arm_features("E", edited))
        else:  # unchanged by an edit of the envelope
            blocks.append(np.repeat(model.arm_features(arm, x), m, axis=0))
    return np.concatenate(blocks, axis=1)


def single_bands() -> list[list[int]]:
    return [[b] for b in range(N_BANDS)]


def attribution_of(
    model: ProbeModel, base_x: np.ndarray, band_x: np.ndarray, target: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """(base probabilities (K,), attribution (N_BANDS,), target) from the features of the
    window (1, F) and of its single-band variants (N_BANDS, F): the target's drop."""
    base = model.predict_features(base_x)[0]
    target = int(base.argmax()) if target is None else int(target)
    return base, base[target] - model.predict_features(band_x)[:, target], target


def occlusion(
    model: ProbeModel, window: np.ndarray, shaft: float, axis: str, target: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """(base probabilities (K,), attribution (N_BANDS,), target class) of one window (C, L):
    the drop of the target's probability when each band is removed alone."""
    base_x = model.features(model.inputs(window[None], np.array([shaft])))
    band_x = variant_features(model, window, shaft, axis, single_bands())
    return attribution_of(model, base_x, band_x, target)


def deletion_curve(
    model: ProbeModel, window: np.ndarray, shaft: float, axis: str, order: np.ndarray, target: int
) -> np.ndarray:
    """(DELETION_STEPS + 1,) probability of ``target`` after removing the first 0, 1/16,
    ..., all of the bands in ``order``."""
    counts = np.round(np.linspace(0, len(order), DELETION_STEPS + 1)).astype(int)
    sets = [list(order[:c]) for c in counts]
    return model.predict_features(variant_features(model, window, shaft, axis, sets))[:, target]


def area(curve: np.ndarray) -> float:
    """Normalized area under a deletion curve (trapezoid on [0, 1])."""
    curve = np.asarray(curve, dtype=float)
    return float(np.sum((curve[1:] + curve[:-1]) / 2) / (len(curve) - 1))


# ----------------------------------------------------------------------------- the run


def explain_windows(
    model: ProbeModel,
    windows: np.ndarray,
    shaft: np.ndarray,
    faults: list[str],
    geometry_: dict,
    axes: tuple[str, ...],
    seed: int = 0,
    log=print,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per window and axis: PAS of the model and of its randomized copy, and the deletion
    areas in attribution order and in random orders. Returns the table and the attributions
    (axis -> (n, N_BANDS)) of the model."""
    rng = np.random.default_rng(seed)
    random_model = randomized(model, seed)
    records, attributions = [], {a: np.full((len(windows), N_BANDS), np.nan) for a in axes}
    started = time.perf_counter()
    for i, (window, f_shaft, fault) in enumerate(zip(windows, shaft, faults, strict=True)):
        for axis in axes:
            edges = band_edges(axis, model.rate)
            unit = f_shaft if axis == "hz" else 1.0
            mask = physics_mask(edges, defect_lines(geometry_, fault, unit))
            base_x = model.features(model.inputs(window[None], np.array([f_shaft])))
            band_x = variant_features(model, window, f_shaft, axis, single_bands())
            base, attribution, target = attribution_of(model, base_x, band_x)
            _, random_attr, _ = attribution_of(random_model, base_x, band_x)
            attributions[axis][i] = attribution
            ranked = np.argsort(-attribution, kind="stable")
            curve = deletion_curve(model, window, f_shaft, axis, ranked, target)
            random_areas = [
                area(deletion_curve(model, window, f_shaft, axis, rng.permutation(N_BANDS), target))
                for _ in range(RANDOM_ORDERS)
            ]
            records.append(
                {
                    "window": i,
                    "fault": fault,
                    "axis": axis,
                    "target": target,
                    "probability": float(base[target]),
                    "physics_bands": int(mask.sum()),
                    "pas": pas(attribution, mask),
                    "pas_randomized": pas(random_attr, mask),
                    "positive_mass": float(np.clip(attribution, 0, None).sum()),
                    "deletion_area": area(curve),
                    "deletion_area_random": float(np.mean(random_areas)),
                }
            )
        if (i + 1) % 10 == 0 or i + 1 == len(windows):
            log(f"[explain] {i + 1}/{len(windows)} windows, {time.perf_counter() - started:.0f} s")
    return pd.DataFrame(records), attributions


def _p(result) -> float:
    """The p-value of a SciPy test result."""
    return float(result.pvalue)


def summarize(table: pd.DataFrame) -> dict:
    """Per axis: PAS of the model and of the randomized model (median, mean, share > 1, the
    paired Wilcoxon test model > randomized) and the deletion areas (attribution order <
    random order)."""
    out = {}
    for axis, group in table.groupby("axis"):
        both = group[["pas", "pas_randomized"]].dropna()
        values = {c: group[c].to_numpy(dtype=float) for c in group.columns if c != "fault"
                  and c != "axis"}  # fmt: skip
        scored = values["pas"][np.isfinite(values["pas"])]
        entry = {
            "windows": int(len(group)),
            "scored": int(len(both)),
            "pas_median": float(np.nanmedian(values["pas"])),
            "pas_mean": float(np.nanmean(values["pas"])),
            "pas_above_1": float(np.mean(scored > 1)) if len(scored) else float("nan"),
            "pas_randomized_median": float(np.nanmedian(values["pas_randomized"])),
            "pas_randomized_mean": float(np.nanmean(values["pas_randomized"])),
            "deletion_area": float(np.mean(values["deletion_area"])),
            "deletion_area_random": float(np.mean(values["deletion_area_random"])),
        }
        if len(both) >= 2 and not np.allclose(both["pas"], both["pas_randomized"]):
            entry["pas_vs_randomized_p"] = _p(
                stats.wilcoxon(both["pas"], both["pas_randomized"], alternative="greater")
            )
        areas = group[["deletion_area", "deletion_area_random"]]
        if len(areas) >= 2 and not np.allclose(areas.iloc[:, 0], areas.iloc[:, 1]):
            entry["deletion_vs_random_p"] = _p(
                stats.wilcoxon(
                    areas["deletion_area"], areas["deletion_area_random"], alternative="less"
                )
            )
        entry["by_fault"] = {
            str(f): {
                "windows": int(len(g)),
                "pas_median": float(np.nanmedian(g["pas"].to_numpy(dtype=float))),
            }
            for f, g in group.groupby("fault")
        }
        out[str(axis)] = entry
    return out


def fit_probe(dataset: str, encoder_name: str, arm_list: tuple[str, ...], window_root=None,
              feature_root=None, device: str = "cpu", checkpoint=None):  # fmt: skip
    """(model, store, explained rows): the Gate-1 probe of ``dataset`` fit on its training
    part from the feature cache; the rows are its validation part (CWRU: 3 hp)."""
    from pdmfm import probe, protocols
    from pdmfm.encoders import get_encoder
    from pdmfm.features import FeatureStore

    if dataset != "cwru":
        raise ValueError("the physics probe runs on CWRU (seeded faults of known type)")
    store = WindowStore.open(dataset, window_root)
    split = protocols.cwru_gate1(store.manifest)
    feature_store = FeatureStore(feature_root)
    blocks, done = [], np.ones(len(store), dtype=bool)
    for arm in arm_list:
        cached = feature_store.find(store, encoder_name, arm)
        done &= cached.done
        blocks.append(cached.pooled(np.arange(len(store))))
    x = np.concatenate(blocks, axis=1)
    y = store.manifest["class_key"].astype(str).to_numpy()
    groups = store.manifest["load_hp"].astype(str).to_numpy()
    train = split.train & done
    classifier, _ = probe.select_and_fit("classification", x[train], y[train], groups[train])
    model = ProbeModel(
        rate=store.rate,
        arm_list=tuple(arm_list),
        channels={a: arms.arm_channels(store, a) for a in arm_list},
        encoder=get_encoder(encoder_name, device=device, checkpoint=checkpoint),
        classifier=classifier,
        classes=[str(c) for c in classifier.classes_],  # pyright: ignore[reportAttributeAccessIssue]
    )
    return model, store, np.flatnonzero(split.val & done), x


def run(
    dataset: str,
    encoder_name: str,
    arm_list: tuple[str, ...],
    run_id: str,
    max_windows: int | None = None,
    out_root: Path | None = None,
    window_root: Path | None = None,
    feature_root: Path | None = None,
    seed: int = 0,
    log=print,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> dict:
    model, store, rows, cached = fit_probe(
        dataset, encoder_name, arm_list, window_root, feature_root, device, checkpoint
    )
    manifest = store.manifest
    labels = manifest["label"].astype(str).to_numpy()
    truth = manifest["class_key"].astype(str).to_numpy()
    windows = np.asarray(store.signals[rows], dtype=np.float32)
    shaft = manifest["shaft_hz"].to_numpy(dtype=float)[rows]
    online = model.predict(windows, shaft)
    from_cache = model.predict_features(cached[rows])
    classes = np.array(model.classes)
    predicted = classes[online.argmax(1)]
    agreement = float(np.mean(online.argmax(1) == from_cache.argmax(1)))
    correct = predicted == truth[rows]
    faulty = np.array([label in FAULT_KIND for label in labels[rows]])
    chosen = np.flatnonzero(correct & faulty)
    if max_windows is not None and len(chosen) > max_windows:
        chosen = np.sort(np.random.default_rng(seed).choice(chosen, max_windows, replace=False))
    log(
        f"[explain] {dataset} {encoder_name} {'+'.join(arm_list)}: {len(rows)} windows, accuracy "
        f"{correct.mean():.3f} (online and cached features agree on {agreement:.3f}); "
        f"explaining {len(chosen)} correct faulty windows"
    )
    axes = ("hz", "order") if "E" in arm_list else ("hz",)
    table, attributions = explain_windows(
        model, windows[chosen], shaft[chosen], list(labels[rows][chosen]), geometry(dataset),
        axes, seed, log,
    )  # fmt: skip
    table.insert(0, "row", rows[chosen][table["window"].to_numpy()])
    out = (Path(out_root) if out_root else REPO_ROOT / "runs" / "explain") / run_id
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "windows.csv", index=False)
    np.savez_compressed(out / "attributions.npz", rows=rows[chosen], **attributions)  # pyright: ignore[reportArgumentType]
    summary = {
        "dataset": dataset,
        "encoder": encoder_name,
        "arms": list(arm_list),
        "explained": "CWRU 3 hp windows; probe fit on 0-2 hp",
        "accuracy": float(correct.mean()),
        "online_cached_agreement": agreement,
        "bands": N_BANDS,
        "tolerance": TOLERANCE,
        "harmonics": HARMONICS,
        "axes": summarize(table),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(out / "summary.json", summary)
    (out / "summary.md").write_text(_markdown(run_id, summary), encoding="utf-8")
    log(f"[explain] -> {out}")
    return summary


def _markdown(run_id: str, summary: dict) -> str:
    lines = [
        f"# Band occlusion and physics score: {run_id}",
        "",
        f"{summary['dataset']}, {summary['encoder']} {'+'.join(summary['arms'])}; "
        f"{summary['explained']}. Probe accuracy {summary['accuracy']:.3f}; online and cached "
        f"features agree on {summary['online_cached_agreement']:.3f} of the windows.",
        "",
        "| Axis | Windows scored | PAS median (mean) | PAS > 1 | Randomized PAS median | "
        "p (PAS > randomized) | Deletion area: attribution / random | p (smaller) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for axis, s in summary["axes"].items():
        lines.append(
            f"| {axis} | {s['scored']}/{s['windows']} | {s['pas_median']:.2f} "
            f"({s['pas_mean']:.2f}) | {s['pas_above_1']:.2f} | {s['pas_randomized_median']:.2f} | "
            f"{s.get('pas_vs_randomized_p', float('nan')):.3g} | {s['deletion_area']:.3f} / "
            f"{s['deletion_area_random']:.3f} | {s.get('deletion_vs_random_p', float('nan')):.3g} |"
        )
    lines += ["", "PAS by fault (median):", ""]
    for axis, s in summary["axes"].items():
        parts = [
            f"{f} {v['pas_median']:.2f} (n = {v['windows']})" for f, v in s["by_fault"].items()
        ]
        lines.append(f"- {axis}: " + "; ".join(parts))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.explain", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    parser.add_argument("--dataset", default="cwru", choices=("cwru",))
    parser.add_argument("--encoder", default="mantis", help="encoder of the cached features")
    parser.add_argument("--arms", nargs="+", default=["F", "E"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-windows", type=int, help="explain a random subset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default: runs/explain/")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (online re-encoding)")
    parser.add_argument(
        "--checkpoint", type=Path, help="own PatchTST only: its model folder (runs/patchtst/...)"
    )
    parser.add_argument("--force", action="store_true", help="run on a busy machine")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        claim_resources("explain", min_idle_cpus=2.0, min_free_ram_gb=2.0, force=args.force)
    summary = run(
        args.dataset, args.encoder, tuple(args.arms), args.run_id, args.max_windows,
        args.out, seed=args.seed, device=args.device, checkpoint=args.checkpoint,
    )  # fmt: skip
    print(json.dumps(summary["axes"], indent=2))


if __name__ == "__main__":
    main()
