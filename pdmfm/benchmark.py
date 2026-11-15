"""Benchmark cached-feature training and estimate a bounded GPU experiment manifest.

Run on a CUDA node with existing caches, for example:
``python -m pdmfm.benchmark --dataset nonastreda --encoder mantis --arms F --fits 120``.
No experiment launches are submitted. Estimates include validation and input staging.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pdmfm import protocols
from pdmfm.model.inputs import assemble
from pdmfm.train import TrainConfig, fit, predict
from pdmfm.utils import write_json_atomic
from pdmfm.windows import WindowStore


def run(dataset, encoder, arms, out, fits=1, budget=110.0):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; no GPU savings can be verified on CPU")
    store = WindowStore.open(dataset)
    split = protocols.SOURCE_FOLDS[dataset](store.manifest, 1)
    inputs = assemble(
        dataset, encoder, tuple(arms), train_units=split.units(store.manifest, "train")
    )
    present = np.any([m.present for m in inputs.modalities.values()], axis=0)
    if inputs.y_class is not None:
        present &= inputs.y_class >= 0
    elif inputs.y_reg is not None:
        present &= np.isfinite(inputs.y_reg)
    train, val = np.flatnonzero(split.train & present), np.flatnonzero(split.val & present)
    if not len(train) or not len(val):
        raise ValueError("benchmark requires labeled training and validation observations")
    cfg = TrainConfig(epochs=2, device="cuda", d=128)
    # Warm up CUDA/library kernels separately from the recorded pilot.
    fit(inputs, train[:256], val[:128], replace(cfg, epochs=1), 0, log=lambda *_: None)
    records, fitted = [], None
    for precision, staging in (("fp32", "cpu"), ("auto", "auto")):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        fitted = fit(inputs, train, val, replace(cfg, precision=precision, cache_device=staging), 0)
        torch.cuda.synchronize()
        records.append(
            {
                "precision": precision,
                "staging": staging,
                "seconds": time.perf_counter() - started,
                "peak_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    assert fitted is not None
    reference = predict(fitted.model, inputs, val[:128], replace(cfg, precision="fp32"))
    accelerated = predict(fitted.model, inputs, val[:128], cfg)
    differences = {}
    for name in ("combined", "fused_nig"):
        if name in reference:
            a, b = reference[name], accelerated[name]
            error = float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a**2)), 1e-6))
            differences[name] = error
            if not np.isfinite(b).all() or error > 0.05:
                raise ValueError(f"{name}: accelerated inference exceeds 5% relative RMS tolerance")
    estimated = records[-1]["seconds"] / cfg.epochs * 50 * fits / 3600
    report = {
        "dataset": dataset,
        "encoder": encoder,
        "arms": list(arms),
        "pilot_epochs": cfg.epochs,
        "records": records,
        "relative_rms_error": differences,
        "planned_50_epoch_fits": fits,
        "estimated_gpu_hours": estimated,
        "budget_hours": budget,
        "within_budget": estimated <= budget,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(out), report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(protocols.SOURCE_FOLDS), required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--arms", nargs="+", default=["F"])
    p.add_argument("--fits", type=int, default=1)
    p.add_argument("--budget", type=float, default=110.0)
    p.add_argument("--out", type=Path, default=Path("runs/benchmark/gpu.json"))
    args = p.parse_args(argv)
    if args.fits < 1 or args.budget <= 0:
        p.error("fits and budget must be positive")
    print(run(args.dataset, args.encoder, args.arms, args.out, args.fits, args.budget))


if __name__ == "__main__":
    main()
