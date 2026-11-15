"""Read-only preprocessing/cache/CV preflight before allocating Ruche GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pdmfm import arms, protocols
from pdmfm import texts as text_specs
from pdmfm.features import FeatureStore
from pdmfm.utils import write_json_atomic
from pdmfm.windows import WindowStore


def inspect(dataset, encoder=None, arm_list=("F",), window_root=None, feature_root=None):
    store = WindowStore.open(dataset, window_root)
    if store.limit is not None or not len(store):
        raise ValueError(f"{dataset}: full, nonempty store required")
    if len(store.levels) != len(store) or len(store.slow) != len(store):
        raise ValueError(f"{dataset}: window/descriptor row counts differ")
    manifest = store.manifest
    expected = set(text_specs.load(dataset).descriptions)
    missing = expected - set(manifest.class_key)
    if missing:
        raise ValueError(f"{dataset}: missing configured classes: {sorted(missing)}")
    if dataset == "cwru":
        for load, part in manifest.groupby("load_hp"):
            if expected - set(part.class_key):
                raise ValueError(f"CWRU load {load}: missing configured classes")
    if not np.isfinite(manifest[["t_start", "t_end"]].to_numpy(float)).all():
        raise ValueError(f"{dataset}: nonfinite window times")
    if (manifest.t_end <= manifest.t_start).any():
        raise ValueError(f"{dataset}: invalid window intervals")
    if manifest.groupby("bag").unit.nunique().gt(1).any():
        raise ValueError(f"{dataset}: a recording bag spans multiple physical units")
    report = {
        "dataset": dataset,
        "window_store": store.key,
        "windows": len(store),
        "rate_hz": store.rate,
        "channels": list(store.channels),
        "class_bags": manifest.drop_duplicates("bag").class_key.value_counts().to_dict(),
        "cv": protocols.audit_folds(dataset, store.manifest),
    }
    for start in range(0, len(store), 128):
        if not np.isfinite(store.signals[start : start + 128]).all():
            raise ValueError(f"{dataset}: nonfinite signal samples")
    if not np.isfinite(store.levels).all():
        raise ValueError(f"{dataset}: nonfinite level descriptors")
    if np.isinf(store.slow).any():
        raise ValueError(f"{dataset}: infinite slow-channel summaries")
    for kind in store.spec.image_kinds:
        have = store.manifest[f"image_{kind}"].ne("")
        age = store.manifest.loc[have, f"age_{kind}"].to_numpy(float)
        if not np.isfinite(age).all() or (age <= 0).any():
            raise ValueError(f"{dataset}: invalid photo availability")
        if dataset in ("nonastreda", "matwi"):
            start = manifest.groupby("bag").t_start.transform("min")
            # The snapshot pass cannot see a photo acquired during that same pass.
            if (age <= (manifest.t_start - start).loc[have].to_numpy(float)).any():
                raise ValueError(f"{dataset}: photo acquired during its prediction pass")
    if encoder:
        caches = FeatureStore(feature_root)
        selected = [a for a in arm_list if a not in arms.unavailable(store)]
        if not selected:
            raise ValueError(f"{dataset}: none of the requested arms are applicable")
        report["features"] = []
        for arm in selected:
            features = caches.find(store, encoder, arm)
            if len(features.features) != len(store) or features.done.shape != (len(store),):
                raise ValueError(f"{dataset} {encoder}:{arm}: cache row counts differ")
            if not features.done.all():
                raise ValueError(f"{dataset} {encoder}:{arm}: incomplete extraction")
            for start in range(0, len(store), 512):
                if not np.isfinite(features.features[start : start + 512]).all():
                    raise ValueError(f"{dataset} {encoder}:{arm}: nonfinite cached tokens")
            report["features"].append(features.folder.name)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--encoder")
    p.add_argument("--arms", nargs="+", default=["F"])
    p.add_argument("--out", type=Path, default=Path("runs/preflight/report.json"))
    args = p.parse_args(argv)
    reports, errors = [], []
    for dataset in args.datasets:
        try:
            reports.append(inspect(dataset, args.encoder, tuple(args.arms)))
        except (FileNotFoundError, ValueError) as error:
            errors.append({"dataset": dataset, "error": str(error)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, {"ok": not errors, "datasets": reports, "errors": errors})
    for error in errors:
        print(error)
    print(f"{len(reports)} passed; {len(errors)} failed: {args.out}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
