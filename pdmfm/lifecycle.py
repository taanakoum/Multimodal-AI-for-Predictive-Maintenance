"""Versioned milling life targets; retrospective labels never become model inputs.

Build with ``python -m pdmfm.lifecycle --dataset matwi``. MATWI targets describe remaining
recorded operation, not independently verified mechanical failure. Nonastreda's endpoint
is confirmed dulled wear; its gap-aware cutting clock is estimated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm.paths import interim_dir
from pdmfm.utils import stable_hash, write_json_atomic

VERSION = 1


def confirmed_dulled(labels: list[str]) -> int | None:
    return next(
        (
            i
            for i, label in enumerate(labels)
            if label == "dulled" and labels[i + 1 : i + 4].count("dulled") >= 2
        ),
        None,
    )


def targets(table: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Label a complete ordered lifecycle at pass end, retaining censored observations.

    Inputs require unit, bag, duration_s. Nonastreda additionally supplies class_key.
    Missing interior operation is represented by gap_before_s and creates no samples.
    """
    if endpoint not in ("confirmed_dulled", "recorded_life_end"):
        raise ValueError("unknown endpoint rule")
    out = table.copy().reset_index(drop=True)
    if out.duplicated(["unit", "bag"]).any():
        raise ValueError("lifecycle keys must be unique")
    durations = out["duration_s"].to_numpy(float)
    gaps = (
        np.asarray(out["gap_before_s"], dtype=float)
        if "gap_before_s" in out
        else np.zeros(len(out))
    )
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("recording durations must be positive and finite")
    if not np.isfinite(gaps).all() or (gaps < 0).any():
        raise ValueError("operation gaps must be nonnegative and finite")
    for _, group in out.groupby("unit", sort=False):
        rows = group.index
        end = np.cumsum(durations[rows] + gaps[rows])
        start = end - durations[rows]
        failure = (
            len(rows) - 1
            if endpoint == "recorded_life_end"
            else confirmed_dulled(group["class_key"].tolist())
        )
        out.loc[rows, "operation_start_s"] = start
        out.loc[rows, "operation_end_s"] = end
        out.loc[rows, "endpoint_s"] = np.nan if failure is None else end[failure]
        out.loc[rows, "event_observed"] = failure is not None
        out.loc[rows, "rul_s"] = (
            np.nan if failure is None else np.where(end <= end[failure], end[failure] - end, np.nan)
        )
    out["endpoint_rule"] = endpoint
    out["target_kind"] = "rul_operating_seconds_proxy"
    out.attrs["signature"] = {"version": VERSION, "endpoint": endpoint}
    return out


def nonastreda_table(reader=None, clock: str = "gap") -> pd.DataFrame:
    from pdmfm.datasets.nonastreda import Nonastreda, trim_bounds

    if clock not in ("gap", "recorded"):
        raise ValueError("clock must be gap or recorded")
    reader = reader or Nonastreda()
    all_records = reader.all_records()
    # Legacy force-cache representatives can be reused; only availability changes.
    legacy = reader._load_records()
    representative = {rid: rid for rid in all_records}
    available = {rid: True for rid in all_records}
    for group in reader.duplicate_groups():
        cached = next(rid for rid in group if rid in legacy)
        for rid in group:
            representative[rid] = cached
            available[rid] = rid == group[-1]
    rows, previous, durations = [], {}, {}
    for r in sorted(all_records.values(), key=lambda r: (int(r.tool[1:]), r.run, r.block)):
        lo, hi = trim_bounds(r.force, causal=True)
        duration = (hi - lo) / 1000.0
        past = durations.setdefault(r.tool, [])
        missing = max(0, r.run - previous.get(r.tool, r.run) - 1) * 4
        gap = missing * float(np.median(past)) if clock == "gap" and missing else 0.0
        rows.append(
            {
                "unit": r.tool,
                "bag": r.record_id,
                "source_bag": representative[r.record_id],
                "duration_s": duration,
                "gap_before_s": gap,
                "force_present": available[r.record_id],
                "class_key": ("sharp", "used", "dulled")[r.label],
            }
        )
        past.append(duration)
        previous[r.tool] = r.run
    out = targets(pd.DataFrame(rows), "confirmed_dulled")
    out.attrs["signature"].update(clock=clock, duplicates="latest", trim="fixed-5s")
    return out


def matwi_table(reader=None) -> pd.DataFrame:
    from pdmfm.datasets.matwi import MATWI, sensor_clock

    reader = reader or MATWI()
    photos = reader.table()
    rows = []
    for info in reader.index():
        start, duration = sensor_clock(Path(info.meta["path"]))
        own = photos.loc[(photos.Set == int(info.unit[3:])) & photos.ImageFile.notna()].copy()
        own["stamp"] = pd.to_datetime(own.ImageDateTime)
        earlier = own.loc[own.stamp < start].sort_values("stamp")
        previous_photo = "" if not len(earlier) else str(earlier.iloc[-1].ImageFile)
        rows.append(
            {
                "unit": info.unit,
                "bag": info.recording_id,
                "source_bag": info.recording_id,
                "duration_s": duration,
                "wall_start": start,
                "wall_end": start + pd.Timedelta(seconds=duration),
                "gap_before_s": 0.0,
                "force_present": True,
                "previous_photo": previous_photo,
                "class_key": info.label or "",
            }
        )
    frame = pd.DataFrame(rows).sort_values(["unit", "wall_start"]).reset_index(drop=True)
    for _, group in frame.groupby("unit"):
        if (group.wall_start.iloc[1:].to_numpy() < group.wall_end.iloc[:-1].to_numpy()).any():
            raise ValueError("overlapping MATWI recordings")
    out = targets(frame, "recorded_life_end")
    out.attrs["signature"].update(clock="recorded", physical_failure_verified=False)
    return out


def build(dataset: str, clock: str = "gap", reader=None, out: Path | None = None):
    from pdmfm.windows import source_fingerprint

    if dataset not in ("nonastreda", "matwi"):
        raise ValueError("lifecycle conversion supports nonastreda and matwi")
    table = nonastreda_table(reader, clock) if dataset == "nonastreda" else matwi_table(reader)
    signature = {
        **table.attrs["signature"],
        "dataset": dataset,
        "source": source_fingerprint(dataset),
        "records": stable_hash(table.to_dict("records")),
    }
    key = stable_hash(signature)[:12]
    folder = Path(out) if out else interim_dir("lifecycle") / dataset / key
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "manifest.csv", index=False)
    write_json_atomic(
        folder / "meta.json", {"signature": signature, "key": key, "records": len(table)}
    )
    table.attrs["signature"] = signature
    return table, folder


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("nonastreda", "matwi"), required=True)
    p.add_argument("--clock", choices=("gap", "recorded"), default="gap")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    table, folder = build(args.dataset, args.clock, out=args.out)
    print(f"{len(table)} observations, {table.unit.nunique()} tools: {folder}")


if __name__ == "__main__":
    main()
