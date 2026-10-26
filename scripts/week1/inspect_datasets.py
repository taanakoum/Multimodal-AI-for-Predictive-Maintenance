"""Week-1 checks on the downloaded datasets: formats, rates, durations, image sizes, quirks.

    python scripts/week1/inspect_datasets.py cwru mbhm najafi_thermal kaist nonastreda

Prints a report per dataset and writes it to ``$PDMFM_DATA_ROOT/../logs/inspect_<name>.json``.
Read-only: nothing in the data folders is modified (archives are read in place).
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from pdmfm.paths import data_root, extracted_dir, raw_dir
from pdmfm.sources import CWRU_FILES


def inspect_cwru() -> dict:
    files = {}
    for number, (kind, size, load) in sorted(CWRU_FILES.items()):
        mat = loadmat(raw_dir("cwru") / f"{number}.mat")
        keys = sorted(k for k in mat if not k.startswith("__"))
        own_de = f"X{number:03d}_DE_time"
        rate = 48_000 if kind == "normal" else 12_000
        prefix = f"X{number:03d}"
        rpm_key = next((k for k in keys if k.startswith(prefix) and k.endswith("RPM")), None)
        files[number] = {
            "class": kind,
            "size_in": size,
            "load_hp": load,
            "variables": keys,
            "foreign_variables": [k for k in keys if not k.startswith(f"X{number:03d}")],
            "samples": int(mat[own_de].size),
            "seconds": round(mat[own_de].size / rate, 2),
            "rpm": float(mat[rpm_key].squeeze()) if rpm_key else None,
        }
    return {
        "files": files,
        "files_with_foreign_variables": [n for n, f in files.items() if f["foreign_variables"]],
        "files_without_rpm": [n for n, f in files.items() if f["rpm"] is None],
    }


def inspect_mbhm() -> dict:
    con = sqlite3.connect(raw_dir("mbhm") / "metadata.sqlite")
    rows = con.execute(
        """select c.dataset, count(distinct c.condition_id), count(f.file_id)
           from condition c left join file_info f on f.condition_id = c.condition_id
           group by c.dataset"""
    ).fetchall()
    columns = [r[1] for r in con.execute("pragma table_info('condition')")]
    per_dataset = {name: {"conditions": nc, "windows": nw} for name, nc, nw in rows}
    overlapping = {"PU", "XJTU"}  # datasets whose units the plan holds out
    usable = {k: v for k, v in per_dataset.items() if k not in overlapping}
    return {
        "condition_columns": columns,
        "has_unit_identifier": bool({"bearing", "unit", "file", "source_file"} & set(columns)),
        "per_dataset": per_dataset,
        "usable_after_dropping": sorted(overlapping),
        "usable_windows": sum(v["windows"] for v in usable.values()),
    }


def inspect_najafi_thermal() -> dict:
    root = extracted_dir("najafi_thermal")
    folders = {d.name: len(list(d.glob("*.bmp"))) for d in sorted(root.iterdir()) if d.is_dir()}
    sizes = Counter(f"{Image.open(p).size} {Image.open(p).mode}" for p in root.rglob("*.bmp"))
    return {"bmp_per_folder": folders, "bmp_sizes": dict(sizes)}


def _zip_listing(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = [i for i in archive.infolist() if not i.is_dir()]
        exts = Counter(Path(i.filename).suffix.lower() for i in names)
        return {
            "files": len(names),
            "extensions": dict(exts),
            "first": [i.filename for i in names[:8]],
            "bytes": sum(i.file_size for i in names),
        }


def inspect_kaist() -> dict:
    from nptdms import TdmsFile

    report = {}
    raw = raw_dir("kaist")
    for name in ("vibration.zip", "current,temp.zip", "acoustic.zip"):
        if (raw / name).exists():
            report[name] = _zip_listing(raw / name)
    # Open one vibration MAT and one TDMS straight from the archives.
    with zipfile.ZipFile(raw / "vibration.zip") as archive:
        mat_name = next(n for n in archive.namelist() if n.lower().endswith(".mat"))
        mat = loadmat(io.BytesIO(archive.read(mat_name)))
        report["vibration_sample"] = {
            "file": mat_name,
            "variables": {k: list(np.shape(v)) for k, v in mat.items() if not k.startswith("__")},
        }
    with zipfile.ZipFile(raw / "current,temp.zip") as archive:
        tdms_name = next(n for n in archive.namelist() if n.lower().endswith(".tdms"))
        tdms = TdmsFile.read(io.BytesIO(archive.read(tdms_name)))
        channels = {}
        for group in tdms.groups():
            for channel in group.channels():
                props = dict(channel.properties)
                channels[f"{group.name}/{channel.name}"] = {
                    "samples": len(channel),
                    "wf_increment": props.get("wf_increment"),
                    "unit": props.get("unit_string") or props.get("NI_UnitDescription"),
                }
        report["tdms_sample"] = {"file": tdms_name, "channels": channels}
    return report


def inspect_nonastreda() -> dict:
    raw = raw_dir("nonastreda")
    report = {}
    for folder in ("tool", "chip", "work", "spec/x", "scal/x"):
        paths = sorted((raw / folder).glob("*"))
        sizes = Counter(f"{Image.open(p).size}" for p in paths[:: max(1, len(paths) // 64)])
        report[folder] = {"files": len(paths), "sizes_sampled": dict(sizes)}
    return report


# Lifetimes in minutes (= number of 1-minute snapshots) published by Wang et al. (2020).
XJTU_SY_LIFETIMES = {
    "Bearing1_1": 123,
    "Bearing1_2": 161,
    "Bearing1_3": 158,
    "Bearing1_4": 122,
    "Bearing1_5": 52,
    "Bearing2_1": 491,
    "Bearing2_2": 161,
    "Bearing2_3": 533,
    "Bearing2_4": 42,
    "Bearing2_5": 339,
    "Bearing3_1": 2538,
    "Bearing3_2": 2496,
    "Bearing3_3": 371,
    "Bearing3_4": 1515,
    "Bearing3_5": 114,
}


def inspect_xjtu_sy() -> dict:
    root = extracted_dir("xjtu_sy") / "XJTU-SY_Bearing_Datasets"
    bearings = {}
    for condition in sorted(p for p in root.iterdir() if p.is_dir()):
        for bearing in sorted(p for p in condition.iterdir() if p.is_dir()):
            snapshots = list(bearing.glob("*.csv"))
            bearings[bearing.name] = {
                "condition": condition.name,
                "snapshots": len(snapshots),
                "published_lifetime_min": XJTU_SY_LIFETIMES.get(bearing.name),
            }
    sample = root / "35Hz12kN" / "Bearing1_1" / "1.csv"
    lines = sample.read_text(encoding="utf-8").splitlines()
    data = np.loadtxt(sample, delimiter=",", skiprows=1)
    return {
        "bearings": bearings,
        "lifetime_mismatches": {
            k: v for k, v in bearings.items() if v["snapshots"] != v["published_lifetime_min"]
        },
        "sample_header": lines[0],
        "sample_shape": list(data.shape),
        "sample_rms_g": [float(np.sqrt((data[:, i] ** 2).mean())) for i in range(data.shape[1])],
    }


CHECKS = {
    "cwru": inspect_cwru,
    "mbhm": inspect_mbhm,
    "najafi_thermal": inspect_najafi_thermal,
    "kaist": inspect_kaist,
    "nonastreda": inspect_nonastreda,
    "xjtu_sy": inspect_xjtu_sy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", choices=sorted(CHECKS))
    args = parser.parse_args()
    logs = data_root().parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        report = CHECKS[name]()
        text = json.dumps(report, indent=2, default=str)
        (logs / f"inspect_{name}.json").write_text(text, encoding="utf-8")
        print(f"===== {name}\n{text[:4000]}")


if __name__ == "__main__":
    main()
