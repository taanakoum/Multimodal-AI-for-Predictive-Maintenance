"""Portable CPU pipeline demonstration using synthetic raw files and stand-in encoders.

No datasets, pretrained weights or sibling repositories are required. Outputs demonstrate
the interfaces only and are not scientific model results.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import savemat

from pdmfm import anomaly, features, lifecycle, planb, rul, windows
from pdmfm.encoders import StatsEncoder
from pdmfm.train import TrainConfig
from pdmfm.utils import stable_hash, write_json_atomic


class DemoText:
    name = "qwen3_embedding-demo"
    dim = 16

    def signature(self):
        return {"encoder": "synthetic-text", "version": 1}

    def encode(self, sentences):
        rows = [
            np.random.default_rng(int(stable_hash(s)[:8], 16)).standard_normal(self.dim)
            for s in sentences
        ]
        x = np.asarray(rows, np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)


class DemoImage:
    name = "dinov2-demo"

    def signature(self, kind):
        return {"encoder": "synthetic-image", "kind": kind, "version": 1}

    def encode_paths(self, paths, kind):
        values = []
        for path in paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), np.float32) / 255
            stats = np.concatenate([pixels.mean((0, 1)), pixels.std((0, 1))])
            values.append(np.tile(stats, (3, 1)))
        return np.asarray(values)


def raw_nonastreda(root: Path):
    """Ten tools, eight passes, native MAT/CSV layout and post-pass photographs."""
    root.mkdir(parents=True, exist_ok=True)
    rows, regression, raw = [], [], np.empty((80, 4), object)
    rng = np.random.default_rng(7)
    for tool in range(1, 11):
        for p in range(8):
            rid = f"T{tool}R{p // 4 + 1}B{p % 4 + 1}"
            label = ("sharp", "used", "dulled")[min(p // 2, 2)]
            wear = 30 + p * 20
            force = rng.normal(40 + p * 4, 2, (3, 9000))
            raw[(tool - 1) * 8 + p] = [rid + ".jpg", label, 0, force]
            rows.append({"id": rid, "image_label": label})
            regression.append({"id": rid, "flank_wear": wear, "gaps": 0, "overhang": 0})
            for kind, suffix in (("tool", ".jpg"), ("chip", ".jpg"), ("work", ".png")):
                folder = root / kind
                folder.mkdir(exist_ok=True)
                Image.new("RGB", (32, 16), (wear, 60, 40)).save(folder / (rid + suffix))
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(regression).to_csv(root / "labels_reg.csv", index=False)
    savemat(root / "forces_xyz_raw.mat", {"baseDatastore": raw})


def run(out: Path):
    out = Path(out).resolve()
    root = out / "data"
    if (out / "demo.json").exists():
        raise ValueError("demo already exists; choose a fresh --out directory")
    old = os.environ.get("PDMFM_DATA_ROOT")
    os.environ["PDMFM_DATA_ROOT"] = str(root)
    torch.set_num_threads(2)
    try:
        raw_nonastreda(root / "raw/nonastreda")
        store = windows.build("nonastreda")
        features.FeatureStore().extract(store, StatsEncoder(), "F")
        features.extract_images("nonastreda", DemoImage())
        features.extract_texts(DemoText(), ["nonastreda"])
        planb.run(
            "nonastreda",
            "stats",
            ("F",),
            "classification",
            [1],
            [0],
            TrainConfig(epochs=2, d=16),
            out_root=out,
        )
        table, _ = lifecycle.build("nonastreda", out=out / "lifecycle")
        rul.run(
            "stats",
            ("F",),
            "rul",
            [1],
            [0],
            rul.RULConfig(epochs=2, d=16),
            "nonastreda",
            out_root=out,
            lifecycle=table,
        )
        anomaly.run("nonastreda", ["stats:F"], "anomaly", out_root=out)
        write_json_atomic(
            out / "demo.json",
            {
                "synthetic": True,
                "pretrained_weights": False,
                "description": "Interface demonstration only",
            },
        )
    finally:
        if old is None:
            os.environ.pop("PDMFM_DATA_ROOT", None)
        else:
            os.environ["PDMFM_DATA_ROOT"] = old
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/demo"))
    args = p.parse_args(argv)
    print(run(args.out))


if __name__ == "__main__":
    main()
