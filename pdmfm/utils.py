"""Small helpers shared by the pipeline stages: stable hashes and atomic writes.

Every cache key in the project is ``stable_hash(signature)``; changing it would orphan every
existing window store and feature cache. Atomic writes go to a temporary file first and are
renamed into place, so a job killed in the middle of a save never leaves a damaged file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def stable_hash(obj) -> str:
    """SHA-256 of the canonical JSON of ``obj`` (sorted keys; non-JSON values as str)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def run_identity(folder: Path, settings: dict) -> str:
    """Refuse stale/legacy experiment results instead of silently skipping their fits."""
    key = stable_hash(settings)
    path = Path(folder) / "identity.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["key"] != key:
            raise ValueError(f"{folder}: other settings or data; use another run id")
    elif (Path(folder) / "results.csv").exists():
        raise ValueError(f"{folder}: legacy results have no identity; use another run id")
    else:
        write_json_atomic(path, {"key": key, "settings": settings})
    return key


def feature_identity(
    dataset: str, root: Path | None = None, encoder: str | None = None, arms=None
) -> str:
    """Hash cache metadata and text-bank identity without reading large feature arrays."""
    from pdmfm.paths import interim_dir

    root = Path(root) if root else interim_dir("features")
    records = []
    for base in (root / dataset, root / "images", root / "texts"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            first = path.relative_to(base).parts[0]
            if base.name in ("images", "texts") and not first.startswith(dataset + "-"):
                continue
            if path.name == "meta.json":
                if base.name == dataset and encoder is not None:
                    import fnmatch

                    signature = json.loads(path.read_text(encoding="utf-8"))["signature"]
                    candidate = signature["encoder"]["encoder"]
                    if not fnmatch.fnmatchcase(candidate, encoder.replace("{fold}", "*")):
                        continue
                    if arms is not None and signature["arm"] not in arms:
                        continue
                records.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
            elif path.suffix == ".npz":
                records.append(
                    (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
                )
    return stable_hash(records)
