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
