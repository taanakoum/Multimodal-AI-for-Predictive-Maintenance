"""Where data, caches and model weights live.

Everything large stays outside the repository. ``PDMFM_DATA_ROOT`` points at the data
folder (``D:\\Project14\\data`` on the development machine, ``$WORKDIR/pdmfm_data`` on
Ruche); without it, ``<repo>/data`` is used, which ``.gitignore`` excludes.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """The data folder, read from ``PDMFM_DATA_ROOT`` at call time."""
    configured = os.environ.get("PDMFM_DATA_ROOT")
    return Path(configured).expanduser() if configured else REPO_ROOT / "data"


def raw_dir(dataset: str) -> Path:
    """Downloaded files of one dataset, exactly as the publisher ships them."""
    return data_root() / "raw" / dataset


def extracted_dir(dataset: str) -> Path:
    """Archives of one dataset unpacked (zip, rar)."""
    return data_root() / "extracted" / dataset


def interim_dir(name: str) -> Path:
    """Intermediate products: window shards, manifests, feature caches."""
    return data_root() / "interim" / name
