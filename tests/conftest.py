"""Shared fixtures and markers.

* Tests marked ``data`` read the real downloaded datasets under ``PDMFM_DATA_ROOT``; they
  are skipped automatically when that dataset is not present (always the case in CI).
* Tests marked ``network`` need internet and run only with ``PDMFM_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import os

import pytest

from pdmfm.paths import extracted_dir, raw_dir


def dataset_available(name: str, extracted: bool = False) -> bool:
    folder = extracted_dir(name) if extracted else raw_dir(name)
    return folder.is_dir() and any(folder.iterdir())


def pytest_collection_modifyitems(config, items):
    run_network = os.environ.get("PDMFM_NETWORK_TESTS") == "1"
    for item in items:
        if "network" in item.keywords and not run_network:
            item.add_marker(pytest.mark.skip(reason="set PDMFM_NETWORK_TESTS=1 to run"))
        marker = item.get_closest_marker("data")
        if marker is not None:
            name = marker.args[0]
            extracted = marker.kwargs.get("extracted", False)
            if not dataset_available(name, extracted):
                item.add_marker(pytest.mark.skip(reason=f"dataset {name} not downloaded"))
