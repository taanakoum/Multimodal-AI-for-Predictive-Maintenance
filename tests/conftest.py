"""Shared fixtures and markers.

* Tests marked ``data`` read the real downloaded datasets under ``PDMFM_DATA_ROOT``; they
  are skipped automatically when that dataset is not present (always the case in CI).
* Tests marked ``network`` need internet and run only with ``PDMFM_NETWORK_TESTS=1``.
* Tests marked ``weights(repo)`` load a pretrained checkpoint; they are skipped when it is
  not in the Hugging Face cache (always the case in CI, which never downloads weights).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pdmfm.paths import extracted_dir, raw_dir


def pytest_addoption(parser):
    for kind in ("data", "weights", "gpu", "network"):
        parser.addoption(
            f"--run-{kind}", action="store_true", help=f"opt into {kind} acceptance tests"
        )


def pytest_configure(config):
    import torch

    torch.set_num_threads(int(os.environ.get("PDMFM_TEST_THREADS", "2")))
    # Unit/integration tests must never download a model opportunistically.
    os.environ["HF_HUB_OFFLINE"] = "1"


def dataset_available(name: str, extracted: bool = False) -> bool:
    folder = extracted_dir(name) if extracted else raw_dir(name)
    return folder.is_dir() and any(folder.iterdir())


def weights_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:  # noqa: BLE001 - no hub, or a broken cache: treat as absent
        return False


def pytest_collection_modifyitems(config, items):
    run_network = config.getoption("--run-network") or os.environ.get("PDMFM_NETWORK_TESTS") == "1"
    for item in items:
        primary = next(
            (
                k
                for k in ("unit", "integration", "e2e", "acceptance")
                if k in Path(str(item.path)).parts
            ),
            None,
        )
        resource = any(item.get_closest_marker(k) for k in ("data", "weights", "gpu", "network"))
        primary = "acceptance" if resource else primary
        if primary is None:
            raise pytest.UsageError(f"{item.nodeid}: missing primary test category")
        item.add_marker(getattr(pytest.mark, primary))
        if primary == "acceptance" and not resource and not config.getoption("--run-data"):
            item.add_marker(pytest.mark.skip(reason="set --run-data for dataset acceptance"))
        weights = item.get_closest_marker("weights")
        if weights is not None and not config.getoption("--run-weights"):
            item.add_marker(pytest.mark.skip(reason="set --run-weights"))
        if weights is not None and not weights_cached(weights.args[0]):
            item.add_marker(pytest.mark.skip(reason=f"{weights.args[0]} not in the HF cache"))
        if "network" in item.keywords and not run_network:
            item.add_marker(pytest.mark.skip(reason="set PDMFM_NETWORK_TESTS=1 to run"))
        marker = item.get_closest_marker("data")
        if marker is not None:
            if not config.getoption("--run-data"):
                item.add_marker(pytest.mark.skip(reason="set --run-data"))
            name = marker.args[0]
            extracted = marker.kwargs.get("extracted", False)
            if not dataset_available(name, extracted):
                item.add_marker(pytest.mark.skip(reason=f"dataset {name} not downloaded"))
        if item.get_closest_marker("gpu"):
            if not config.getoption("--run-gpu"):
                item.add_marker(pytest.mark.skip(reason="set --run-gpu"))
            else:
                import torch

                if not torch.cuda.is_available():
                    raise pytest.UsageError("--run-gpu requested but CUDA is unavailable")
