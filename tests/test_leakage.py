"""Leakage checks (pdmfm.leakage): the context lookup and the dataset-identity probe."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from synthetic import build_planb_root

from pdmfm import leakage
from pdmfm.windows import WindowStore


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("leakage")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PDMFM_DATA_ROOT", str(root))
        build_planb_root(root, datasets=("nonastreda", "xjtu_sy", "cwru", "matwi"))
        yield root


def windows(root):
    return root / "interim" / "windows"


def test_one_context_sentence_carries_nothing(root):
    result = leakage.context_check("nonastreda", windows(root), n=100)
    assert result["contexts"] == 1
    assert result["classes"]["difference"] == 0.0 and not result["classes"]["informative"]
    assert result["regression"]["difference"] == 0.0 and result["regression"]["target"] == "wear_um"


def test_a_context_that_names_the_class_is_flagged(root, monkeypatch):
    manifest = WindowStore.open("nonastreda", windows(root)).manifest.copy()
    manifest["context"] = "A milling machine; the tool is " + manifest["class_key"].astype(str)
    monkeypatch.setattr(
        leakage.WindowStore, "open", lambda *a, **k: SimpleNamespace(manifest=manifest)
    )
    result = leakage.context_check("nonastreda", n=200)
    assert result["classes"]["context"] == 1.0 and result["classes"]["informative"]
    assert result["regression"]["informative"]  # wear follows the class here


def test_regression_only_datasets(root):
    result = leakage.context_check("xjtu_sy", windows(root), n=100)
    assert "classes" not in result and result["regression"]["target"] == "rul"
    with pytest.raises(ValueError, match="source datasets"):
        leakage.context_check("cwru", windows(root))


def test_dataset_identity_probe(root):
    features = root / "interim" / "features"
    found = leakage.dataset_probe(
        ["nonastreda", "xjtu_sy", "cwru"], "stats:F", windows(root), features
    )
    assert found["datasets"] == ["nonastreda", "xjtu_sy", "cwru"]
    assert found["chance"] == pytest.approx(1 / 3) and found["balanced_accuracy"] > 0.9
    relative = leakage.dataset_probe(
        ["nonastreda", "xjtu_sy"], "levels:relative", windows(root), features
    )
    assert 0.0 <= relative["balanced_accuracy"] <= 1.0
    missing = leakage.dataset_probe(["nonastreda", "xjtu_sy"], "mantis:F", windows(root), features)
    assert set(missing["skipped"]) == {"nonastreda", "xjtu_sy"} and np.isnan(
        missing["balanced_accuracy"]
    )
    with pytest.raises(ValueError, match="held-out target"):
        leakage.dataset_probe(["nonastreda", "kaist"], "stats:F", windows(root), features)


def test_window_vectors_average_channels_and_tokens(root):
    store = WindowStore.open("nonastreda", windows(root))
    rows = np.arange(5)
    x = leakage.window_vectors(store, "levels:absolute", rows, None)
    np.testing.assert_allclose(x, store.levels[rows].mean(axis=1), rtol=1e-6)
    with pytest.raises(ValueError, match="levels:absolute or levels:relative"):
        leakage.window_vectors(store, "levels:both", rows, None)


def test_run_writes_the_summary(root):
    result = leakage.run(
        "check", ["nonastreda", "xjtu_sy"], ["nonastreda", "xjtu_sy"], ["stats:F"],
        out_root=root / "runs", window_root=windows(root),
        feature_root=root / "interim" / "features", log=quiet,
    )  # fmt: skip
    folder = root / "runs" / "check"
    saved = json.loads((folder / "leakage.json").read_text(encoding="utf-8"))
    assert len(saved["context"]) == 2 and saved["dataset"][0]["feature"] == "stats:F"
    text = (folder / "summary.md").read_text(encoding="utf-8")
    assert "at chance" in text and "Which dataset?" in text
    assert result["context"][0]["dataset"] == "nonastreda"
