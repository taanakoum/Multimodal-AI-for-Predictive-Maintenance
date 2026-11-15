"""Preflight reads real store/cache interfaces and refuses corrupted or partial data."""

import numpy as np
import pytest

from pdmfm import preflight
from pdmfm.features import FeatureStore
from pdmfm.windows import WindowStore
from tests.support.synthetic import build_planb_root


def test_preflight_checks_all_folds_complete_caches_and_numeric_values(tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    build_planb_root(tmp_path, datasets=("nonastreda",))
    report = preflight.inspect("nonastreda", "stats")
    assert len(report["cv"]["folds"]) == 5
    assert report["cv"]["source_units"] == 10
    path = tmp_path / "interim/windows/nonastreda/manifest.csv"
    original = path.read_text(encoding="utf-8")
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False)
    frame["class_key"] = "sharp"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing configured classes"):
        preflight.inspect("nonastreda")
    path.write_text(original, encoding="utf-8")
    store = WindowStore.open("nonastreda")
    features = FeatureStore().find(store, "stats", "F")
    done = features.done.copy()
    done[-1] = False
    np.save(features.folder / "done.npy", done)
    with pytest.raises(ValueError, match="incomplete extraction"):
        preflight.inspect("nonastreda", "stats")
    np.save(features.folder / "done.npy", features.done)
    tokens = np.load(features.folder / "features.npy", mmap_mode="r+")
    value = tokens.flat[0]
    tokens.flat[0] = np.nan
    tokens.flush()
    with pytest.raises(ValueError, match="nonfinite cached tokens"):
        preflight.inspect("nonastreda", "stats")
    tokens.flat[0] = value
    tokens.flush()
    del tokens
    del store
    # Corrupt an otherwise readable signal file; the preflight must examine contents.
    signal = np.memmap(
        tmp_path / "interim/windows/nonastreda/signals.f32", dtype=np.float32, mode="r+"
    )
    signal[0] = np.nan
    signal.flush()
    del signal
    with pytest.raises(ValueError, match="nonfinite signal"):
        preflight.inspect("nonastreda")
