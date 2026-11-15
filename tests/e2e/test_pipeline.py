"""Reproducible CPU demo: stores -> caches -> heads -> RUL/anomaly -> paired report."""

import numpy as np
import pandas as pd

from pdmfm import anomaly, lifecycle, planb, protocols, report, rul
from pdmfm.train import TrainConfig
from pdmfm.windows import WindowStore
from tests.support.synthetic import build_planb_root


def test_cached_multimodal_pipeline_and_milling_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    build_planb_root(tmp_path, datasets=("nonastreda", "matwi"))
    paths = {
        "window_root": tmp_path / "interim/windows",
        "feature_root": tmp_path / "interim/features",
    }
    monkeypatch.setattr(planb, "BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(rul, "BOOTSTRAP_REPLICATES", 20)
    a = planb.run(
        "nonastreda",
        "stats",
        ("F",),
        "demo",
        [1],
        [0],
        TrainConfig(epochs=1, d=16),
        out_root=tmp_path / "runs",
        log=lambda *_: None,
        **paths,
    )
    assert (a / "summary.md").is_file()
    tables = report.load_run(a)
    assert set(tables) == {0} and tables[0].bag.is_unique
    for dataset in ("nonastreda", "matwi"):
        store = WindowStore.open(dataset, paths["window_root"])
        first = store.manifest.groupby("bag", sort=False).first().reset_index()
        life = lifecycle.targets(
            pd.DataFrame(
                {
                    "unit": first.unit,
                    "bag": first.bag,
                    "source_bag": first.bag,
                    "duration_s": np.full(len(first), 20.0),
                    "force_present": True,
                    "class_key": first.class_key,
                }
            ),
            "recorded_life_end",
        )
        out = rul.run(
            "stats",
            ("F",),
            f"rul-{dataset}",
            [1],
            [0],
            rul.RULConfig(epochs=1, d=16),
            dataset,
            lifecycle=life,
            out_root=tmp_path / "runs",
            log=lambda *_: None,
            **paths,
        )
        bags = pd.read_csv(out / "fold1-seed0-bags.csv")
        assert np.isfinite(bags.pred_reg).all()
        assert bags.target_kind.eq("rul_operating_seconds_proxy").all()
        assert (bags.y_reg >= 0).all()
        # The anomaly protocol uses the same unit-disjoint folds and validation thresholds.
        sets = {"rms": (anomaly.rms_scores, anomaly.log_rms(store), np.ones(len(store), bool))}
        result = anomaly.milling_protocol(store, sets)
        assert len(result) == 5 and result.status.eq("ok").any()
        untouched = result.threshold.copy()
        outer = protocols.SOURCE_FOLDS[dataset](store.manifest, 1)
        store.manifest.loc[outer.test, "class_key"] = "dulled"
        changed = anomaly.milling_protocol(store, sets)
        assert changed.loc[changed.fold == 1, "threshold"].iloc[0] == untouched.iloc[0]


def test_matwi_raw_timestamps_to_lifecycle(tmp_path):
    from pdmfm.datasets.matwi import MATWI
    from tests.integration.test_readers import write_matwi

    write_matwi(tmp_path)
    table = lifecycle.matwi_table(MATWI(tmp_path))
    assert table.unit.nunique() == 2
    assert table.groupby("unit").rul_s.last().eq(0).all()
    np.testing.assert_allclose(table.duration_s, 9, atol=0.001)
    assert table.rul_s.notna().all()  # missing wear and unknown insert IDs do not remove RUL


def test_public_demo_uses_native_files_without_external_dependencies(tmp_path, monkeypatch):
    import json

    from pdmfm import demo

    monkeypatch.setattr(planb, "BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(rul, "BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path / "original"))
    out = demo.run(tmp_path / "demo")
    metadata = json.loads((out / "demo.json").read_text())
    assert metadata["synthetic"] and not metadata["pretrained_weights"]
    assert (out / "classification/summary.md").is_file()
    assert (out / "rul/summary.md").is_file()
    assert (out / "anomaly/results.csv").is_file()
    import os

    assert os.environ["PDMFM_DATA_ROOT"] == str(tmp_path / "original")
