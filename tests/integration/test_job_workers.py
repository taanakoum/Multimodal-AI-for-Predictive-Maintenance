"""Real subprocess isolation, memory/thread shares, failure propagation and resumption."""

import json
import sys

import pandas as pd
import pytest

from pdmfm import jobs
from pdmfm.utils import stable_hash


def test_workers_preserve_gpu_mask_and_resume_only_successes(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    target = tmp_path / "probe.py"
    target.write_text(
        "import os,sys,json\nfrom pathlib import Path\n"
        "path=Path(sys.argv[1])\n"
        "data={k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',"
        "'PDMFM_GPU_MEMORY_FRACTION']}\n"
        "previous=json.loads(path.read_text()) if path.exists() else {}\n"
        "data['attempts']=previous.get('attempts',0)+1\n"
        "path.write_text(json.dumps(data))\n"
        "sys.exit(int(sys.argv[2]))\n"
    )
    path = tmp_path / "manifest.json"
    tasks = [
        {
            "id": f"fold{i}",
            "argv": [sys.executable, str(target), str(tmp_path / f"probe{i}.json"), str(i)],
        }
        for i in (0, 1)
    ]
    path.write_text(json.dumps({"tasks": tasks}))
    assert jobs.launch(path, workers=2, require_cuda=False) == 1
    assert jobs.launch(path, workers=2, require_cuda=False) == 1
    first = json.loads((tmp_path / "probe0.json").read_text())
    second = json.loads((tmp_path / "probe1.json").read_text())
    assert first["CUDA_VISIBLE_DEVICES"] == "3"
    assert first["OMP_NUM_THREADS"] == "4"
    assert float(first["PDMFM_GPU_MEMORY_FRACTION"]) == 0.45
    assert first["attempts"] == 1 and second["attempts"] == 2


def test_completed_isolated_folds_collect_into_one_pooled_report(tmp_path, monkeypatch):
    from pdmfm import planb
    from pdmfm.train import TrainConfig
    from tests.support.synthetic import build_planb_root

    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(planb, "BOOTSTRAP_REPLICATES", 20)
    build_planb_root(tmp_path, datasets=("nonastreda",))
    manifest = jobs.make("planb", "collect", ["nonastreda"], list(range(1, 6)), encoder="stats")
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(manifest))
    path.with_suffix("").mkdir()
    out = tmp_path / "collected"
    with pytest.raises(ValueError, match="incomplete"):
        jobs.collect(path, "nonastreda", out, run_root=tmp_path / "runs")
    for task in manifest["tasks"]:
        planb.run(
            "nonastreda",
            "stats",
            ("F",),
            task["id"],
            [task["fold"]],
            [0],
            TrainConfig(epochs=1, d=16),
            photos=False,
            out_root=tmp_path / "runs/planb",
            log=lambda *_: None,
        )
        state = {"identity": stable_hash({"manifest": manifest, "task": task}), "returncode": 0}
        (path.with_suffix("") / f"{task['id']}.json").write_text(json.dumps(state))
    jobs.collect(path, "nonastreda", out, run_root=tmp_path / "runs")
    assert len(list(out.glob("fold*-seed0.json"))) == 5
    assert len(list(out.glob("fold*-seed0-bags.csv"))) == 5
    assert (out / "pooled.csv").is_file() and (out / "collection.json").is_file()
    bags = pd.concat([pd.read_csv(p) for p in out.glob("*-bags.csv")])
    assert bags.unit.nunique() == 10 and not bags.bag.duplicated().any()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "summary.md").write_text("another experiment")
    with pytest.raises(ValueError, match="fresh directory"):
        jobs.collect(path, "nonastreda", unrelated, run_root=tmp_path / "runs")
