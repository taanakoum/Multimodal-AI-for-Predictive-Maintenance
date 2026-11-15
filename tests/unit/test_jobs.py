"""Production scheduling policy: one seed, isolated folds, valid training CLIs."""

import importlib
import json

import pytest

from pdmfm import jobs


@pytest.mark.parametrize("stage", jobs.STAGES)
def test_each_generated_task_is_accepted_by_its_real_cli(stage, monkeypatch):
    manifest = jobs.make(stage, "audit", ["nonastreda", "matwi"], [1, 2], epochs=1)
    assert manifest["seed"] == 0
    assert len({t["id"] for t in manifest["tasks"]}) == len(manifest["tasks"])
    assert len(manifest["tasks"]) == (2 if stage == "pretrain" else 4)
    module = importlib.import_module(f"pdmfm.{stage}")
    for task in manifest["tasks"]:
        assert "fold" in task["id"]
        if stage not in ("gate2", "tune"):
            pos = task["argv"].index("--seeds")
            assert task["argv"][pos + 1 : pos + 2] == ["0"]
    # Validate actual arguments, without reaching resource checks or training.
    import argparse

    original = argparse.ArgumentParser.parse_args
    parsed = []

    def stop(parser, argv=None, namespace=None):
        parsed.append(original(parser, argv, namespace))
        raise InterruptedError("parsed")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    for task in manifest["tasks"]:
        with pytest.raises(InterruptedError):
            module.main(task["argv"][3:])
    assert len(parsed) == len(manifest["tasks"])
    if stage == "pretrain":
        assert parsed[0].datasets == ["cwru", "nonastreda", "matwi"]


def test_incomplete_gate1_and_unsupported_rul_are_rejected(tmp_path):
    report = tmp_path / "gate1.json"
    report.write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        jobs.make("planb", "audit", ["matwi"], [1], gate1=report)
    with pytest.raises(ValueError, match="run-to-failure"):
        jobs.make("rul", "audit", ["paderborn"], [1])
    with pytest.raises(ValueError, match="unique"):
        jobs.make("planb", "audit", ["matwi"], [1, 1])


def test_fold_specific_gate1_and_initialization_paths(tmp_path):
    report = tmp_path / "gate1.json"
    report.write_text(
        json.dumps({"complete": True, "per_fold_winner": {"1": "mantis:F", "2": "moment:T+F"}})
    )
    template = str(tmp_path / "pre-shared-fold{fold}" / "fold{fold}-seed{seed}" / "best.pt")
    for fold in (1, 2):
        checkpoint = tmp_path / f"pre-shared-fold{fold}" / f"fold{fold}-seed0" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
    tasks = jobs.make(
        "planb",
        "final",
        ["matwi"],
        [1, 2],
        gate1=report,
        init=template,
    )["tasks"]
    assert tasks[0]["argv"][tasks[0]["argv"].index("--encoder") + 1] == "mantis"
    assert tasks[1]["argv"][tasks[1]["argv"].index("--encoder") + 1] == "moment"
    assert tasks[1]["argv"][-1] == template.format(fold=2, seed=0)
