"""Independent arithmetic and adversarial checks from the November 2026 audit."""

import numpy as np
import pandas as pd
import pytest
import torch

from pdmfm import lifecycle, losses
from pdmfm.compute import InputBank, autocast
from pdmfm.model.inputs import Modality, TaskInputs
from pdmfm.train import selection_score
from pdmfm.utils import run_identity


def test_selection_weights_bags_equally():
    inputs = TaskInputs("toy", pd.DataFrame({"bag": ["a", "b", "b", "b"]}), {}, y_reg=np.zeros(4))
    pred = {"fused_nig": np.array([[2, 1, 2, 1], [0, 1, 2, 1], [0, 1, 2, 1], [0, 1, 2, 1]], float)}
    assert selection_score(inputs, np.arange(4), pred, 0, 1) == pytest.approx(1 - np.sqrt(2))


def test_confirmed_wear_rejects_isolated_and_unconfirmed_dulled_labels():
    assert lifecycle.confirmed_dulled(["sharp", "dulled", "used", "sharp"]) is None
    assert lifecycle.confirmed_dulled(["sharp", "dulled", "used", "dulled", "dulled"]) == 1
    assert lifecycle.confirmed_dulled(["dulled", "dulled"]) is None


def test_lifetime_conversion_by_hand_and_future_target_separation():
    table = pd.DataFrame(
        {
            "unit": ["a"] * 3,
            "bag": ["x", "y", "z"],
            "duration_s": [10.0, 20.0, 30.0],
            "gap_before_s": [0.0, 40.0, 0.0],
        }
    )
    out = lifecycle.targets(table, "recorded_life_end")
    np.testing.assert_allclose(out.operation_end_s, [10, 70, 100])
    np.testing.assert_allclose(out.rul_s, [90, 30, 0])
    table.loc[2, "duration_s"] = 40
    changed = lifecycle.targets(table, "recorded_life_end")
    np.testing.assert_allclose(changed.operation_end_s[:2], out.operation_end_s[:2])
    assert changed.rul_s[0] == 100  # retrospective targets may change with the future


def test_censored_and_post_endpoint_passes_have_no_regression_label():
    table = pd.DataFrame(
        {
            "unit": ["a"] * 4 + ["b"],
            "bag": list("abcde"),
            "duration_s": [10.0] * 5,
            "class_key": ["dulled"] * 3 + ["used", "sharp"],
        }
    )
    out = lifecycle.targets(table, "confirmed_dulled")
    assert out.rul_s[0] == 0 and out.rul_s.iloc[1:].isna().all()
    assert not out.event_observed.iloc[-1]


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_lifecycle_rejects_invalid_durations(bad):
    with pytest.raises(ValueError, match="durations"):
        lifecycle.targets(
            pd.DataFrame({"unit": ["a"], "bag": ["a"], "duration_s": [bad]}), "recorded_life_end"
        )


def test_resumption_requires_full_settings_identity(tmp_path):
    run_identity(tmp_path, {"data": "v1", "precision": "fp32"})
    run_identity(tmp_path, {"data": "v1", "precision": "fp32"})
    with pytest.raises(ValueError, match="other settings"):
        run_identity(tmp_path, {"data": "v2", "precision": "fp32"})


def test_cpu_bank_matches_unstaged_inputs_and_masks_independently():
    m = Modality(
        "force",
        np.arange(24, dtype=np.float32).reshape(4, 2, 3),
        np.zeros(2, int),
        np.zeros(2, int),
        np.array([1, 0, 1, 1], bool),
    )
    inputs = TaskInputs("toy", pd.DataFrame(index=range(4)), {"force": m})
    bank = InputBank(inputs, "cpu")
    np.testing.assert_array_equal(bank.batch(np.array([3, 1]))["force"]["tokens"], m.tokens[[3, 1]])
    assert not bank.batch(np.array([0]), ("force",))["force"]["present"].any()


def test_low_precision_evidential_math_keeps_finite_gradients():
    raw = torch.zeros(4, 4, dtype=torch.bfloat16, requires_grad=True)
    params = losses.nig_parameters(raw)
    assert params.dtype == torch.float32
    losses.nig_loss(params, torch.ones(4)).mean().backward()
    assert torch.isfinite(raw.grad).all()
    with autocast("cpu", "auto"):
        assert (torch.ones(2) + 1).dtype == torch.float32


def test_milling_report_does_not_infer_normalized_rul_from_small_values():
    from pdmfm.report import tasks_of

    table = pd.DataFrame(
        {
            "y_reg": [0.1, 0.8],
            "pred_reg": [0.2, 0.7],
            "target_kind": ["rul_operating_seconds_proxy"] * 2,
        }
    )
    assert tasks_of(table) == ["reg"]


def test_gpu_benchmark_refuses_to_report_cpu_measurements(monkeypatch, tmp_path):
    from pdmfm import benchmark

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        benchmark.run("nonastreda", "stats", ("F",), tmp_path / "gpu.json")
    assert not (tmp_path / "gpu.json").exists()


def test_shared_pretraining_decisions_are_fold_local_and_order_independent(tmp_path):
    import json

    from pdmfm.pretrain import shared_choices

    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, fusion, width in zip(paths, ("concat", "perceiver"), (128, 256), strict=True):
        path.write_text(
            json.dumps(
                {
                    "per_fold": {
                        "1": {"overrides": {"fusion": fusion, "d": width}},
                        "2": {"overrides": {"fusion": "concat", "d": 128}},
                    }
                }
            )
        )
    expected = {1: {"fusion": "late"}, 2: {"fusion": "concat", "d": 128}}
    assert shared_choices(paths) == expected
    assert shared_choices(list(reversed(paths))) == expected
