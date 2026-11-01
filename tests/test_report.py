"""Paired comparisons across runs (pdmfm.report): statistics on known values, alignment of
bag tables, and the report files."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pdmfm import report
from pdmfm import stats as S


def write_run(folder, tables: dict[tuple[int, int], pd.DataFrame]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for (fold, seed), table in tables.items():
        table.assign(fold=fold).to_csv(folder / f"fold{fold}-seed{seed}-bags.csv", index=False)


def class_table(units: int, accuracy: float, seed: int, k: int = 3, bags: int = 20):
    """Bags of ``units`` units; the prediction is right with probability ``accuracy``."""
    rng = np.random.default_rng(seed)
    unit = np.repeat([f"U{i}" for i in range(units)], bags)
    y = rng.integers(0, k, len(unit))
    wrong = (y + rng.integers(1, k, len(unit))) % k
    pred = np.where(rng.random(len(unit)) < accuracy, y, wrong)
    return pd.DataFrame(
        {"bag": [f"b{i}" for i in range(len(unit))], "unit": unit, "y_class": y,
         "pred_class": pred, "majority": 0, "persistence": y, "schedule": wrong}
    )  # fmt: skip


# ----------------------------------------------------------------------------- statistics


def test_holm_adjustment():
    adjusted = report.holm([0.01, 0.04, 0.03, float("nan")])
    np.testing.assert_allclose(adjusted[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adjusted[3])
    assert report.holm([0.6, 0.9]) == [1.0, 1.0]  # capped at 1, monotone


def test_wilcoxon_p_values():
    a = np.arange(10.0)
    assert report.wilcoxon_p(a, a) == 1.0
    assert np.isnan(report.wilcoxon_p(np.array([1.0]), np.array([0.0])))
    assert report.wilcoxon_p(a + 1, a) == pytest.approx(2 / 1024)  # exact, all 10 positive
    assert np.isnan(report.wilcoxon_p(np.array([np.nan, 1.0]), np.array([0.0, 0.0])))


def test_sufficient_statistics_reproduce_the_metrics():
    table = class_table(4, 0.6, 0).rename(columns={"y_class": "y", "pred_class": "pred"})
    units = sorted(table["unit"].unique())
    counts = report.unit_statistics(table, "class", 3, units)
    assert counts.shape == (4, 3, 3) and counts.sum() == len(table)
    pooled = report.metric_from(counts.sum(axis=0), "class")
    assert pooled == pytest.approx(S.macro_f1(table["y"], table["pred"]))
    first = table[table["unit"] == "U0"]
    assert report.metric_from(counts[0], "class") == pytest.approx(
        S.macro_f1(first["y"], first["pred"])
    )
    reg = pd.DataFrame({"unit": ["a", "a", "b"], "y": [1.0, 2.0, 3.0], "pred": [1.0, 4.0, 3.0]})
    sse = report.unit_statistics(reg, "reg", 0, ["a", "b"])
    np.testing.assert_allclose(sse, [[4.0, 2.0], [0.0, 1.0]])
    assert report.metric_from(sse.sum(axis=0), "reg") == pytest.approx(np.sqrt(4 / 3))


def _methods(*tables_per_method):
    return [
        report.Method(name, [t.rename(columns={"y_class": "y", "pred_class": "pred"})
                             .assign(fold=1) for t in tables])
        for name, tables in tables_per_method
    ]  # fmt: skip


def test_a_better_method_is_found_and_a_copy_is_not():
    good = class_table(12, 0.95, 0)
    poor = good.assign(pred_class=class_table(12, 0.4, 0)["pred_class"])
    methods = _methods(("good", [good]), ("poor", [poor]), ("copy", [good]))
    table = report.compare(methods, "class", n=300).set_index("method")
    assert table.loc["poor", "difference_high"] < 0  # interval excludes zero
    assert table.loc["poor", "units_better"] == 0 and table.loc["poor", "wilcoxon_p"] < 0.01
    assert table.loc["copy", "difference_low"] == table.loc["copy", "difference_high"] == 0.0
    assert table.loc["copy", "wilcoxon_p"] == 1.0
    assert table.loc["good", "low"] <= table.loc["good", "estimate"] <= table.loc["good", "high"]
    # Equal bags per unit: the unit average of per-unit macro-F1 is close to the pooled value.
    assert abs(table.loc["good", "unit_average"] - table.loc["good", "estimate"]) < 0.1
    assert table["holm_p"].iloc[1:].notna().all() and np.isnan(table["holm_p"].iloc[0])


def test_seeds_are_averaged_not_counted():
    one = class_table(6, 0.7, 1)
    single = report.compare(_methods(("a", [one]), ("b", [one])), "class", n=50)
    double = report.compare(_methods(("a", [one, one]), ("b", [one])), "class", n=50)
    assert single.iloc[0]["estimate"] == pytest.approx(double.iloc[0]["estimate"])
    assert double.iloc[0]["seeds"] == 2


def test_alignment_keeps_shared_labelled_bags_and_checks_labels():
    a = class_table(3, 0.8, 0)
    b = a.iloc[5:].copy()  # a run that lacks some bags
    a.loc[0, "y_class"] = -1  # an unlabelled bag
    methods = report.methods_for(
        {"a": {0: a.assign(fold=1)}, "b": {0: b.assign(fold=1)}}, "class", "a"
    )
    aligned, info = report.align(methods, "class")
    assert info["shared_bags"] == len(a) - 5
    assert all(len(t) == len(a) - 5 for m in aligned for t in m.tables)
    assert [m.name for m in aligned] == ["a", "b", "majority", "persistence", "schedule"]
    bad = b.assign(y_class=(b["y_class"] + 1) % 3)
    with pytest.raises(ValueError, match="labels differ"):
        report.align(
            report.methods_for({"a": {0: a.assign(fold=1)}, "b": {0: bad.assign(fold=1)}},
                               "class", "a"), "class",
        )  # fmt: skip


# ----------------------------------------------------------------------------- the report


def test_run_writes_the_tables_for_classes(tmp_path):
    # Same labels in both seeds (same draw of y), different predictions.
    model = {(f, s): class_table(4, 0.9 - 0.1 * s, 10 * f) for f in (1, 2) for s in (0, 1)}
    other = {(f, 0): t.assign(pred_class=0) for (f, s), t in model.items() if s == 0}
    write_run(tmp_path / "model", model)
    write_run(tmp_path / "other", other)
    with pytest.raises(FileNotFoundError):
        report.load_run(tmp_path / "missing")
    tables = report.run(
        "check", {"model": tmp_path / "model", "other": tmp_path / "other"},
        n=100, out_root=tmp_path / "out", log=lambda *_: None,
    )  # fmt: skip
    assert set(tables) == {"class"}
    folder = tmp_path / "out" / "check"
    assert (folder / "class.csv").is_file()
    text = (folder / "report.md").read_text(encoding="utf-8")
    assert "Reference: **model**" in text and "| other | 1 |" in text
    meta = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert meta["alignment"]["class"]["units"] == 4


def test_rul_runs_get_the_after_fpt_table(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(5):
        life = np.linspace(1.0, 0.0, 30)
        life[:10] = 1.0
        for i, y in enumerate(life):
            rows.append({"bag": f"B{unit}/{i}", "unit": f"B{unit}", "y_reg": y,
                         "pred_reg": y + rng.normal(0, 0.1), "train_mean": 0.8,
                         "schedule_reg": 0.7})  # fmt: skip
    table = pd.DataFrame(rows)
    write_run(tmp_path / "rul", {(1, 0): table})
    write_run(tmp_path / "flat", {(1, 0): table.assign(pred_reg=0.8)})
    tables = report.run(
        "rul", {"rul": tmp_path / "rul", "flat": tmp_path / "flat"}, n=100,
        out_root=tmp_path / "out", log=lambda *_: None,
    )  # fmt: skip
    assert set(tables) == {"reg", "reg_post"}
    post = tables["reg_post"].set_index("method")
    assert post.loc["flat", "difference"] > 0  # the constant is worse after the FPT
    assert set(post.index) == {"rul", "flat", "training mean", "schedule"}
    # Per unit, the constant prediction's RMSE is the unit's spread around 0.8.
    assert post.loc["flat", "unit_average"] > post.loc["rul", "unit_average"]
    with pytest.raises(ValueError, match="not among the runs"):
        report.run("x", {"rul": tmp_path / "rul"}, reference="nope", out_root=tmp_path)


def test_command_line(tmp_path, capsys):
    write_run(tmp_path / "a", {(1, 0): class_table(3, 0.9, 0)})
    write_run(tmp_path / "b", {(1, 0): class_table(3, 0.5, 0)})
    report.main(
        ["--name", "cli", "--run", f"a={tmp_path / 'a'}", "--run", f"b={tmp_path / 'b'}",
         "--replicates", "50", "--no-builtins", "--out", str(tmp_path / "out")]
    )  # fmt: skip
    printed = capsys.readouterr().out
    assert "Paired comparison: cli" in printed and "majority" not in printed
    with pytest.raises(SystemExit):
        report.main(["--name", "x", "--run", "nofolder"])
