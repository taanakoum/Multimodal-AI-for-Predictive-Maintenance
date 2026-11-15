"""Native milling adapters and lifecycle targets, with tiny real-format files."""

import numpy as np
import pandas as pd
import pytest

from pdmfm import lifecycle
from pdmfm.datasets.matwi import MATWI, _seconds, acquisition_rate, sensor_clock
from tests.integration.test_readers import write_matwi


def test_matwi_adhesion_and_unidentified_inserts_are_not_flank_targets(tmp_path):
    write_matwi(tmp_path)
    table = pd.read_csv(tmp_path / "labels.csv")
    table.loc[0, "type"] = "adhesion"
    table.to_csv(tmp_path / "labels.csv", index=False)
    reader = MATWI(tmp_path)
    infos = reader.index()
    assert infos[0].label is None and np.isnan(infos[0].meta["wear_um"])
    assert infos[0].meta["annotated_wear_um"] == 30
    assert all(i.label is None for i in infos if i.unit == "Set02")
    life = lifecycle.matwi_table(reader)
    assert len(life) == 8 and life.rul_s.notna().all()
    assert life.loc[life.groupby("unit").head(1).index, "previous_photo"].eq("").all()


def test_matwi_clock_uses_acquisition_not_filename_and_rejects_backwards(tmp_path):
    path = tmp_path / "unrelated-name.csv"
    path.write_text(
        "0,0,0,0,0,2023-01-01 00:00:00\n"
        "0,0,0,0,0,2023-01-01 00:00:01\n"
        "0,0,0,0,0,2023-01-01 00:00:02\n"
    )
    _, duration = sensor_clock(path)
    assert duration == 3
    path.write_text("0,0,0,0,0,2023-01-01 00:00:02\n0,0,0,0,0,2023-01-01 00:00:01\n")
    with pytest.raises(ValueError, match="timestamps"):
        sensor_clock(path)


def test_matwi_clock_is_independent_of_pandas_datetime_resolution():
    stamps = pd.Series(pd.date_range("2023-01-01", periods=5, freq="1ms"))
    seconds = _seconds(stamps.astype("datetime64[us]"))
    np.testing.assert_allclose(seconds, 1672531200 + np.arange(5) / 1000, rtol=0, atol=1e-6)
    assert seconds[0] > 1e9
    assert acquisition_rate(stamps) == pytest.approx(1000)


@pytest.mark.parametrize("offsets", [[0, 1, 1, 3], [0, 1, 0, 3], [0, 1, 2, 9]])
def test_matwi_full_clock_rejects_interior_corruption(offsets):
    stamps = pd.Series(pd.Timestamp("2023-01-01") + pd.to_timedelta(offsets, unit="ms"))
    with pytest.raises(ValueError, match="acquisition timestamps"):
        acquisition_rate(stamps)


def test_gap_durations_use_past_and_duplicates_never_move_signals_backwards():
    from pdmfm.datasets.nonastreda import ForceRecord

    records = {
        f"T1R{r}B{b}": ForceRecord(
            f"T1R{r}B{b}", "T1", r, b, np.ones((1000 * seconds, 3), np.float32), 2
        )
        for r, b, seconds in [(1, 1, 1), (1, 2, 1), (3, 1, 2), (3, 2, 2)]
    }

    class Reader:
        def all_records(self):
            return records

        def _load_records(self):
            return {k: v for k, v in records.items() if k != "T1R1B2"}

        def duplicate_groups(self):
            return [["T1R1B1", "T1R1B2"]]

    table = lifecycle.nonastreda_table(Reader())
    assert table.gap_before_s.tolist() == [0, 0, 4, 0]
    assert table.force_present.tolist() == [False, True, True, True]
    assert table.source_bag[:2].tolist() == ["T1R1B1", "T1R1B1"]
    recorded = lifecycle.nonastreda_table(Reader(), "recorded")
    assert recorded.gap_before_s.eq(0).all()


def test_lifecycle_export_has_content_identity(tmp_path):
    write_matwi(tmp_path)
    table, folder = lifecycle.build("matwi", reader=MATWI(tmp_path), out=tmp_path / "life")
    assert (folder / "meta.json").is_file()
    loaded = pd.read_csv(folder / "manifest.csv")
    np.testing.assert_allclose(loaded.rul_s, table.rul_s)
    assert table.attrs["signature"]["physical_failure_verified"] is False
