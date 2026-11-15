from collections import Counter

import pytest

from pdmfm import sources
from pdmfm.sources import SOURCES, get_source


def test_registry_entries_are_complete():
    for name, source in SOURCES.items():
        assert source.name == name
        assert source.role in ("source", "target", "source+target", "optional")
        assert source.licence and source.citation and source.homepage.startswith("http")
        if source.kind in ("direct", "gdrive"):
            assert source.files, name
            assert len({f.relpath for f in source.files}) == len(source.files)
        if source.kind == "mendeley":
            assert {"dataset_id", "version"} <= set(source.params)


def test_cwru_is_the_ten_class_four_load_set():
    classes = Counter((kind, size) for kind, size, _ in sources.CWRU_FILES.values())
    assert len(classes) == 10
    assert set(classes.values()) == {4}
    loads = Counter(load for *_, load in sources.CWRU_FILES.values())
    assert loads == {0: 10, 1: 10, 2: 10, 3: 10}


def test_paderborn_split_covers_all_bearings_once():
    everything = sources.PADERBORN_HEALTHY + sources.PADERBORN_ARTIFICIAL + sources.PADERBORN_REAL
    assert len(everything) == len(set(everything)) == 32
    assert set(sources.PADERBORN_SOURCE).isdisjoint(sources.PADERBORN_TARGET)
    assert set(sources.PADERBORN_SOURCE) | set(sources.PADERBORN_TARGET) == set(everything)
    # Real damage never enters the source pool.
    assert set(sources.PADERBORN_REAL) <= set(sources.PADERBORN_TARGET)


def test_unknown_dataset_names_the_known_ones():
    with pytest.raises(KeyError, match="kaist"):
        get_source("no-such-dataset")
